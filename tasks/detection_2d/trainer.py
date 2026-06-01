"""
trainer.py
Local SGD cho bài toán Object Detection (Kịch bản 2 & 3).

Tier 1 (AUV/AUV) chỉ chạy local SGD thuần (DetectionTrainer), KHÔNG dùng KD.
KD (Knowledge Distillation với Teacher YOLO12l) được di chuyển lên Tier 3 (Gateway).

Payload truyền đi: LoRA adapters + cv3.x.2 output conv, nén INT8.
"""
import torch
import copy
import logging
import numpy as np
from ultralytics.models.yolo.detect import DetectionTrainer
from config.settings import fed_cfg


def _fl_prepare_model_for_train(model: torch.nn.Module, device) -> torch.nn.Module:
    """
    Đưa DetectionModel lên device và khởi tạo lại criterion.
    Ultralytics gắn v8DetectionLoss trên model.criterion; buffer `proj` lấy device
    lúc init — nếu model/criterion lệch device sẽ lỗi matmul CUDA vs CPU.
    """
    from ultralytics.utils.torch_utils import unwrap_model

    if isinstance(device, str):
        device = torch.device(device)
    model = model.to(device)
    unwrapped = unwrap_model(model)
    if hasattr(unwrapped, "init_criterion"):
        unwrapped.criterion = unwrapped.init_criterion()
    elif getattr(unwrapped, "criterion", None) is not None:
        crit = unwrapped.criterion
        crit.to(device)
        if hasattr(crit, "proj"):
            crit.proj = crit.proj.to(device)
    return model


def _count_inference_tensors(module: torch.nn.Module) -> tuple[int, list[str]]:
    """Count tensors carrying inference-mode flag and return a few sample names."""
    count = 0
    sample_names = []

    for name, param in module.named_parameters():
        if hasattr(param, 'is_inference') and param.is_inference():
            count += 1
            if len(sample_names) < 5:
                sample_names.append(f"param:{name}")

    for name, buf in module.named_buffers():
        if hasattr(buf, 'is_inference') and buf.is_inference():
            count += 1
            if len(sample_names) < 5:
                sample_names.append(f"buffer:{name}")

    return count, sample_names

class CustomDetectionTrainer(DetectionTrainer):
    def __init__(self, overrides=None, _callbacks=None, student_wrapper=None,
                 cached_optimizer_state: dict = None):
        super().__init__(overrides=overrides, _callbacks=_callbacks)
        self.student_wrapper = student_wrapper
        # State dict bởi tên tham số từ round trước (None = round đầu tiên)
        self.cached_optimizer_state = cached_optimizer_state
        self._fl_injected_model = None

    def _setup_train(self):
        from ultralytics.utils import LOGGER

        # Tắt triệt để cảnh báo của YOLO (trong lúc _setup_train, YOLO sẽ quét và lật lại requires_grad)
        original_warning = LOGGER.warning
        LOGGER.warning = lambda *args, **kwargs: None

        injected = self._fl_injected_model
        try:
            super()._setup_train()
        finally:
            LOGGER.warning = original_warning

        # Ultralytics _setup_train có thể thay self.model; khôi phục FL model + criterion đồng device.
        if injected is not None:
            self.model = _fl_prepare_model_for_train(injected, self.device)
        else:
            self.model = _fl_prepare_model_for_train(self.model, self.device)

        # Inject optimizer state từ round trước (keyed by param name) để simulate
        # continuous training. Phải chạy SAU super()._setup_train() vì lúc đó
        # self.optimizer mới được build xong.
        
        # [FIX] Force accumulate = 1 so that small auvs (e.g. 1 batch) 
        # still trigger optimizer.step() and populate self.optimizer.state.
        self.accumulate = 1
        
        if self.cached_optimizer_state:
            print(f"[OptimState] Warm-start cache detected: {len(self.cached_optimizer_state)} tensors.")
            self._restore_optimizer_state(self.cached_optimizer_state)
        else:
            print("[OptimState] Cold-start optimizer (no cache from previous FL round).")

    def _restore_optimizer_state(self, named_state: dict):
        """
        Restore AdamW exp_avg / exp_avg_sq từ round trước vào optimizer hiện tại.
        Key bằng tên tham số (str) thay vì tensor id để vượt qua việc model object
        bị tạo mới mỗi round.
        Các tham số không có trong cache (ví dụ round đầu) được bỏ qua.
        """
        # Map: id(tensor) -> param_name, chỉ cho các param đang được train
        id_to_name = {id(p): n for n, p in self.model.named_parameters()}

        # Flatten tất cả params trong optimizer theo thứ tự
        all_params = [p for g in self.optimizer.param_groups for p in g['params']]

        restored = 0
        for param in all_params:
            name = id_to_name.get(id(param))
            if name is None or name not in named_state:
                continue
            cached = named_state[name]
            self.optimizer.state[param] = {
                k: v.clone().to(param.device) if isinstance(v, torch.Tensor) else v
                for k, v in cached.items()
            }
            restored += 1

        if restored:
            print(f"[OptimState] Restored {restored} param states from previous round.")

    def get_named_optimizer_state(self) -> dict:
        """
        Extract trạng thái optimizer hiện tại thành dict {param_name: state},
        lưu trên CPU để tái sử dụng round sau.
        """
        id_to_name = {id(p): n for n, p in self.model.named_parameters()}
        print(f"[DEBUG OptimState] Model params: {len(id_to_name)}, Optimizer state items: {len(self.optimizer.state)}")
        
        named_state = {}
        for param, state in self.optimizer.state.items():
            name = id_to_name.get(id(param))
            if name is None:
                continue
            named_state[name] = {
                k: v.cpu().clone() if isinstance(v, torch.Tensor) else v
                for k, v in state.items()
            }
        print(f"[DEBUG OptimState] Extracted {len(named_state)} tensors")
        return named_state

    def validate(self):
        """
        Bỏ qua validate nếu args.val = False (khi chạy FL Local SGD).
        Nếu args.val = True (khi train Teacher), kiểm tra val_period để chạy định kỳ.
        """
        if not self.args.val:
            return {}, 0.0
            
        val_period = getattr(self, 'val_period', 1)
        # Chỉ evaluate nếu epoch hiện tại chia hết cho val_period, HOẶC là epoch cuối cùng
        # Chú ý: self.epoch bắt đầu từ 0
        if (self.epoch + 1) % val_period != 0 and self.epoch != self.epochs - 1:
            return {}, 0.0
            
        return super().validate()

    def final_eval(self):
        """Bỏ qua bước Validate dư thừa ở cuối quá trình Local SGD để tiết kiệm thời gian."""
        if not self.args.val:
            from ultralytics.utils.torch_utils import strip_optimizer
            model = self.best if self.best.exists() else None
            if self.last.exists():
                strip_optimizer(self.last)
            if model:
                strip_optimizer(self.best)
                self.run_callbacks("on_fit_epoch_end")
        else:
            super().final_eval()

    def build_optimizer(self, model, name='auto', lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        optimizer = super().build_optimizer(model, name, lr, momentum, decay, iterations)

        if self.student_wrapper and not self.student_wrapper.full_param:
            payload_keys = set(self.student_wrapper.trainable_state_dict().keys())
            for k, v in model.named_parameters():
                if k in payload_keys:
                    v.requires_grad = True
                else:
                    v.requires_grad = False

        # ---------------------------------------------------------------
        # Differential LR: Head params học nhanh hơn LoRA (head_lr_multiplier lần).
        # Mặc định = 1.0 → không ảnh hưởng gì (giữ nguyên hành vi FL cũ).
        # Set trainer.head_lr_multiplier = 5.0 để Head học với lr * 5.
        # ---------------------------------------------------------------
        head_lr_multiplier = getattr(self, 'head_lr_multiplier', 1.0)
        if head_lr_multiplier != 1.0:
            id_to_name = {id(p): n for n, p in model.named_parameters()}
            # Detect head nằm ở layer cuối (yolo12n: model.22, YOLO12l: model.21)
            head_patterns = ('model.21.', 'model.22.', 'model.23.')
            new_groups = []
            for group in optimizer.param_groups:
                head_p  = [p for p in group['params']
                           if any(h in id_to_name.get(id(p), '') for h in head_patterns)
                           and 'lora_' not in id_to_name.get(id(p), '')]
                head_p_ids = set(id(x) for x in head_p)
                other_p = [p for p in group['params'] if id(p) not in head_p_ids]
                if other_p:
                    g = {k: v for k, v in group.items() if k != 'params'}
                    g['params'] = other_p
                    new_groups.append(g)
                if head_p:
                    g = {k: v for k, v in group.items() if k != 'params'}
                    g['params'] = head_p
                    g['lr'] = group.get('lr', lr) * head_lr_multiplier
                    new_groups.append(g)
            optimizer.param_groups = new_groups
            print(f"[DiffLR] Head LR boosted ×{head_lr_multiplier} → "
                  f"LoRA lr={lr:.2e} | Head lr={lr * head_lr_multiplier:.2e}")

        # Loại bỏ các parameter đã bị đóng băng (requires_grad=False) khỏi optimizer param_groups
        real_trained_count = 0
        for group in optimizer.param_groups:
            group['params'] = [p for p in group['params'] if p.requires_grad]
            real_trained_count += len(group['params'])

        print(f"\n[CustomDetectionTrainer] Đã lọc optimizer! Cố định Backbone, CHỈ CÒN {real_trained_count} tensors được học.")

        return optimizer


    def optimizer_step(self):
        # Apply Proximal term to gradients before step (FedProx)
        if getattr(self, 'fedprox_mu', 0.0) > 0.0 and getattr(self, 'global_weights', None) is not None:
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None and name in self.global_weights:
                    prox_term = param.data - self.global_weights[name].to(param.device)
                    param.grad.data.add_(prox_term, alpha=self.fedprox_mu)

        super().optimizer_step()


def local_sgd_od(
    student_model,
    auv_yaml: str,
    auv_id: int,
    epochs: int = 2,
    batch_size: int = 16,
    lr: float = 0.01,
    device: str = "cpu",
    fedprox_mu: float = 0.0,
    global_weights: dict = None,
    local_teacher = None,
    cached_optimizer_state: dict = None,
) -> tuple:
    """
    Thực hiện Local SGD cho OD tại AUV (Tier 1).
    KHÔNG sử dụng KD — Teacher chỉ chạy tại Gateway (Tier 3).

    student_model          : tasks.detection_2d.models.yolo_wrapper.StudentModel
    auv_yaml            : đường dẫn data.yaml của auv
    cached_optimizer_state : dict {param_name: {exp_avg, exp_avg_sq, step}} từ round trước
                             None → optimizer bắt đầu lạnh (round đầu tiên).

    Returns:
        (new_state, delta_norm, train_loss, new_optimizer_state)
        new_optimizer_state : dict cần lưu vào AUVWorker cho round tiếp theo.
    """
    has_optim_cache = cached_optimizer_state is not None and len(cached_optimizer_state) > 0
    print(
        f"[LocalSGD][AUV {auv_id}] lr0={lr:.8f}, epochs={epochs}, "
        f"optimizer_cache={'ON' if has_optim_cache else 'OFF'}"
    )

    student_infer_before, student_names_before = _count_inference_tensors(student_model.yolo.model)
    if student_infer_before > 0:
        print(
            f"[InferenceCheck][AUV {auv_id}] Student has {student_infer_before} inference tensors before strip. "
            f"Samples: {student_names_before}"
        )

    # 1. Snapshot trạng thái trước khi train
    student_model.strip_inference_tensors()

    student_infer_after, student_names_after = _count_inference_tensors(student_model.yolo.model)
    if student_infer_after > 0:
        raise RuntimeError(
            f"[InferenceCheck][AUV {auv_id}] Student still has {student_infer_after} inference tensors "
            f"after strip. Samples: {student_names_after}"
        )

    if local_teacher is not None and hasattr(local_teacher, 'yolo'):
        teacher_infer_before, teacher_names_before = _count_inference_tensors(local_teacher.yolo.model)
        if teacher_infer_before > 0:
            print(
                f"[InferenceCheck][AUV {auv_id}] Teacher has {teacher_infer_before} inference tensors before strip. "
                f"Samples: {teacher_names_before}"
            )

        # Teacher có thể đã đi qua đường đánh giá trước đó.
        for p in local_teacher.yolo.model.parameters():
            p.data = p.data.clone().detach()
        for b in local_teacher.yolo.model.buffers():
            b.data = b.data.clone().detach()

        teacher_infer_after, teacher_names_after = _count_inference_tensors(local_teacher.yolo.model)
        if teacher_infer_after > 0:
            raise RuntimeError(
                f"[InferenceCheck][AUV {auv_id}] Teacher still has {teacher_infer_after} inference tensors "
                f"after strip. Samples: {teacher_names_after}"
            )

    state_before = copy.deepcopy(student_model.trainable_state_dict())

    # 2. Chuẩn bị overrides cho Ultralytics Trainer
    overrides = {
        'model': "yolo12n.pt", # Dummy, will be overwritten by _fl_injected_model
        'data': auv_yaml,
        'cache': getattr(fed_cfg, 'CACHE_DATASET', True),
        'epochs': epochs,
        'batch': batch_size,
        'workers': getattr(fed_cfg, 'DATALOADER_WORKERS', 4),
        'close_mosaic': 0,
        'lr0': lr,
        'optimizer': 'AdamW',
        'warmup_epochs': 0.0,
        'warmup_bias_lr': lr,  # Tránh nhảy lr=0.1 mặc định của YOLO gây loss nổ
        'warmup_momentum': 0.937,
        'lrf': 1.0,
        'cos_lr': False,
        'device': device,
        'amp': False,  # Vô hiệu hóa FP16 để tránh overflow (Inf/NaN) khi train LoRA/KD
        'project': 'runs/fl_auvs',
        'name': f'auv_{auv_id}',
        'exist_ok': True,
        'verbose': False,  # Ngăn YOLO in bảng kiến trúc
        'save': False,
        'val': False,
        'plots': False,    # Vô hiệu hóa tính năng vẽ biểu đồ để tránh lỗi Plotting error do bỏ qua val
        'workers': 0,
    }

    # 3. Khởi tạo Trainer phù hợp
    if local_teacher is not None:
        from tasks.detection_2d.knowledge_compression.knowledge_distillation import KDDetectionTrainer
        trainer = KDDetectionTrainer(overrides=overrides)
        trainer.student_wrapper = student_model
        trainer.set_teacher(local_teacher.yolo.model)
        trainer.kd_lambda = 1.0  # Hoặc trọng số tuỳ chỉnh
        trainer._fl_injected_model = student_model.yolo.model
        # KDDetectionTrainer không hỗ trợ cached_optimizer_state (không cần thiết cho FedKD)
    else:
        trainer = CustomDetectionTrainer(
            overrides=overrides,
            student_wrapper=student_model,
            cached_optimizer_state=cached_optimizer_state,
        )

    trainer._fl_injected_model = student_model.yolo.model
    trainer.model = student_model.yolo.model
    trainer.fedprox_mu = fedprox_mu
    trainer.global_weights = global_weights
    # Diff LR: LoRA/Backbone dùng lr0=2e-4, Head học nhanh hơn 5× → 1e-3
    trainer.head_lr_multiplier = 5.0

    # HẠN ĐỊNH: Xác định các keys sẽ được truyền qua mạng (LoRA + Head) và coi
    # toàn bộ phần còn lại là "đóng băng"; sử dụng `trainable_state_dict()`
    # để tránh lệ thuộc vào trạng thái `requires_grad` do trainer có thể thay đổi.
    payload_keys = set(student_model.trainable_state_dict().keys())
    frozen_weights_before = {}
    for k, v in student_model.yolo.model.state_dict().items():
        if k not in payload_keys:
            frozen_weights_before[k] = v.clone().detach()

    # Chạy train
    trainer.train()

    # 3b. Extract optimizer state ngay sau khi train xong (trước khi trainer bị xóa)
    # Chỉ CustomDetectionTrainer mới hỗ trợ get_named_optimizer_state()
    new_optimizer_state = None
    if isinstance(trainer, CustomDetectionTrainer):
        try:
            new_optimizer_state = trainer.get_named_optimizer_state()
            if new_optimizer_state is not None:
                print(f"[OptimState] Saved {len(new_optimizer_state)} tensors for next FL round.")
        except Exception as e:
            print(f"[OptimState] Không thể extract optimizer state: {e}")

    # Đảm bảo không có bất kỳ trọng số nào ngoài LoRA và head bị thay đổi
    if not student_model.full_param:
        for k, v in student_model.yolo.model.named_parameters():
            if k in frozen_weights_before:
                diff = torch.abs(frozen_weights_before[k].to(v.device) - v).max().item()
                if diff > 1e-6:
                    raise RuntimeError(f"CƠ CHẾ NGẦM PHÁT HIỆN: Lớp '{k}' dự kiến bị đóng băng nhưng đã thay đổi (max diff: {diff})!")

    # 4. Lấy state sau khi train
    state_after = student_model.trainable_state_dict()

    # 5. Tính delta norm (cho Lazy Communication Filter — Eq. 40)
    delta_norm = 0.0
    for k in state_before:
        if k in state_after:
            diff = state_after[k].float() - state_before[k].float()
            delta_norm += torch.sum(diff ** 2).item()

    # 6. Đọc training loss thực tế từ file results.csv của YOLO
    import pandas as pd
    from pathlib import Path
    train_loss = 0.0
    try:
        csv_path = Path(trainer.save_dir) / 'results.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            last_row = df.iloc[-1]
            train_loss = float(last_row.get('train/box_loss', 0.0)) + \
                         float(last_row.get('train/cls_loss', 0.0)) + \
                         float(last_row.get('train/dfl_loss', 0.0))
    except Exception as e:
        print(f"[Trainer] Không thể đọc results.csv để lấy loss: {e}")

    return state_after, delta_norm, train_loss, new_optimizer_state


def evaluate_od(student_model, test_yaml: str, device: str = "cpu") -> dict:
    """
    Đánh giá mAP@0.5:0.95 và mAP@0.5 của Student model trên tập test.
    Returns: dict chứa các metrics
    """
    import copy
    # [CRITICAL FIX] Lưu lại kiến trúc mạng gốc chưa bị Fuse (gộp BatchNorm)
    unfused_model = copy.deepcopy(student_model.yolo.model)
    
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    
    results = student_model.yolo.val(
        data=test_yaml,
        device=device,
        verbose=False,
        split='val',
        half=False,  # [FIX] Ngăn cast model sang FP16 in-place
        workers=0,   # [CRITICAL FIX] Tránh đóng băng multiprocessing/SHM
        batch=16     # Giới hạn batch size để tránh tràn VRAM
    )
    
    # Khôi phục lại mạng chưa Fuse cho các vòng FL tiếp theo!
    student_model.yolo.model = unfused_model
    torch.cuda.empty_cache()
    gc.collect()
    
    # Lấy precision và recall (mean)
    mp = float(np.mean(results.box.mp)) if hasattr(results.box, 'mp') else 0.0
    mr = float(np.mean(results.box.mr)) if hasattr(results.box, 'mr') else 0.0
    
    return {
        'mAP50-95': float(results.box.map),
        'mAP50': float(results.box.map50),
        'Prec': mp,
        'Rec': mr
    }


def evaluate_od_on_auv_train(student_model, auv_yaml: str, device: str = "cpu") -> dict:
    """
    Đánh giá Student model trên chính tập TRAIN của auv (split='train').
    Dùng để kiểm tra xem auv có thực sự cải thiện sau mỗi vòng FL.
    Dùng YOLO built-in val với split='train' — không ảnh hưởng tập val/test.

    Returns: dict chứa mAP50-95, mAP50, Prec, Rec trên train split của auv đó.
    """
    try:
        import copy
        unfused_model = copy.deepcopy(student_model.yolo.model)
        
        import gc
        torch.cuda.empty_cache()
        gc.collect()
        
        results = student_model.yolo.val(
            data=auv_yaml,
            device=device,
            verbose=False,
            split='train',  # Eval ngay trên tập train của auv
            half=False,
            workers=0,
            batch=16
        )
        
        student_model.yolo.model = unfused_model
        torch.cuda.empty_cache()
        gc.collect()
        mp = float(np.mean(results.box.mp)) if hasattr(results.box, 'mp') else 0.0
        mr = float(np.mean(results.box.mr)) if hasattr(results.box, 'mr') else 0.0
        return {
            'local_mAP50-95': float(results.box.map),
            'local_mAP50':    float(results.box.map50),
            'local_Prec':     mp,
            'local_Rec':      mr,
        }
    except Exception as e:
        print(f"[evaluate_od_on_auv_train] Lỗi khi eval trên train split: {e}")
        return {
            'local_mAP50-95': 0.0,
            'local_mAP50':    0.0,
            'local_Prec':     0.0,
            'local_Rec':      0.0,
        }

