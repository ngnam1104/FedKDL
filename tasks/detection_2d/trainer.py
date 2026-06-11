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
    Đưa DetectionModel lên device và đảm bảo criterion.proj cùng device.
    Ultralytics gắn v8DetectionLoss trên model.criterion; buffer `proj` lấy device
    lúc init — nếu model/criterion lệch device sẽ lỗi matmul CUDA vs CPU.
    
    KHÔNG gọi init_criterion() ở đây vì model.hyp có thể chưa được set
    thành SimpleNamespace (gây AttributeError: 'dict' object has no attribute 'box').
    """
    if isinstance(device, str):
        device = torch.device(device)
    model = model.to(device)
    
    # Chỉ move proj tensor sang device, không reinit criterion
    crit = getattr(model, 'criterion', None)
    if crit is not None:
        if hasattr(crit, 'to'):
            crit.to(device)
        if hasattr(crit, 'proj'):
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
                 cached_optimizer_state: dict = None, global_c: dict = None, local_c: dict = None,
                 cached_train_loader=None, cached_val_loader=None):
        super().__init__(overrides=overrides, _callbacks=_callbacks)
        self.student_wrapper = student_wrapper
        # State dict bởi tên tham số từ round trước (None = round đầu tiên)
        self.cached_optimizer_state = cached_optimizer_state
        self._fl_injected_model = None
        self.topk_grad_ratio = None
        # SCAFFOLD
        self.global_c = global_c
        self.local_c = local_c
        self.cached_train_loader = cached_train_loader
        self.cached_val_loader = cached_val_loader

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """Reuse the AUV loader so image RAM cache survives across FL rounds."""
        cached = self.cached_train_loader if mode == "train" else self.cached_val_loader
        if cached is not None:
            if hasattr(cached, "reset"):
                cached.reset()
            return cached
        return super().get_dataloader(dataset_path, batch_size, rank, mode)

    def validate(self):
        """
        Ngăn chặn Ultralytics Validator gọi model.fuse() làm mất trọng số LoRA.
        Thay thế hàm fuse() bằng một hàm rỗng trả về chính model.
        Đồng thời TẮT half() (FP16) để tránh tràn số khi nhân lora_B @ lora_A.
        """
        if hasattr(self, 'args'):
            self.args.half = False
            
        if self.model and hasattr(self.model, 'fuse'):
            self.model.fuse = lambda: self.model
        if self.ema and self.ema.ema and hasattr(self.ema.ema, 'fuse'):
            self.ema.ema.fuse = lambda: self.ema.ema
            
        return super().validate()

    def setup_model(self):
        ckpt = super().setup_model()
        # [FIX BUG] Ultralytics khởi tạo model trên CPU ở round đầu tiên,
        # khiến `model.criterion.proj` (1 tensor thường) bị kẹt ở CPU.
        # Khi _setup_train tạo EMA, nó copy luôn tensor lỗi này sang EMA.
        # Ta phải fix device NGAY TẠI ĐÂY (trước khi tạo EMA).
        self.model = _fl_prepare_model_for_train(self.model, self.device)
        return ckpt


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

        if injected is not None:
            # Restore lại model đã inject LoRA sau khi _setup_train cố override nó
            self.model = injected
            # Đảm bảo device của injected model
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
            # [CRITICAL FIX] Ultralytics final_eval sẽ tự động gọi fuse() làm hỏng LoRAConv2d.
            # Ta bỏ qua bước final_eval mặc định này. Epoch cuối cùng đã in ra mAP chính xác rồi.
            print("\n[CustomDetectionTrainer] Bỏ qua final_eval() mặc định của Ultralytics vì hàm fuse() sẽ làm hỏng LoRAConv2d.")
            print("[CustomDetectionTrainer] Metrics ở epoch cuối cùng (phía trên) chính là kết quả chính xác nhất!")
            from ultralytics.utils.torch_utils import strip_optimizer
            if self.last.exists(): strip_optimizer(self.last)
            if self.best.exists(): strip_optimizer(self.best)

    def build_optimizer(self, model, name='auto', lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        optimizer = super().build_optimizer(model, name, lr, momentum, decay, iterations)

        if self.student_wrapper and not self.student_wrapper.full_param:
            payload_keys = set(self.student_wrapper.trainable_state_dict().keys())
            for k, v in model.named_parameters():
                if k in payload_keys or 'bn' in k:
                    v.requires_grad = True
                else:
                    v.requires_grad = False

        # Differential LR is configured by each caller through trainer attributes
        # and ultimately by config/settings.py. Defaults preserve Ultralytics-style
        # behavior when a caller does not request a split LR.
        head_lr_multiplier = getattr(self, 'head_lr_multiplier', 1.0)   # Default: Head LR = lr0 × 1.0
        lora_lr_multiplier = getattr(self, 'lora_lr_multiplier', 0.25)  # Default: LoRA LR = lr0 × 0.25
        
        if head_lr_multiplier != 1.0 or lora_lr_multiplier != 1.0:
            id_to_name = {id(p): n for n, p in model.named_parameters()}
            # Detect head nằm ở layer cuối (yolo12n: model.22, YOLO12l: model.21)
            head_patterns = ('model.21.', 'model.22.', 'model.23.')
            new_groups = []
            
            for group in optimizer.param_groups:
                head_p = []
                lora_p = []
                other_p = []
                
                for p in group['params']:
                    name = id_to_name.get(id(p), '')
                    if 'lora_' in name:
                        lora_p.append(p)
                    elif any(h in name for h in head_patterns):
                        head_p.append(p)
                    else:
                        other_p.append(p)
                        
                if other_p:
                    g = {k: v for k, v in group.items() if k != 'params'}
                    g['params'] = other_p
                    new_groups.append(g)
                if head_p:
                    g = {k: v for k, v in group.items() if k != 'params'}
                    g['params'] = head_p
                    g['lr'] = group.get('lr', lr) * head_lr_multiplier
                    new_groups.append(g)
                if lora_p:
                    g = {k: v for k, v in group.items() if k != 'params'}
                    g['params'] = lora_p
                    g['lr'] = group.get('lr', lr) * lora_lr_multiplier
                    new_groups.append(g)
                    
            optimizer.param_groups = new_groups
            print(f"[DiffLR] Head LR boosted ×{head_lr_multiplier}, LoRA scaled ×{lora_lr_multiplier} → "
                  f"LoRA lr={lr * lora_lr_multiplier:.2e} | Head lr={lr * head_lr_multiplier:.2e} | Base={lr:.2e}")

        # Loại bỏ các parameter đã bị đóng băng (requires_grad=False) khỏi optimizer param_groups
        real_trained_count = 0
        for group in optimizer.param_groups:
            group['params'] = [p for p in group['params'] if p.requires_grad]
            real_trained_count += len(group['params'])

        print(f"\n[CustomDetectionTrainer] Đã lọc optimizer! Cố định Backbone, CHỈ CÒN {real_trained_count} tensors được học.")

        return optimizer


    def optimizer_step(self):
        # 1. Cache Named Parameters and GPU Tensors (Thực hiện 1 lần duy nhất)
        if not hasattr(self, '_cached_named_params'):
            # Lọc sẵn các tham số đang được train (có requires_grad = True)
            self._cached_named_params = [(name, param) for name, param in self.model.named_parameters() if param.requires_grad]
            
            # Đưa toàn bộ FedProx weights lên GPU một lần (tránh chuyển đổi mỗi batch gây nghẽn PCIe)
            if getattr(self, 'fedprox_mu', 0.0) > 0.0 and getattr(self, 'global_weights', None) is not None:
                device = self._cached_named_params[0][1].device if self._cached_named_params else "cpu"
                self._gpu_global_weights = {k: v.to(device) for k, v in self.global_weights.items()}
            
            # Đưa toàn bộ SCAFFOLD control variates lên GPU một lần
            if getattr(self, 'global_c', None) is not None and getattr(self, 'local_c', None) is not None:
                device = self._cached_named_params[0][1].device if self._cached_named_params else "cpu"
                self._gpu_global_c = {k: v.to(device) for k, v in self.global_c.items()}
                self._gpu_local_c = {k: v.to(device) for k, v in self.local_c.items()}

        # Apply Proximal term to gradients before step (FedProx)
        if getattr(self, 'fedprox_mu', 0.0) > 0.0 and hasattr(self, '_gpu_global_weights'):
            for name, param in self._cached_named_params:
                if param.grad is not None and name in self._gpu_global_weights:
                    prox_term = param.data - self._gpu_global_weights[name]
                    param.grad.data.add_(prox_term, alpha=self.fedprox_mu)

        # [SCAFFOLD] Inject Control Variates
        if hasattr(self, '_gpu_global_c') and hasattr(self, '_gpu_local_c'):
            for name, param in self._cached_named_params:
                if param.grad is not None and name in self._gpu_global_c and name in self._gpu_local_c:
                    gc_tensor = self._gpu_global_c[name]
                    lc_tensor = self._gpu_local_c[name]
                    param.grad.data.add_(gc_tensor - lc_tensor)

        import torch
        if getattr(self, 'grad_diagnostics', False):
            raw_grad_norm = 0.0
            _grad_inf_nan = False
            for _, param in self._cached_named_params:
                if param.grad is not None:
                    g = param.grad.detach()
                    if not torch.isfinite(g).all():
                        _grad_inf_nan = True
                        break
                    raw_grad_norm += g.norm().item() ** 2
            raw_grad_norm = raw_grad_norm ** 0.5

            if _grad_inf_nan:
                print(
                    f"\n[GRAD EXPLOSION] Batch #{getattr(self, '_batch_count', '?')} | "
                    f"NaN/Inf trong gradient. AMP GradScaler will skip this optimizer step. "
                    f"If repeated, set fed_cfg.LOCAL_AMP=False or lower LOCAL_LR."
                )
            elif raw_grad_norm > 100.0:
                print(
                    f"\n[GRAD HIGH NORM] Batch #{getattr(self, '_batch_count', '?')} | "
                    f"Grad norm = {raw_grad_norm:.1f} (>100). Clip will cap it at max_norm=10."
                )

        # Tăng bộ đếm batch nội bộ để log có ngữ cảnh
        self._batch_count = getattr(self, '_batch_count', 0) + 1

        # ── Gradient Clipping (sau khi đã log) ─────────────────────────────
        trainable_params = [p for _, p in self._cached_named_params]
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=10.0)

        topk_ratio = getattr(self, "topk_grad_ratio", None)
        if topk_ratio is not None and topk_ratio < 1.0:
            self.scaler.unscale_(self.optimizer)
            with torch.no_grad():
                for p in trainable_params:
                    if p.grad is not None:
                        numel = p.grad.numel()
                        k = max(1, int(numel * topk_ratio))
                        if k < numel:
                            # Use topk to find threshold for top K elements
                            threshold = torch.topk(p.grad.abs().flatten(), k)[0][-1]
                            p.grad[p.grad.abs() < threshold] = 0.0
            
            # The clip_grad_norm_ was already called above, but if we modified gradients we might want to clip again,
            # though usually it's fine since we only zeroed out some elements (norm decreases).
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            if self.ema:
                self.ema.update(self.model)
        else:
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
    global_c: dict = None,
    local_c: dict = None,
    cached_train_loader=None,
    cached_val_loader=None,
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

    # [CRITICAL FIX] Dù là Full Parameter hay LoRA, việc dùng AdamW KHÔNG CÓ Warm-up 
    # (do local epochs quá ngắn) sẽ tạo ra step size khổng lồ ở các batch đầu tiên
    # do mẫu số v_t tiến về 0, chắc chắn làm nổ mạng (Loss = NaN).
    # Ta chuyển sang SGD để đảm bảo an toàn tuyến tính.
    # (Dùng lại biến lr được truyền từ simulator thay vì hardcode để giữ được cơ chế Global LR Decay)
    opt_choice = 'SGD'
    print(f"[DiffLR] Chuyển Optimizer sang SGD (lr={lr:.4f}) để chống nổ Loss do cold-start.")
    try:
        from config.settings import fed_cfg
        local_cache_dataset = getattr(fed_cfg, 'LOCAL_CACHE_DATASET', getattr(fed_cfg, 'CACHE_DATASET', True))
        local_amp = getattr(fed_cfg, 'LOCAL_AMP', True)
        local_workers = getattr(fed_cfg, 'LOCAL_DATALOADER_WORKERS', 8)
        local_augment = getattr(fed_cfg, 'LOCAL_AUGMENT', False)
        grad_diagnostics = getattr(fed_cfg, 'GRAD_DIAGNOSTICS', False)
    except Exception:
        fed_cfg = None
        local_cache_dataset = True
        local_amp = True
        local_workers = 8
        local_augment = False
        grad_diagnostics = False

    # 2. Chuẩn bị overrides cho Ultralytics Trainer
    overrides = {
        'model': "yolo12n.pt", # Dummy, will be overwritten by _fl_injected_model
        'data': auv_yaml,
        'cache': local_cache_dataset,
        'epochs': epochs,
        'batch': batch_size,
        'mosaic': 0.0,
        'mixup': 0.0,
        'augment': local_augment,
        'hsv_h': getattr(fed_cfg, 'LOCAL_HSV_H', 0.01) if fed_cfg else 0.01,
        'hsv_s': getattr(fed_cfg, 'LOCAL_HSV_S', 0.30) if fed_cfg else 0.30,
        'hsv_v': getattr(fed_cfg, 'LOCAL_HSV_V', 0.20) if fed_cfg else 0.20,
        'translate': getattr(fed_cfg, 'LOCAL_TRANSLATE', 0.05) if fed_cfg else 0.05,
        'scale': getattr(fed_cfg, 'LOCAL_SCALE', 0.15) if fed_cfg else 0.15,
        'fliplr': getattr(fed_cfg, 'LOCAL_FLIPLR', 0.50) if fed_cfg else 0.50,
        'flipud': 0.0,
        'degrees': 0.0,
        'shear': 0.0,
        'perspective': 0.0,
        'close_mosaic': 0,
        'lr0': lr,
        'optimizer': opt_choice,
        'warmup_epochs': 0.0,
        'warmup_bias_lr': lr,  # Tránh nhảy lr=0.1 mặc định của YOLO gây loss nổ
        'warmup_momentum': 0.937,
        'lrf': 1.0,
        'cos_lr': False,
        'device': device,
        'amp': local_amp,
        'project': 'runs/fl_auvs',
        'name': f'auv_{auv_id}',
        'exist_ok': True,
        'verbose': False,  # Ngăn YOLO in bảng kiến trúc
        'save': False,     # Không lưu weight cục bộ
        'val': False,      # Không đánh giá cục bộ
        'plots': False,    # Không vẽ đồ thị cục bộ
        'workers': local_workers,
    }

    # 3. Khởi tạo Trainer phù hợp
    if local_teacher is not None:
        from tasks.detection_2d.knowledge_compression.knowledge_distillation import KDDetectionTrainer
        trainer = KDDetectionTrainer(overrides=overrides)
        trainer.student_wrapper = student_model
        trainer.set_teacher(local_teacher.yolo.model)

        trainer.stu_lambda = getattr(fed_cfg, 'LOCAL_KD_STU_LAMBDA', 0.20)
        trainer.kd_lambda = getattr(fed_cfg, 'LOCAL_KD_LAMBDA', 1.0)
        
        trainer._fl_injected_model = student_model.yolo.model
        trainer.head_lr_multiplier = getattr(fed_cfg, 'LOCAL_KD_HEAD_LR_MULT', 3.0)
        # KDDetectionTrainer không hỗ trợ cached_optimizer_state (không cần thiết cho FedKD)
    else:
        trainer = CustomDetectionTrainer(
            overrides=overrides,
            student_wrapper=student_model,
            cached_optimizer_state=cached_optimizer_state,
            global_c=global_c,
            local_c=local_c,
            cached_train_loader=cached_train_loader,
            cached_val_loader=cached_val_loader,
        )

    trainer._fl_injected_model = student_model.yolo.model
    trainer.model = student_model.yolo.model
    trainer.fedprox_mu = fedprox_mu
    trainer.global_weights = global_weights
    trainer.grad_diagnostics = grad_diagnostics
    
    if getattr(student_model, 'full_param', False):
        trainer.head_lr_multiplier = 1.0
    else:
        # Diff LR:
        try:
            from config.settings import fed_cfg
            trainer.head_lr_multiplier = getattr(fed_cfg, 'LOCAL_HEAD_LR_MULT', 3.0)
            trainer.lora_lr_multiplier = getattr(fed_cfg, 'LOCAL_LORA_LR_MULT', 1.5)
        except Exception:
            trainer.head_lr_multiplier = 3.0
            trainer.lora_lr_multiplier = 1.5

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

    # Đảm bảo không có bất kỳ trọng số nào ngoài LoRA và head bị thay đổi.
    # BN belongs to the transmitted/aggregated payload, so it is not part of
    # frozen_weights_before. Keep the name guard for compatibility with older
    # checkpoints whose payload selection may differ.
    if not student_model.full_param:
        for k, v in student_model.yolo.model.named_parameters():
            if k in frozen_weights_before and 'bn' not in k:
                diff = torch.abs(frozen_weights_before[k].to(v.device) - v).max().item()
                if diff > 1e-6:
                    raise RuntimeError(f"CƠ CHẾ NGẦM PHÁT HIỆN: Lớp '{k}' dự kiến bị đóng băng nhưng đã thay đổi (max diff: {diff})!")

    # 4. Lấy state sau khi train
    state_after = student_model.trainable_state_dict()

    # [SCAFFOLD] Tính toán Control Variates Update
    if global_c is not None and local_c is not None:
        delta_c_i = {}
        try:
            steps = epochs * len(trainer.train_loader)
        except Exception:
            steps = epochs * 10  # fallback
            
        for k in state_before:
            if k in state_after and k in global_c and k in local_c:
                # Tìm lr_multiplier cho param k
                lr_mult = 1.0
                if 'lora_' in k:
                    lr_mult = getattr(trainer, 'lora_lr_multiplier', 1.0)
                elif any(h in k for h in ('model.21.', 'model.22.', 'model.23.')):
                    lr_mult = getattr(trainer, 'head_lr_multiplier', 1.0)
                
                eff_lr = lr * lr_mult * steps
                if eff_lr > 0:
                    x = state_before[k].to(device)
                    y = state_after[k].to(device)
                    # c_i^+ = c_i - c + (x - y) / (K * eta_l)
                    c_i_plus = local_c[k].to(device) - global_c[k].to(device) + (x - y) / eff_lr
                    delta_c_i[k] = (c_i_plus - local_c[k].to(device)).cpu()
                    # Cập nhật local_c trực tiếp bằng c_i^+
                    local_c[k] = c_i_plus.cpu()
        
        state_after['__scaffold_delta_c__'] = delta_c_i

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

    student_model._cached_train_loader = getattr(trainer, "train_loader", None)
    student_model._cached_val_loader = getattr(trainer, "test_loader", None)
    return state_after, delta_norm, train_loss, new_optimizer_state


def evaluate_od(student_model, test_yaml: str, device: str = "cpu") -> dict:
    """
    Đánh giá mAP@0.5:0.95 và mAP@0.5 của Student model trên tập test.
    Returns: dict chứa các metrics
    """
    import copy
    # [CRITICAL FIX] Lưu lại kiến trúc mạng gốc chưa bị Fuse (gộp BatchNorm) và chưa bị Bake
    unfused_model = copy.deepcopy(student_model.yolo.model)
    
    # [CRITICAL FIX 2] Phải BAKE LoRA vào Conv gốc trước khi chạy val()!
    # Nếu không bake, hàm model.fuse() bên trong val() sẽ xóa sổ class LoRAConv2d, vứt bỏ toàn bộ trọng số LoRA!
    student_model.bake_lora()
    
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    
    try:
        results = student_model.yolo.val(
            data=test_yaml,
            device=device,
            verbose=False,
            split='val',
            half=False,  # [FIX] Ngăn cast model sang FP16 in-place
            workers=0,   # [CRITICAL FIX] Tránh đóng băng multiprocessing/SHM
            batch=16     # Giới hạn batch size để tránh tràn VRAM
        )
    finally:
        # Khôi phục lại mạng chưa Fuse cho các vòng FL tiếp theo kể cả khi val() lỗi.
        student_model.yolo.model = unfused_model
        torch.cuda.empty_cache()
        gc.collect()
    
    # Lấy precision và recall (mean)
    mp = float(np.mean(results.box.mp)) if hasattr(results.box, 'mp') else 0.0
    mr = float(np.mean(results.box.mr)) if hasattr(results.box, 'mr') else 0.0
    
    # [FAIL-FAST] Nếu mAP50 rớt về 0.0, chắc chắn model đã bị hỏng trọng số!
    map50 = float(results.box.map50)
    if map50 < 1e-5:
        raise RuntimeError(
            f"[CRITICAL ERROR] Model evaluation resulted in {map50:.4f} mAP50! "
            f"This indicates the model weights have been severely corrupted "
            f"(e.g. exploding gradients or SVD scale divergence). Aborting FL."
        )
        
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
        
        student_model.bake_lora()
        
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
