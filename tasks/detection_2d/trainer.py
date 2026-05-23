"""
trainer.py
Local SGD cho bài toán Object Detection (Kịch bản 2 & 3).

Tier 1 (Sensor/AUV) chỉ chạy local SGD thuần (DetectionTrainer), KHÔNG dùng KD.
KD (Knowledge Distillation với Teacher YOLO12l) được di chuyển lên Tier 3 (Gateway).

Payload truyền đi: LoRA adapters + cv3.x.2 output conv, nén INT8.
"""
import torch
import copy
import logging
import numpy as np
from ultralytics.models.yolo.detect import DetectionTrainer
from config.settings import fed_cfg

class CustomDetectionTrainer(DetectionTrainer):
    def __init__(self, overrides=None, _callbacks=None, student_wrapper=None):
        super().__init__(overrides=overrides, _callbacks=_callbacks)
        self.student_wrapper = student_wrapper

    def _setup_train(self):
        from ultralytics.utils import LOGGER
        
        # Tắt triệt để cảnh báo của YOLO (trong lúc _setup_train, YOLO sẽ quét và lật lại requires_grad)
        original_warning = LOGGER.warning
        LOGGER.warning = lambda *args, **kwargs: None
        
        try:
            super()._setup_train()
        finally:
            LOGGER.warning = original_warning

    def validate(self):
        """Bỏ qua validate giữa các epoch để tiết kiệm thời gian (Lần 1)."""
        return {}, 0.0

    def final_eval(self):
        """Bỏ qua bước Validate dư thừa ở cuối quá trình Local SGD để tiết kiệm thời gian (Lần 2)."""
        from ultralytics.utils.torch_utils import strip_optimizer
        model = self.best if self.best.exists() else None
        if self.last.exists():
            strip_optimizer(self.last)
        if model:
            strip_optimizer(self.best)
            self.run_callbacks("on_fit_epoch_end")

    def build_optimizer(self, model, name='auto', lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        optimizer = super().build_optimizer(model, name, lr, momentum, decay, iterations)
        
        if self.student_wrapper and not self.student_wrapper.full_param:
            payload_keys = set(self.student_wrapper.trainable_state_dict().keys())
            for k, v in model.named_parameters():
                if k in payload_keys:
                    v.requires_grad = True
                else:
                    v.requires_grad = False
                
        # Loại bỏ các parameter đã bị đóng băng (requires_grad=False) khỏi optimizer param_groups
        # Điều này đảm bảo PyTorch hoàn toàn bỏ qua chúng trong quá trình step()
        for group in optimizer.param_groups:
            group['params'] = [p for p in group['params'] if p.requires_grad]
            
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
    client_yaml: str,
    client_id: int,
    epochs: int = 2,
    batch_size: int = 16,
    lr: float = 0.01,
    device: str = "cpu",
    fedprox_mu: float = 0.0,
    global_weights: dict = None,
) -> tuple:
    """
    Thực hiện Local SGD cho OD tại Sensor (Tier 1).
    KHÔNG sử dụng KD — Teacher chỉ chạy tại Gateway (Tier 3).

    student_model : tasks.detection_2d.models.yolo_wrapper.StudentModel
    client_yaml   : đường dẫn data.yaml của client

    Returns:
        (new_state, delta_norm)  — new_state là absolute state dict (LoRA + Head partial).
                                   delta_norm là L2 norm của sự thay đổi (để Lazy Filter).
    """
    # 1. Snapshot trạng thái trước khi train
    state_before = copy.deepcopy(student_model.trainable_state_dict())

    # 2. Chuẩn bị overrides cho Ultralytics Trainer
    overrides = {
        'model': "yolo11n.pt",
        'data': client_yaml,
        'epochs': epochs,
        'batch': batch_size,
        'lr0': lr,
        'device': device,
        'project': 'runs/fl_clients',
        'name': f'client_{client_id}',
        'exist_ok': True,
        'verbose': False,  # Ngăn YOLO in bảng kiến trúc
        'save': False,
        'val': False,
        'plots': False,    # Vô hiệu hóa tính năng vẽ biểu đồ để tránh lỗi Plotting error do bỏ qua val
        'workers': 0,
    }

    # 3. Khởi tạo CustomDetectionTrainer
    trainer = CustomDetectionTrainer(overrides=overrides, student_wrapper=student_model)
    trainer.model = student_model.yolo.model
    trainer.fedprox_mu = fedprox_mu
    trainer.global_weights = global_weights

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

    return state_after, delta_norm, train_loss


def evaluate_od(student_model, test_yaml: str, device: str = "cpu") -> dict:
    """
    Đánh giá mAP@0.5:0.95 và mAP@0.5 của Student model trên tập test.
    Returns: dict chứa các metrics
    """
    results = student_model.yolo.val(
        data=test_yaml,
        device=device,
        verbose=False,
        split='val',
    )
    
    # Lấy precision và recall (mean)
    mp = float(np.mean(results.box.mp)) if hasattr(results.box, 'mp') else 0.0
    mr = float(np.mean(results.box.mr)) if hasattr(results.box, 'mr') else 0.0
    
    return {
        'mAP50-95': float(results.box.map),
        'mAP50': float(results.box.map50),
        'Prec': mp,
        'Rec': mr
    }

