"""
trainer.py
Local SGD cho bài toán Object Detection (Kịch bản 2 & 3).

Tier 1 (Sensor/AUV) chỉ chạy local SGD thuần (DetectionTrainer), KHÔNG dùng KD.
KD (Knowledge Distillation với Teacher YOLO12l) được di chuyển lên Tier 3 (Gateway).

Payload truyền đi: LoRA adapters + cv3.x.2 output conv, nén INT8.
"""
import torch
import copy
from ultralytics.models.yolo.detect import DetectionTrainer
from config.settings import fed_cfg


def local_sgd_od(
    student_model,
    client_yaml: str,
    client_id: int,
    epochs: int = 2,
    batch_size: int = 16,
    lr: float = 0.01,
    device: str = "cpu",
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
        'model': "yolo26n.pt",
        'data': client_yaml,
        'epochs': epochs,
        'batch': batch_size,
        'lr0': lr,
        'device': device,
        'project': 'runs/fl_clients',
        'name': f'client_{client_id}',
        'exist_ok': True,
        'verbose': False,
        'save': False,
        'val': False,
        'workers': 0,
    }

    # 3. Khởi tạo DetectionTrainer (thuần, không KD)
    trainer = DetectionTrainer(overrides=overrides)
    trainer.model = student_model.yolo.model
    trainer.train()

    # 4. Lấy state sau khi train
    state_after = student_model.trainable_state_dict()

    # 5. Tính delta norm (cho Lazy Communication Filter — Eq. 40)
    delta_norm = 0.0
    for k in state_before:
        if k in state_after:
            diff = state_after[k].float() - state_before[k].float()
            delta_norm += torch.sum(diff ** 2).item()

    return state_after, delta_norm


def evaluate_od(student_model, test_yaml: str, device: str = "cpu") -> float:
    """
    Đánh giá mAP@0.5:0.95 của Student model trên tập test.
    Returns: mAP score (float)
    """
    results = student_model.yolo.val(
        data=test_yaml,
        device=device,
        verbose=False,
        split='val',
    )
    return results.box.map

