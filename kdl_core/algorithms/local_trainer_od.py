"""
local_trainer_od.py
Local SGD cho bài toán Object Detection (Kịch bản 2 & 3).
Sử dụng KDDetectionTrainer (fl_core/knowledge_compression/knowledge_distillation.py)
và Lazy Communication Filter (fl_core/algorithms/lazy_filter.py).
"""
import torch
import copy
from kdl_core.knowledge_compression.knowledge_distillation import KDDetectionTrainer
from kdl_core.algorithms.lazy_filter import lazy_filter
from config.settings import fed_cfg


def local_sgd_od(
    student_model,
    teacher_model,
    client_yaml: str,
    client_id: int,
    epochs: int = 2,
    batch_size: int = 16,
    lr: float = 0.01,
    device: str = "cpu",
    kd_lambda: float = 1.0,
    delta_threshold: float = None,
    use_kd: bool = True,
):
    """
    Thực hiện Local SGD cho OD, tích hợp KD + Lazy Filter.

    student_model : kdl_core.models.yolo_wrapper.StudentModel
    teacher_model : kdl_core.models.yolo_wrapper.TeacherModel (hoặc None)
    client_yaml   : đường dẫn data.yaml của client
    delta_threshold: None → dùng DELTA_SKIP từ FedKDLConfig

    Returns:
        (new_state, delta_norm)  nếu pass filter → cần gửi
        (None, delta_norm)       nếu bị lazy filter giữ lại
    """
    if delta_threshold is None:
        delta_threshold = fed_cfg.DELTA_SKIP

    # 1. Snapshot trạng thái trước khi train
    state_before = copy.deepcopy(student_model.trainable_state_dict())

    # 2. Chuẩn bị overrides cho Ultralytics Trainer
    overrides = {
        'model': student_model.yolo.model,
        'data': client_yaml,
        'epochs': epochs,
        'batch': batch_size,
        'lr0': lr,
        'device': device,
        'kd_lambda': kd_lambda,
        'project': 'runs/fl_clients',
        'name': f'client_{client_id}',
        'exist_ok': True,
        'verbose': False,
        'save': False,
        'val': False,
    }

    # 3. Khởi tạo KDDetectionTrainer
    trainer = KDDetectionTrainer(overrides=overrides)
    trainer.set_teacher(teacher_model.yolo.model if (use_kd and teacher_model) else None)
    trainer.model = student_model.yolo.model
    trainer.train()

    # 4. Lấy state sau khi train
    state_after = student_model.trainable_state_dict()

    # 5. Áp dụng Lazy Communication Filter
    filtered_state, delta = lazy_filter(state_before, state_after, delta_threshold)
    return filtered_state, delta


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
