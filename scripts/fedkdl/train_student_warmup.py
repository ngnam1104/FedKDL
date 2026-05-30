"""
train_student_warmup.py
Warm-up Student Model (YOLO11n) với LoRA trên Proxy Data trong 2 epochs.

Dùng đúng pattern của repo (CustomDetectionTrainer + snapshot+rollback frozen weights).
"""
import sys
import torch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.detection_2d.models.yolo_wrapper import StudentModel
from tasks.detection_2d.trainer import CustomDetectionTrainer
from config.settings import fed_cfg


def main():
    print("==================================================")
    print("[Student Warmup LoRA] Warm-up YOLO11n + LoRA trên Proxy Data")
    print("==================================================")

    rank = fed_cfg.LORA_RANK
    print(f"-> LoRA Rank: {rank}")

    # Kiểm tra proxy data yaml
    proxy_yaml = REPO_ROOT / "datasets/proxy_kd_data.yaml"
    if not proxy_yaml.exists():
        proxy_yaml = REPO_ROOT / "datasets/URPC2020.yaml"
        print(f"[Warning] proxy_kd_data.yaml không tồn tại. Dùng fallback: {proxy_yaml.name}")

    save_path = REPO_ROOT / "yolo11n_warmup.pt"
    if save_path.exists():
        print(f"[Student Warmup LoRA] {save_path.name} đã tồn tại, BỎ QUA.")
        return

    # 1. Khởi tạo Student LoRA
    print(f"-> Loading yolo11n.pt và tiêm LoRA (rank={rank})...")
    student = StudentModel(
        ckpt="yolo11n.pt",
        rank=rank,
        nc=4,
        full_param=False,
        use_lora=True,
    )

    # 2. Snapshot frozen weights TRƯỚC khi train
    payload_keys = set(student.trainable_state_dict().keys())
    frozen_weights_before = {}
    for k, v in student.yolo.model.state_dict().items():
        if k not in payload_keys:
            frozen_weights_before[k] = v.clone().detach()

    print(f"-> Trainable keys: {len(payload_keys)}, Frozen keys: {len(frozen_weights_before)}")

    # 3. Cấu hình trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    overrides = {
        'model': "yolo11n.pt",
        'data': str(proxy_yaml),
        'epochs': 2,
        'batch': 16,
        'workers': 2,
        'lr0': 1e-3,
        'optimizer': 'AdamW',
        'warmup_epochs': 0.0,
        'lrf': 1.0,
        'cos_lr': False,
        'device': device,
        'amp': False,
        'project': str(REPO_ROOT / "runs/student_warmup_lora"),
        'name': 'yolo11n_lora_warmup',
        'exist_ok': True,
        'verbose': True,
        'save': True,
        'val': False,
        'plots': False,
        'close_mosaic': 0,
    }

    trainer = CustomDetectionTrainer(
        overrides=overrides,
        student_wrapper=student,
        cached_optimizer_state=None,
    )
    trainer.model = student.yolo.model

    print(f"\n-> Bắt đầu warm-up 2 epochs trên: {proxy_yaml.name}")
    trainer.train()

    # 4. ROLLBACK frozen weights (Ultralytics sẽ train full, ta rollback sau)
    print("\n[Rollback] Khôi phục frozen weights về giá trị gốc...")
    with torch.no_grad():
        state_dict = student.yolo.model.state_dict()
        for k, v_before in frozen_weights_before.items():
            if k in state_dict:
                state_dict[k].copy_(v_before)
    student.yolo.model.load_state_dict(state_dict)
    print("[OK] Rollback hoàn tất. Chỉ LoRA + Head weights được giữ lại sau warm-up.")

    # 5. Lưu model sau rollback
    best_weights = REPO_ROOT / "runs/student_warmup_lora/yolo11n_lora_warmup/weights/best.pt"
    last_weights = REPO_ROOT / "runs/student_warmup_lora/yolo11n_lora_warmup/weights/last.pt"
    chosen = best_weights if best_weights.exists() else last_weights

    if chosen.exists():
        import shutil
        shutil.copy(chosen, save_path)
        print(f"\n[Thành công] Đã lưu Student LoRA warmup tại: {save_path}")
    else:
        print("\n[Lỗi] Không tìm thấy best.pt hoặc last.pt!")


if __name__ == "__main__":
    main()
