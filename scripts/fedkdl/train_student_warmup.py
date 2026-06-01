"""
train_student_warmup.py
Warm-up Student Model (yolo12n) với LoRA trên Proxy Data trong 2 epochs.

Dùng đúng pattern của repo (CustomDetectionTrainer + snapshot+rollback frozen weights).
"""
import sys
import argparse
import torch
from pathlib import Path
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.detection_2d.models.yolo_wrapper import StudentModel
from tasks.detection_2d.trainer import CustomDetectionTrainer
from config.settings import fed_cfg


def run_warmup(epochs: int):
    print("==================================================")
    print(f"[Student Warmup LoRA] Warm-up YOLO12n + LoRA trong {epochs} epochs trên toàn bộ dữ liệu (Full YAML)")
    print("==================================================")

    rank = fed_cfg.LORA_RANK
    print(f"-> LoRA Rank: {rank}")

    # Sử dụng full dataset để warmup
    full_yaml = REPO_ROOT / "datasets/URPC2020.yaml"

    save_path = REPO_ROOT / "yolo12n_warmup.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"-> Loading yolo12n.pt và tiêm LoRA (rank={rank})...")
    student = StudentModel(
        ckpt="yolo12n.pt",
        rank=rank,
        nc=4,
        full_param=False,
        use_lora=True,
    )

    payload_keys = set(student.trainable_state_dict().keys())
    frozen_weights_before = {}
    for k, v in student.yolo.model.state_dict().items():
        if k not in payload_keys:
            frozen_weights_before[k] = v.clone().detach()

    overrides = {
        'model': "yolo12n.pt",
        'data': str(full_yaml),
        'epochs': epochs,
        'batch': 16,
        'workers': 2,
        'lr0': 2e-4,
        'warmup_bias_lr': 2e-4,
        'optimizer': 'AdamW',
        'warmup_epochs': 0.0,
        'lrf': 1.0,
        'cos_lr': False,
        'device': device,
        'amp': False,
        'project': str(REPO_ROOT / "runs/student_warmup_lora"),
        'name': 'yolo12n_lora_warmup',
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
    trainer._fl_injected_model = student.yolo.model
    trainer.model = student.yolo.model
    trainer.head_lr_multiplier = 5.0

    print(f"\n-> Bắt đầu warm-up {epochs} epochs trên: {full_yaml.name} (LoRA lr=2e-4 | Head lr=1e-3)")
    trainer.train()

    print("\n[Rollback] Khôi phục frozen weights về giá trị gốc (nếu có sai lệch ngầm)...")
    with torch.no_grad():
        state_dict = student.yolo.model.state_dict()
        for k, v_before in frozen_weights_before.items():
            if k in state_dict:
                state_dict[k].copy_(v_before)
    student.yolo.model.load_state_dict(state_dict)

    ckpt = {"model": student.yolo.model.half(), "epoch": epochs}
    torch.save(ckpt, save_path)
    print(f"\n[Thành công] Đã lưu Student LoRA warmup tại: {save_path}")

    try:
        from verify_teacher_lora_eval import merge_lora_into_base
    except ImportError:
        from bake_teacher_lora import bake_lora_into_model as merge_lora_into_base

    print("\n[Đánh giá] Đánh giá chất lượng Student LoRA (đã merge LoRA)...")
    merge_lora_into_base(student.yolo.model)
    full_yaml_test = REPO_ROOT / "datasets/URPC2020.yaml"
    student.yolo.val(
        data=str(full_yaml_test),
        imgsz=640,
        batch=16,
        device=device,
        verbose=True,
        split="val",
    )


def run_centralized_lora(epochs: int):
    print("\n" + "="*50)
    print(f"[Centralized LoRA] Train LoRA + Head trong {epochs} epochs (Upper Bound)")
    print("="*50)
    
    full_yaml = REPO_ROOT / "datasets/URPC2020.yaml"
    rank = fed_cfg.LORA_RANK
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    student = StudentModel(
        ckpt="yolo12n.pt",
        rank=rank,
        nc=4,
        full_param=False,
        use_lora=True,
    )
    
    overrides = {
        'model': "yolo12n.pt",
        'data': str(full_yaml),
        'epochs': epochs,
        'batch': 16,
        'workers': 4,
        'lr0': 2e-4,
        'warmup_bias_lr': 2e-4,
        'optimizer': 'AdamW',
        'warmup_epochs': 0.0,
        'lrf': 1.0,
        'cos_lr': False,
        'device': device,
        'amp': False,
        'project': str(REPO_ROOT / "runs" / "centralized"),
        'name': 'lora_finetune',
        'exist_ok': True,
        'verbose': True,
        'save': True,
        'val': True,
        'plots': True,
        'close_mosaic': 0,
    }

    trainer = CustomDetectionTrainer(
        overrides=overrides,
        student_wrapper=student,
        cached_optimizer_state=None,
    )
    trainer._fl_injected_model = student.yolo.model
    trainer.model = student.yolo.model
    trainer.head_lr_multiplier = 5.0

    trainer.train()
    
    save_path = REPO_ROOT / "yolo12n_lora_centralized.pt"
    ckpt = {"model": student.yolo.model.half(), "epoch": epochs}
    torch.save(ckpt, save_path)
    print(f"[Thành công] Đã lưu mô hình LoRA Centralized tại: {save_path}")


def run_centralized_full(epochs: int):
    print("\n" + "="*50)
    print(f"[Centralized Full] Train 100% Parameter (No LoRA) trong {epochs} epochs")
    print("="*50)
    
    full_yaml = REPO_ROOT / "datasets/URPC2020.yaml"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = YOLO("yolo12n.pt")
    
    model.train(
        data=str(full_yaml),
        epochs=epochs,
        batch=16,
        workers=4,
        device=device,
        project=str(REPO_ROOT / "runs" / "centralized"),
        name="full_finetune",
        exist_ok=True,
        verbose=True,
        save=True,
        val=True,
        plots=True
    )
    
    print(f"[Thành công] Đã lưu mô hình Full Finetune tại: runs/centralized/full_finetune/weights/best.pt")


def main():
    parser = argparse.ArgumentParser(description="Chạy Warmup hoặc Centralized Baselines")
    parser.add_argument("--mode", type=str, default="all", choices=["warmup", "centralized_lora", "centralized_full", "all"],
                        help="Chế độ chạy (mặc định: all)")
    parser.add_argument("--epochs-warmup", type=int, default=3, help="Số epoch cho warmup")
    parser.add_argument("--epochs-centralized", type=int, default=200, help="Số epoch cho centralized tests")
    args = parser.parse_args()

    if args.mode in ["warmup", "all"]:
        run_warmup(epochs=args.epochs_warmup)
        
    if args.mode in ["centralized_lora", "all"]:
        run_centralized_lora(epochs=args.epochs_centralized)
        
    if args.mode in ["centralized_full", "all"]:
        run_centralized_full(epochs=args.epochs_centralized)


if __name__ == "__main__":
    main()
