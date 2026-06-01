"""
train_student_warmup.py (Giờ kiêm luôn vai trò Centralized Baseline Runner)
Chạy Centralized Baseline để làm benchmark (Full Parameter và LoRA).
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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200, help="Số epoch cần train")
    return parser.parse_args()

def run_lora_finetune(epochs: int, full_yaml: Path):
    print("\n" + "="*60)
    print(f"[Centralized LoRA] Train LoRA + Head trong {epochs} epochs")
    print("="*60)
    
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
    
    # Save model weights
    save_path = REPO_ROOT / "yolo12n_lora_centralized.pt"
    ckpt = {"model": student.yolo.model.half(), "epoch": epochs}
    torch.save(ckpt, save_path)
    print(f"[Thành công] Đã lưu mô hình LoRA tại: {save_path}")

def run_full_finetune(epochs: int, full_yaml: Path):
    print("\n" + "="*60)
    print(f"[Centralized Full] Train 100% Parameter (No LoRA) trong {epochs} epochs")
    print("="*60)
    
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
    args = get_args()
    full_yaml = REPO_ROOT / "datasets/URPC2020.yaml"
    
    if not full_yaml.exists():
        raise FileNotFoundError(f"[LỖI] Không tìm thấy file {full_yaml}")

    # Run both experiments sequentially
    run_lora_finetune(epochs=args.epochs, full_yaml=full_yaml)
    run_full_finetune(epochs=args.epochs, full_yaml=full_yaml)

if __name__ == "__main__":
    main()
