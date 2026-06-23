"""
train_teacher_lora.py
=======================
Huấn luyện Teacher (YOLO12l) với LoRA theo đúng config hiện tại.

Sau khi train xong:
  → yolo12l_lora_pretrained.pt   (raw LoRA weights — dùng cho TeacherModel + KD_PROJ_WEIGHT)

Pipeline đề xuất sau khi chạy xong:
  1. Kiểm tra mAP trực tiếp (in ra cuối script)
  2. Nếu muốn dùng Teacher như clean checkpoint cho non-projection KD:
       python scripts/fedkdl/bake_teacher_lora.py

Chạy:
  python scripts/fedkdl/train_teacher_lora.py
  python scripts/fedkdl/train_teacher_lora.py --epochs 30 --lr 5e-4
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.settings import fed_cfg
from tasks.detection_2d.models.yolo_wrapper import TeacherModel, StudentModel
from tasks.detection_2d.trainer import CustomDetectionTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train Teacher (YOLO12l) with LoRA")
    p.add_argument("--ckpt", default="yolo12l_pretrained.pt",
                   help="Base teacher checkpoint (clean, no LoRA)")
    p.add_argument("--data", default="datasets/URPC2020.yaml")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--nc", type=int, default=4)
    p.add_argument("--out", default="yolo12l_lora_pretrained.pt",
                   help="Đường dẫn lưu checkpoint đầu ra")
    p.add_argument("--device", default="0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = REPO_ROOT / args.ckpt
    data_yaml = str(REPO_ROOT / args.data)
    out_path = REPO_ROOT / args.out

    rank = fed_cfg.LORA_RANK
    lora_strategy = getattr(fed_cfg, "LORA_STRATEGY", "adaptive")
    lora_targets = list(getattr(fed_cfg, "LORA_TARGETS", ("Conv",)))

    print("=" * 60)
    print("  TEACHER LoRA PRE-TRAINING")
    print("=" * 60)
    print(f"  Base ckpt : {ckpt_path}")
    print(f"  Data      : {data_yaml}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  LR        : {args.lr}")
    print(f"  LoRA rank : {rank}  strategy={lora_strategy}  targets={lora_targets}")
    print(f"  Output    : {out_path}")
    print("=" * 60)

    if not ckpt_path.exists():
        if ckpt_path.name in ("yolo12l.pt", "yolo12n.pt", "yolo11l.pt", "yolov8l.pt"):
            print(f"[Info] Base checkpoint {ckpt_path.name} sẽ được tự động tải về bởi Ultralytics.")
        else:
            print(f"[Error] Không tìm thấy base checkpoint: {ckpt_path}")
            print("  → Hãy chạy pretrain.py trước để có yolo12l_pretrained.pt")
            print("  → Hoặc dùng CKPT=yolo12l.pt để train từ pretrain COCO gốc.")
            sys.exit(1)

    # 1. Khởi tạo Teacher với LoRA theo đúng config hiện tại
    teacher = StudentModel(
        ckpt=str(ckpt_path),
        rank=rank,
        nc=args.nc,
        lora_targets=lora_targets,
        lora_strategy=lora_strategy,
    )

    # Đếm LoRA layers
    from tasks.detection_2d.models.lora import LoRAConv2d
    n_lora = sum(1 for m in teacher.yolo.model.modules() if isinstance(m, LoRAConv2d))
    total = sum(p.numel() for p in teacher.yolo.model.parameters())
    trainable = sum(p.numel() for p in teacher.yolo.model.parameters() if p.requires_grad)
    print(f"\n[TeacherLoRA] {n_lora} LoRAConv2d layers")
    print(f"[TeacherLoRA] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # 2. Train qua CustomDetectionTrainer (giống student training pipeline)
    device_str = args.device
    if not torch.cuda.is_available() and device_str != "cpu":
        print("[WARN] CUDA không khả dụng, chuyển sang CPU")
        device_str = "cpu"

    overrides = {
        "model": str(ckpt_path),
        "data": data_yaml,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": device_str,
        "lr0": args.lr,
        "optimizer": "AdamW",
        "warmup_bias_lr": args.lr,
        "lrf": 0.1,               # Cosine decay xuống lr * 0.1 cuối run
        "warmup_epochs": 2,
        "weight_decay": 5e-4,
        "momentum": 0.937,
        "patience": 30,           # Early-stop sau 30 epochs không cải thiện
        "project": str(REPO_ROOT / "runs/teacher_lora"),
        "name": f"yolo12l_lora_r{rank}_{lora_strategy}",
        "exist_ok": True,
        "verbose": True,
        "workers": 4,
        "plots": False,
    }

    trainer = CustomDetectionTrainer(
        overrides=overrides,
        student_wrapper=teacher,
        cached_optimizer_state=None,
    )
    trainer._fl_injected_model = teacher.yolo.model
    trainer.model = teacher.yolo.model

    # Set explicit LR multipliers: Base=2e-4 -> Head=5e-4, LoRA=1e-4
    trainer.head_lr_multiplier = 2.5
    trainer.lora_lr_multiplier = 0.5

    print(f"\n[TeacherLoRA] Bắt đầu huấn luyện {args.epochs} epochs...\n")
    trainer.train()

    # 3. Lưu raw LoRA checkpoint (giữ LoRAConv2d — dùng cho TeacherModel trực tiếp)
    best_pt = REPO_ROOT / f"runs/teacher_lora/yolo12l_lora_r{rank}_{lora_strategy}/weights/best.pt"
    if best_pt.exists():
        import shutil
        shutil.copy(best_pt, out_path)
        print(f"\n[TeacherLoRA] ✅ Đã lưu tại: {out_path}")
        print(f"[TeacherLoRA] Kích thước: {out_path.stat().st_size / 1024 / 1024:.1f} MB")
    else:
        print(f"\n[TeacherLoRA] ⚠️  Không tìm thấy best.pt tại {best_pt}")

    del teacher, trainer
    gc.collect()
    torch.cuda.empty_cache()

    print("\n[TeacherLoRA] HOÀN THÀNH!")
    print("  Tiếp theo (nếu cần clean checkpoint):")
    print("    python scripts/fedkdl/bake_teacher_lora.py --src", args.out)


if __name__ == "__main__":
    main()
