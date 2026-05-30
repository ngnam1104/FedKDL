"""
train_student_warmup.py
Warm-up Student Model (YOLO11n) với LoRA trên Proxy Data trong 2 epochs.

Cách tiếp cận ĐÚNG: Monkey-patch DetectionTrainer.setup_model()
  - Inject LoRA + đóng băng AFTER model load, BEFORE optimizer build
"""
import sys
import torch
from pathlib import Path
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.detection_2d.models.lora import inject_lora
from config.settings import fed_cfg


def _make_lora_setup_hook(rank, lora_targets):
    """Monkey-patch DetectionTrainer.setup_model để inject LoRA trước khi Optimizer được build."""
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    original_setup_model = DetectionTrainer.setup_model

    def patched_setup_model(self):
        original_setup_model(self)

        print("\n[LoRA Hook] Injecting LoRA into Student model...")
        n_injected = inject_lora(self.model, target_layer_names=lora_targets, rank=rank)
        print(f"[LoRA Hook] Injected LoRA into {n_injected} Conv2d layers.")

        frozen = trainable = 0
        for name, param in self.model.named_parameters():
            is_lora = 'lora_A' in name or 'lora_B' in name
            is_head = 'model.21' in name or 'model.22' in name or 'model.23' in name
            if is_lora or is_head:
                param.requires_grad_(True)
                trainable += param.numel()
            else:
                param.requires_grad_(False)
                frozen += param.numel()

        total = trainable + frozen
        print(f"[LoRA Hook] Trainable: {trainable:,} / {total:,} params ({trainable/total*100:.1f}%)")

    DetectionTrainer.setup_model = patched_setup_model


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

    # Layer targets
    lora_targets = ['C2f', 'C3k2', 'A2C2f', 'C2fAttn']

    # Patch Ultralytics Trainer
    print(f"-> Monkey-patching DetectionTrainer để inject LoRA (rank={rank})...")
    _make_lora_setup_hook(rank=rank, lora_targets=lora_targets)

    print(f"-> Bắt đầu warm-up 2 epochs trên: {proxy_yaml.name}")
    student_yolo = YOLO("yolo11n.pt")
    student_yolo.train(
        data=str(proxy_yaml),
        epochs=2,       # Warm-up siêu nhẹ (2 epochs) để tránh overfitting proxy data
        imgsz=640,
        batch=16,
        workers=2,
        optimizer="AdamW",
        lr0=1e-3,
        device="cuda" if torch.cuda.is_available() else "cpu",
        project=str(REPO_ROOT / "runs/student_warmup_lora"),
        name="yolo11n_lora_warmup",
        exist_ok=True,
        close_mosaic=0,
    )

    # Lưu checkpoint
    best_weights = REPO_ROOT / "runs/student_warmup_lora/yolo11n_lora_warmup/weights/best.pt"
    last_weights = REPO_ROOT / "runs/student_warmup_lora/yolo11n_lora_warmup/weights/last.pt"
    chosen = best_weights if best_weights.exists() else last_weights

    if chosen.exists():
        import shutil
        shutil.copy(chosen, save_path)
        print(f"\n[Thành công] Đã lưu Student LoRA warmup tại: {save_path} (nguồn: {chosen.name})")
    else:
        print("\n[Lỗi] Không tìm thấy best.pt hoặc last.pt!")


if __name__ == "__main__":
    main()
