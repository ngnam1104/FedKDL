"""
train_teacher_lora.py
Script huấn luyện Teacher Model (YOLO12l) với LoRA để phục vụ cho Feature KD.

Cách tiếp cận ĐÚNG: Monkey-patch DetectionTrainer.setup_model()
  - Ultralytics sẽ tự load yolo12l.pt lên (bình thường)
  - Ngay sau đó, ta inject LoRA và đóng băng Backbone VÀO CHÍNH model đó
  - Ultralytics build Optimizer SAU setup_model → chỉ thấy ~955K LoRA params
  - Đảm bảo 100% chỉ LoRA được train, không cần file tạm.
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
        # 1. Để Ultralytics load model như bình thường (full YOLO12l)
        original_setup_model(self)

        # 2. Inject LoRA ngay sau khi model được load
        print("\n[LoRA Hook] Injecting LoRA into Ultralytics model...")
        n_injected = inject_lora(self.model, target_layer_names=lora_targets, rank=rank)
        print(f"[LoRA Hook] Injected LoRA into {n_injected} Conv2d layers.")

        # 3. Đóng băng tất cả tham số gốc, chỉ để lora_A, lora_B và Detection Head trainable
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
        print(f"[LoRA Hook] Frozen:    {frozen:,} / {total:,} params ({frozen/total*100:.1f}%)")

    DetectionTrainer.setup_model = patched_setup_model


def main():
    print("==================================================")
    print("[Teacher LoRA] Train YOLO12l với LoRA (Monkey-patch Ultralytics)")
    print("==================================================")

    rank = fed_cfg.LORA_RANK
    teacher_ckpt = "yolo12l.pt"
    yaml_path = REPO_ROOT / "datasets/URPC2020.yaml"
    save_path = REPO_ROOT / "yolo12l_lora_pretrained.pt"

    best_weights = REPO_ROOT / "runs/teacher_lora/yolo12l_lora_urpc/weights/best.pt"
    last_weights = REPO_ROOT / "runs/teacher_lora/yolo12l_lora_urpc/weights/last.pt"

    if save_path.exists():
        print(f"[Skip] {save_path.name} đã tồn tại.")
        return

    # Layer targets (C3k2 = YOLO12 block, A2C2f = YOLO12 attention block)
    lora_targets = ['C2f', 'C3k2', 'A2C2f', 'C2fAttn']

    # Patch Ultralytics Trainer — inject LoRA ngay sau setup_model, trước build_optimizer
    print(f"-> Monkey-patching DetectionTrainer để inject LoRA (rank={rank})...")
    _make_lora_setup_hook(rank=rank, lora_targets=lora_targets)

    # Khởi tạo YOLO và train — setup_model đã bị patch → LoRA sẽ được inject đúng
    lora_yolo = YOLO(teacher_ckpt)
    lora_yolo.train(
        data=str(yaml_path),
        epochs=100,
        imgsz=640,
        batch=8,
        workers=2,
        optimizer="AdamW",
        lr0=1e-3,
        device="cuda" if torch.cuda.is_available() else "cpu",
        project=str(REPO_ROOT / "runs/teacher_lora"),
        name="yolo12l_lora_urpc",
        exist_ok=True,
        resume=False,
    )

    # Lưu model
    chosen = best_weights if best_weights.exists() else last_weights
    if chosen.exists():
        import shutil
        shutil.copy(chosen, save_path)
        print(f"\n[Thành công] Đã lưu Teacher LoRA model tại: {save_path} (nguồn: {chosen.name})")
    else:
        print("\n[Lỗi] Không tìm thấy best.pt hoặc last.pt!")


if __name__ == "__main__":
    main()
