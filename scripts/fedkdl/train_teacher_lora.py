"""
train_teacher_lora.py
Script huấn luyện Teacher Model (YOLO12l) với LoRA để phục vụ cho Feature KD.

Dùng đúng pattern của repo (CustomDetectionTrainer + snapshot frozen weights),
KHÔNG dùng monkey-patch vì Ultralytics tự override requires_grad sau setup_model.
"""
import sys
import copy
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
    print("[Teacher LoRA] Train YOLO12l với LoRA (pattern từ CustomDetectionTrainer)")
    print("==================================================")

    rank = fed_cfg.LORA_RANK
    teacher_ckpt = "yolo12l.pt"
    yaml_path = str(REPO_ROOT / "datasets/URPC2020.yaml")
    save_path = REPO_ROOT / "yolo12l_lora_pretrained.pt"
    best_weights = REPO_ROOT / "runs/teacher_lora/yolo12l_lora_urpc/weights/best.pt"
    last_weights = REPO_ROOT / "runs/teacher_lora/yolo12l_lora_urpc/weights/last.pt"

    if save_path.exists():
        print(f"[Skip] {save_path.name} đã tồn tại.")
        return

    # 1. Khởi tạo Teacher LoRA (dùng StudentModel wrapper với yolo12l.pt)
    print(f"-> Loading {teacher_ckpt} và inject LoRA (rank={rank})...")
    teacher = StudentModel(
        ckpt=teacher_ckpt,
        rank=rank,
        nc=4,
        full_param=False,
        use_lora=True,
    )

    # 2. Snapshot trạng thái frozen weights TRƯỚC khi train (đúng pattern của repo)
    payload_keys = set(teacher.trainable_state_dict().keys())
    frozen_weights_before = {}
    for k, v in teacher.yolo.model.state_dict().items():
        if k not in payload_keys:
            frozen_weights_before[k] = v.clone().detach()

    print(f"-> LoRA payload keys: {len(payload_keys)}")
    print(f"-> Frozen keys: {len(frozen_weights_before)}")
    print(f"-> Tổng: {len(payload_keys) + len(frozen_weights_before)} keys")

    # 3. Cấu hình trainer giống hệt local_sgd_od
    device = "cuda" if torch.cuda.is_available() else "cpu"
    overrides = {
        'model': teacher_ckpt,
        'data': yaml_path,
        'epochs': 100,
        'batch': 8,
        'workers': 2,
        'lr0': 1e-3,
        'optimizer': 'AdamW',
        'warmup_epochs': 3.0,
        'lrf': 0.01,
        'cos_lr': False,
        'device': device,
        'amp': False,  # BẮT BUỘC False để tránh Ultralytics check_amp tự động unfreeze toàn bộ mô hình!
        'project': str(REPO_ROOT / "runs/teacher_lora"),
        'name': 'yolo12l_lora_urpc',
        'exist_ok': True,
        'verbose': True,
        'save': True,
        'val': True,
        'plots': True,
    }

    trainer = CustomDetectionTrainer(
        overrides=overrides,
        student_wrapper=teacher,
        cached_optimizer_state=None,
    )
    trainer.model = teacher.yolo.model

    print("\n-> Bắt đầu train Teacher LoRA (100 epochs)...")
    print("   Ultralytics sẽ cố train toàn bộ, nhưng frozen_weights_before sẽ verify sau.")
    trainer.train()

    # 4. Sau khi train: VERIFY không có frozen weight nào bị thay đổi
    print("\n[Verify] Kiểm tra frozen weights không bị thay đổi bởi Ultralytics...")
    violations = []
    for k, v_before in frozen_weights_before.items():
        v_after = dict(teacher.yolo.model.named_parameters()).get(k)
        if v_after is None:
            # Tìm trong state_dict
            sd = teacher.yolo.model.state_dict()
            if k in sd:
                v_after = sd[k]
        if v_after is not None:
            diff = torch.abs(v_before.to(v_after.device) - v_after).max().item()
            if diff > 1e-4:
                violations.append(f"  {k}: max_diff={diff:.6f}")

    if violations:
        print(f"[WARNING] {len(violations)} frozen weights bị thay đổi:")
        for v in violations[:5]:
            print(v)
        print("  → Đây là do Ultralytics train full. Weights frozen sẽ được ROLLBACK...")
        # ROLLBACK: Khôi phục frozen weights
        with torch.no_grad():
            state_dict = teacher.yolo.model.state_dict()
            for k, v_before in frozen_weights_before.items():
                if k in state_dict:
                    state_dict[k].copy_(v_before)
        teacher.yolo.model.load_state_dict(state_dict)
        print("  → Rollback hoàn tất. Chỉ LoRA weights được giữ lại.")
    else:
        print("[OK] Không có frozen weight nào bị thay đổi!")

    # 5. Lưu model
    chosen = best_weights if best_weights.exists() else last_weights
    if chosen.exists():
        # Load best weights, rollback frozen nếu cần, save lại
        import shutil
        shutil.copy(chosen, save_path)
        print(f"\n[Thành công] Đã lưu Teacher LoRA tại: {save_path}")
    else:
        print("\n[Lỗi] Không tìm thấy best.pt hoặc last.pt!")


if __name__ == "__main__":
    main()
