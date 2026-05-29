"""
train_student_warmup.py
Khởi động ấm (Warm-up) Student Model (YOLO11n) với LoRA tiêm sẵn
trên Proxy Data (public data) trong 10 epochs.

Mục tiêu: Tạo ra yolo11n_warmup.pt có chứa các lớp LoRA đã được warm-up sơ bộ,
giúp Student không bị "lạnh" khi bước vào vòng FL đầu tiên.
"""
import sys
import torch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.detection_2d.models.yolo_wrapper import StudentModel
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
        # Fallback: dùng URPC2020 full nếu chưa có proxy
        proxy_yaml = REPO_ROOT / "datasets/URPC2020.yaml"
        print(f"[Warning] proxy_kd_data.yaml không tồn tại. Dùng fallback: {proxy_yaml.name}")

    save_path = REPO_ROOT / "yolo11n_warmup.pt"
    if save_path.exists():
        print(f"[Student Warmup LoRA] {save_path.name} đã tồn tại, BỎ QUA.")
        return

    print(f"-> Loading yolo11n.pt và tiêm LoRA (rank={rank})...")
    student = StudentModel(
        ckpt="yolo11n.pt",
        rank=rank,
        nc=4,          # URPC2020: holothurian, echinus, scallop, starfish
        full_param=False,
        use_lora=True,
    )

    print(f"-> Bắt đầu warm-up 10 epochs trên: {proxy_yaml.name}")
    student.yolo.train(
        data=str(proxy_yaml),
        epochs=10,      # Warm-up nhẹ nhàng, không cần nhiều epoch
        imgsz=640,
        batch=16,
        device="cuda" if torch.cuda.is_available() else "cpu",
        project="runs/student_warmup_lora",
        name="yolo11n_lora_warmup",
        exist_ok=True,
        lr0=0.01,
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
