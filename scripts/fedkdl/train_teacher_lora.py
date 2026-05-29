"""
train_teacher_lora.py
Script huấn luyện Teacher Model (YOLO12l) với LoRA để phục vụ cho Feature KD.
Mục tiêu: Tạo ra yolo12l_lora_pretrained.pt có chứa các lớp LoRA.
"""
import os
import sys
import torch
from pathlib import Path
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.detection_2d.models.yolo_wrapper import StudentModel
from config.settings import fed_cfg

def main():
    print("==================================================")
    print("[Teacher LoRA] Tiêm LoRA vào YOLO12l và Huấn luyện")
    print("==================================================")
    
    # 1. Khởi tạo Teacher với LoRA (sử dụng logic của StudentModel nhưng nhét trọng số Teacher vào)
    # Rank sẽ sử dụng mặc định LORA_RANK = 16 (giống Student)
    rank = fed_cfg.LORA_RANK
    teacher_ckpt = "yolo12l.pt"
    
    print(f"-> Loading {teacher_ckpt}...")
    
    # Thủ thuật: Dùng StudentModel wrapper nhưng truyền yolo12l.pt
    # Điều này sẽ giúp ta tự động thay thế Head và inject_lora với rank tương ứng
    teacher_lora = StudentModel(
        ckpt=teacher_ckpt, 
        rank=rank, 
        nc=4, # URPC2020 có 4 classes (holothurian, echinus, scallop, starfish)
        full_param=False, 
        use_lora=True
    )
    
    print("-> Sẵn sàng huấn luyện Teacher với LoRA...")
    
    # 2. Huấn luyện (Fine-tune) Teacher trên toàn bộ URPC2020
    # Chú ý: Train trên toàn bộ dataset chứ không phải 20% proxy data
    yaml_path = REPO_ROOT / "datasets/URPC2020.yaml"
    
    results = teacher_lora.yolo.train(
        data=str(yaml_path),
        epochs=200,  # 200 epochs để có Teacher LoRA cực mạnh
        imgsz=640,
        batch=8,     # Giảm batch size xuống 8 để tránh tràn VRAM trên A30
        workers=2,   # Giảm workers để tránh tràn RAM hệ thống (Exit code 137)
        device="cuda" if torch.cuda.is_available() else "cpu",
        project="runs/teacher_lora",
        name="yolo12l_lora_urpc",
        exist_ok=True,
        resume=True,  # Tự động resume nếu bị sập giữa chừng
    )
    
    # 3. Lưu mô hình lại
    save_path = REPO_ROOT / "yolo12l_lora_pretrained.pt"
    # Ưu tiên best.pt, fallback về last.pt nếu chưa có best
    best_weights = REPO_ROOT / "runs/teacher_lora/yolo12l_lora_urpc/weights/best.pt"
    last_weights = REPO_ROOT / "runs/teacher_lora/yolo12l_lora_urpc/weights/last.pt"
    chosen = best_weights if best_weights.exists() else last_weights
    if chosen.exists():
        import shutil
        shutil.copy(chosen, save_path)
        print(f"\n[Thành công] Đã lưu Teacher LoRA model tại: {save_path} (nguồn: {chosen.name})")
    else:
        print("\n[Lỗi] Không tìm thấy best.pt hoặc last.pt!")

if __name__ == "__main__":
    main()
