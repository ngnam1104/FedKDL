"""
train_teacher_lora.py
Khởi tạo và huấn luyện Teacher model (YOLO12l) với LoRA.
Mục tiêu: Tạo ra global_teacher_lora.pt đã chứa Domain Adaptation (URPC).
Sử dụng rank = 8 cho neck_head, rank = 2 cho mid backbone.
"""
import torch
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
from tasks.detection_2d.models.yolo_wrapper import StudentModel

def main():
    print("🚀 Bắt đầu huấn luyện Teacher (YOLO12l) với LoRA")
    
    # 1. Khởi tạo Teacher Model (YOLO12l) kèm LoRA
    # Strategy 'adaptive': bỏ qua 0-3, rank 2 cho 4-9, rank 8 cho >= 10
    teacher_model = StudentModel(
        ckpt="yolo12l.pt",
        rank=8,
        nc=4, # 4 classes cho URPC
        full_param=False,
        use_lora=True
    )
    
    print("\n✅ Đã tiêm LoRA vào Teacher.")
    
    # Đóng băng weights gốc, chỉ train lora_A và lora_B
    # StudentModel đã tự động làm việc này nếu full_param=False
    
    # 2. Bắt đầu Train nhẹ (10-20 epochs) trên URPC dataset
    epochs = 15 # Train nhẹ 15 epochs
    print(f"\n🏃 Bắt đầu train Teacher {epochs} epochs trên URPC Dataset...")
    
    # Dataset config: Thay bằng file yaml dataset URPC của bạn
    dataset_yaml = "dataset.yaml" # Thay bằng path thật nếu cần
    if not os.path.exists(dataset_yaml):
        # Tạo file dummy dataset yaml để chạy tạm nếu không có
        print(f"⚠️ Không tìm thấy {dataset_yaml}, hãy đảm bảo bạn trỏ đúng file config data.")
        dataset_yaml = "urpc.yaml" 
        
    try:
        teacher_model.yolo.train(
            data=dataset_yaml,
            epochs=epochs,
            imgsz=640,
            batch=8, # Teacher model khá nặng, batch 8 là vừa phải
            device="cuda" if torch.cuda.is_available() else "cpu",
            project="runs/fedkdl",
            name="global_teacher_lora",
            exist_ok=True,
            val=True
        )
        print("\n✅ Huấn luyện Teacher hoàn tất.")
        
        # 3. Lưu lại kết quả
        best_pt_path = Path("runs/fedkdl/global_teacher_lora/weights/best.pt")
        output_path = Path("global_teacher_lora.pt")
        
        if best_pt_path.exists():
            import shutil
            shutil.copy(best_pt_path, output_path)
            print(f"🎉 Đã lưu Teacher model đã train LoRA tại: {output_path}")
        else:
            print("⚠️ Không tìm thấy file best.pt sau khi train.")
            
    except Exception as e:
        print(f"❌ Có lỗi trong quá trình train: {e}")

if __name__ == "__main__":
    main()
