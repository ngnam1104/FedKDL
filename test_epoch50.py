import sys
from pathlib import Path
from ultralytics import YOLO
import torch

# Cần import lora để PyTorch có thể deserialize các lớp LoRAConv2d trong file pt
import tasks.detection_2d.models.lora

def main():
    ckpt_path = "epoch50.pt"
    data_yaml = "datasets/URPC2020.yaml"
    
    if not Path(ckpt_path).exists():
        print(f"❌ Không tìm thấy file: {ckpt_path}")
        return
        
    print(f"🚀 Đang tải mô hình từ: {ckpt_path}")
    # Load model
    model = YOLO(ckpt_path)
    
    print(f"\n📊 Bắt đầu evaluate trên tập validation ({data_yaml})...")
    # Tắt half/amp nếu có lỗi, nhưng mặc định nên để YOLO tự lo
    metrics = model.val(
        data=data_yaml,
        split="val",
        batch=1,         # Giảm xuống 1 để dùng ít VRAM nhất có thể (< 1GB)
        half=True,       # Ép dùng FP16 để giảm nửa VRAM
        imgsz=640,
        device="cuda" if torch.cuda.is_available() else "cpu",
        plots=False,  
        verbose=True
    )
    
    print("\n" + "="*50)
    print("🏆 KẾT QUẢ EVALUATION (EPOCH 50)")
    print("="*50)
    print(f"  - mAP@50     : {metrics.box.map50:.4f}")
    print(f"  - mAP@50-95  : {metrics.box.map:.4f}")
    print(f"  - Precision  : {metrics.box.mp:.4f}")
    print(f"  - Recall     : {metrics.box.mr:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
