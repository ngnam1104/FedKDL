import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ultralytics import YOLO

def extract_cls_OLD(p):
    train_out = p[1] if (isinstance(p, tuple) and len(p) > 1 and isinstance(p[1], dict)) else p
    if isinstance(train_out, dict):
        if 'cls' in train_out: return train_out['cls']
        elif 'pred_scores' in train_out: return train_out['pred_scores']
        
    if isinstance(p, torch.Tensor):
        return p[:, 4:, :]
    return None

def extract_cls_NEW(p):
    train_out = p[1] if (isinstance(p, tuple) and len(p) > 1 and isinstance(p[1], dict)) else p
    if isinstance(train_out, dict):
        if 'scores' in train_out: return train_out['scores']
        elif 'cls' in train_out: return train_out['cls']
        elif 'pred_scores' in train_out: return train_out['pred_scores']
        
    if isinstance(p, torch.Tensor):
        return p[:, 4:, :]
    return None

def main():
    print("=== KIỂM CHỨNG BUG KD TRÊN YOLOv12 (SHAPE MISMATCH DO DICT KEY) ===\n")
    
    print("1. Tải Teacher Model (yolo12l_lora_pretrained.pt) ở Eval Mode...")
    try:
        teacher = YOLO('yolo12l_lora_pretrained.pt')
        teacher.model.eval()
    except Exception as e:
        print(f"Không thể tải model: {e}")
        return

    imgs = torch.rand(2, 3, 640, 640)
    print("2. Chạy Forward pass trên ảnh ảo...\n")
    with torch.no_grad():
        t_preds = teacher.model(imgs)
        
    print("3. Phân tích Output format của Teacher YOLOv12:")
    print(f"   - Type của t_preds: {type(t_preds)}")
    if isinstance(t_preds, tuple):
        train_out = t_preds[1]
        print(f"   - Type của phần tử thứ 2 (train_out): {type(train_out)}")
        if isinstance(train_out, dict):
            print(f"   - CÁC KEYS TRONG DICT: {list(train_out.keys())}\n")
            
    t_cls_old = extract_cls_OLD(t_preds)
    print("4. Trích xuất logit bằng Logic CŨ (Chỉ check 'cls' và 'pred_scores'):")
    print(f"   -> Kết quả t_cls: {t_cls_old} (None)\n")
    
    t_cls_new = extract_cls_NEW(t_preds)
    print("5. Trích xuất logit bằng Logic MỚI (Đã thêm check 'scores'):")
    if t_cls_new is not None:
        print(f"   -> Kết quả t_cls: tensor, shape = {t_cls_new.shape}\n")
    else:
        print("   -> Kết quả t_cls: None\n")
        
    print("=== HẬU QUẢ VỚI KNOWLEDGE DISTILLATION ===")
    print("- Vì Logic CŨ trả về None, lệnh `if s_cls.shape == t_cls.shape` chắc chắn là FALSE.")
    print("- KD_Loss ngay lập tức bị ép về 0.0 (Bị vô hiệu hóa ngầm).")
    print("- Student_Loss = (YOLO_Loss * 0.5) + 0.0.")
    print("- Do đó, Student học với tốc độ CHẬM ĐI MỘT NỬA so với No-KD, và không học được gì từ Teacher!")

if __name__ == '__main__':
    main()
