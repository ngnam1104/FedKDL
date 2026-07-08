import os
import sys
import torch

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import detection_2d.compat
from ultralytics import YOLO

def main():
    model_path = os.path.join(project_root, 'demo', 'student_lora_best.pt')
    print(f"--- Kiểm tra kiến trúc của {model_path} ---")
    
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # 1. Kiểm tra cấu trúc file checkpoint
    print(f"\n[1] Checkpoint keys: {list(ckpt.keys())}")
    
    state = None
    if 'model' in ckpt:
        state = ckpt['model'].state_dict() if hasattr(ckpt['model'], 'state_dict') else ckpt['model']
    else:
        print("Không tìm thấy key 'model' trong checkpoint.")
        return

    # 2. Kiểm tra các tham số LoRA
    lora_keys = [k for k in state.keys() if 'lora_' in k]
    print(f"\n[2] Tổng số tensor: {len(state)}")
    print(f"Số lượng tensor LoRA (chứa 'lora_'): {len(lora_keys)}")
    
    if len(lora_keys) > 0:
        print(f"Ví dụ một số LoRA keys: {lora_keys[:5]}")
        # Kiểm tra giá trị của lora_A, lora_B có toàn số 0 không
        print("\nKiểm tra giá trị các tensor LoRA:")
        for k in lora_keys[:4]:
            mean_val = state[k].abs().mean().item()
            print(f" - {k}: abs_mean = {mean_val:.6f}")
    else:
        print("\nWARNING: KHÔNG TÌM THẤY BẤT KỲ TENSOR LORA NÀO TRONG TRỌNG SỐ!")
        
    # 3. Kiểm tra số class (nc)
    model_nc = getattr(ckpt['model'], 'nc', 'Không rõ') if hasattr(ckpt['model'], 'nc') else 'Không rõ'
    yaml_nc = ckpt.get('model').yaml.get('nc', 'Không rõ') if hasattr(ckpt.get('model'), 'yaml') else 'Không rõ'
    print(f"\n[3] Số classes (nc): model.nc = {model_nc}, yaml['nc'] = {yaml_nc}")

if __name__ == '__main__':
    main()
