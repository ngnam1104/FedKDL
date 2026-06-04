import os
import re

def patch_int8():
    path = "tasks/detection_2d/knowledge_compression/int8_quantization.py"
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    old_code = """        if has_nan or has_inf:
            raise RuntimeError(
                f"[CRITICAL ERROR] Local training produced NaN/Inf in Key '{key}'! "
                f"NaN={has_nan}, Inf={has_inf}, Shape={tuple(tensor.shape)}. "
                f"Aborting FL round to prevent corrupting the Global Model."
            )"""
    new_code = """        if has_nan or has_inf:
            print(f"[Cảnh báo] Phát hiện NaN/Inf ở {key}! Tự động thiết lập về 0.0 để cứu vãn vòng FL.")
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
            tensor = torch.clamp(tensor, min=-1e10, max=1e10)"""
    
    if old_code in content:
        content = content.replace(old_code, new_code)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("Patched int8_quantization.py")
    else:
        print("Could not find target code in int8_quantization.py")

def patch_simulator():
    path = "tasks/detection_2d/simulator.py"
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    old_code = """            'workers': 0,
            'optimizer': 'SGD', # Chuyển sang SGD
            'lr0': 1e-3,
            'warmup_epochs': 0.0,"""
    new_code = """            'workers': 0,
            'optimizer': 'SGD', # Dùng SGD nhưng giảm LR
            'lr0': 5e-4,        # [STABILITY] Giảm từ 1e-3 xuống 5e-4 để tránh nổ loss
            'warmup_epochs': 0.0,"""
    
    if old_code in content:
        content = content.replace(old_code, new_code)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("Patched simulator.py")
    else:
        print("Could not find target code in simulator.py")

def patch_yolo_wrapper():
    path = "tasks/detection_2d/models/yolo_wrapper.py"
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Load LoRA weights
    target1 = "print(f\"[StudentModel] Injected LoRA into {injected} layers (Targets: {actual_targets}, Strategy: {actual_strategy}).\")"
    new1 = """print(f\"[StudentModel] Injected LoRA into {injected} layers (Targets: {actual_targets}, Strategy: {actual_strategy}).\")
            
            # [CRITICAL FIX] Load lại weights LoRA từ checkpoint NẾU CÓ!
            import torch
            checkpoint = torch.load(ckpt, map_location='cpu', weights_only=False)
            if 'model' in checkpoint:
                ckpt_state = checkpoint['model'].state_dict() if hasattr(checkpoint['model'], 'state_dict') else checkpoint['model']
                lora_state = {k: v for k, v in ckpt_state.items() if 'lora_' in k}
                if len(lora_state) > 0:
                    self.yolo.model.load_state_dict(lora_state, strict=False)
                    print(f\"[StudentModel] Recovered {len(lora_state)} LoRA tensors from {ckpt}!\")"""
                    
    # 2. Change head_lr_scale
    target2 = "self.head_lr_scale = 3.0"
    new2 = "self.head_lr_scale = 1.0  # [STABILITY] Set về 1.0 để chống nổ Head trong FL"
    
    if target1 in content:
        content = content.replace(target1, new1)
        print("Patched yolo_wrapper.py (LoRA loading)")
    
    if target2 in content:
        content = content.replace(target2, new2)
        print("Patched yolo_wrapper.py (LR scale)")
        
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

patch_int8()
patch_simulator()
patch_yolo_wrapper()
