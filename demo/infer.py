import os
import sys

# Thêm thư mục gốc của project vào sys.path để Python nhận diện được package detection_2d
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ultralytics import YOLO
import detection_2d.compat  # Register shims for tasks.detection_2d

def main():
    # Thêm thư mục gốc của project vào sys.path để Python nhận diện được package detection_2d
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Trỏ đúng vào model đã finetune trong thư mục demo (do user đã copy sang)
    model_path = os.path.join(script_dir, 'student_lora_best.pt')
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found!")
        return

    print(f"Loading model from {model_path}...")
    from detection_2d.models.yolo_wrapper import StudentModel
    
    # Sử dụng wrapper StudentModel để load đúng các tham số LoRA
    # Cung cấp rõ lora_targets=['Conv'] vì tên file 'student_lora_best.pt' 
    # không chứa chữ '12n', khiến hàm is_nano bị False.
    
    # --- DIAGNOSTIC PRINT ---
    import torch
    print("Pre-loading checkpoint directly to inspect:")
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    m = ckpt.get('ema') or ckpt.get('model')
    conv2d_count = sum(1 for _, mod in m.named_modules() if isinstance(mod, torch.nn.Conv2d))
    print(f"Direct loaded model has {conv2d_count} torch.nn.Conv2d instances.")
    
    import detection_2d.models.lora as lora_mod
    lora_count = sum(1 for _, mod in m.named_modules() if isinstance(mod, lora_mod.LoRAConv2d))
    print(f"Direct loaded model has {lora_count} lora_mod.LoRAConv2d instances.")
    
    print("Class of LoRAConv2d in checkpoint:")
    for _, mod in m.named_modules():
        if "LoRAConv2d" in str(type(mod)):
            print(f"  Type: {type(mod)}")
            print(f"  Is lora_mod.LoRAConv2d? {isinstance(mod, lora_mod.LoRAConv2d)}")
            break
    # ------------------------
            
    student = StudentModel(ckpt=model_path, use_lora=True, lora_targets=['Conv'])
    # Bake LoRA vào base weights TRƯỚC KHI gọi val() để tránh model.fuse() xóa mất LoRA
    print("Baking LoRA weights...")
    student.bake_lora()
    
    model = student.yolo


    # Assuming the dataset configuration for validation is available.
    # Typically, you need to pass data='path/to/data.yaml' if not embedded in the model.
    # But YOLO evaluates on validation set automatically if you just call val() or you can predict on images.
    print("Running validation inference...")
    
    # Try running validation
    # Adjust 'data' argument if your dataset yaml is located elsewhere
    try:
        # Sử dụng đường dẫn tường minh thay vì lấy từ model checkpoint (có thể là đường dẫn cũ trên Kaggle)
        data_yaml = os.path.join(project_root, 'datasets', 'URPC2020.yaml')
        if not os.path.exists(data_yaml):
            print(f"Dataset config not found at {data_yaml}. Trying fallback...")
            data_yaml = os.path.join(project_root, 'datasets', 'URPC2020', 'data.yaml')

        metrics = model.val(data=data_yaml, half=False)
        print("Validation metrics:")
        print(f"mAP50: {metrics.box.map50}")
        print(f"mAP50-95: {metrics.box.map}")
    except Exception as e:
        print(f"Error during validation: {e}")
        print("Note: Please make sure the dataset configuration file is present in your datasets folder.")

if __name__ == '__main__':
    main()
