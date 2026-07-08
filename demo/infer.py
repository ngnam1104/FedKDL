import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import detection_2d.compat  # Register tasks.detection_2d -> detection_2d shims
from ultralytics import YOLO
from detection_2d.models.yolo_wrapper import StudentModel

def main():
    model_path = os.path.join(script_dir, 'student_lora_best.pt')
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found!")
        return

    print(f"Loading model from {model_path}...")
    
    # Use StudentModel to load LoRA parameters correctly
    # Force lora_targets=['Conv'] because the filename does not contain '12n'
    student = StudentModel(ckpt=model_path, use_lora=True, lora_targets=['Conv'])
    
    # Bake LoRA into base weights BEFORE calling val() to prevent model.fuse() from discarding LoRA
    print("Baking LoRA weights...")
    student.bake_lora()
    
    model = student.yolo

    print("Running validation inference...")
    try:
        data_yaml = os.path.join(project_root, 'datasets', 'URPC2020.yaml')
        if not os.path.exists(data_yaml):
            print(f"Dataset config not found at {data_yaml}. Trying fallback...")
            data_yaml = os.path.join(project_root, 'datasets', 'URPC2020', 'data.yaml')

        # Use half=False to avoid FP16 precision loss with LoRA weights
        metrics = model.val(data=data_yaml, half=False)
        print("Validation metrics:")
        print(f"mAP50: {metrics.box.map50}")
        print(f"mAP50-95: {metrics.box.map}")
    except Exception as e:
        print(f"Error during validation: {e}")
        print("Note: Please make sure the dataset configuration file is present in your datasets folder.")

if __name__ == '__main__':
    main()
