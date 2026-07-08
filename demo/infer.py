import os
from ultralytics import YOLO
import detection_2d.compat  # Register shims for tasks.detection_2d

def main():
    # Lấy thư mục chứa file infer.py hiện tại
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'student_lora_best.pt')
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found!")
        return

    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)

    # Assuming the dataset configuration for validation is available.
    # Typically, you need to pass data='path/to/data.yaml' if not embedded in the model.
    # But YOLO evaluates on validation set automatically if you just call val() or you can predict on images.
    print("Running validation inference...")
    
    # Try running validation
    # Adjust 'data' argument if your dataset yaml is located elsewhere
    try:
        metrics = model.val()
        print("Validation metrics:")
        print(f"mAP50: {metrics.box.map50}")
        print(f"mAP50-95: {metrics.box.map}")
    except Exception as e:
        print(f"Error during validation: {e}")
        print("Note: You might need to specify the data=... argument in model.val(data='your_dataset.yaml')")

if __name__ == '__main__':
    main()
