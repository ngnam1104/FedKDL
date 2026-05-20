import os
import random
import yaml
from pathlib import Path
import numpy as np

def create_client_datasets_yolo(
    base_data_yaml: str,
    output_dir: str,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42
):
    """
    Tạo các cấu hình data.yaml cho từng client (Non-IID Dirichlet).
    Dành riêng cho YOLO (Ultralytics).
    
    base_data_yaml: Đường dẫn tới file yaml gốc của dataset (vd: URPC2020.yaml)
    output_dir: Thư mục lưu các file yaml của từng client
    num_clients: Số lượng Sensor Nodes
    alpha: Tham số Dirichlet (càng nhỏ càng Non-IID)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(base_data_yaml):
        # Trả về chế độ dry-run / synthetic
        print(f"Warning: Không tìm thấy {base_data_yaml}. Chạy ở chế độ synthetic/dry-run.")
        return _create_synthetic_client_yamls(output_dir, num_clients)
        
    with open(base_data_yaml, 'r') as f:
        base_cfg = yaml.safe_load(f)
        
    # Phân tích dataset gốc
    train_path = base_cfg.get('train', '')
    if isinstance(train_path, str) and train_path.endswith('.txt'):
        with open(train_path, 'r') as f:
            all_images = [line.strip() for line in f.readlines()]
    else:
        # Nếu path là thư mục, lấy danh sách ảnh
        dataset_dir = Path(base_data_yaml).parent
        img_dir = dataset_dir / train_path
        all_images = [str(p) for p in img_dir.glob('**/*.jpg')]
        
    num_classes = base_cfg.get('nc', 1)
    
    # Do YOLO label lưu ở file .txt rời, việc tính toán phân phối nhãn rất tốn kém IO.
    # Để đơn giản trong mô phỏng, ta sẽ partition số lượng ảnh theo Dirichlet thay vì nhãn,
    # hoặc giả định một phân phối nhãn xấp xỉ.
    # Trong môi trường dưới nước, ta dùng Dirichlet cho số lượng mẫu (Quantity Skew).
    
    proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
    proportions = proportions / proportions.sum()
    
    num_samples = len(all_images)
    client_splits = (proportions * num_samples).astype(int)
    
    # Sửa lỗi làm tròn
    client_splits[-1] = num_samples - sum(client_splits[:-1])
    
    random.shuffle(all_images)
    
    client_yamls = []
    current_idx = 0
    for i in range(num_clients):
        client_imgs = all_images[current_idx : current_idx + client_splits[i]]
        current_idx += client_splits[i]
        
        # Ghi file txt chứa danh sách ảnh cho client i
        client_txt = os.path.join(output_dir, f"client_{i}_train.txt")
        with open(client_txt, 'w') as f:
            f.write("\n".join(client_imgs))
            
        # Ghi file yaml cho client i
        client_cfg = base_cfg.copy()
        client_cfg['train'] = client_txt  # Override tập train
        
        # Validation giữ nguyên để so sánh công bằng
        client_yaml = os.path.join(output_dir, f"client_{i}.yaml")
        with open(client_yaml, 'w') as f:
            yaml.safe_dump(client_cfg, f)
            
        client_yamls.append(client_yaml)
        
    print(f"Đã tạo {num_clients} client datasets tại {output_dir}")
    return client_yamls

def _create_synthetic_client_yamls(output_dir, num_clients):
    """
    Tạo dummy yaml cho dry-run khi không có dữ liệu thật.
    Sử dụng coco8 mặc định của Ultralytics làm fallback.
    """
    client_yamls = []
    for i in range(num_clients):
        dummy_yaml = os.path.join(output_dir, f"dummy_client_{i}.yaml")
        with open(dummy_yaml, 'w') as f:
            # Fallback dùng tập COCO8 siêu nhỏ có sẵn trong ultralytics
            f.write("path: coco8\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n")
            f.write("nc: 80\n")
            f.write("names: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '...']\n")
        client_yamls.append(dummy_yaml)
    return client_yamls
