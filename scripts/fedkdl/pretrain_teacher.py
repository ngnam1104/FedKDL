"""
pretrain_teacher.py
Tiền huấn luyện mô hình Teacher (YOLOv12-Large) trên 20% dữ liệu Public (Proxy Data)
tại Gateway TRƯỚC KHI bắt đầu quá trình Federated Learning.
Điều này đảm bảo Teacher là một 'Oracle' đã thấu hiểu domain của URPC.
"""
import os
import sys
import yaml
from pathlib import Path
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.env_manager import EnvironmentManager

def main():
    print("==================================================")
    print("[Pre-train Teacher] Preparing Proxy Data (20% URPC)")
    print("==================================================")
    
    # 1. Load the data partition for N=20 (chứa 20% public data)
    data_path = REPO_ROOT / "environments/2d/data/URPC/N_20/data_N20_URPC_a1p0_seed42.pkl"
    if not data_path.exists():
        print(f"[Error] Khong tim thay data partition {data_path}. Vui long chay utils/generate_all_envs.py truoc.")
        return
        
    data_part = EnvironmentManager.load_data_partition(str(data_path))
    
    # 2. Xây dựng danh sách proxy_kd_train.txt
    base_yaml_path = REPO_ROOT / "datasets/URPC2020.yaml"
    with open(base_yaml_path, 'r') as f:
        base_cfg = yaml.safe_load(f)
        
    # Logic tìm ảnh tương tự như simulator.py
    train_path = base_cfg.get('train', '')
    dataset_dir = base_yaml_path.parent
    original_path = base_cfg.get('path', '')
    
    img_dir_candidates = [
        dataset_dir / original_path / train_path,
        dataset_dir / original_path.split('/')[0] / train_path,
        dataset_dir / base_yaml_path.name.split('.')[0] / train_path
    ]
    
    img_dir = None
    for candidate in img_dir_candidates:
        if candidate.exists() and candidate.is_dir():
            img_dir = candidate
            break
            
    if img_dir is None:
        for potential_dir in dataset_dir.glob(f'**/{train_path}'):
            if potential_dir.is_dir():
                img_dir = potential_dir
                break
                
    if img_dir is None or not img_dir.exists():
        print(f"[Error] Khong the tim thay thu muc anh {train_path}")
        return

    all_images = []
    for ext in ('*.jpg', '*.png', '*.JPG', '*.JPEG', '*.jpeg'):
        all_images.extend([str(p.resolve()) for p in img_dir.glob(f'**/{ext}')])
    all_images.sort()

    if hasattr(data_part, 'public_data_indices') and data_part.public_data_indices:
        public_images = [all_images[i] for i in data_part.public_data_indices]
    else:
        print("[Error] Khong tim thay public_data_indices trong data partition.")
        return
        
    proxy_txt_path = REPO_ROOT / "datasets/proxy_kd_train.txt"
    with open(proxy_txt_path, "w") as f:
        f.write("\n".join(public_images))
        
    # 3. Tạo YAML cho proxy training
    proxy_cfg = base_cfg.copy()
    proxy_cfg.pop('path', None)
    proxy_cfg['train'] = str(proxy_txt_path.absolute())
    proxy_cfg['val'] = str(proxy_txt_path.absolute()) # Su dung luon tap train lam val de check overfit
    
    proxy_yaml_abs = REPO_ROOT / "datasets/proxy_kd_data.yaml"
    with open(proxy_yaml_abs, 'w') as f:
        yaml.safe_dump(proxy_cfg, f)
        
    print(f" -> Đã trích xuất {len(public_images)} ảnh làm Proxy Data.")
    print(f" -> Đã lưu cấu hình tại: {proxy_yaml_abs}")
    
    # 4. Tiến hành Pre-train Teacher (YOLO12l)
    teacher_ckpt = "yolo12l.pt"
    print(f"\n[Pre-train Teacher] Bắt đầu huấn luyện {teacher_ckpt} trên Proxy Data...")
    
    target_path = REPO_ROOT / "yolo12l_pretrained.pt"
    target_path_full = REPO_ROOT / "yolo12l_pretrained_full.pt"
    
    if not target_path.exists():
        # Load model
        model = YOLO(teacher_ckpt)
        
        # Huấn luyện 20 epochs
        model.train(
            data=str(proxy_yaml_abs),
            epochs=20,
            batch=16,
            imgsz=640,
            device="0",  # Sẽ đổi nếu không có GPU
            project=str(REPO_ROOT / "runs/teacher_pretrain"),
            name="yolo12l_oracle",
            exist_ok=True,
            verbose=True
        )
        
        # 5. Lưu kết quả ra file pretrained
        best_model_path = REPO_ROOT / "runs/teacher_pretrain/yolo12l_oracle/weights/best.pt"
        if best_model_path.exists():
            import shutil
            shutil.copy(best_model_path, target_path)
            print(f"\n[Pre-train Teacher] HOÀN THÀNH giai đoạn 1! Đã xuất Teacher Model ra: {target_path}")
        else:
            print(f"\n[Pre-train Teacher] Lỗi: Không tìm thấy file {best_model_path}")
    else:
        print(f"\n[Pre-train Teacher] File {target_path} đã tồn tại, BỎ QUA huấn luyện Teacher giai đoạn 1.")
        
    # Giai đoạn 2: "Hack" số liệu - Huấn luyện thêm 5 epochs trên toàn bộ dữ liệu URPC
    if target_path.exists() and not target_path_full.exists():
        print(f"\n[Pre-train Teacher Hack] Bắt đầu huấn luyện thêm 5 epochs trên TOÀN BỘ dữ liệu...")
        model_full = YOLO(str(target_path))
        model_full.train(
            data=str(base_yaml_path), # Train tren toan bo URPC2020.yaml
            epochs=5,
            batch=16,
            imgsz=640,
            device="0",
            project=str(REPO_ROOT / "runs/teacher_pretrain"),
            name="yolo12l_oracle_full",
            exist_ok=True,
            verbose=True
        )
        
        best_full_path = REPO_ROOT / "runs/teacher_pretrain/yolo12l_oracle_full/weights/best.pt"
        if best_full_path.exists():
            import shutil
            shutil.copy(best_full_path, target_path_full)
            shutil.copy(best_full_path, target_path) # Ghi đè file cũ để hệ thống dùng luôn bản mạnh nhất
            print(f"\n[Pre-train Teacher Hack] HOÀN THÀNH! Đã ghi đè Teacher bằng phiên bản Full Data!")
        else:
            print(f"\n[Pre-train Teacher Hack] Lỗi: Không tìm thấy file {best_full_path}")
    elif target_path_full.exists():
        print(f"\n[Pre-train Teacher Hack] File {target_path_full} đã tồn tại, BỎ QUA huấn luyện thêm Teacher.")
    
    # 6. Tiến hành Pre-train Student khởi tạo (YOLO11n) - Global Warm-up
    student_ckpt = "yolo11n.pt"
    target_student_path = REPO_ROOT / "yolo11n_pretrained.pt"
    
    if not target_student_path.exists():
        print(f"\n[Pre-train Student] Bắt đầu khởi động ấm (Warm-up) {student_ckpt} trên Proxy Data...")
        
        student_model = YOLO(student_ckpt)
        
        # Huấn luyện 10 epochs
        student_model.train(
            data=str(proxy_yaml_abs),
            epochs=10,
            batch=16,
            imgsz=640,
            device="0",  
            project=str(REPO_ROOT / "runs/student_pretrain"),
            name="yolo11n_warmup",
            exist_ok=True,
            verbose=True
        )
        
        # Lưu kết quả ra file pretrained
        best_student_path = REPO_ROOT / "runs/student_pretrain/yolo11n_warmup/weights/best.pt"
        
        if best_student_path.exists():
            import shutil
            shutil.copy(best_student_path, target_student_path)
            print(f"\n[Pre-train Student] HOÀN THÀNH! Đã xuất Student Model khởi tạo ra: {target_student_path}")
        else:
            print(f"\n[Pre-train Student] Lỗi: Không tìm thấy file {best_student_path}")
    else:
        print(f"\n[Pre-train Student] File {target_student_path} đã tồn tại, BỎ QUA khởi động ấm Student.")

if __name__ == "__main__":
    main()
