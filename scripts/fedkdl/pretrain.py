"""
pretrain.py
1. Tiền huấn luyện mô hình Teacher (YOLO12l) trên toàn bộ dữ liệu (5 epochs).
2. Tiền huấn luyện mô hình Student (YOLO11n) - Khởi động ấm (Warm-up) 
   trên 20% dữ liệu Public (Proxy Data) (3 epochs).
"""
import os
import sys
import yaml
import gc
import torch
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
    
    # 1. Load the data partition for N=30 (chứa 20% public data)
    data_path = REPO_ROOT / "environments/2d/data/URPC/N_30/data_N30_URPC_a2p0_seed42.pkl"
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
    
    # 4. Giai đoạn 2: Tiến hành huấn luyện thêm Teacher (YOLO12l) trên TOÀN BỘ dữ liệu (5 epochs)
    teacher_ckpt = REPO_ROOT / "yolo12l_pretrained.pt"
    target_teacher_path_full = REPO_ROOT / "yolo12l_pretrained_full.pt"
    
    if teacher_ckpt.exists() and not target_teacher_path_full.exists():
        print(f"\n[Pre-train Teacher Hack] Bắt đầu huấn luyện thêm 5 epochs trên TOÀN BỘ dữ liệu từ checkpoint {teacher_ckpt}...")
        teacher_model = YOLO(str(teacher_ckpt))
        teacher_model.train(
            data=str(base_yaml_path), # Toàn bộ URPC2020.yaml
            epochs=5,
            batch=16,
            imgsz=640,
            device="0",
            project=str(REPO_ROOT / "runs/teacher_pretrain"),
            name="yolo12l_oracle_full",
            exist_ok=True,
            verbose=True,
            workers=4,
            plots=False
        )
        
        best_teacher_path = REPO_ROOT / "runs/teacher_pretrain/yolo12l_oracle_full/weights/best.pt"
        if best_teacher_path.exists():
            import shutil
            shutil.copy(best_teacher_path, target_teacher_path_full)
            shutil.copy(best_teacher_path, teacher_ckpt) # Ghi đè file cũ để Simulator dùng bản mạnh nhất
            print(f"\n[Pre-train Teacher Hack] HOÀN THÀNH! Đã ghi Teacher bằng phiên bản Full Data!")
        else:
            print(f"\n[Pre-train Teacher Hack] Lỗi: Không tìm thấy file {best_teacher_path}")
            
        del teacher_model
        gc.collect()
        torch.cuda.empty_cache()
    elif target_teacher_path_full.exists():
        print(f"\n[Pre-train Teacher Hack] File {target_teacher_path_full} đã tồn tại, BỎ QUA huấn luyện thêm Teacher.")
    else:
        print(f"\n[Pre-train Teacher Hack] Lỗi: Không tìm thấy {teacher_ckpt} để train tiếp. Vui lòng chuẩn bị file này trước.")

    # 5. Tiến hành Pre-train Student khởi tạo (YOLO11n) - Global Warm-up
    student_ckpt = "yolo11n.pt"
    target_student_path = REPO_ROOT / "yolo11n_pretrained.pt"
    
    if not target_student_path.exists():
        print(f"\n[Pre-train Student] Bắt đầu khởi động ấm (Warm-up) {student_ckpt} trên Proxy Data (3 epochs)...")
        
        print("\n=== [Kiểm tra bộ nhớ (RAM/VRAM) trước khi Train Student] ===")
        try:
            import psutil
            vm = psutil.virtual_memory()
            print(f"[-] System RAM: Dùng {vm.used / (1024**3):.2f} GB / Tổng {vm.total / (1024**3):.2f} GB (Trống: {vm.available / (1024**3):.2f} GB)")
        except ImportError:
            print("[-] System RAM: (Thư viện psutil chưa cài đặt, không thể đo lường)")
            
        import torch
        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            vram_total = torch.cuda.get_device_properties(device_id).total_memory
            vram_allocated = torch.cuda.memory_allocated(device_id)
            vram_reserved = torch.cuda.memory_reserved(device_id)
            vram_free = vram_total - vram_reserved
            print(f"[-] GPU VRAM  : Đã cấp phát {vram_allocated / (1024**3):.2f} GB, Đã giữ {vram_reserved / (1024**3):.2f} GB / Tổng {vram_total / (1024**3):.2f} GB")
            print(f"[-] GPU VRAM (Trống thực tế): {vram_free / (1024**3):.2f} GB")
        else:
            print("[-] GPU VRAM: Không tìm thấy GPU CUDA.")
        print("============================================================\n")
        
        student_model = YOLO(student_ckpt)
        
        # Huấn luyện 3 epochs thay vì 10
        student_model.train(
            data=str(proxy_yaml_abs),
            epochs=3,
            batch=16,
            imgsz=640,
            device="0",  
            project=str(REPO_ROOT / "runs/student_pretrain"),
            name="yolo11n_warmup",
            exist_ok=True,
            verbose=True,
            workers=2,    # Giảm số luồng load data để tránh tràn RAM hệ thống
            plots=False   # Tắt vẽ biểu đồ ở cuối epoch để tránh lỗi Crash
        )
        
        # Lưu kết quả ra file pretrained
        best_student_path = REPO_ROOT / "runs/student_pretrain/yolo11n_warmup/weights/best.pt"
        
        if best_student_path.exists():
            import shutil
            shutil.copy(best_student_path, target_student_path)
            print(f"\n[Pre-train Student] HOÀN THÀNH! Đã xuất Student Model khởi tạo ra: {target_student_path}")
        else:
            print(f"\n[Pre-train Student] Lỗi: Không tìm thấy file {best_student_path}")
            
        del student_model
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print(f"\n[Pre-train Student] File {target_student_path} đã tồn tại, BỎ QUA khởi động ấm Student.")

if __name__ == "__main__":
    main()
