"""
pretrain.py
1. Tiền huấn luyện mô hình Teacher (YOLO12l) trên toàn bộ dữ liệu (5 epochs).
2. Tiền huấn luyện mô hình Student (yolo12n) - Khởi động ấm (Warm-up) 
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
    data_path = REPO_ROOT / "environments/2d/data/URPC/N_30/data_N30_URPC_a1p0_seed42.pkl"
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
    
    # 4. Giai đoạn 2: Tiến hành huấn luyện thêm Teacher (YOLO12l) trên TOÀN BỘ dữ liệu
    teacher_ckpt = REPO_ROOT / "yolo12l_pretrained.pt"
    target_teacher_path_full = REPO_ROOT / "yolo12l_pretrained_full.pt"

    best_teacher_path = REPO_ROOT / "runs/teacher_pretrain/yolo12l_oracle_full/weights/best.pt"

    # ── CHIẾN LƯỢC TRAINING MỚI ─────────────────────────────────────────────────
    # Phân tích results.csv cho thấy:
    #   - Model tốt nhất (best.pt) đạt được ở epoch ~43 (mAP50=0.749, Recall=0.682)
    #   - Sau epoch 43, val loss bắt đầu tăng nhẹ trong khi train loss vẫn giảm
    #     → dấu hiệu OVERFIT, KHÔNG NÊN resume từ last.pt (epoch 96)
    #
    # Giải pháp: Fine-tune từ best.pt với một run MỚI, thêm augmentation mạnh
    # để mô hình thoát khỏi local minima và cải thiện Recall:
    #   - mixup=0.15: trộn ảnh để mô hình dám detect ở vùng nhập nhòe
    #   - copy_paste=0.1: copy-paste object để tăng đa dạng hóa
    #   - cls=0.3 (giảm từ 0.5): giảm penalty phân loại, ép mô hình dám detect khi không chắc
    #   - lr0=0.001: LR nhỏ hơn để fine-tune ổn định, không phá vỡ weights tốt
    #   - patience=80: đủ lớn để không bị early stop quá sớm
    #   - epochs=200: đủ dài để vượt qua giai đoạn plateau và đạt điểm mới
    # ────────────────────────────────────────────────────────────────────────────

    if best_teacher_path.exists() and not target_teacher_path_full.exists():
        print(f"\n[Pre-train Teacher] TÌM THẤY BEST.PT! Bắt đầu Fine-tune từ checkpoint tốt nhất...")
        print(f"  → Source: {best_teacher_path}")
        print(f"  → Strategy: Fine-tune mới với Augmentation mạnh để boost Recall")
        print(f"  → Epochs: 200 | LR: 0.001 | Patience: 80 | Augmentation: mixup+copy_paste")

        teacher_model = YOLO(str(best_teacher_path))
        teacher_model.train(
            data=str(base_yaml_path),       # Toàn bộ URPC2020.yaml
            epochs=50,                      # Huấn luyện thêm 50 epoch
            patience=0,                     # Vô hiệu hóa Early Stopping
            batch=16,
            imgsz=640,
            device="0",
            lr0=0.001,                      # LR nhỏ hơn để fine-tune ổn định
            lrf=0.01,                       # Cosine anneal xuống 0.001*0.01 = 1e-5 ở cuối
            momentum=0.937,
            weight_decay=0.0005,
            # Augmentation mạnh hơn để thoát overfit và tăng Recall
            mixup=0.15,                     # Trộn ảnh → mô hình dám detect vùng nhập nhòe
            copy_paste=0.1,                 # Copy-paste object → đa dạng hóa vị trí object
            mosaic=1.0,                     # Giữ nguyên mosaic
            degrees=5.0,                    # Xoay nhẹ để robust hơn
            # Giảm cls weight để mô hình dám detect khi không chắc → tăng Recall
            cls=0.3,                        # Giảm từ 0.5 xuống 0.3
            box=7.5,                        # Giữ nguyên box regression weight
            project=str(REPO_ROOT / "runs/teacher_pretrain"),
            name="yolo12l_oracle_finetune_v2", # Tên run mới, không ghi đè run cũ
            exist_ok=True,
            verbose=True,
            workers=4,
            plots=False,
        )

        best_finetune_path = REPO_ROOT / "runs/teacher_pretrain/yolo12l_oracle_finetune_v2/weights/best.pt"
        last_finetune_path = REPO_ROOT / "runs/teacher_pretrain/yolo12l_oracle_finetune_v2/weights/last.pt"

        if best_finetune_path.exists():
            import shutil
            shutil.copy(best_finetune_path, target_teacher_path_full)
            shutil.copy(best_finetune_path, teacher_ckpt)   # Ghi đè để Simulator dùng bản mạnh nhất
            print(f"\n[Pre-train Teacher] HOÀN THÀNH FINE-TUNE!")
            print(f"  → Đã xuất Teacher (best) vào: {teacher_ckpt}")

            if last_finetune_path.exists():
                import shutil as _shutil
                last_ckpt = REPO_ROOT / "yolo12l_last_pretrained.pt"
                _shutil.copy(last_finetune_path, last_ckpt)
                print(f"  → Đã lưu last.pt vào: {last_ckpt}")
        else:
            print(f"\n[Pre-train Teacher] Lỗi: Không tìm thấy {best_finetune_path}")

        del teacher_model
        gc.collect()
        torch.cuda.empty_cache()

    elif teacher_ckpt.exists() and not target_teacher_path_full.exists():
        print(f"\n[Pre-train Teacher] Không tìm thấy best.pt. Bắt đầu huấn luyện TỪ ĐẦU (200 epochs)...")
        teacher_model = YOLO(str(teacher_ckpt))
        teacher_model.train(
            data=str(base_yaml_path),
            epochs=50,
            patience=0,
            batch=16,
            imgsz=640,
            device="0",
            lr0=0.01,
            cls=0.3,
            mixup=0.15,
            copy_paste=0.1,
            degrees=5.0,
            project=str(REPO_ROOT / "runs/teacher_pretrain"),
            name="yolo12l_oracle_full",
            exist_ok=True,
            verbose=True,
            workers=4,
            plots=False,
        )

        best_out = REPO_ROOT / "runs/teacher_pretrain/yolo12l_oracle_full/weights/best.pt"
        if best_out.exists():
            import shutil
            shutil.copy(best_out, target_teacher_path_full)
            shutil.copy(best_out, teacher_ckpt)
            print(f"\n[Pre-train Teacher] HOÀN THÀNH! Đã xuất Teacher vào: {teacher_ckpt}")
        else:
            print(f"\n[Pre-train Teacher] Lỗi: Không tìm thấy {best_out}")

        del teacher_model
        gc.collect()
        torch.cuda.empty_cache()

    elif target_teacher_path_full.exists():
        print(f"\n[Pre-train Teacher] File {target_teacher_path_full} đã tồn tại, BỎ QUA huấn luyện thêm Teacher.")
    else:
        print(f"\n[Pre-train Teacher] Lỗi: Không tìm thấy {teacher_ckpt} hoặc best.pt. Vui lòng chuẩn bị file trước.")

    # 5. Giai đoạn 3: Khởi động ấm (Warm-up) Student (yolo12n) trên Proxy Data
    student_ckpt = REPO_ROOT / "yolo12n.pt"
    warmup_ckpt = REPO_ROOT / "yolo12n_warmup.pt"
    
    if not warmup_ckpt.exists():
        print("\n[Pre-train Student] Bắt đầu Warm-up Student trên Proxy Data (1 epochs)...")
        from ultralytics import YOLO
        student_model = YOLO(str(student_ckpt))
        student_model.train(
            data=str(proxy_yaml_abs),
            epochs=1,  # Chỉ 1 epoch để định hình Detection Head

            batch=16,
            device="0",
            lr0=0.01,
            project=str(REPO_ROOT / "runs/student_warmup"),
            name="yolo12n_proxy_warmup",
            exist_ok=True,
            verbose=False,
            workers=4,
            plots=False,
        )
        
        warmup_out = REPO_ROOT / "runs/student_warmup/yolo12n_proxy_warmup/weights/best.pt"
        if warmup_out.exists():
            import shutil
            shutil.copy(warmup_out, warmup_ckpt)
            print(f"\n[Pre-train Student] HOÀN THÀNH! Đã xuất Student Warm-up vào: {warmup_ckpt}")
        else:
            print(f"\n[Pre-train Student] Lỗi: Không tìm thấy {warmup_out}")
            
        del student_model
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print(f"\n[Pre-train Student] File {warmup_ckpt} đã tồn tại, BỎ QUA quá trình Warm-up.")
if __name__ == "__main__":
    main()
