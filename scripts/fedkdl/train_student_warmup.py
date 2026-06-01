"""
train_student_warmup.py
Warm-up Student Model (yolo12n) với LoRA trên Proxy Data trong 2 epochs.

Dùng đúng pattern của repo (CustomDetectionTrainer + snapshot+rollback frozen weights).
"""
import sys
import torch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.detection_2d.models.yolo_wrapper import StudentModel
from tasks.detection_2d.trainer import CustomDetectionTrainer
from config.settings import fed_cfg


def ensure_proxy_data_exists(min_images: int = 100):
    proxy_yaml = REPO_ROOT / "datasets/proxy_kd_data.yaml"
    proxy_txt = REPO_ROOT / "datasets/proxy_kd_train.txt"
    if proxy_yaml.exists() and proxy_txt.exists():
        # Kiểm tra số ảnh hiện có đủ không, tránh dùng lại file cũ từ dry-test
        n_lines = sum(1 for line in open(proxy_txt, encoding='utf-8') if line.strip())
        if n_lines >= min_images:
            print(f"[Proxy Data] Tái sử dụng proxy đã có: {proxy_yaml.name} ({n_lines} ảnh)")
            return proxy_yaml
        else:
            print(f"[Proxy Data] Cảnh báo: proxy_kd_train.txt chỉ có {n_lines} ảnh (< {min_images}) — đang tạo lại...")

    print("[Proxy Data] Đang dựng proxy data (20%) từ môi trường mới nhất...")
    import pickle
    import yaml
    data_dir = REPO_ROOT / "environments/2d/data/URPC"
    pkl_files = list(data_dir.glob("**/*.pkl"))
    if not pkl_files:
        raise RuntimeError("[LỖI] Không tìm thấy data partition nào. BẠN CẦN CHẠY `python utils/generate_all_envs.py` trước khi chạy script này!")
        
    latest_pkl = max(pkl_files, key=lambda p: p.stat().st_mtime)
    print(f"-> Sử dụng phân vùng: {latest_pkl.name}")
    
    with open(latest_pkl, "rb") as f:
        data_part = pickle.load(f)
        
    if not hasattr(data_part, 'public_data_indices') or not data_part.public_data_indices:
        raise RuntimeError("[LỖI] Phân vùng không có public_data_indices (proxy data).")
        
    base_yaml_path = REPO_ROOT / "datasets/URPC2020.yaml"
    if not base_yaml_path.exists():
        raise RuntimeError(f"[LỖI] Không tìm thấy file {base_yaml_path}")
        
    with open(base_yaml_path, 'r') as f:
        base_cfg = yaml.safe_load(f)
        
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
        print(f"  [warn] Không tìm thấy thư mục {train_path} trực tiếp. Thực hiện quét toàn bộ ảnh trong {dataset_dir}...")
        all_images = []
        for ext in ('*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG'):
            for p in dataset_dir.rglob(ext):
                if 'train' in p.parts:
                    all_images.append(str(p))
        if not all_images:
            raise RuntimeError("[LỖI] Không tìm thấy ảnh nào trong dataset (chứa 'train' trong đường dẫn).")
        all_images = sorted(set(all_images))
    else:
        all_images = []
        for ext in ('*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG'):
            all_images.extend([str(p) for p in img_dir.glob(f'**/{ext}')])
        all_images = sorted(set(all_images))
    
    public_images = [all_images[i] for i in data_part.public_data_indices if i < len(all_images)]
    
    proxy_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(proxy_txt, "w") as f:
        f.write("\n".join(public_images))
        
    print(f"-> Đã xuất {len(public_images)} ảnh proxy ra {proxy_txt.name}")
    
    p_cfg = base_cfg.copy()
    if 'path' in p_cfg and 'val' in p_cfg:
        base_dir = dataset_dir / p_cfg['path']
        if isinstance(p_cfg['val'], str):
            p_cfg['val'] = str((base_dir / p_cfg['val']).resolve())
        elif isinstance(p_cfg['val'], list):
            p_cfg['val'] = [str((base_dir / v).resolve()) for v in p_cfg['val']]
            
    p_cfg.pop('path', None)
    p_cfg['train'] = str(proxy_txt.absolute())
    
    with open(proxy_yaml, 'w') as f:
        yaml.safe_dump(p_cfg, f)
        
    print(f"-> Đã tạo file proxy config: {proxy_yaml.name}")
    return proxy_yaml


def main():
    print("==================================================")
    print("[Student Warmup LoRA] Warm-up YOLO12n + LoRA trên Proxy Data")
    print("==================================================")

    rank = fed_cfg.LORA_RANK
    print(f"-> LoRA Rank: {rank}")

    # Xây dựng hoặc tái sử dụng proxy data yaml
    proxy_yaml = ensure_proxy_data_exists()

    save_path = REPO_ROOT / "yolo12n_warmup.pt"
    if save_path.exists():
        print(f"[Student Warmup LoRA] {save_path.name} đã tồn tại. Sẽ tiến hành đánh giá để kiểm tra thay vì train lại...")
        student_test = StudentModel(ckpt=str(save_path), rank=rank, nc=4, use_lora=True)
        try:
            from verify_teacher_lora_eval import merge_lora_into_base
        except ImportError:
            from bake_teacher_lora import bake_lora_into_model as merge_lora_into_base
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("\n[Đánh giá] Kiểm tra chất lượng Student LoRA đã warmup...")
        merge_lora_into_base(student_test.yolo.model)
        full_yaml_test = REPO_ROOT / "datasets/URPC2020.yaml"
        student_test.yolo.val(
            data=str(full_yaml_test),
            imgsz=640,
            batch=16,
            device=device,
            verbose=True,
            split="val"
        )
        return

    # 1. Khởi tạo Student LoRA
    print(f"-> Loading yolo12n.pt và tiêm LoRA (rank={rank})...")
    student = StudentModel(
        ckpt="yolo12n.pt",
        rank=rank,
        nc=4,
        full_param=False,
        use_lora=True,
    )

    # 2. Snapshot frozen weights TRƯỚC khi train
    payload_keys = set(student.trainable_state_dict().keys())
    frozen_weights_before = {}
    for k, v in student.yolo.model.state_dict().items():
        if k not in payload_keys:
            frozen_weights_before[k] = v.clone().detach()

    print(f"-> Trainable keys: {len(payload_keys)}, Frozen keys: {len(frozen_weights_before)}")

    # 3. Cấu hình trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full_yaml = REPO_ROOT / "datasets/URPC2020.yaml"
    overrides = {
        'model': "yolo12n.pt",
        'data': str(full_yaml),
        'epochs': 2,
        'batch': 16,
        'workers': 2,
        # Nhất quán với FL rounds: LoRA/Backbone lr=2e-4, Head lr=1e-3 (×5)
        'lr0': 2e-4,
        'warmup_bias_lr': 2e-4,  # Tránh YOLO default 0.1 gây loss nổ
        'optimizer': 'AdamW',
        'warmup_epochs': 0.0,
        'lrf': 1.0,
        'cos_lr': False,
        'device': device,
        'amp': False,
        'project': str(REPO_ROOT / "runs/student_warmup_lora"),
        'name': 'yolo12n_lora_warmup',
        'exist_ok': True,
        'verbose': True,
        'save': True,
        'val': False,
        'plots': False,
        'close_mosaic': 0,
    }

    trainer = CustomDetectionTrainer(
        overrides=overrides,
        student_wrapper=student,
        cached_optimizer_state=None,
    )
    trainer._fl_injected_model = student.yolo.model
    trainer.model = student.yolo.model
    # Diff LR: LoRA/Backbone 2e-4, Head 1e-3 (×5) — nhất quán với FL rounds
    trainer.head_lr_multiplier = 5.0

    print(f"\n-> Bắt đầu warm-up 2 epochs trên: {full_yaml.name} (LoRA lr=2e-4 | Head lr=1e-3)")
    trainer.train()

    # 4. ROLLBACK frozen weights (lớp bảo vệ dự phòng 2 lớp)
    print("\n[Rollback] Khôi phục frozen weights về giá trị gốc (nếu có sai lệch ngầm)...")
    with torch.no_grad():
        state_dict = student.yolo.model.state_dict()
        for k, v_before in frozen_weights_before.items():
            if k in state_dict:
                state_dict[k].copy_(v_before)
    student.yolo.model.load_state_dict(state_dict)
    print("[OK] Rollback hoàn tất. Chỉ LoRA + Head weights được giữ lại sau warm-up.")

    # 5. Lưu model sau rollback đúng cách (lưu state_dict của model in-memory)
    # LƯU Ý: Không dùng shutil.copy(best.pt) vì file đó có thể chưa được rollback hoặc chứa rác.
    ckpt = {"model": student.yolo.model.half(), "epoch": 3}
    torch.save(ckpt, save_path)
    print(f"\n[Thành công] Đã lưu Student LoRA warmup tại: {save_path}")

    # 6. Eval chuẩn để kiểm tra chất lượng sau warmup
    try:
        from verify_teacher_lora_eval import merge_lora_into_base
    except ImportError:
        from bake_teacher_lora import bake_lora_into_model as merge_lora_into_base

    print("\n[6] Đánh giá chất lượng Student LoRA (đã merge LoRA)...")
    merge_lora_into_base(student.yolo.model)
    student.yolo.val(
        data=str(full_yaml),
        imgsz=640,
        batch=16,
        device=device,
        verbose=True,
        split="val",
    )


if __name__ == "__main__":
    main()
