"""
train_student_warmup.py
Warm-up Student Model (yolo12n) với LoRA trên Proxy Data trong 2 epochs.

Dùng đúng pattern của repo (CustomDetectionTrainer + snapshot+rollback frozen weights).
"""
import sys
import argparse
import torch
from pathlib import Path
from ultralytics import YOLO

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


def run_warmup(epochs: int):
    print("==================================================")
    print(f"[Student Warmup LoRA] Warm-up YOLO12n + LoRA trong {epochs} epochs")
    print("==================================================")

    rank = fed_cfg.LORA_RANK
    print(f"-> LoRA Rank: {rank}")

    # Xây dựng hoặc tái sử dụng proxy data yaml (hoặc bạn có thể đổi thành full_yaml nếu muốn)
    proxy_yaml = ensure_proxy_data_exists()
    # Nếu muốn dùng full dataset để warmup, uncomment dòng dưới:
    # proxy_yaml = REPO_ROOT / "datasets/URPC2020.yaml"

    save_path = REPO_ROOT / "yolo12n_warmup.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"-> Loading yolo12n.pt và tiêm LoRA (rank={rank})...")
    student = StudentModel(
        ckpt="yolo12n.pt",
        rank=rank,
        nc=4,
        full_param=False,
        use_lora=True,
    )

    payload_keys = set(student.trainable_state_dict().keys())
    frozen_weights_before = {}
    for k, v in student.yolo.model.state_dict().items():
        if k not in payload_keys:
            frozen_weights_before[k] = v.clone().detach()

    overrides = {
        'model': "yolo12n.pt",
        'data': str(proxy_yaml),
        'epochs': epochs,
        'batch': 16,
        'workers': 2,
        'lr0': 2e-4,
        'warmup_bias_lr': 2e-4,
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
    trainer.head_lr_multiplier = 5.0

    print(f"\n-> Bắt đầu warm-up {epochs} epochs trên: {proxy_yaml.name} (LoRA lr=2e-4 | Head lr=1e-3)")
    trainer.train()

    print("\n[Rollback] Khôi phục frozen weights về giá trị gốc (nếu có sai lệch ngầm)...")
    with torch.no_grad():
        state_dict = student.yolo.model.state_dict()
        for k, v_before in frozen_weights_before.items():
            if k in state_dict:
                state_dict[k].copy_(v_before)
    student.yolo.model.load_state_dict(state_dict)

    ckpt = {"model": student.yolo.model.half(), "epoch": epochs}
    torch.save(ckpt, save_path)
    print(f"\n[Thành công] Đã lưu Student LoRA warmup tại: {save_path}")

    try:
        from verify_teacher_lora_eval import merge_lora_into_base
    except ImportError:
        from bake_teacher_lora import bake_lora_into_model as merge_lora_into_base

    print("\n[Đánh giá] Đánh giá chất lượng Student LoRA (đã merge LoRA)...")
    merge_lora_into_base(student.yolo.model)
    full_yaml_test = REPO_ROOT / "datasets/URPC2020.yaml"
    student.yolo.val(
        data=str(full_yaml_test),
        imgsz=640,
        batch=16,
        device=device,
        verbose=True,
        split="val",
    )


def run_centralized_lora(epochs: int):
    print("\n" + "="*50)
    print(f"[Centralized LoRA] Train LoRA + Head trong {epochs} epochs (Upper Bound)")
    print("="*50)
    
    full_yaml = REPO_ROOT / "datasets/URPC2020.yaml"
    rank = fed_cfg.LORA_RANK
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    student = StudentModel(
        ckpt="yolo12n.pt",
        rank=rank,
        nc=4,
        full_param=False,
        use_lora=True,
    )
    
    overrides = {
        'model': "yolo12n.pt",
        'data': str(full_yaml),
        'epochs': epochs,
        'batch': 16,
        'workers': 4,
        'lr0': 2e-4,
        'warmup_bias_lr': 2e-4,
        'optimizer': 'AdamW',
        'warmup_epochs': 0.0,
        'lrf': 1.0,
        'cos_lr': False,
        'device': device,
        'amp': False,
        'project': str(REPO_ROOT / "runs" / "centralized"),
        'name': 'lora_finetune',
        'exist_ok': True,
        'verbose': True,
        'save': True,
        'val': True,
        'plots': True,
        'close_mosaic': 0,
    }

    trainer = CustomDetectionTrainer(
        overrides=overrides,
        student_wrapper=student,
        cached_optimizer_state=None,
    )
    trainer._fl_injected_model = student.yolo.model
    trainer.model = student.yolo.model
    trainer.head_lr_multiplier = 5.0

    trainer.train()
    
    save_path = REPO_ROOT / "yolo12n_lora_centralized.pt"
    ckpt = {"model": student.yolo.model.half(), "epoch": epochs}
    torch.save(ckpt, save_path)
    print(f"[Thành công] Đã lưu mô hình LoRA Centralized tại: {save_path}")


def run_centralized_full(epochs: int):
    print("\n" + "="*50)
    print(f"[Centralized Full] Train 100% Parameter (No LoRA) trong {epochs} epochs")
    print("="*50)
    
    full_yaml = REPO_ROOT / "datasets/URPC2020.yaml"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = YOLO("yolo12n.pt")
    
    model.train(
        data=str(full_yaml),
        epochs=epochs,
        batch=16,
        workers=4,
        device=device,
        project=str(REPO_ROOT / "runs" / "centralized"),
        name="full_finetune",
        exist_ok=True,
        verbose=True,
        save=True,
        val=True,
        plots=True
    )
    
    print(f"[Thành công] Đã lưu mô hình Full Finetune tại: runs/centralized/full_finetune/weights/best.pt")


def main():
    parser = argparse.ArgumentParser(description="Chạy Warmup hoặc Centralized Baselines")
    parser.add_argument("--mode", type=str, default="all", choices=["warmup", "centralized_lora", "centralized_full", "all"],
                        help="Chế độ chạy (mặc định: all)")
    parser.add_argument("--epochs-warmup", type=int, default=3, help="Số epoch cho warmup")
    parser.add_argument("--epochs-centralized", type=int, default=200, help="Số epoch cho centralized tests")
    args = parser.parse_args()

    if args.mode in ["warmup", "all"]:
        run_warmup(epochs=args.epochs_warmup)
        
    if args.mode in ["centralized_lora", "all"]:
        run_centralized_lora(epochs=args.epochs_centralized)
        
    if args.mode in ["centralized_full", "all"]:
        run_centralized_full(epochs=args.epochs_centralized)


if __name__ == "__main__":
    main()
