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


def run_warmup(epochs: int):
    print("==================================================")
    print(f"[Student Warmup LoRA] Warm-up YOLO12n + LoRA trong {epochs} epochs trên toàn bộ dữ liệu (Full YAML)")
    print("==================================================")

    rank = fed_cfg.LORA_RANK
    print(f"-> LoRA Rank: {rank}")

    # Sử dụng full dataset để warmup
    full_yaml = REPO_ROOT / "datasets/URPC2020.yaml"

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

    # [CRITICAL HOTFIX] 
    # Khi thay đổi nc=4, toàn bộ Detection Head được random khởi tạo mới.
    # Hàm _is_payload_key hiện tại của FL chỉ trả về True cho lớp đầu ra cuối cùng (.cv2.x.2),
    # khiến các lớp Conv trung gian của Head bị đóng băng với trọng số ngẫu nhiên -> mAP = 0!
    # Do đó, TRONG QUÁ TRÌNH WARMUP, ta phải cho phép train TOÀN BỘ Detection Head.
    original_is_payload = student._is_payload_key
    def _warmup_is_payload(k: str) -> bool:
        if original_is_payload(k): return True
        head_idx = len(student.yolo.model.model) - 1
        if f'model.{head_idx}.' in k: return True
        return False
    student._is_payload_key = _warmup_is_payload


    payload_keys = set(student.trainable_state_dict().keys())
    frozen_weights_before = {}
    for k, v in student.yolo.model.state_dict().items():
        if k not in payload_keys:
            frozen_weights_before[k] = v.clone().detach()

    overrides = {
        'model': "yolo12n.pt",
        'data': str(full_yaml),
        'epochs': epochs,
        'batch': 16,
        'workers': 2,
        'lr0': 2e-3,
        'warmup_bias_lr': 2e-3,
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
    trainer.head_lr_multiplier = 1.0

    print(f"\n-> Bắt đầu warm-up {epochs} epochs trên: {full_yaml.name} (LoRA lr=2e-3 | Head lr=2e-3)")
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

    print("\n[Đánh giá] Đánh giá chất lượng Student LoRA (đã merge LoRA)...")
    student.bake_lora()
    full_yaml_test = REPO_ROOT / "datasets/URPC2020.yaml"
    student.yolo.val(
        data=str(full_yaml_test),
        imgsz=640,
        batch=16,
        device=device,
        verbose=True,
        split="val",
    )


def run_centralized_lora(epochs: int, patience: int = 30, resume: bool = False):
    print("\n" + "="*50)
    print(f"[Centralized LoRA] Train LoRA + Head trong {epochs} epochs (Upper Bound)")
    if resume: print("[Resume] Tiếp tục train từ last.pt...")
    print("="*50)
    
    full_yaml = REPO_ROOT / "datasets/URPC2020.yaml"
    rank = fed_cfg.LORA_RANK
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ckpt_path = "yolo12n.pt"
    if resume:
        last_pt = REPO_ROOT / "runs" / "centralized" / "lora_finetune" / "weights" / "last.pt"
        if last_pt.exists():
            ckpt_path = str(last_pt)
        else:
            print(f"[Cảnh báo] Không tìm thấy {last_pt}. Train từ đầu!")
            resume = False
            
    student = StudentModel(
        ckpt=ckpt_path,
        rank=rank,
        nc=4,
        full_param=False,
        use_lora=True,
    )
    
    # [CRITICAL HOTFIX] 
    # Giống như Warmup, vì Centralized LoRA khởi chạy từ yolo12n.pt (khởi tạo lại Head cho nc=4),
    # ta phải mở khóa TOÀN BỘ Detection Head để các layer trung gian không bị kẹt ở trạng thái ngẫu nhiên.
    original_is_payload = student._is_payload_key
    def _centralized_is_payload(k: str) -> bool:
        if original_is_payload(k): return True
        head_idx = len(student.yolo.model.model) - 1
        if f'model.{head_idx}.' in k: return True
        return False
    student._is_payload_key = _centralized_is_payload

    overrides = {
        'model': ckpt_path,
        'data': str(full_yaml),
        'epochs': epochs,
        'batch': 16,
        'workers': 4,
        # [OPTIMIZATION] LoRA cần LR lớn hơn Full Finetune khoảng 5-10 lần để hội tụ cùng tốc độ.
        # Ở Full FT ta dùng 1e-3, nên LoRA nên dùng 2e-3.
        'lr0': 2e-3,  
        'warmup_bias_lr': 2e-3,
        'optimizer': 'AdamW',
        'warmup_epochs': 3.0,
        'lrf': 0.01,
        'cos_lr': True,
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
        'resume': resume,
        'patience': patience,
    }

    trainer = CustomDetectionTrainer(
        overrides=overrides,
        student_wrapper=student,
        cached_optimizer_state=None,
    )
    trainer._fl_injected_model = student.yolo.model
    trainer.model = student.yolo.model
    trainer.head_lr_multiplier = 1.0  # Head LR = 2e-3 * 1.0 = 2e-3 (Cân bằng với LoRA)

    trainer.train()
    
    save_path = REPO_ROOT / "yolo12n_lora_centralized.pt"
    ckpt = {"model": student.yolo.model.half(), "epoch": epochs}
    torch.save(ckpt, save_path)
    print(f"[Thành công] Đã lưu mô hình LoRA Centralized tại: {save_path}")


def run_centralized_full(epochs: int, patience: int = 30, resume: bool = False):
    print("\n" + "="*50)
    print(f"[Centralized Full] Train 100% Parameter (No LoRA) trong {epochs} epochs")
    if resume: print("[Resume] Tiếp tục train từ last.pt...")
    print("="*50)
    
    full_yaml = REPO_ROOT / "datasets/URPC2020.yaml"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ckpt_path = "yolo12n.pt"
    if resume:
        last_pt = REPO_ROOT / "runs" / "centralized" / "full_finetune" / "weights" / "last.pt"
        if last_pt.exists():
            ckpt_path = str(last_pt)
        else:
            print(f"[Cảnh báo] Không tìm thấy {last_pt}. Train từ đầu!")
            resume = False
            
    model = YOLO(ckpt_path)
    
    model.train(
        data=str(full_yaml),
        epochs=epochs,
        batch=16,
        workers=4,
        device=device,
        project=str(REPO_ROOT / "runs" / "centralized"),
        name="full_finetune",
        optimizer="AdamW", # Bắt buộc phải set optimizer thì YOLO mới không đè mất lr0
        lr0=0.001,  # Set lr0 to 0.001 (giống Top-K)
        exist_ok=True,
        verbose=True,
        save=True,
        val=True,
        plots=True,
        resume=resume,
        patience=patience
    )
    
    print(f"[Thành công] Đã lưu mô hình Full Finetune tại: runs/centralized/full_finetune/weights/best.pt")


def run_centralized_topk_grad(epochs: int, patience: int = 30, resume: bool = False, topk_ratio: float = 0.05):
    print("\n" + "="*50)
    print(f"[Centralized Top-K Grad] Train với Top-{topk_ratio*100}% Gradient trong {epochs} epochs")
    if resume: print("[Resume] Tiếp tục train từ last.pt...")
    print("="*50)
    
    full_yaml = REPO_ROOT / "datasets/URPC2020.yaml"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ckpt_path = "yolo12n.pt"
    if resume:
        last_pt = REPO_ROOT / "runs" / "centralized" / "topk_grad_finetune" / "weights" / "last.pt"
        if last_pt.exists():
            ckpt_path = str(last_pt)
        else:
            print(f"[Cảnh báo] Không tìm thấy {last_pt}. Train từ đầu!")
            resume = False

    # Khởi tạo mô hình student KHÔNG dùng LoRA (full parameter mode)
    print(f"-> Loading yolo12n.pt (Full Parameter, cắt Gradient {topk_ratio*100}%)...")
    student = StudentModel(
        ckpt=ckpt_path,
        rank=0,
        nc=4,
        full_param=True,
        use_lora=False,
    )

    overrides = {
        'model': ckpt_path,
        'data': str(full_yaml),
        'epochs': epochs,
        'batch': 16,
        'workers': 4,
        'lr0': 1e-3,
        'warmup_bias_lr': 1e-3,
        'optimizer': 'AdamW',
        'lrf': 0.01,
        'cos_lr': True,
        'device': device,
        'project': str(REPO_ROOT / "runs" / "centralized"),
        'name': 'topk_grad_finetune',
        'exist_ok': True,
        'verbose': True,
        'save': True,
        'val': True,
        'plots': True,
        'resume': resume,
        'patience': patience,
    }

    trainer = CustomDetectionTrainer(
        overrides=overrides,
        student_wrapper=student,
        cached_optimizer_state=None,
    )
    trainer._fl_injected_model = student.yolo.model
    trainer.model = student.yolo.model
    trainer.topk_grad_ratio = topk_ratio

    trainer.train()
    
    save_path = REPO_ROOT / "yolo12n_topk_grad_centralized.pt"
    ckpt = {"model": student.yolo.model.half(), "epoch": epochs}
    torch.save(ckpt, save_path)
    print(f"[Thành công] Đã lưu mô hình Top-K Grad Centralized tại: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Chạy Warmup hoặc Centralized Baselines")
    parser.add_argument("--mode", type=str, default="all", choices=["warmup", "centralized_lora", "centralized_full", "centralized_topk", "all"],
                        help="Chế độ chạy (mặc định: all)")
    parser.add_argument("--epochs-warmup", type=int, default=3, help="Số epoch cho warmup")
    parser.add_argument("--epochs-centralized", type=int, default=150, help="Số epoch cho centralized tests")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--resume", action="store_true", help="Resume training từ last.pt nếu server bị sập")
    parser.add_argument("--topk-ratio", type=float, default=0.05, help="Tỉ lệ gradient giữ lại cho chế độ topk (mặc định 0.05 = 5%)")
    args = parser.parse_args()

    if args.mode in ["warmup", "all"]:
        run_warmup(epochs=args.epochs_warmup)
        
    if args.mode in ["centralized_lora", "all"]:
        run_centralized_lora(epochs=args.epochs_centralized, patience=args.patience, resume=args.resume)
        
    if args.mode in ["centralized_full", "all"]:
        run_centralized_full(epochs=args.epochs_centralized, patience=args.patience, resume=args.resume)
        
    if args.mode in ["centralized_topk", "all"]:
        run_centralized_topk_grad(epochs=args.epochs_centralized, patience=args.patience, resume=args.resume, topk_ratio=args.topk_ratio)


if __name__ == "__main__":
    main()
