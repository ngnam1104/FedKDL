"""
bake_teacher_lora.py
=====================
Tái lưu Teacher LoRA checkpoint dưới dạng YOLO thuần túy (không cần LoRAConv2d).

Luồng:
  1. Load yolo12l_lora_pretrained.pt vào kiến trúc StudentModel (có LoRAConv2d)
  2. Bake LoRA vào base weights: W_eff = W_base + scaling * (lora_B @ lora_A)
  3. Thay toàn bộ LoRAConv2d → Conv2d thường
  4. Lưu tạm → yolo12l_lora_baked.pt
  5. Eval trên URPC2020.yaml để xác nhận mAP ~0.73
  6. Rename baked → yolo12l_lora_pretrained.pt (backup file cũ thành .bak)

Chạy:
  python scripts/fedkdl/bake_teacher_lora.py
  python scripts/fedkdl/bake_teacher_lora.py --no-rename   # chỉ tạo file baked, không đổi tên
"""
import sys
import argparse
import torch
import torch.nn as nn
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from detection_2d.models.yolo_wrapper import StudentModel
from detection_2d.models.lora import LoRAConv2d
from config.settings import fed_cfg


def load_teacher_with_lora(ckpt_path: str, rank: int, nc: int = 4) -> StudentModel:
    """Load Teacher weights vào kiến trúc StudentModel (có LoRA)."""
    print(f"\n[1] Khởi tạo Teacher LoRA Architecture (yolo12l.pt + LoRA rank={rank})...")
    teacher = StudentModel(
        ckpt="yolo12l.pt",
        rank=rank,
        nc=nc,
        full_param=False,
        use_lora=True,
    )

    print(f"\n[2] Loading state_dict từ: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict):
        if "model" in ckpt:
            raw_sd = ckpt["model"].state_dict() if hasattr(ckpt["model"], "state_dict") else ckpt["model"]
        elif "state_dict" in ckpt:
            raw_sd = ckpt["state_dict"]
        else:
            raw_sd = ckpt
    else:
        raw_sd = ckpt.state_dict() if hasattr(ckpt, "state_dict") else ckpt

    model_sd = teacher.yolo.model.state_dict()
    matched = {k: v for k, v in raw_sd.items() if k in model_sd and v.shape == model_sd[k].shape}
    print(f"   Matched keys: {len(matched)}/{len(model_sd)}")

    model_sd.update(matched)
    teacher.yolo.model.load_state_dict(model_sd, strict=False)
    return teacher


def bake_lora_into_model(model: nn.Module) -> int:
    """
    Gộp LoRA vào Conv weight gốc và THAY THẾ LoRAConv2d → Conv2d thường.
    Sau bước này, không còn LoRAConv2d nào trong model — fuse() an toàn.
    """
    merged_count = 0
    for parent_name, parent_module in list(model.named_modules()):
        for child_name, child_module in list(parent_module.named_children()):
            if not isinstance(child_module, LoRAConv2d):
                continue

            with torch.no_grad():
                lora_weight = (child_module.lora_B @ child_module.lora_A).view(
                    child_module.weight.shape
                ) * child_module.scaling
                baked_weight = child_module.weight.data + lora_weight

                new_conv = nn.Conv2d(
                    in_channels=child_module.in_channels,
                    out_channels=child_module.out_channels,
                    kernel_size=child_module.kernel_size,
                    stride=child_module.stride,
                    padding=child_module.padding,
                    dilation=child_module.dilation,
                    groups=child_module.groups,
                    bias=child_module.bias is not None,
                    padding_mode=child_module.padding_mode,
                )
                new_conv.weight.data = baked_weight
                if child_module.bias is not None:
                    new_conv.bias.data = child_module.bias.data.clone()

            setattr(parent_module, child_name, new_conv)
            merged_count += 1

    return merged_count


def main():
    parser = argparse.ArgumentParser("Bake LoRA and re-save as clean YOLO checkpoint")
    parser.add_argument("--src", type=str, default="yolo12l_lora_pretrained.pt",
                        help="File nguồn chứa Teacher LoRA weights")
    parser.add_argument("--dst", type=str, default="yolo12l_lora_baked.pt",
                        help="File đích tạm thời (sẽ được đổi tên thành src sau khi val)")
    parser.add_argument("--data", type=str, default="datasets/URPC2020.yaml",
                        help="Dataset YAML để eval sau khi bake")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--nc", type=int, default=4)
    parser.add_argument("--no-rename", action="store_true",
                        help="Không đổi tên — giữ nguyên file dst riêng biệt")
    args = parser.parse_args()

    rank = args.rank or fed_cfg.LORA_RANK
    src_path = REPO_ROOT / args.src
    dst_path = REPO_ROOT / args.dst
    data_yaml = str(REPO_ROOT / args.data)

    print("=" * 60)
    print("  BAKE TEACHER LoRA → Clean YOLO Checkpoint")
    print("=" * 60)
    print(f"  Src  : {src_path}")
    print(f"  Dst  : {dst_path}")
    print(f"  Data : {data_yaml}")
    print(f"  Rank : {rank} | nc: {args.nc}")
    print("=" * 60)

    # 1. Load model đúng kiến trúc (StudentModel với LoRAConv2d)
    teacher = load_teacher_with_lora(str(src_path), rank=rank, nc=args.nc)

    # 2. Kiểm tra LoRA trước khi bake
    n_lora_before = sum(1 for m in teacher.yolo.model.modules() if isinstance(m, LoRAConv2d))
    nonzero = sum(
        1 for m in teacher.yolo.model.modules()
        if isinstance(m, LoRAConv2d)
        and (m.lora_A.data.abs().max() > 1e-7 or m.lora_B.data.abs().max() > 1e-7)
    )
    print(f"\n[3] Trước bake: {n_lora_before} LoRAConv2d layers ({nonzero} non-zero)")
    if nonzero == 0:
        print("   ⚠️  CẢNH BÁO: LoRA weights đều bằng 0! File nguồn có thể bị lỗi.")

    # 3. Bake LoRA → Conv2d thường
    print("\n[4] Baking LoRA weights vào base Conv2d...")
    baked = bake_lora_into_model(teacher.yolo.model)
    n_lora_after = sum(1 for m in teacher.yolo.model.modules() if isinstance(m, LoRAConv2d))
    print(f"   Baked {baked} layers. LoRAConv2d còn lại: {n_lora_after}")

    # 4. Lưu dưới dạng Ultralytics checkpoint (FP16 giống best.pt)
    print(f"\n[5] Lưu clean checkpoint → {dst_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher.yolo.model.to(device)
    ckpt = {
        "model": teacher.yolo.model.half(),
        "epoch": -1,
        "best_fitness": None,
        "optimizer": None,
        "ema": None,
        "updates": None,
        "date": None,
        "version": "baked_lora",
        "nc": args.nc,
    }
    torch.save(ckpt, dst_path)
    size_mb = dst_path.stat().st_size / 1024 / 1024
    print(f"   ✅ Đã lưu! Kích thước: {size_mb:.1f} MB")

    # 5. Eval để xác nhận mAP đúng (load thẳng bằng YOLO — không cần merge thêm)
    print(f"\n[6] Eval trên {args.data} để xác nhận mAP sau bake...")
    from ultralytics import YOLO
    yolo_eval = YOLO(str(dst_path))
    metrics = yolo_eval.val(
        data=data_yaml,
        imgsz=640,
        batch=8,
        device=device,
        verbose=True,
        split="val",
    )

    map50   = metrics.box.map50
    map5095 = metrics.box.map
    prec    = metrics.box.mp
    rec     = metrics.box.mr

    print("\n" + "=" * 60)
    print("  KẾT QUẢ EVAL (sau bake)")
    print("=" * 60)
    print(f"  mAP50    : {map50:.4f}  (mong đợi ~0.73)")
    print(f"  mAP50-95 : {map5095:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print("=" * 60)

    if map50 < 0.5:
        print("\n⚠️  mAP50 thấp hơn mong đợi — KHÔNG rename. Kiểm tra lại file nguồn.")
        return

    # 6. Rename: baked → pretrained (backup file cũ)
    if not args.no_rename:
        import shutil
        backup_path = src_path.with_suffix(".pt.bak")
        print(f"\n[7] Rename:")
        print(f"   Backup file cũ → {backup_path.name}")
        shutil.copy(src_path, backup_path)
        print(f"   Thay thế {src_path.name} bằng file baked mới...")
        shutil.move(str(dst_path), str(src_path))
        print(f"   ✅ Xong! {src_path.name} bây giờ là file CLEAN (không còn LoRAConv2d).")
        print(f"   (Backup: {backup_path.name} — có thể xóa sau khi xác nhận)")
    else:
        print(f"\n[7] --no-rename: Giữ nguyên {dst_path.name} riêng biệt.")

    print("\n✅ HOÀN THÀNH! Teacher LoRA đã được bake và sẵn sàng dùng.")


if __name__ == "__main__":
    main()
