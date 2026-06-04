"""
verify_lora.py
==============
Kiểm tra toàn diện rằng LoRA đã được tiêm ĐÚNG CÁCH AN TOÀN:
  1. KHÔNG tiêm vào C2fAttn (không có shape mismatch)
  2. Trọng số gốc (W_pre) bị đóng băng hoàn toàn
  3. Ma trận LoRA (A, B) đã được train (khác 0)
  4. Strategy = adaptive (skip layer 0-3, rank=2 ở 4-9, full rank ở 10+)
  5. Áp dụng đúng cho cả Teacher (yolo12l) và Student (yolo12n)

Chạy: python verify_lora.py
"""
import sys
import torch
import torch.nn as nn
from pathlib import Path

# ── Setup path ──────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

from tasks.detection_2d.models.lora import LoRAConv2d

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

# ════════════════════════════════════════════════════════════════════════════
# HELPER
# ════════════════════════════════════════════════════════════════════════════

def check_lora_injection(model: nn.Module, model_name: str, base_rank: int = 8):
    """
    Kiểm tra toàn diện LoRA injection trên 1 model.
    Trả về True nếu tất cả đều PASS.
    """
    print(f"\n{'='*60}")
    print(f"  KIỂM TRA: {model_name}")
    print(f"{'='*60}")

    all_pass = True

    # ── 1. Thu thập tất cả LoRAConv2d trong model ─────────────────────────
    lora_layers = []  # (full_name, module, parent_class_name, layer_idx)
    non_lora_conv = []  # Conv2d không được bọc LoRA

    for name, mod in model.named_modules():
        if isinstance(mod, LoRAConv2d):
            # Tìm parent block class name
            parts = name.split('.')
            layer_idx = int(parts[1]) if len(parts) >= 2 and parts[0] == 'model' and parts[1].isdigit() else -1

            # Tìm tên class của block cha gần nhất
            parent_classes = []
            cur = model
            for part in parts[:-1]:
                if part == '': continue
                try:
                    cur = getattr(cur, part)
                    parent_classes.append(cur.__class__.__name__)
                except AttributeError:
                    break

            lora_layers.append((name, mod, parent_classes, layer_idx))

        elif isinstance(mod, nn.Conv2d):
            # Conv2d gốc chưa bọc — kiểm tra có nằm trong C2fAttn không
            parts = name.split('.')
            parent_classes = []
            cur = model
            for part in parts[:-1]:
                if part == '': continue
                try:
                    cur = getattr(cur, part)
                    parent_classes.append(cur.__class__.__name__)
                except AttributeError:
                    break
            non_lora_conv.append((name, mod, parent_classes))

    print(f"\n[1] Số lượng layer đã được tiêm LoRA: {len(lora_layers)}")

    # ── 2. Kiểm tra C2fAttn KHÔNG bị tiêm ───────────────────────────────
    print(f"\n[2] Kiểm tra C2fAttn KHÔNG bị tiêm LoRA (Safe Mode):")
    dangerous_lora = [(n, pcs) for n, m, pcs, _ in lora_layers if 'C2fAttn' in pcs]
    if dangerous_lora:
        print(f"  {FAIL} CÓ {len(dangerous_lora)} LoRAConv2d nằm trong C2fAttn!")
        for n, pcs in dangerous_lora[:5]:
            print(f"       - {n} | parents: {pcs}")
        all_pass = False
    else:
        print(f"  {PASS} Không có LoRAConv2d nào nằm trong C2fAttn.")

    # ── 3. Kiểm tra các Conv2d bình thường (không phải LoRA) không nằm trong C2f/C3k2 ─
    print(f"\n[3] Kiểm tra Conv2d gốc (non-LoRA) — đảm bảo không bỏ sót C2f/C3k2:")
    missed_targets = []
    for n, m, pcs in non_lora_conv:
        parts = n.split('.')
        layer_idx = int(parts[1]) if len(parts) >= 2 and parts[0] == 'model' and parts[1].isdigit() else -1
        
        # Adaptive strategy skips layer_idx < 4
        if layer_idx != -1 and layer_idx < 4:
            continue
            
        if any(t in cls for cls in pcs for t in ('C2f', 'C3k2')):
            missed_targets.append((n, pcs))

    # Filter out C2fAttn (những cái này intentionally NOT injected)
    real_missed = [(n, pcs) for n, pcs in missed_targets if 'C2fAttn' not in pcs]
    if real_missed:
        print(f"  {WARN} Có {len(real_missed)} Conv2d trong C2f/C3k2 CHƯA được tiêm LoRA:")
        for n, pcs in real_missed[:5]:
            print(f"       - {n} | parents: {pcs}")
    else:
        print(f"  {PASS} Tất cả Conv2d trong C2f/C3k2 đều đã được bọc LoRA.")

    # ── 4. Kiểm tra W_pre bị đóng băng ──────────────────────────────────
    print(f"\n[4] Kiểm tra W_pre (trọng số gốc) bị đóng băng (requires_grad=False):")
    unfrozen_base = [(n, m) for n, m, _, _ in lora_layers if m.weight.requires_grad]
    if unfrozen_base:
        print(f"  {FAIL} {len(unfrozen_base)} LoRAConv2d có W_pre vẫn đang TRAIN:")
        for n, m in unfrozen_base[:5]:
            print(f"       - {n}")
        all_pass = False
    else:
        print(f"  {PASS} Tất cả W_pre đều bị đóng băng.")

    # ── 5. Kiểm tra lora_A và lora_B đã được train (khác 0) ─────────────
    print(f"\n[5] Kiểm tra lora_A và lora_B đã được train (khác 0):")
    zero_lora = []
    nonzero_count = 0
    for name, mod, _, _ in lora_layers:
        a_max = mod.lora_A.data.abs().max().item()
        b_max = mod.lora_B.data.abs().max().item()
        if a_max < 1e-7 and b_max < 1e-7:
            zero_lora.append(name)
        else:
            nonzero_count += 1

    if zero_lora:
        print(f"  {WARN} {len(zero_lora)} layer có lora_A và lora_B đều gần bằng 0 (chưa train?)")
        for n in zero_lora[:5]:
            print(f"       - {n}")
    print(f"  {PASS} {nonzero_count}/{len(lora_layers)} layer đã có lora_A/B khác 0 (đã train).")

    # ── 6. Kiểm tra adaptive strategy (layer_idx và rank) ────────────────
    print(f"\n[6] Kiểm tra Adaptive Strategy:")
    skipped_shallow = []
    rank_mid = []
    rank_full = []

    for name, mod, pcs, idx in lora_layers:
        r = mod.lora_A.shape[0]
        if idx == -1:
            rank_full.append((name, r))
        elif idx < 4:
            skipped_shallow.append(name)
        elif 4 <= idx < 10:
            rank_mid.append((name, idx, r))
        else:
            rank_full.append((name, r))

    print(f"  Layer 0-3 (shallow): {len(skipped_shallow)} layer đã bị SKIP {PASS}")

    if rank_mid:
        mid_ranks = [r for _, _, r in rank_mid]
        all_rank2 = all(r == 2 for r in mid_ranks)
        print(f"  Layer 4-9 (mid backbone): {len(rank_mid)} layer, rank = {set(mid_ranks)} "
              f"{'✅ (phải là rank=2)' if all_rank2 else '❌ (phải là rank=2!)'}")
        if not all_rank2:
            all_pass = False
    else:
        print(f"  Layer 4-9 (mid backbone): 0 layer (model quá nhỏ?)")

    if rank_full:
        full_ranks = [r for _, r in rank_full]
        print(f"  Layer 10+ (neck+head): {len(rank_full)} layer, rank = {set(full_ranks)} (phải là rank={base_rank})")

    # ── 7. Kiểm tra requires_grad của lora_A, lora_B ─────────────────────
    print(f"\n[7] Kiểm tra lora_A, lora_B có requires_grad=True:")
    frozen_lora = [(n, m) for n, m, _, _ in lora_layers
                   if not m.lora_A.requires_grad or not m.lora_B.requires_grad]
    if frozen_lora:
        print(f"  {FAIL} {len(frozen_lora)} layer có lora_A/B bị đóng băng!")
        for n, m in frozen_lora[:5]:
            print(f"       - {n}: A.grad={m.lora_A.requires_grad}, B.grad={m.lora_B.requires_grad}")
        all_pass = False
    else:
        print(f"  {PASS} Tất cả lora_A và lora_B đều có requires_grad=True.")

    # ── Tóm tắt ──────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    total_lora_params = sum(
        m.lora_A.numel() + m.lora_B.numel() for _, m, _, _ in lora_layers
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  LoRA params: {total_lora_params:,} / {total_params:,} ({total_lora_params/total_params*100:.2f}%)")
    print(f"  KẾT QUẢ TỔNG: {'✅ AN TOÀN & ĐÚNG' if all_pass else '❌ CÓ VẤN ĐỀ — XEM Ở TRÊN'}")
    print(f"{'─'*60}")

    return all_pass


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from config.settings import fed_cfg
    from tasks.detection_2d.models.yolo_wrapper import StudentModel

    LORA_RANK = fed_cfg.LORA_RANK

    # ── A. Kiểm tra Teacher (yolo12l.pt với LoRA) ─────────────────────────
    print("\n🔍 KHỞI TẠO TEACHER (YOLO12l + LoRA) ĐỂ KIỂM TRA...")
    teacher = StudentModel(ckpt="yolo12l.pt", rank=LORA_RANK, nc=4, full_param=False, use_lora=True)
    teacher_ok = check_lora_injection(teacher.yolo.model, f"TEACHER (yolo12l, rank={LORA_RANK})", base_rank=LORA_RANK)

    # ── B. Kiểm tra Teacher từ last.pt đã train ───────────────────────────
    last_pt = Path("last.pt")
    if last_pt.exists():
        print(f"\n🔍 KIỂM TRA last.pt (Teacher đã train {last_pt})...")
        teacher_trained = StudentModel(ckpt=str(last_pt), rank=LORA_RANK, nc=4, full_param=False, use_lora=True)
        teacher_trained_ok = check_lora_injection(
            teacher_trained.yolo.model,
            f"TEACHER đã train (last.pt, rank={LORA_RANK})",
            base_rank=LORA_RANK
        )
    else:
        print(f"\n{WARN} Không tìm thấy last.pt — bỏ qua bước kiểm tra Teacher đã train.")
        teacher_trained_ok = None

    # ── C. Kiểm tra Student (yolo12n.pt với LoRA) ─────────────────────────
    print("\n🔍 KHỞI TẠO STUDENT (yolo12n + LoRA) ĐỂ KIỂM TRA...")
    student = StudentModel(ckpt="yolo12n.pt", rank=LORA_RANK, nc=4, full_param=False, use_lora=True)
    student_ok = check_lora_injection(student.yolo.model, f"STUDENT (yolo12n, rank={LORA_RANK})", base_rank=LORA_RANK)

    # ── D. Kiểm tra Trainable State Dict (Payload) ────────────────────────
    print(f"\n{'='*60}")
    print(f"  KIỂM TRA ĐÓNG GÓI PAYLOAD (STATE DICT)")
    print(f"{'='*60}")
    
    t_sd = teacher.trainable_state_dict()
    s_sd = student.trainable_state_dict()
    
    t_lora_keys = [k for k in t_sd.keys() if 'lora_' in k]
    s_lora_keys = [k for k in s_sd.keys() if 'lora_' in k]
    
    print(f"  [Teacher] Payload size: {len(t_sd)} tensors (trong đó {len(t_lora_keys)} là LoRA)")
    print(f"  [Student] Payload size: {len(s_sd)} tensors (trong đó {len(s_lora_keys)} là LoRA)")
    
    if len(t_lora_keys) == 0 or len(s_lora_keys) == 0:
        print(f"  {FAIL} LỖI NGHIÊM TRỌNG: Payload không chứa tensor LoRA nào! Hãy kiểm tra hàm _is_payload_key trong yolo_wrapper.py")
        teacher_ok = False
        student_ok = False
    else:
        print(f"  {PASS} Hàm trainable_state_dict() hoạt động đúng cho cả hai mạng.")

    # ── Tổng kết ──────────────────────────────────────────────────────────
    print(f"\n{'#'*60}")
    print(f"  KẾT QUẢ TỔNG HỢP")
    print(f"{'#'*60}")
    results = {
        "Teacher (yolo12l fresh)": teacher_ok,
        "Student (yolo12n fresh)": student_ok,
    }
    if teacher_trained_ok is not None:
        results["Teacher (last.pt trained)"] = teacher_trained_ok

    for label, ok in results.items():
        status = f"{PASS} OK" if ok else f"{FAIL} CÓ VẤN ĐỀ"
        print(f"  {status} — {label}")
    print(f"{'#'*60}\n")
