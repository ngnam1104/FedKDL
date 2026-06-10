"""
check_kd_alignment.py
Kiểm tra sự đồng nhất giữa LoRA Backbone của Teacher (YOLO12l) và Student (YOLO12n)
để đảm bảo KD hoạt động đúng cách.
"""
import sys, os
sys.path.insert(0, os.getcwd())
import torch
from tasks.detection_2d.models.lora import LoRAConv2d

def inspect_lora_layers(model, label):
    layers = {}
    for name, m in model.named_modules():
        if isinstance(m, LoRAConv2d):
            parts = name.split('.')
            layer_idx = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else -1
            rank = m.lora_A.shape[0]
            in_f = m.lora_A.shape[1]
            out_f = m.lora_B.shape[0]
            layers[name] = {'layer_idx': layer_idx, 'rank': rank, 'in': in_f, 'out': out_f}

    print(f"\n=== {label} LoRA Summary ===")
    print(f"  Total LoRA layers: {len(layers)}")

    # Rank distribution by backbone layer index
    rank_by_layer = {}
    for name, info in layers.items():
        idx = info['layer_idx']
        r = info['rank']
        rank_by_layer.setdefault(idx, set()).add(r)

    print(f"  Rank distribution per backbone layer index:")
    for idx in sorted(rank_by_layer.keys()):
        print(f"    layer {idx:>2}: ranks={rank_by_layer[idx]}")

    # Max / min rank
    all_ranks = [info['rank'] for info in layers.values()]
    print(f"  Rank range: min={min(all_ranks)}, max={max(all_ranks)}")
    return layers

def main():
    sys.stdout.reconfigure(encoding='utf-8')

    # --- Load Teacher (YOLO12l) ---
    print("Loading Teacher checkpoint...")
    teacher_ckpt = torch.load('yolo12l_lora_pretrained.pt', map_location='cpu', weights_only=False)
    teacher_model = teacher_ckpt.get('model', None)
    if teacher_model is None:
        print("ERROR: cannot find model in teacher ckpt")
        return

    teacher_layers = inspect_lora_layers(teacher_model, "Teacher (YOLO12l)")

    # --- Load Student (YOLO12n with LoRA injected) ---
    print("\nLoading Student (yolo12n.pt + LoRA injection)...")
    from tasks.detection_2d.models.yolo_wrapper import StudentModel
    from config.settings import fed_cfg
    student = StudentModel(ckpt='yolo12n.pt', rank=fed_cfg.LORA_RANK, nc=4, full_param=False, use_lora=True)
    student_layers = inspect_lora_layers(student.yolo.model, "Student (YOLO12n)")

    # --- Alignment analysis ---
    print("\n=== KD ALIGNMENT ANALYSIS ===")

    # 1. Rank consistency
    t_ranks = set(info['rank'] for info in teacher_layers.values())
    s_ranks = set(info['rank'] for info in student_layers.values())
    print(f"\n[Rank Consistency]")
    print(f"  Teacher ranks used: {t_ranks}")
    print(f"  Student ranks used: {s_ranks}")
    if t_ranks == s_ranks:
        print("  -> ALIGNED: same rank set")
    else:
        print("  -> MISMATCH: different rank sets!")

    # 2. Strategy alignment — compare which backbone indices have LoRA
    t_indices = set(info['layer_idx'] for info in teacher_layers.values())
    s_indices = set(info['layer_idx'] for info in student_layers.values())
    print(f"\n[Backbone Coverage]")
    print(f"  Teacher LoRA at layers: {sorted(t_indices)}")
    print(f"  Student LoRA at layers: {sorted(s_indices)}")

    only_teacher = t_indices - s_indices
    only_student = s_indices - t_indices
    both = t_indices & s_indices
    print(f"  Both inject:   {sorted(both)}")
    if only_teacher:
        print(f"  Teacher only (not in Student): {sorted(only_teacher)}")
    if only_student:
        print(f"  Student only (not in Teacher): {sorted(only_student)}")

    # 3. KD compatibility summary
    print(f"\n[KD Compatibility]")
    print(f"  Teacher has {len(teacher_layers)} LoRA layers (YOLO12l backbone)")
    print(f"  Student has {len(student_layers)} LoRA layers (YOLO12n backbone)")
    print(f"  KD works via LoRA-Proj: Teacher projects LoRA A/B -> Student subspace.")
    print(f"  Key requirement: SAME RANK at matching logical depth layers.")
    if t_ranks == s_ranks:
        print(f"  -> PASS: Both use rank={t_ranks}, SVD projection will work correctly.")
    else:
        print(f"  -> FAIL: Rank mismatch will break SVD cross-model projection!")
        print(f"     Fix: Set student LORA_RANK to match teacher max rank = {max(t_ranks)}")

if __name__ == '__main__':
    main()
