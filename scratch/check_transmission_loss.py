"""
check_transmission_loss.py
Chẩn đoán mất mát tham số qua pipeline: local_train → INT8 pack → unpack → SVD → global
"""
import torch
import copy

# --- Setup path ---
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tasks.detection_2d.models.yolo_wrapper import StudentModel
from tasks.detection_2d.knowledge_compression.int8_quantization import pack_payload, unpack_payload
from federated_core.aggregator import svd_lora_aggregate

CKPT = "yolo12n.pt"   # use base model — warmup not available locally
DEVICE = "cpu"
RANK = 4

print("=" * 60)
print("[1] Load warmup model")
print("=" * 60)
student = StudentModel(CKPT, rank=RANK, nc=4, full_param=False, use_lora=True)
original_state = student.trainable_state_dict()

n_params = sum(v.numel() for v in original_state.values())
print(f"   Payload keys: {len(original_state)}")
print(f"   Total params: {n_params:,}")

# ------- [A] INT8 round-trip error -------
print()
print("=" * 60)
print("[A] INT8 Pack → Unpack round-trip error")
print("=" * 60)
packed_bytes, packed_kb = pack_payload(original_state)
unpacked_state = unpack_payload(packed_bytes, original_state)

errors = {}
for k in original_state:
    orig = original_state[k].float()
    back = unpacked_state[k].float()
    abs_err = (orig - back).abs()
    errors[k] = {
        'max': abs_err.max().item(),
        'mean': abs_err.mean().item(),
        'orig_range': (orig.min().item(), orig.max().item()),
    }

print(f"   Payload size: {packed_kb:.1f} KB")
print(f"   Errors per tensor:")
for k, e in sorted(errors.items(), key=lambda x: x[1]['max'], reverse=True)[:10]:
    print(f"     {k:60s} | max_err={e['max']:.6f} | mean_err={e['mean']:.6f} | range=[{e['orig_range'][0]:.4f}, {e['orig_range'][1]:.4f}]")

all_max = max(e['max'] for e in errors.values())
all_mean = sum(e['mean'] for e in errors.values()) / len(errors)
print(f"\n   OVERALL: max_err={all_max:.6f}  mean_err={all_mean:.6f}")

# ------- [B] SVD round-trip error (simulate 2 identical clients → aggregate) -------
print()
print("=" * 60)
print("[B] SVD aggregate (2 identical clients) → should be identity")
print("=" * 60)

# Simulate 2 clients with slightly perturbed states (like after 1 training step)
s1 = copy.deepcopy(unpacked_state)
s2 = copy.deepcopy(unpacked_state)

# Add tiny noise to simulate local training
for k in s1:
    if 'lora_' in k:
        s1[k] = s1[k] + torch.randn_like(s1[k]) * 0.001
        s2[k] = s2[k] + torch.randn_like(s2[k]) * 0.001

agg = svd_lora_aggregate([s1, s2], [0.5, 0.5])

# Compare effective LoRA weight W = B @ A before and after SVD
lora_B_keys = [k for k in agg if 'lora_B' in k]
svd_errors = []
for b_key in lora_B_keys:
    a_key = b_key.replace('lora_B', 'lora_A')
    W_s1 = (s1[b_key].float() @ s1[a_key].float())
    W_s2 = (s2[b_key].float() @ s2[a_key].float())
    W_expected = 0.5 * W_s1 + 0.5 * W_s2

    if b_key not in agg or a_key not in agg:
        print(f"   MISSING after SVD: {b_key}")
        continue
    
    W_agg = (agg[b_key].float() @ agg[a_key].float())
    svd_err = (W_expected - W_agg).abs()
    rel_err = svd_err.max().item() / (W_expected.abs().max().item() + 1e-8)
    svd_errors.append(rel_err)

avg_svd_rel_err = sum(svd_errors) / len(svd_errors) if svd_errors else 0
print(f"   SVD round-trip relative error (avg over {len(svd_errors)} LoRA pairs): {avg_svd_rel_err:.6f}")
print(f"   SVD round-trip relative error (max): {max(svd_errors):.6f}")
print(f"   → {'OK (< 1e-4)' if avg_svd_rel_err < 1e-4 else 'WARNING: SVD introduces significant error!'}")

# ------- [C] Simulate ONE FL round: local "train" (add noise) → pack → unpack → SVD → compare -------
print()
print("=" * 60)
print("[C] Full pipeline: simulate 1 FL round update")
print("=" * 60)

LEARNING_SIGNAL = 0.01  # simulate SGD update magnitude
N_CLIENTS = 4

client_states = []
for c in range(N_CLIENTS):
    cs = copy.deepcopy(original_state)
    for k in cs:
        if 'lora_' in k or 'cv3' in k:  # payload keys
            cs[k] = cs[k] + torch.randn_like(cs[k]) * LEARNING_SIGNAL
    client_states.append(cs)

# Pack → unpack each client (simulating INT8 transmission)
unpacked_clients = []
for c, cs in enumerate(client_states):
    pb, _ = pack_payload(cs)
    up = unpack_payload(pb, original_state)
    unpacked_clients.append(up)

# SVD aggregate
weights = [1.0 / N_CLIENTS] * N_CLIENTS
global_new = svd_lora_aggregate(unpacked_clients, weights)

# Compare: expected_update vs actual_update
print(f"   Simulating {N_CLIENTS} clients each adding N(0, {LEARNING_SIGNAL}) noise to LoRA layers...")

actual_diffs = {}
for k in original_state:
    if k not in global_new:
        continue
    orig = original_state[k].float()
    new_v = global_new[k].float()
    diff = (new_v - orig).abs().mean().item()
    actual_diffs[k] = diff

mean_diff = sum(actual_diffs.values()) / len(actual_diffs)
max_diff = max(actual_diffs.values())
print(f"   Mean |global_new - original|: {mean_diff:.6f}")
print(f"   Max  |global_new - original|: {max_diff:.6f}")
print(f"   INT8 mean quantization error: {all_mean:.6f}")
print(f"\n   Ratio update/quant_noise: {mean_diff / (all_mean + 1e-12):.2f}x")
print(f"   → {'✅ Update signal LARGER than quantization noise — training can converge' if mean_diff > all_mean else '❌ Quantization noise DOMINATES update signal — training is stalled!'}")

# ------- [D] Check if lora_A and lora_B are preserved in global after SVD -------
print()
print("=" * 60)
print("[D] Key presence check after SVD aggregation")
print("=" * 60)
orig_keys = set(original_state.keys())
agg_keys = set(global_new.keys())
missing = orig_keys - agg_keys
extra = agg_keys - orig_keys
print(f"   Original payload keys: {len(orig_keys)}")
print(f"   Post-SVD agg keys:     {len(agg_keys)}")
if missing:
    print(f"   ❌ MISSING keys: {missing}")
else:
    print(f"   ✅ All keys preserved")
if extra:
    print(f"   ⚠️  Extra keys (OK if not in original): {extra}")
