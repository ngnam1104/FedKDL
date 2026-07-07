import sys, os
sys.path.insert(0, os.getcwd())
import torch
from ultralytics import YOLO

# Load nano model to inspect head structure
yolo = YOLO('yolo12n.pt')
from ultralytics.nn.tasks import DetectionModel
from detection_2d.models.lora import LoRAConv2d

# Rebuild for nc=4
cfg = yolo.model.yaml.copy()
cfg['nc'] = 4
from ultralytics.nn.tasks import DetectionModel
model = DetectionModel(cfg, ch=3, nc=4, verbose=False)

head_idx = str(len(model.model) - 1)
print(f"Head index: {head_idx}")
print(f"\n--- Params in Detection Head (model.{head_idx}) by branch ---")

branch_sizes = {}
for name, param in model.named_parameters():
    if f'model.{head_idx}.' not in name:
        continue
    # Get second-level branch
    parts = name.split('.')  # model, idx, branch, ...
    branch = parts[2] if len(parts) > 2 else 'other'
    sub = parts[3] if len(parts) > 3 else ''
    key = f"{branch}.{sub}" if sub.isdigit() else branch
    branch_sizes[key] = branch_sizes.get(key, 0) + param.numel()

for k, v in sorted(branch_sizes.items()):
    print(f"  {k:<25} {v:>8,} params  {v/1024:>7.2f} KB")

total_head = sum(branch_sizes.values())
print(f"\n  TOTAL HEAD:              {total_head:>8,} params  {total_head/1024:>7.2f} KB")

# Simulate what we'd get if we open full cv3 + keep cv2 suffix-only
# cv3 = classification branch (more important for KD alignment)
cv3_params = sum(v for k, v in branch_sizes.items() if k.startswith('cv3'))
cv2_final_params = 0
for name, param in model.named_parameters():
    if f'model.{head_idx}.' not in name:
        continue
    if '.cv2.' in name and name.endswith('.2.weight') or name.endswith('.2.bias'):
        if '.cv2.' in name:
            cv2_final_params += param.numel()

# One2one branches
one2one_params = sum(v for k, v in branch_sizes.items() if 'one2one' in k)

print(f"\n--- Budget simulation (LoRA rank=8 = ~127KB) ---")
lora_kb = 127.12
print(f"  LoRA:                   {lora_kb:>7.2f} KB")
print(f"  cv3 full:               {cv3_params/1024:>7.2f} KB")
print(f"  cv2 final only:         {cv2_final_params/1024:>7.2f} KB")
print(f"  one2one:                {one2one_params/1024:>7.2f} KB")
print(f"  ---")
print(f"  TOTAL (cv3+cv2_final+lora): {(lora_kb + cv3_params/1024 + cv2_final_params/1024):.2f} KB / 300 KB")
print(f"  TOTAL + one2one: {(lora_kb + cv3_params/1024 + cv2_final_params/1024 + one2one_params/1024):.2f} KB / 300 KB")
