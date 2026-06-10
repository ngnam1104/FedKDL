import sys, os
sys.path.insert(0, os.getcwd())
from tasks.detection_2d.models.yolo_wrapper import StudentModel
from config.settings import fed_cfg
import io

student = StudentModel(ckpt='yolo12n.pt', rank=fed_cfg.LORA_RANK, nc=4, full_param=False, use_lora=True)

payload_keys = set(student.trainable_state_dict().keys())
head_idx = str(len(student.yolo.model.model) - 1)

buf = io.StringIO()

# --- Trainable params (requires_grad=True) ---
buf.write("=" * 70 + "\n")
buf.write(f"Student Trainable Breakdown (LORA_RANK={fed_cfg.LORA_RANK})\n")
buf.write("=" * 70 + "\n")

trainable = [(n, p) for n, p in student.yolo.model.named_parameters() if p.requires_grad]
total_trainable = sum(p.numel() for _, p in trainable)
total_params = sum(p.numel() for p in student.yolo.model.parameters())

lora_params = sum(p.numel() for n, p in trainable if 'lora_' in n)
head_params  = sum(p.numel() for n, p in trainable if f'model.{head_idx}.' in n)
bn_params    = sum(p.numel() for n, p in trainable if ('bn.weight' in n or 'bn.bias' in n) and f'model.{head_idx}.' not in n)
other_params = total_trainable - lora_params - head_params - bn_params

buf.write(f"Total Params:     {total_params:>10,}\n")
buf.write(f"Trainable Params: {total_trainable:>10,}\n")
buf.write(f"LoRA Params:      {lora_params:>10,}\n")
buf.write(f"Head Trainable:   {head_params:>10,}\n")
buf.write(f"BN Params:        {bn_params:>10,}\n")
buf.write(f"Other:            {other_params:>10,}\n")

payload_total = sum(v.numel() for v in student.trainable_state_dict().values())
buf.write(f"\nPayload (INT8):    {payload_total:>10,} params  |  {payload_total/1024:.2f} KB\n")
buf.write(f"Budget:           {fed_cfg.TARGET_PAYLOAD_KB} KB\n")

# --- Per-layer detail ---
buf.write("\n" + "=" * 70 + "\n")
buf.write("All Trainable Layers (requires_grad=True):\n")
buf.write("=" * 70 + "\n")

for name, param in trainable:
    in_payload = "  [PAYLOAD]" if name in payload_keys else "  [local-only]"
    buf.write(f" - {name}: {list(param.shape)} -> {param.numel():,} params{in_payload}\n")

buf.write("\n" + "=" * 70 + "\n")
buf.write("Payload Keys Only (transmitted over network):\n")
buf.write("=" * 70 + "\n")

payload_dict = student.trainable_state_dict()
lora_total, head_total, bn_total, other_total = 0, 0, 0, 0

for name, tensor in sorted(payload_dict.items()):
    n = tensor.numel()
    tag = ""
    if 'lora_' in name:
        tag = "[LoRA]"
        lora_total += n
    elif f'model.{head_idx}.' in name:
        tag = "[Head]"
        head_total += n
    elif 'bn.' in name:
        tag = "[BN]"
        bn_total += n
    else:
        tag = "[?]"
        other_total += n
    buf.write(f" - {name}: {list(tensor.shape)} -> {n:,} params  {tag}\n")

buf.write("\n--- Payload Breakdown ---\n")
buf.write(f"  LoRA:  {lora_total:>8,} params  |  {lora_total/1024:>7.2f} KB\n")
buf.write(f"  Head:  {head_total:>8,} params  |  {head_total/1024:>7.2f} KB\n")
buf.write(f"  BN:    {bn_total:>8,} params  |  {bn_total/1024:>7.2f} KB\n")
buf.write(f"  Other: {other_total:>8,} params  |  {other_total/1024:>7.2f} KB\n")
buf.write(f"  TOTAL: {payload_total:>8,} params  |  {payload_total/1024:>7.2f} KB  {'OK' if payload_total/1024 <= fed_cfg.TARGET_PAYLOAD_KB else 'OVER'}\n")

output = buf.getvalue()
print(output)

with open('trainable_layers_n.txt', 'w', encoding='utf-8') as f:
    f.write(output)
print(f"\n-> Saved to trainable_layers_n.txt")
