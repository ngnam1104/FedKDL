import sys, os
sys.path.insert(0, os.getcwd())
from tasks.detection_2d.models.yolo_wrapper import StudentModel
from config.settings import fed_cfg

student = StudentModel(ckpt='yolo12n.pt', rank=fed_cfg.LORA_RANK, nc=4, full_param=False, use_lora=True)
payload_dict = student.trainable_state_dict()

lora_p, head_p, bn_p, other_p = 0, 0, 0, 0
head_idx = str(len(student.yolo.model.model) - 1)

for k, v in payload_dict.items():
    if 'lora_' in k:
        lora_p += v.numel()
    elif ('model.' + head_idx + '.') in k:
        head_p += v.numel()
    elif 'bn.weight' in k or 'bn.bias' in k:
        bn_p += v.numel()
    else:
        other_p += v.numel()
        print('OTHER:', k, v.shape)

total = lora_p + head_p + bn_p + other_p
budget = fed_cfg.TARGET_PAYLOAD_KB
status = "OK" if total/1024 <= budget else "OVER"
print(f"LoRA:  {lora_p:>8,} params  | {lora_p/1024:>7.2f} KB INT8")
print(f"Head:  {head_p:>8,} params  | {head_p/1024:>7.2f} KB INT8")
print(f"BN:    {bn_p:>8,} params  | {bn_p/1024:>7.2f} KB INT8")
print(f"Other: {other_p:>8,} params  | {other_p/1024:>7.2f} KB INT8")
print(f"-----")
print(f"Total: {total:>8,} params  | {total/1024:>7.2f} KB INT8")
print(f"Budget: {budget} KB -> {status}")
