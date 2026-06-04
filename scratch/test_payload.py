import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from tasks.detection_2d.models.yolo_wrapper import StudentModel
from tasks.detection_2d.knowledge_compression.int8_quantization import pack_payload

print("Initializing YOLOv12n...")
student = StudentModel(ckpt='yolo12n_warmup.pt', rank=4, nc=4, use_lora=True)
payload, payload_kb = pack_payload(student.trainable_state_dict())
print(f"Payload Size (Float32 BN): {payload_kb:.2f} KB")

# Test FP16 BN
import struct
import numpy as np
def pack_payload_fp16(state_dict):
    buf = bytearray()
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]
        if 'bn' in key or 'running' in key or 'tracked' in key:
            tensor_f16 = tensor.half().cpu().numpy()
            buf.extend(tensor_f16.tobytes())
        else:
            # Fake INT8 header + data for size estimation
            buf.extend(struct.pack('fi', 1.0, 0))
            buf.extend(np.zeros(tensor.numel(), dtype=np.int8).tobytes())
    data = bytes(buf)
    return data, len(data) / 1024.0

payload_f16, payload_f16_kb = pack_payload_fp16(student.trainable_state_dict())
print(f"Payload Size (Float16 BN): {payload_f16_kb:.2f} KB")
