import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from ultralytics import YOLO
from tasks.detection_2d.models.yolo_wrapper import StudentModel

print("Initializing StudentModel (YOLOv12n) with LoRA...")
student = StudentModel(ckpt="yolo12n.pt", rank=8, nc=4, use_lora=True)

total_params = sum(p.numel() for p in student.yolo.model.parameters())
trainable_params = sum(p.numel() for p in student.yolo.model.parameters() if p.requires_grad)

lora_params = sum(p.numel() for n, p in student.yolo.model.named_parameters() if 'lora_' in n and p.requires_grad)
head_trainable_params = sum(p.numel() for n, p in student.yolo.model.named_parameters() if student._is_payload_key(n) and 'lora_' not in n and p.requires_grad)

print(f"\nStudent Trainable Breakdown:")
print(f"Total Params: {total_params:,}")
print(f"Trainable Params: {trainable_params:,}")
print(f"LoRA Params: {lora_params:,}")
print(f"Head Trainable Params: {head_trainable_params:,}")

print("\nAll Payload Trainable Layers:")
for n, p in student.yolo.model.named_parameters():
    if student._is_payload_key(n) and p.requires_grad:
        print(f" - {n}: {list(p.shape)} -> {p.numel():,} params")
