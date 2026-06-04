import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from ultralytics import YOLO
from tasks.detection_2d.models.yolo_wrapper import StudentModel

print("Initializing TeacherModel (YOLOv12l) as used in pretraining...")
# We use StudentModel wrapper just like in train_teacher_lora.py
teacher_pretrain = StudentModel(ckpt="yolo12l.pt", rank=8, nc=4, use_lora=True)

head_trainable_params = sum(p.numel() for n, p in teacher_pretrain.yolo.model.named_parameters() if teacher_pretrain._is_payload_key(n) and 'lora_' not in n and p.requires_grad)

print(f"\nTeacher Pretrain Trainable Params: {head_trainable_params:,}")
for n, p in teacher_pretrain.yolo.model.named_parameters():
    if teacher_pretrain._is_payload_key(n) and p.requires_grad:
        print(f" - {n}: {list(p.shape)} -> {p.numel():,} params")
