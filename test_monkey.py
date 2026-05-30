import torch
from ultralytics import YOLO

yolo = YOLO('yolo11n.pt')
for p in yolo.model.parameters():
    p.requires_grad = False

print('Before train:', sum(p.numel() for p in yolo.model.parameters() if p.requires_grad))

yolo.train(model=yolo.model, data='coco8.yaml', epochs=1, imgsz=640)
