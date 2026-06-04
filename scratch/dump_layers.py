import torch
from tasks.detection_2d.models.yolo_wrapper import StudentModel

student_l = StudentModel(ckpt='yolo12l_pretrained.pt', rank=2, nc=4, use_lora=True)
with open('trainable_layers.txt', 'w', encoding='utf-8') as f:
    f.write('--- YOLOv12l Trainable Layers ---\n')
    for n, p in student_l.yolo.model.named_parameters():
        if p.requires_grad:
            f.write(f'{n}: {list(p.shape)}\n')

student_n = StudentModel(ckpt='yolo12n_warmup.pt', rank=2, nc=4, use_lora=True)
with open('trainable_layers_n.txt', 'w', encoding='utf-8') as f:
    f.write('--- YOLOv12n Trainable Layers ---\n')
    for n, p in student_n.yolo.model.named_parameters():
        if p.requires_grad:
            f.write(f'{n}: {list(p.shape)}\n')

print('Done generating trainable_layers.txt and trainable_layers_n.txt')
