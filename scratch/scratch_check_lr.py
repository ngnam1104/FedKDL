import torch
import os

files = [
    'yolo12l_lora_pretrained.pt',
    'yolo12l_pretrained.pt',
    'runs/centralized/topk_grad_finetune/weights/last.pt',
    'runs/centralized/full_finetune/weights/last.pt'
]

for f in files:
    path = os.path.join('d:/Documents/HUST/2022-2026/Research_Thesis/FedKDL', f)
    if not os.path.exists(path):
        print(f'\n--- {f} ---')
        print('File not found.')
        continue
        
    print(f'\n--- {f} ---')
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        if 'train_args' in ckpt:
            args = ckpt['train_args']
            print(f"lr0: {args.get('lr0')}")
            print(f"lrf: {args.get('lrf')}")
            print(f"optimizer: {args.get('optimizer')}")
        else:
            print('No train_args found in this checkpoint.')
    except Exception as e:
        print(f'Error loading: {e}')
