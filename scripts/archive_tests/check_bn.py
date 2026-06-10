import torch
import sys

sys.stdout.reconfigure(encoding='utf-8')
ckpt_path = r'd:\Documents\HUST\2022-2026\Research_Thesis\FedKDL\yolo12l_lora_pretrained.pt'
try:
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = ckpt.get('model', None)
    if model is None:
        model = ckpt.get('ema')
    
    if hasattr(model, 'model'):
        bn_count = 0
        trainable_bn = 0
        for name, param in model.named_parameters():
            if 'bn.weight' in name or 'bn.bias' in name:
                bn_count += 1
                if param.requires_grad:
                    trainable_bn += 1
        print(f'Tìm thấy {bn_count} BN params (weight/bias).')
        print(f'Có {trainable_bn} BN params ĐANG MỞ (requires_grad=True).')
        if trainable_bn > 0:
            print('-> BatchNorm trong checkpoint thực sự ĐƯỢC MỞ.')
        else:
            print('-> BatchNorm trong checkpoint ĐANG BỊ ĐÓNG.')
    else:
        print('Không tìm thấy model architecture trong ckpt.')
except Exception as e:
    print('Lỗi:', e)
