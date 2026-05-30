import torch
from ultralytics import YOLO

print('Đang tải mô hình...')
# Tải mô hình đã train LoRA
model_trained = YOLO('last.pt').model

# Tải mô hình gốc chưa train
model_orig = YOLO('yolo12l.pt').model

# 1. KIỂM TRA ĐÓNG BĂNG TRỌNG SỐ (Frozen Weights)
# Lấy trọng số của một lớp Conv bất kỳ trong Backbone (vd: lớp đầu tiên model.0)
w_trained = model_trained.model[0].conv.weight
w_orig = model_orig.model[0].conv.weight

# So sánh từng con số thập phân
is_frozen_same = torch.allclose(w_trained, w_orig)
print(f'-> Backbone Conv (model.0) bị đóng băng 100%? : {"✅ ĐÚNG" if is_frozen_same else "❌ SAI (Đã bị train mất)"}')

# 2. KIỂM TRA XEM CÓ LORA BÊN TRONG KHÔNG
has_lora = any('lora_' in name for name, _ in model_trained.named_parameters())
print(f'-> Mô hình có chứa tham số LoRA (lora_A, lora_B)? : {"✅ CÓ" if has_lora else "❌ KHÔNG CÓ"}')

# 3. KIỂM TRA SỐ LƯỢNG THAM SỐ TRAIN ĐƯỢC
trainable_params = sum(p.numel() for p in model_trained.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model_trained.parameters())
print(f'-> Số lượng tham số train được: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)')
