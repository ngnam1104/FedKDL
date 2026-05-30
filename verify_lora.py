import torch
from ultralytics import YOLO

print('Đang tải mô hình...')
# Tải mô hình đã train LoRA
model_trained = YOLO('last.pt').model

# Tải mô hình gốc chưa train
model_orig = YOLO('yolo12l.pt').model

# 1. KIỂM TRA ĐÓNG BĂNG TRỌNG SỐ (Frozen Weights) BẰNG CÁCH TRỪ THAM SỐ
print("\n--- KIỂM TRA ĐÓNG BĂNG TRỌNG SỐ (TOÀN BỘ BASE MODEL) ---")
# Lấy trọng số của một lớp Conv bất kỳ trong Backbone (vd: lớp đầu tiên model.0)
w_trained = model_trained.model[0].conv.weight
w_orig = model_orig.model[0].conv.weight

diff = torch.abs(w_trained - w_orig)
max_diff = diff.max().item()
mean_diff = diff.mean().item()

is_frozen_same = max_diff < 1e-6
print(f'-> Backbone Conv (model.0) bị đóng băng 100%? : {"✅ ĐÚNG" if is_frozen_same else "❌ SAI (Đã bị train mất)"}')
print(f'   + Chênh lệch lớn nhất (Max Diff): {max_diff}')
print(f'   + Chênh lệch trung bình (Mean Diff): {mean_diff}')

# (Tùy chọn) Kiểm tra toàn bộ các layer không phải LoRA
diff_layers = []
for (name, param_trained), (name_orig, param_orig) in zip(model_trained.named_parameters(), model_orig.named_parameters()):
    if 'lora_' not in name and 'lora_' not in name_orig:
        if param_trained.shape == param_orig.shape:
            layer_diff = torch.abs(param_trained - param_orig).max().item()
            if layer_diff > 1e-6:
                diff_layers.append((name, layer_diff))

if diff_layers:
    print(f'❌ CẢNH BÁO: Có {len(diff_layers)} layer gốc (base weights) đã bị thay đổi!')
    print('   Một số layer bị đổi (tối đa 10):')
    for name, diff in diff_layers[:10]:
        print(f'     - {name}: Max diff = {diff}')
else:
    print('✅ TẤT CẢ các layer gốc (không phải LoRA) đều được giữ nguyên 100%.')

# 2. KIỂM TRA XEM CÓ LORA BÊN TRONG KHÔNG
print("\n--- KIỂM TRA CÁC THAM SỐ LORA ---")
lora_params = {name: param for name, param in model_trained.named_parameters() if 'lora_' in name}
if lora_params:
    print('✅ Mô hình có chứa tham số LoRA.')
    # Kiểm tra xem tham số LoRA có khác 0 không (đã được train)
    all_zeros = True
    for name, param in lora_params.items():
        if torch.abs(param).max().item() > 1e-6:
            all_zeros = False
            break
    if all_zeros:
        print('⚠️ Tham số LoRA đều bằng 0 (chưa được train hiệu quả hoặc do khởi tạo).')
    else:
        print('✅ Tham số LoRA đã được cập nhật (khác 0).')
else:
    print('❌ KHÔNG CÓ tham số LoRA nào trong mô hình.')

# 3. KIỂM TRA SỐ LƯỢNG THAM SỐ TRAIN ĐƯỢC
print("\n--- THỐNG KÊ THAM SỐ ---")
trainable_params = sum(p.numel() for p in model_trained.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model_trained.parameters())
print(f'-> Số lượng tham số train được: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)')
