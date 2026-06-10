mport sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from tasks.detection_2d.models.yolo_wrapper import StudentModel
from config.settings import fed_cfg

def check_bn():
    print("=== 1. CHECK TEACHER BATCHNORM ===")
    ckpt_path = PROJECT_ROOT / 'yolo12l_lora_pretrained.pt'
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model = ckpt.get('model', ckpt.get('ema'))
        
        if hasattr(model, 'model'):
            bn_count, trainable_bn = 0, 0
            for name, param in model.named_parameters():
                if 'bn.weight' in name or 'bn.bias' in name:
                    bn_count += 1
                    if param.requires_grad:
                        trainable_bn += 1
            print(f"Tổng số BN params (weight/bias): {bn_count}")
            print(f"Số BN params ĐANG MỞ (requires_grad=True): {trainable_bn}")
            if trainable_bn > 0:
                print("Kết luận: BatchNorm trong Teacher THỰC SỰ ĐƯỢC MỞ!\n")
            else:
                print("Kết luận: BatchNorm trong Teacher ĐANG BỊ ĐÓNG!\n")
    except Exception as e:
        print(f"Lỗi đọc Teacher: {e}\n")

def check_payload():
    print("=== 2. CHECK THỰC TẾ PAYLOAD CỦA NANO (RANK=16) ===")
    # Khởi tạo Nano với cấu hình hiện tại (rank=16, open BN, open Head)
    rank = fed_cfg.LORA_RANK
    print(f"Sử dụng LORA_RANK = {rank} theo settings.py")
    
    # Khởi tạo fake model để lấy tham số
    # (Để bypass lỗi yolo12n.pt nếu không có sẵn, dùng try/except hoặc torch mock, nhưng chắc có yolo12n.pt)
    try:
        student = StudentModel(ckpt="yolo12n.pt", rank=rank, nc=4, full_param=False, use_lora=True)
        payload_dict = student.trainable_state_dict()
        
        total_params = 0
        for name, tensor in payload_dict.items():
            total_params += tensor.numel()
            
        fp32_bytes = total_params * 4
        int8_bytes = total_params * 1 # Sau khi lượng tử hoá
        
        print(f"Tổng số tham số truyền qua mạng (Payload Tensors): {len(payload_dict)} tensors")
        print(f"Tổng số Parameter (số thực): {total_params:,}")
        print(f"Kích thước Payload (FP32 chưa nén): {fp32_bytes / 1024:.2f} KB")
        print(f"Kích thước Payload (INT8 lượng tử hoá): {int8_bytes / 1024:.2f} KB")
        print(f"-> So với ngân sách {fed_cfg.TARGET_PAYLOAD_KB} KB, payload này là " + ("ĐẠT" if (int8_bytes/1024) <= fed_cfg.TARGET_PAYLOAD_KB else "VƯỢT"))
        print("\n")
    except Exception as e:
        print(f"Lỗi kiểm tra Payload: {e}\n")

def check_lr():
    print("=== 3. PHÂN TÍCH LEARNING RATE (NGUYÊN NHÂN INF/NAN) ===")
    print("Theo code trong train_student_warmup.py và CustomDetectionTrainer:")
    print("- Base lr0 trong Warmup: 2e-3 (0.002)")
    print("- Tỉ lệ head_lr_multiplier: 1.0 (-> Head LR = 0.002)")
    print("- Tỉ lệ lora_lr_multiplier: 0.25 (-> LoRA LR = 0.0005)")
    print("- Optimizer Warmup: AdamW (với warmup_epochs=0.0 tức là KHÔNG CÓ cold-start warmup phase của YOLO)")
    print("\n[PHÂN TÍCH]:")
    print("Đối với mô hình cực nhỏ (YOLO12n) có params ít hơn rất nhiều so với YOLO12l,")
    print("mật độ Gradient tác động lên từng tham số là rất lớn. Việc sử dụng LR = 2e-3 với AdamW ngay từ Epoch 0")
    print("(không có số bước warmup momentum) dễ làm nổ Gradient. Các biến số LoRA (lora_A, lora_B) khi tính toán (A @ B)")
    print("bị văng giá trị lên hàng triệu. Khi Ultralytics cố lưu checkpoint bằng model.half() (FP16),")
    print("giới hạn của FP16 là ~65,504. Giá trị văng vượt quá 65,504 sẽ lập tức bị ép thành 'inf' hoặc 'nan',")
    print("làm lây nhiễm toàn bộ EMA Model.")
    
if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    check_bn()
    check_payload()
    check_lr()
