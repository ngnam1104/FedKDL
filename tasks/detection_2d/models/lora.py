"""
lora.py
LoRA (Low-Rank Adaptation) injection cho các Conv2d layers của YOLO.
Dùng cho Domain Adaptation từ terrestrial (COCO) sang underwater (URPC 2020).

Domain shift chính cần adapt:
 - Màu sắc (red channel bị hấp thụ, scene xanh lam/lục)
 - Độ tương phản thấp (tán xạ ánh sáng)
 - Texture mờ (nhiễu hạt phù sa)
→ Inject LoRA vào shallow/mid backbone layers thay vì chỉ ở cuối.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAConv2d(nn.Module):
    """
    Wrap nn.Conv2d với LoRA (Low-Rank Adaptation).
    W_eff = W_pre + (alpha/rank) * B @ A
    Chỉ A và B là trainable; W_pre bị đóng băng.
    """
    def __init__(self, conv: nn.Conv2d, rank: int = 4, alpha: float = None):
        super().__init__()
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.padding_mode = conv.padding_mode

        # Đóng băng trọng số gốc
        self.weight = conv.weight
        self.weight.requires_grad = False
        if conv.bias is not None:
            self.bias = conv.bias
            self.bias.requires_grad = False
        else:
            self.register_parameter('bias', None)

        # Ma trận LoRA: A (rank x in_features), B (out x rank)
        in_features = (self.in_channels * self.kernel_size[0]
                       * self.kernel_size[1] // self.groups)
        
        # [CRITICAL FIX v14] Đóng băng ma trận A (FFA-LoRA)
        # Nếu để cả A và B cùng train ở các AUV khác nhau (Non-IID),
        # khi Server tính trung bình (A1+A2)/2 và (B1+B2)/2, tích B_avg @ A_avg sẽ sinh ra
        # các cross-terms rác (B1@A2, B2@A1) phá nát hoàn toàn Feature Map!
        # Giải pháp: Khóa cứng A ở mức khởi tạo ngẫu nhiên, chỉ train B. Tính chất tuyến tính
        # được bảo toàn tuyệt đối khi FedAvg.
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features), requires_grad=False)
        self.lora_B = nn.Parameter(torch.zeros(self.out_channels, rank), requires_grad=True)
        alpha = alpha if alpha is not None else float(rank)
        self.scaling = alpha / rank

        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        lora_weight = (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
        return F.conv2d(
            x, self.weight + lora_weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups
        )


def inject_lora(module: nn.Module,
                target_layer_names=None,
                rank: int = 4,
                alpha: float = None) -> int:
    """
    Tiêm LoRA vào các Conv2d bên trong các block được chỉ định.

    target_layer_names: List tên class module mục tiêu.
        - None → mặc định ['C2f', 'C3k2', 'C2fAttn'] (YOLO26 / YOLOv8/11)
        - Truyền ['Conv'] để inject toàn bộ backbone (cho domain shift nặng)

    Returns: số lượng Conv2d đã được wrap bằng LoRAConv2d.
    """
    if target_layer_names is None:
        # C2f = YOLOv8/11, C3k2 = YOLO26, C2fAttn = YOLOv11 phiên bản Attention
        target_layer_names = ['C2f', 'C3k2', 'C2fAttn']

    count = 0
    for _name, child in module.named_children():
        class_name = child.__class__.__name__
        if any(t in class_name for t in target_layer_names):
            # Inject LoRA vào tất cả Conv2d con
            for sub_name, sub_child in child.named_modules():
                if isinstance(sub_child, nn.Conv2d):
                    parent = _get_parent(child, sub_name)
                    leaf = sub_name.split('.')[-1]
                    setattr(parent, leaf, LoRAConv2d(sub_child, rank=rank, alpha=alpha))
                    count += 1
        else:
            count += inject_lora(child, target_layer_names, rank, alpha)
    return count


def _get_parent(module: nn.Module, path: str) -> nn.Module:
    """Trả về module cha của `path` (phân cách bởi '.')."""
    parts = path.split('.')
    if len(parts) == 1:
        return module
    for part in parts[:-1]:
        module = getattr(module, part)
    return module
