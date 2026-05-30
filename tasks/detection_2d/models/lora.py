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
        
        # FlexLoRA: Cả A và B đều được huấn luyện (requires_grad=True).
        # Phương pháp này sẽ dùng SVD ở server để giải quyết cross-terms khi FedAvg.
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features), requires_grad=True)
        self.lora_B = nn.Parameter(torch.zeros(self.out_channels, rank), requires_grad=True)
        alpha = alpha if alpha is not None else float(rank)
        self.scaling = alpha / rank

        # Khởi tạo Random Gaussian cho A và Zeros cho B
        nn.init.normal_(self.lora_A, mean=0.0, std=1.0 / (in_features ** 0.5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        lora_weight = (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
        return F.conv2d(
            x, self.weight + lora_weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups
        )


def inject_lora(module: nn.Module,
                target_layer_names=None,
                rank: int = 8,
                alpha: float = 8.0,
                strategy: str = "adaptive") -> int:
    """
    Tiêm LoRA vào các Conv2d bên trong các block được chỉ định.
    - strategy = "all": Chèn mọi layer (rank = base_rank)
    - strategy = "neck_head_only": Đóng băng hoàn toàn Backbone (chỉ chèn layer >= 10)
    - strategy = "adaptive": Rank siêu nhỏ ở Backbone, Rank chuẩn ở Neck+Head
    """
    if target_layer_names is None:
        # C2f = YOLOv8/11, C3k2 = YOLO26, C2fAttn = YOLOv11 phiên bản Attention
        target_layer_names = ['C2f', 'C3k2', 'C2fAttn']

    count = 0
    # Lặp qua tất cả submodules (Iterative)
    for name, sub_module in module.named_modules():
        if isinstance(sub_module, nn.Conv2d):
            # Lấy list các class name trên đường dẫn từ root tới sub_module
            path_classes = []
            current = module
            for part in name.split('.')[:-1]:
                if part == '': continue
                current = getattr(current, part)
                path_classes.append(current.__class__.__name__)
            
            # Kiểm tra xem module này có nằm trong block mục tiêu không
            is_target = ('Conv' in target_layer_names) or any(any(t in cls_name for t in target_layer_names) for cls_name in path_classes)
            
            if is_target:
                layer_idx = -1
                parts = name.split('.')
                if len(parts) >= 2 and parts[0] == 'model' and parts[1].isdigit():
                    layer_idx = int(parts[1])
                
                # Quyết định rank
                current_rank = rank
                if strategy == "neck_head_only":
                    if layer_idx != -1 and layer_idx < 10:
                        continue  # Skip backbone
                elif strategy == "adaptive":
                    if layer_idx != -1:
                        if layer_idx < 4:
                            continue  # Skip shallow (0-3)
                        elif 4 <= layer_idx < 10:
                            current_rank = 2  # Mid backbone -> rank 2
                
                parent_name = '.'.join(parts[:-1])
                leaf_name = parts[-1]
                parent = _get_parent(module, parent_name)
                
                setattr(parent, leaf_name, LoRAConv2d(sub_module, rank=current_rank, alpha=alpha))
                count += 1
                
    return count


def _get_parent(module: nn.Module, path: str) -> nn.Module:
    """Trả về module cha của `path` (phân cách bởi '.')."""
    parts = path.split('.')
    if len(parts) == 1:
        return module
    for part in parts[:-1]:
        module = getattr(module, part)
    return module
