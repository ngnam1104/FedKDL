import torch
import numpy as np
from typing import Tuple, Dict, Any, List

def flatten_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, List[Tuple[str, torch.Size, int]]]:
    """
    Biến đổi một state_dict thành mảng 1D duy nhất và metadata để phục hồi.
    """
    tensors = []
    shapes = []
    for k, v in state_dict.items():
        shapes.append((k, v.shape, v.numel()))
        tensors.append(v.flatten())
    
    if tensors:
        flat_tensor = torch.cat(tensors)
    else:
        flat_tensor = torch.tensor([])
        
    return flat_tensor, shapes

def unflatten_state_dict(flat_tensor: torch.Tensor, shapes: List[Tuple[str, torch.Size, int]]) -> Dict[str, torch.Tensor]:
    """
    Khôi phục state_dict từ mảng 1D.
    """
    state_dict = {}
    offset = 0
    for k, shape, numel in shapes:
        state_dict[k] = flat_tensor[offset:offset+numel].reshape(shape)
        offset += numel
    return state_dict

class TopKCompressor:
    """
    Top-K Gradient Compressor với Error Feedback.
    """
    def __init__(self, total_params: int, rho_s: float = 0.05):
        self.total_params = total_params
        self.rho_s = rho_s
        self.K = max(1, int(rho_s * total_params))
        # Error buffer khởi tạo = 0 trên CPU để tiết kiệm VRAM
        self.error_buffer = torch.zeros(total_params)

    def compress(self, delta_theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Nén gradient update Δθ với Error Feedback.
        """
        flat = delta_theta.view(-1).detach().cpu()

        # Bước 1: Compensate với error buffer
        compensated = flat + self.error_buffer

        # Bước 2: Top-K selection theo magnitude
        _, topk_indices = torch.topk(compensated.abs(), self.K, largest=True)
        topk_values = compensated[topk_indices]

        # Bước 3: Tái tạo sparse vector và cập nhật error buffer
        sparse_vec = torch.zeros(self.total_params)
        sparse_vec[topk_indices] = topk_values
        self.error_buffer = compensated - sparse_vec

        return topk_indices, topk_values

    def decompress(self, topk_indices: torch.Tensor, topk_values: torch.Tensor) -> torch.Tensor:
        """
        Tái tạo dense gradient từ sparse representation.
        """
        dense = torch.zeros(self.total_params)
        dense[topk_indices] = topk_values
        return dense

    def reset_error_buffer(self):
        self.error_buffer = torch.zeros(self.total_params)

class SparseFloatPayload:
    """
    Payload hoàn chỉnh sau Top-K: (indices, float32 values, metadata).
    Dành riêng cho baseline không dùng INT8.
    """
    def __init__(self, topk_indices: torch.Tensor, topk_values: torch.Tensor, total_params: int, shapes: List[Tuple[str, torch.Size, int]]):
        self.indices = topk_indices
        self.values = topk_values
        self.total_params = total_params
        self.shapes = shapes
        
        # Calculate size in bytes
        # Values (Float32): K * 4 bytes
        # Indices (int): K * ceil(log2(total_params))/8 bytes
        K = len(topk_indices)
        index_bytes = np.ceil(np.log2(total_params + 1)) / 8.0
        self.payload_bytes = K * 4 + K * index_bytes
        self.payload_bits = self.payload_bytes * 8.0

    def decompress(self) -> torch.Tensor:
        dense = torch.zeros(self.total_params)
        dense[self.indices] = self.values
        return dense
