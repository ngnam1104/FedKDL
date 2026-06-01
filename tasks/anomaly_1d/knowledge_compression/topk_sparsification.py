"""
topk_sparsification.py
Top-K Sparsification với Error Feedback cho gradient compression.

Triển khai theo bài báo gốc Omeke et al. 2026 (Scenario 1):
    - Giữ lại K tọa độ có |value| lớn nhất (ρ_s = 0.05 → K = 5% params)
    - Error Feedback: cộng residual từ round trước vào Δθ hiện tại
    - Payload ≈ 65 kbit với autoencoder ~54,000 params
"""

import torch
import numpy as np
from typing import Tuple, Optional


class TopKCompressor:
    """
    Top-K Gradient Compressor với Error Feedback.

    Attributes:
        rho_s:        Tỷ lệ giữ lại (0.05 = 5%).
        error_buffer: Residual tích lũy từ round trước (initialized = 0).
    """

    def __init__(self, total_params: int, rho_s: float = 0.05):
        """
        Args:
            total_params: Tổng số parameters của model.
            rho_s:        Sparsity ratio — tỷ lệ tọa độ được giữ lại.
        """
        self.total_params = total_params
        self.rho_s = rho_s
        self.K = max(1, int(rho_s * total_params))
        # Error buffer khởi tạo = 0
        self.error_buffer = torch.zeros(total_params)

    def compress(
        self, delta_theta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Nén gradient update Δθ với Error Feedback.

        Bước 1: compensated = Δθ + error_buffer (cộng residual cũ)
        Bước 2: Lấy Top-K theo |compensated|
        Bước 3: error_buffer ← compensated - sparse_vector (lưu residual mới)

        Args:
            delta_theta: (total_params,) float tensor — model update.

        Returns:
            topk_indices: (K,) long tensor — chỉ số Top-K.
            topk_values:  (K,) float tensor — giá trị tương ứng.
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

    def decompress(
        self, topk_indices: torch.Tensor, topk_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Tái tạo dense gradient từ sparse representation.

        Returns:
            dense: (total_params,) float tensor.
        """
        dense = torch.zeros(self.total_params)
        dense[topk_indices] = topk_values
        return dense

    def reset_error_buffer(self):
        """Reset error buffer (dùng khi model bị replace bởi global model)."""
        self.error_buffer = torch.zeros(self.total_params)

    def payload_bits(self) -> int:
        """
        Ước tính kích thước payload (bits) trước khi quantize.
        Values: K × 32 bits (float32)
        Indices: K × ceil(log2(total_params)) bits
        """
        index_bits = int(np.ceil(np.log2(self.total_params + 1)))
        return self.K * (32 + index_bits)
