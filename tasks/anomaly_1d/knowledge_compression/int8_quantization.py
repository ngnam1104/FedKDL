"""
int8_quantization.py
Lượng tử hóa INT8 Affine (Symmetric/Asymmetric) cho gradient compression.

Triển khai Eq. 37 trong Research Proposal:
    Q(X) = clamp(round(X/Δ) + Z, -128, 127)
    Δ = (max(X) - min(X)) / 255

Dùng sau Top-K để encode topk_values → INT8.
Payload cuối: ≈ 1.3 kbit (với K ≈ 68 entries cho AE 1350 params).
"""

import torch
import numpy as np
from typing import Tuple
from dataclasses import dataclass


@dataclass
class QuantizedTensor:
    """Tensor đã lượng tử hóa với metadata."""
    data_int8: torch.Tensor   # INT8 values
    scale: float              # Δ (scale factor)
    zero_point: int           # Z (zero point)
    original_shape: tuple     # Shape gốc


def quantize_tensor(x: torch.Tensor) -> QuantizedTensor:
    """
    Lượng tử hóa tensor sang INT8 — Eq. 37.

    Q(X) = clamp(round(X/Δ) + Z, -128, 127)
    Δ = (max(X) - min(X)) / 255

    Args:
        x: Float tensor (bất kỳ shape).

    Returns:
        QuantizedTensor với data_int8, scale, zero_point.
    """
    x_flat = x.view(-1).float()
    if torch.isnan(x_flat).any() or torch.isinf(x_flat).any():
        x_flat = torch.nan_to_num(x_flat, nan=0.0, posinf=0.0, neginf=0.0)
        
    x_min = x_flat.min().item()
    x_max = x_flat.max().item()

    # Tránh chia cho 0 nếu tensor hằng số
    if abs(x_max - x_min) < 1e-8:
        scale = 1.0
        zero_point = 0
        data_int8 = torch.zeros_like(x_flat, dtype=torch.int8)
        return QuantizedTensor(data_int8, scale, zero_point, tuple(x.shape))

    # Tính scale Δ và zero point Z
    scale = (x_max - x_min) / 255.0
    zero_point = int(round(-x_min / scale)) - 128
    zero_point = int(np.clip(zero_point, -128, 127))

    # Lượng tử hóa
    x_quantized = torch.round(x_flat / scale).long() + zero_point
    x_clamped = torch.clamp(x_quantized, -128, 127).to(torch.int8)

    return QuantizedTensor(x_clamped, scale, zero_point, tuple(x.shape))


def dequantize_tensor(qt: QuantizedTensor) -> torch.Tensor:
    """
    Giải lượng tử hóa INT8 → Float32.

    X̂ = (Q - Z) × Δ

    Returns:
        Float tensor có shape gốc.
    """
    data_float = (qt.data_int8.float() - qt.zero_point) * qt.scale
    return data_float.view(qt.original_shape)


def compute_payload_bits(
    topk_indices: torch.Tensor,
    topk_values_qt: QuantizedTensor,
    total_params: int,
) -> int:
    """
    Tính kích thước payload thực tế sau INT8 quantization (bits).

    Components:
        - Values (INT8):  K × 8 bits
        - Indices:        K × ceil(log2(total_params)) bits
        - Scale (float32): 32 bits
        - Zero point (int8): 8 bits
        - Total header:   40 bits

    Returns:
        payload_bits: Tổng số bits.
    """
    K = len(topk_indices)
    index_bits = int(np.ceil(np.log2(total_params + 1)))
    value_bits = K * 8           # INT8
    idx_bits   = K * index_bits  # sparse indices
    header_bits = 32 + 8         # scale (f32) + zero_point (i8)
    return value_bits + idx_bits + header_bits


class SparseINT8Payload:
    """
    Payload hoàn chỉnh sau Top-K + INT8: (indices, quantized values, metadata).
    Đây là đơn vị truyền qua kênh âm thanh.
    """

    def __init__(self, topk_indices: torch.Tensor,
                 topk_values: torch.Tensor, total_params: int):
        self.indices = topk_indices                      # (K,) long
        self.qt = quantize_tensor(topk_values)           # INT8 quantized
        self.total_params = total_params
        self.payload_bits = compute_payload_bits(
            topk_indices, self.qt, total_params)
        self.payload_bytes = self.payload_bits / 8.0

    def decompress(self) -> torch.Tensor:
        """Tái tạo dense float gradient từ payload."""
        from tasks.anomaly_1d.knowledge_compression.topk_sparsification import TopKCompressor
        values_float = dequantize_tensor(self.qt)
        dense = torch.zeros(self.total_params)
        dense[self.indices] = values_float
        return dense

    def __repr__(self):
        return (f"SparseINT8Payload(K={len(self.indices)}, "
                f"payload={self.payload_bytes:.0f}B / "
                f"{self.payload_bits:.0f}b)")


# ──────────────────────────────────────────────────────────────────────
#  Pack / Unpack toàn bộ state dict (LoRA + Head) cho Kịch bản 2 & 3
# ──────────────────────────────────────────────────────────────────────

import struct
from typing import Dict, Tuple


def pack_payload(state_dict: Dict[str, torch.Tensor]) -> Tuple[bytes, float]:
    """
    Nén toàn bộ {LoRA + Head} state dict thành bytes INT8.
    Mỗi tensor: 8 bytes metadata (scale f32 + zp i32) + INT8 data.

    Returns:
        (payload_bytes, size_kb)
    """
    buf = bytearray()
    for key, tensor in state_dict.items():
        qt = quantize_tensor(tensor)
        # 8 bytes header: scale (float32=4B) + zero_point (int32=4B)
        buf.extend(struct.pack('fi', qt.scale, qt.zero_point))
        buf.extend(qt.data_int8.flatten().cpu().numpy().tobytes())
    data = bytes(buf)
    return data, len(data) / 1024.0


def unpack_payload(payload: bytes,
                   template: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Giải nén bytes → float32 state dict (dùng shape từ template).

    Args:
        payload : bytes từ pack_payload
        template: state dict mẫu có đúng key và shape
    Returns:
        Dict[key → float32 tensor]
    """
    offset = 0
    recovered = {}
    for key, tmpl in template.items():
        scale, zero_point = struct.unpack_from('fi', payload, offset)
        offset += 8
        numel = tmpl.numel()
        q_bytes = payload[offset: offset + numel]
        offset += numel
        q_arr = np.frombuffer(q_bytes, dtype=np.int8).copy()
        q_tensor = torch.from_numpy(q_arr).reshape(tmpl.shape)
        recovered[key] = dequantize_tensor(
            QuantizedTensor(q_tensor, scale, zero_point, tuple(tmpl.shape))
        )
    return recovered
