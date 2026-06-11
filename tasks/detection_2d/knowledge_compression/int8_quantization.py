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


def _quant_bounds(bits: int) -> tuple[int, int, int]:
    """Return (n_levels, q_min, q_max) for uniform affine quantization."""
    n_levels = (1 << bits) - 1
    q_min = -(1 << (bits - 1))
    q_max = (1 << (bits - 1)) - 1
    return n_levels, q_min, q_max


def quantize_tensor(x: torch.Tensor, bits: int | None = None) -> QuantizedTensor:
    """
    Lượng tử hóa tensor sang INT8 — Eq. 37.

    Q(X) = clamp(round(X/Δ) + Z, -128, 127)
    Δ = (max(X) - min(X)) / 255

    Args:
        x: Float tensor (bất kỳ shape).

    Returns:
        QuantizedTensor với data_int8, scale, zero_point.
    """
    x_flat = x.reshape(-1).float()
    if bits is None:
        from config.settings import fed_cfg
        bits = fed_cfg.QUANTIZATION_BITS
    n_levels, q_min, q_max = _quant_bounds(bits)

    # [GUARD] Sanitize NaN/Inf để tránh crash khi quantize
    # NaN/Inf có thể xuất hiện khi training gradient explosion hoặc optimizer state bị stale
    nan_count = torch.isnan(x_flat).sum().item()
    inf_count = torch.isinf(x_flat).sum().item()
    
    # Check for extremely large values that might overflow float32 during dequantization
    # Float32 max is 3.4e38. We clamp to 1e30 to be safe.
    max_val = x_flat.abs().max().item() if x_flat.numel() > 0 else 0
    large_count = (x_flat.abs() > 1e30).sum().item()
    
    if nan_count > 0 or inf_count > 0 or large_count > 0:
        import warnings
        warnings.warn(
            f"[quantize_tensor] Detected {nan_count} NaN, {inf_count} Inf, and {large_count} massive values (>1e30) "
            f"in tensor shape {tuple(x.shape)}. Sanitizing before quantization."
        )
        # [DEBUG] Tạm tắt ép kiểu NaN/Inf về 0 để thấy lỗi vỡ thật sự
        # x_flat = torch.nan_to_num(x_flat, nan=0.0, posinf=0.0, neginf=0.0)
        x_flat = torch.clamp(x_flat, min=-1e30, max=1e30)

    x_min = x_flat.min().item()
    x_max = x_flat.max().item()

    # Tránh chia cho 0 nếu tensor hằng số — GIỮ NGUYÊN giá trị gốc (exact float)
    if abs(x_max - x_min) < 1e-8:
        constant_val = x_min
        if constant_val == 0.0:
            # Zero tensor: encode as zeros, scale = 0 → dequant = (0 - 0) * 0 = 0 ✅
            scale = 0.0
            zero_point = 0
            data_int8 = torch.zeros_like(x_flat, dtype=torch.int8)
        else:
            # Non-zero constant: encode as ones, scale = constant_val
            # → dequant = (1 - 0) * constant_val = constant_val ✅ exact float
            scale = float(constant_val)
            zero_point = 0
            data_int8 = torch.ones_like(x_flat, dtype=torch.int8)
        return QuantizedTensor(data_int8, scale, zero_point, tuple(x.shape))

    # Tính scale Δ và zero point Z
    scale = (x_max - x_min) / float(n_levels)
    zero_point = int(round(-x_min / scale)) + q_min

    # Lượng tử hóa
    x_quantized = torch.round(x_flat / scale).long() + zero_point
    x_clamped = torch.clamp(x_quantized, q_min, q_max).to(torch.int8)

    return QuantizedTensor(x_clamped, scale, zero_point, tuple(x.shape))


def dequantize_tensor(qt: QuantizedTensor) -> torch.Tensor:
    """
    Giải lượng tử hóa INT8 → Float32.

    X̂ = (Q - Z) × Δ

    Returns:
        Float tensor có shape gốc.
    """
    data_float = (qt.data_int8.float() - qt.zero_point) * qt.scale
    return data_float.reshape(qt.original_shape)


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
        from tasks.detection_2d.knowledge_compression.topk_sparsification import TopKCompressor
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
    # [CRITICAL FIX] BẮT BUỘC SORT KEY để đồng bộ thứ tự giữa Client (pack) và Server (unpack).
    # SVD Aggregator ở Server sắp xếp lại dict nên nếu không sort sẽ bị lệch byte (gây NaN/Inf).
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]
        # [GUARD] Log cảnh báo nếu phát hiện NaN/Inf để dễ debug root cause
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        if has_nan or has_inf:
            raise RuntimeError(
                f"[CRITICAL ERROR] Local training produced NaN/Inf in Key '{key}'! "
                f"NaN={has_nan}, Inf={has_inf}, Shape={tuple(tensor.shape)}. "
                f"Aborting FL round to prevent corrupting the Global Model."
            )
            
        # [CRITICAL FIX] Bỏ qua Quantization INT8 cho BatchNorm! 
        # Nếu Variance bị ép về INT8 sẽ làm mất các giá trị nhỏ -> làm feature map bùng nổ -> mAP = 0.
        if 'bn' in key or 'running' in key or 'tracked' in key:
            # [HOTFIX] Ghi ra raw bytes dạng Float32 (4 bytes) thay vì Float16 (2 bytes)
            # Float16 tối đa chỉ là 65504, trong khi num_batches_tracked có thể lên tới hàng trăm ngàn.
            # Ép float16 sẽ gây lỗi Inf, làm vỡ model khi giải nén.
            # Số lượng tham số BN cực kỳ nhỏ (vài nghìn) nên dùng float32 chỉ tốn thêm ~4-8 KB.
            tensor_f32 = tensor.float().cpu().numpy()
            buf.extend(tensor_f32.tobytes())
        else:
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
    # [CRITICAL FIX] BẮT BUỘC SORT KEY để đồng bộ với pack_payload
    for key in sorted(template.keys()):
        tmpl = template[key]
        if 'bn' in key or 'running' in key or 'tracked' in key:
            # BatchNorm parameters were stored as raw float32 bytes
            numel = tmpl.numel()
            byte_size = numel * 4  # float32 = 4 bytes
            q_bytes = payload[offset: offset + byte_size]
            offset += byte_size
            
            f32_arr = np.frombuffer(q_bytes, dtype=np.float32).copy()
            recovered[key] = torch.from_numpy(f32_arr).reshape(tmpl.shape)
            
            # Khôi phục kiểu dữ liệu gốc (VD: num_batches_tracked là int64)
            if tmpl.dtype == torch.int64:
                recovered[key] = recovered[key].long()
        else:
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
            # [DEBUG] Check if dequantization produced non-finite values
            if not torch.isfinite(recovered[key]).all():
                print(f"[unpack_payload WARNING] Key '{key}' has Non-finite after dequant! "
                      f"scale={scale:.6e}, zp={zero_point}, "
                      f"int8 range=[{q_tensor.min()}, {q_tensor.max()}], "
                      f"float range=[{recovered[key].min():.6e}, {recovered[key].max():.6e}]")
            
    return recovered


def pack_delta_payload(
    state_dict: Dict[str, torch.Tensor],
    reference_state: Dict[str, torch.Tensor],
) -> Tuple[bytes, float]:
    """Pack a state update relative to the model shared at the start of the link."""
    if set(state_dict) != set(reference_state):
        missing = sorted(set(reference_state) - set(state_dict))
        extra = sorted(set(state_dict) - set(reference_state))
        raise ValueError(
            f"Delta payload keys must match the reference state; missing={missing[:5]}, "
            f"extra={extra[:5]}"
        )
    delta_state = {
        key: state_dict[key].cpu() - reference_state[key].cpu()
        for key in reference_state
    }
    return pack_payload(delta_state)


def unpack_delta_payload(
    payload: bytes,
    reference_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Reconstruct a state from an INT8 update and its exact link reference."""
    decoded_delta = unpack_payload(payload, reference_state)
    return {
        key: reference_state[key].cpu() + decoded_delta[key]
        for key in reference_state
    }
