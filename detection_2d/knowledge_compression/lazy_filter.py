"""
lazy_filter.py
Bộ lọc truyền thông lười (Lazy Communication Filter) — Eq. 40 (Research Proposal).

Nguyên tắc đúng theo Proposal:
    AUV i thuộc tập kích hoạt A_m^(t) khi và chỉ khi:

    sqrt( Σ_l ||A_{l,i}^t - Â_{l,m}^{t-1}||_F² + ||B_{l,i}^t - B̂_{l,m}^{t-1}||_F² )
    + || Θ_head,i^t - Θ̂_head,m^{t-1} || > δ_skip

Tức là: so sánh trọng số LoRA+Head CỦA AUV với trọng số CLUSTER đã aggregate vòng trước,
không phải so với state trước khi train của chính auv đó.
"""
import math
import torch
from typing import Dict, Optional, Tuple


def _frobenius_sq(t1: torch.Tensor, t2: torch.Tensor) -> float:
    """||t1 - t2||_F^2"""
    return torch.sum((t1.float() - t2.float()) ** 2).item()


def compute_cluster_drift_norm(
    auv_state: Dict[str, torch.Tensor],
    cluster_prev_state: Dict[str, torch.Tensor],
) -> float:
    """
    Tính drift norm theo Eq. 40:

        sqrt( Σ_l ||A_{l,i} - Â_{l,m}||_F² + ||B_{l,i} - B̂_{l,m}||_F² )
        + || Θ_head,i - Θ̂_head,m ||

    Phân tách LoRA adapters (keys chứa 'lora_A' hoặc 'lora_B') và Head
    ('detect' hoặc 'head') để tính riêng từng thành phần như Proposal.

    Args:
        auv_state:       state dict (LoRA + Head) của auv i sau local SGD.
        cluster_prev_state: state dict (LoRA + Head) cụm m đã aggregate vòng t-1.

    Returns:
        drift_norm (float): Giá trị drift tổng hợp theo đúng Eq. 40.
    """
    # Tổng bình phương của Frobenius norm LoRA adapters
    lora_sq_sum = 0.0
    head_sq_sum = 0.0

    for k, v in auv_state.items():
        if k not in cluster_prev_state:
            continue
        ref = cluster_prev_state[k]
        sq = _frobenius_sq(v, ref)

        if 'lora_a' in k.lower() or 'lora_b' in k.lower():
            lora_sq_sum += sq
        else:
            # Head và các tensor còn lại
            head_sq_sum += sq

    # Eq. 40: sqrt(Σ LoRA Frobenius²) + || Head delta ||
    lora_norm = math.sqrt(lora_sq_sum)
    head_norm  = math.sqrt(head_sq_sum)
    return lora_norm + head_norm


def lazy_filter(
    auv_state: Dict[str, torch.Tensor],
    cluster_prev_state: Optional[Dict[str, torch.Tensor]],
    threshold: float,
) -> Tuple[Optional[Dict[str, torch.Tensor]], float]:
    """
    Áp dụng Lazy Communication Filter theo Eq. 40.

    Args:
        auv_state:       State dict (LoRA + Head) sau khi train local — θ_i^t.
        cluster_prev_state: State dict cụm đã aggregate vòng trước — θ̂_{m}^{t-1}.
                            None → không có reference (round đầu) → luôn gửi.
        threshold:          δ_skip từ FedKDLConfig.DELTA_SKIP.

    Returns:
        (auv_state, drift)  nếu drift >= threshold  → cần gửi lên Relay.
        (None, drift)          nếu drift <  threshold  → bỏ qua, AUV ngủ đông.
    """
    # Round đầu tiên: chưa có aggregate reference → luôn gửi để bootstrap
    if cluster_prev_state is None:
        return auv_state, float('inf')

    drift = compute_cluster_drift_norm(auv_state, cluster_prev_state)
    if drift < threshold:
        return None, drift
    return auv_state, drift


# ─── Backward-compat helper (Scenario 1, dùng raw state diff) ──────────────

def compute_delta_norm(
    state_before: Dict[str, torch.Tensor],
    state_after: Dict[str, torch.Tensor],
) -> float:
    """
    [Scenario 1 / Legacy] Tính ||Δθ||² trên toàn state dict.
    Dùng khi không có cluster reference (tác vụ 1D autoencoder).
    """
    total = 0.0
    for k in state_before:
        if k in state_after:
            diff = state_after[k].float() - state_before[k].float()
            total += torch.sum(diff ** 2).item()
    return total


def lazy_filter_legacy(
    state_before: Dict[str, torch.Tensor],
    state_after: Dict[str, torch.Tensor],
    threshold: float,
) -> Tuple[Optional[Dict[str, torch.Tensor]], float]:
    """
    [Scenario 1 / Legacy] Lazy filter dựa trên local gradient norm.
    Giữ nguyên để không phá Scenario 1 simulator.
    """
    delta = compute_delta_norm(state_before, state_after)
    if delta < threshold:
        return None, delta
    return state_after, delta
