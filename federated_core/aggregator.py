"""
aggregator.py
FedAvg Aggregation tại tầng Relay (nội cụm) và Gateway (toàn cục).

Eq. 40: θ_relay = θ_global + Σ (n_i / Σn_k) × Δθ_i   [intra-cluster]
Eq. 43: Θ_global = Σ (n_m / N) × θ̃_relay_m           [global]
"""

import torch
import torch.nn as nn
import copy
from typing import Dict, List, Tuple


def fedavg_intra_cluster(
    global_state_dict: Dict[str, torch.Tensor],
    auv_deltas: List[Tuple[torch.Tensor, int]],
    model_template: nn.Module,
) -> Dict[str, torch.Tensor]:
    """
    FedAvg nội cụm tại Relay node  —  Eq. 40.

    Tổng hợp các Δθ_i từ auvs active trong cụm,
    cộng vào θ_global để tạo θ_relay.

    Args:
        global_state_dict: θ^t — trọng số toàn cục đầu round.
        auv_deltas:     List of (delta_theta_flat, n_samples_i).
                           delta_theta_flat: (total_params,) float tensor.
        model_template:    Model dùng để map flat → state_dict.

    Returns:
        relay_state_dict: θ_relay sau FedAvg nội cụm.
    """
    if not auv_deltas:
        return copy.deepcopy(global_state_dict)

    total_samples = sum(n for _, n in auv_deltas)
    if total_samples == 0:
        return copy.deepcopy(global_state_dict)

    # Tính weighted average Δθ
    total_params = sum(p.numel() for p in model_template.parameters())
    device = next(iter(global_state_dict.values())).device if global_state_dict else 'cpu'
    weighted_delta = torch.zeros(total_params, device=device)
    for delta_flat, n_i in auv_deltas:
        weighted_delta += (n_i / total_samples) * delta_flat.to(device)

    # Cộng Δθ vào θ_global (flat → state_dict)
    relay_sd = copy.deepcopy(global_state_dict)
    offset = 0
    for name, param in model_template.named_parameters():
        numel = param.numel()
        relay_sd[name] = relay_sd[name].float() + weighted_delta[offset:offset + numel].view(param.shape)
        offset += numel

    return relay_sd


def fedavg_global(
    relay_state_dicts: List[Dict[str, torch.Tensor]],
    cluster_total_samples: List[int],
) -> Dict[str, torch.Tensor]:
    """
    FedAvg toàn cục tại Gateway  —  Eq. 43.

    Θ^{T,(t+1)} = Σ_m (n_m / N) × θ̃_relay_m

    Args:
        relay_state_dicts:       List of θ̃_relay_m (sau HFL-Selective).
        cluster_total_samples: List of Σ n_i trong cụm m (weights).

    Returns:
        global_state_dict: Mô hình toàn cục mới Θ^{t+1}.
    """
    if not relay_state_dicts:
        raise ValueError("No relay models to aggregate.")

    N = sum(cluster_total_samples)
    if N == 0:
        # Fallback: uniform average
        weights = [1.0 / len(relay_state_dicts)] * len(relay_state_dicts)
    else:
        weights = [n / N for n in cluster_total_samples]

    # Lấy UNION tất cả keys từ mọi relay để tránh KeyError khi relay empty fallback về partial dict
    all_keys = set()
    for relay_sd in relay_state_dicts:
        all_keys.update(relay_sd.keys())

    # Tìm tensor shape từ relay đầu tiên có key đó
    global_sd = {}
    for key in all_keys:
        for relay_sd in relay_state_dicts:
            if key in relay_sd:
                global_sd[key] = torch.zeros_like(relay_sd[key], dtype=torch.float32)
                break

    # Weighted average — bỏ qua relay nào không có key đó (relay empty cluster)
    for relay_sd, w in zip(relay_state_dicts, weights):
        for key in global_sd:
            if key in relay_sd:
                global_sd[key] += w * relay_sd[key].float()

    return global_sd


def svd_lora_aggregate(
    client_sds: List[Dict[str, torch.Tensor]],
    weights: List[float],
) -> Dict[str, torch.Tensor]:
    """
    SVD-based Aggregation for FlexLoRA.
    """
    if not client_sds:
        return {}

    aggregated_sd = {}
    all_keys = set()
    for sd in client_sds:
        all_keys.update(sd.keys())

    # Find all lora_B keys
    lora_B_keys = [k for k in all_keys if 'lora_B' in k]

    for k in all_keys:
        if k in lora_B_keys:
            continue
        if 'lora_A' in k:
            continue
        
        # Standard FedAvg cho các keys khác (Detection Head)
        original_dtype = None
        weighted_sum = None
        for sd, w in zip(client_sds, weights):
            if k in sd:
                if original_dtype is None:
                    original_dtype = sd[k].dtype
                    weighted_sum = torch.zeros_like(sd[k].float())
                weighted_sum += sd[k].float() * w
        if weighted_sum is not None:
            aggregated_sd[k] = weighted_sum.to(original_dtype)

    # SVD cho LoRA keys
    for b_key in lora_B_keys:
        a_key = b_key.replace('lora_B', 'lora_A')
        
        rank = None
        W_avg = None
        original_dtype = None
        
        for sd, w in zip(client_sds, weights):
            if b_key in sd and a_key in sd:
                B_i = sd[b_key].float()
                A_i = sd[a_key].float()
                if original_dtype is None:
                    original_dtype = sd[b_key].dtype
                    rank = B_i.shape[1]
                    out_features = B_i.shape[0]
                    in_features = A_i.shape[1]
                    W_avg = torch.zeros((out_features, in_features), dtype=torch.float32, device=B_i.device)
                
                # B_i: (out, rank), A_i: (rank, in) => W_i: (out, in)
                W_i = torch.matmul(B_i, A_i)
                W_avg += W_i * w

        if W_avg is not None:
            # SVD decomposition
            # W_avg = U @ S @ Vh
            try:
                U, S, Vh = torch.linalg.svd(W_avg, full_matrices=False)
                
                # Xử lý trường hợp M < rank (ví dụ lớp Conv có số channel nhỏ)
                M = S.shape[0]
                if M < rank:
                    B_new = torch.zeros((out_features, rank), dtype=torch.float32, device=B_i.device)
                    A_new = torch.zeros((rank, in_features), dtype=torch.float32, device=A_i.device)
                    B_new[:, :M] = U * S.unsqueeze(0)
                    A_new[:M, :] = Vh
                else:
                    B_new = U[:, :rank] * S[:rank].unsqueeze(0)
                    A_new = Vh[:rank, :]
                
                aggregated_sd[b_key] = B_new.to(original_dtype)
                aggregated_sd[a_key] = A_new.to(original_dtype)
            except Exception as e:
                print(f"[SVD Error] Lỗi phân rã SVD tại {b_key}: {e}. Fallback to standard FedAvg.")
                B_sum = torch.zeros_like(sd[b_key].float())
                A_sum = torch.zeros_like(sd[a_key].float())
                for sd, w in zip(client_sds, weights):
                    if b_key in sd: B_sum += sd[b_key].float() * w
                    if a_key in sd: A_sum += sd[a_key].float() * w
                aggregated_sd[b_key] = B_sum.to(original_dtype)
                aggregated_sd[a_key] = A_sum.to(original_dtype)

    return aggregated_sd
