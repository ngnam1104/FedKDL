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
    client_deltas: List[Tuple[torch.Tensor, int]],
    model_template: nn.Module,
) -> Dict[str, torch.Tensor]:
    """
    FedAvg nội cụm tại Relay node  —  Eq. 40.

    Tổng hợp các Δθ_i từ auvs active trong cụm,
    cộng vào θ_global để tạo θ_relay.

    Args:
        global_state_dict: θ^t — trọng số toàn cục đầu round.
        client_deltas:     List of (delta_theta_flat, n_samples_i).
                           delta_theta_flat: (total_params,) float tensor.
        model_template:    Model dùng để map flat → state_dict.

    Returns:
        relay_state_dict: θ_relay sau FedAvg nội cụm.
    """
    if not client_deltas:
        return copy.deepcopy(global_state_dict)

    total_samples = sum(n for _, n in client_deltas)
    if total_samples == 0:
        return copy.deepcopy(global_state_dict)

    # Tính weighted average Δθ
    total_params = sum(p.numel() for p in model_template.parameters())
    device = next(iter(global_state_dict.values())).device if global_state_dict else 'cpu'
    weighted_delta = torch.zeros(total_params, device=device)
    for delta_flat, n_i in client_deltas:
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
