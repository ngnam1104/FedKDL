"""
aggregator.py
FedAvg Aggregation tại tầng Fog (nội cụm) và Gateway (toàn cục).

Eq. 40: θ_fog = θ_global + Σ (n_i / Σn_k) × Δθ_i   [intra-cluster]
Eq. 43: Θ_global = Σ (n_m / N) × θ̃_fog_m           [global]
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
    FedAvg nội cụm tại Fog node  —  Eq. 40.

    Tổng hợp các Δθ_i từ sensors active trong cụm,
    cộng vào θ_global để tạo θ_fog.

    Args:
        global_state_dict: θ^t — trọng số toàn cục đầu round.
        client_deltas:     List of (delta_theta_flat, n_samples_i).
                           delta_theta_flat: (total_params,) float tensor.
        model_template:    Model dùng để map flat → state_dict.

    Returns:
        fog_state_dict: θ_fog sau FedAvg nội cụm.
    """
    if not client_deltas:
        return copy.deepcopy(global_state_dict)

    total_samples = sum(n for _, n in client_deltas)
    if total_samples == 0:
        return copy.deepcopy(global_state_dict)

    # Tính weighted average Δθ
    total_params = sum(p.numel() for p in model_template.parameters())
    weighted_delta = torch.zeros(total_params)
    for delta_flat, n_i in client_deltas:
        weighted_delta += (n_i / total_samples) * delta_flat.cpu()

    # Cộng Δθ vào θ_global (flat → state_dict)
    fog_sd = copy.deepcopy(global_state_dict)
    offset = 0
    for name, param in model_template.named_parameters():
        numel = param.numel()
        fog_sd[name] = fog_sd[name].float() + weighted_delta[offset:offset + numel].view(param.shape)
        offset += numel

    return fog_sd


def fedavg_global(
    fog_state_dicts: List[Dict[str, torch.Tensor]],
    cluster_total_samples: List[int],
) -> Dict[str, torch.Tensor]:
    """
    FedAvg toàn cục tại Gateway  —  Eq. 43.

    Θ^{T,(t+1)} = Σ_m (n_m / N) × θ̃_fog_m

    Args:
        fog_state_dicts:       List of θ̃_fog_m (sau HFL-Selective).
        cluster_total_samples: List of Σ n_i trong cụm m (weights).

    Returns:
        global_state_dict: Mô hình toàn cục mới Θ^{t+1}.
    """
    if not fog_state_dicts:
        raise ValueError("No fog models to aggregate.")

    N = sum(cluster_total_samples)
    if N == 0:
        # Fallback: uniform average
        weights = [1.0 / len(fog_state_dicts)] * len(fog_state_dicts)
    else:
        weights = [n / N for n in cluster_total_samples]

    # Khởi tạo từ fog đầu tiên
    global_sd = {k: torch.zeros_like(v, dtype=torch.float32)
                 for k, v in fog_state_dicts[0].items()}

    for fog_sd, w in zip(fog_state_dicts, weights):
        for key in global_sd:
            global_sd[key] += w * fog_sd[key].float()

    return global_sd
