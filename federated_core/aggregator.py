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
        # safe_delta = torch.nan_to_num(delta_flat.to(device), nan=0.0, posinf=0.0, neginf=0.0)
        safe_delta = delta_flat.to(device)
        weighted_delta += (n_i / total_samples) * safe_delta

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
    lora_aggregation: str = "svd",
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

    # FedKDL aggregates effective LoRA weights through SVD. The naive-LoRA
    # control intentionally averages A/B independently to expose cross-terms.
    if lora_aggregation == "svd":
        return svd_lora_aggregate(relay_state_dicts, weights)
    if lora_aggregation == "naive":
        return weighted_state_dict_average(relay_state_dicts, weights)
    raise ValueError(f"Unknown LoRA aggregation strategy: {lora_aggregation}")


def _weighted_fedavg_tensors(tensors: List[torch.Tensor], weights: List[float]) -> torch.Tensor:
    if len(tensors) != len(weights):
        raise ValueError(f"Expected one weight per tensor, got {len(tensors)} tensors and {len(weights)} weights.")
    ref = tensors[0]
    out = torch.zeros_like(ref, dtype=torch.float32)
    for tensor, weight in zip(tensors, weights):
        tensor_f32 = tensor.float()
        if not torch.isfinite(tensor_f32).all():
            raise RuntimeError("[CRITICAL ERROR] Non-finite tensor found during FedAvg aggregation.")
        out += float(weight) * tensor_f32
    return out


def _weighted_fedavg_control_dicts(dicts: List[Dict[str, torch.Tensor]], weights: List[float]) -> Dict[str, torch.Tensor]:
    """Weighted average for metadata dictionaries such as SCAFFOLD delta_c."""
    if len(dicts) != len(weights):
        raise ValueError(f"Expected one weight per dict, got {len(dicts)} dicts and {len(weights)} weights.")
    out: Dict[str, torch.Tensor] = {}
    all_keys = set().union(*(d.keys() for d in dicts))
    for key in sorted(all_keys):
        tensors = [d[key] for d in dicts if key in d]
        key_weights = [w for d, w in zip(dicts, weights) if key in d]
        if not tensors:
            continue
        weight_sum = sum(key_weights)
        key_weights = [w / weight_sum for w in key_weights] if weight_sum > 0 else [1.0 / len(tensors)] * len(tensors)
        out[key] = _weighted_fedavg_tensors(tensors, key_weights).to(tensors[0].dtype)
    return out


def weighted_state_dict_average(
    state_dicts: List[Dict[str, torch.Tensor]],
    weights: List[float],
) -> Dict[str, torch.Tensor]:
    """Ordinary weighted FedAvg, including independent LoRA A/B averaging."""
    if not state_dicts:
        return {}
    if len(state_dicts) != len(weights):
        raise ValueError(f"Expected one weight per state, got {len(state_dicts)} states and {len(weights)} weights.")

    total_weight = float(sum(weights))
    if total_weight <= 0.0:
        weights = [1.0 / len(state_dicts)] * len(state_dicts)
    else:
        weights = [float(weight) / total_weight for weight in weights]

    aggregated: Dict[str, torch.Tensor] = {}
    all_keys = set().union(*(state.keys() for state in state_dicts))
    for key in sorted(all_keys):
        values = [state[key] for state in state_dicts if key in state]
        key_weights = [weight for state, weight in zip(state_dicts, weights) if key in state]
        if not values:
            continue
        weight_sum = sum(key_weights)
        key_weights = (
            [weight / weight_sum for weight in key_weights]
            if weight_sum > 0
            else [1.0 / len(values)] * len(values)
        )
        if isinstance(values[0], dict):
            aggregated[key] = _weighted_fedavg_control_dicts(values, key_weights)
        else:
            aggregated[key] = _weighted_fedavg_tensors(values, key_weights).to(values[0].dtype)
    return aggregated


def svd_lora_aggregate(
    client_sds: List[Dict[str, torch.Tensor]],
    weights: List[float],
) -> Dict[str, torch.Tensor]:
    """
    SVD aggregation for FlexLoRA.

    For each LoRA pair, aggregate the effective low-rank matrix W_i = B_i @ A_i,
    then project the weighted average back to rank r with the Eckart-Young optimal
    truncated SVD. Non-LoRA payload keys are aggregated by ordinary FedAvg.
    """
    if not client_sds:
        return {}
    if len(client_sds) != len(weights):
        raise ValueError(f"Expected one weight per client, got {len(client_sds)} clients and {len(weights)} weights.")

    total_weight = float(sum(weights))
    if total_weight <= 0.0:
        weights = [1.0 / len(client_sds)] * len(client_sds)
    else:
        weights = [float(w) / total_weight for w in weights]

    aggregated_sd: Dict[str, torch.Tensor] = {}
    all_keys = set().union(*(sd.keys() for sd in client_sds))
    lora_B_keys = sorted(k for k in all_keys if 'lora_B' in k)
    lora_A_keys = {k.replace('lora_B', 'lora_A') for k in lora_B_keys}

    for key in sorted(all_keys):
        if key in lora_B_keys or key in lora_A_keys:
            continue
        tensors = [sd[key] for sd in client_sds if key in sd]
        key_weights = [w for sd, w in zip(client_sds, weights) if key in sd]
        if not tensors:
            continue
        weight_sum = sum(key_weights)
        key_weights = [w / weight_sum for w in key_weights] if weight_sum > 0 else [1.0 / len(tensors)] * len(tensors)
        if isinstance(tensors[0], dict):
            aggregated_sd[key] = _weighted_fedavg_control_dicts(tensors, key_weights)
        else:
            aggregated_sd[key] = _weighted_fedavg_tensors(tensors, key_weights).to(tensors[0].dtype)

    for b_key in lora_B_keys:
        a_key = b_key.replace('lora_B', 'lora_A')
        pairs = []
        pair_weights = []
        for client_idx, (sd, weight) in enumerate(zip(client_sds, weights)):
            if b_key not in sd or a_key not in sd:
                continue
            B_i = sd[b_key].float()
            A_i = sd[a_key].float()
            if not torch.isfinite(B_i).all() or not torch.isfinite(A_i).all():
                nan_B = torch.isnan(B_i).sum().item()
                inf_B = torch.isinf(B_i).sum().item()
                nan_A = torch.isnan(A_i).sum().item()
                inf_A = torch.isinf(A_i).sum().item()
                print(
                    f"[SVD AGG WARNING] Client #{client_idx} has Non-finite LoRA for {b_key}: "
                    f"B(NaN={nan_B}, Inf={inf_B}, max={B_i.abs().max():.4e}), "
                    f"A(NaN={nan_A}, Inf={inf_A}, max={A_i.abs().max():.4e}). "
                    f"SKIPPING this client for this key."
                )
                continue
            pairs.append((B_i, A_i))
            pair_weights.append(weight)

        if not pairs:
            continue

        weight_sum = sum(pair_weights)
        pair_weights = [w / weight_sum for w in pair_weights] if weight_sum > 0 else [1.0 / len(pairs)] * len(pairs)
        rank = pairs[0][0].shape[1]
        out_features = pairs[0][0].shape[0]
        in_features = pairs[0][1].shape[1]
        original_dtype_B = client_sds[0][b_key].dtype if b_key in client_sds[0] else pairs[0][0].dtype
        original_dtype_A = client_sds[0][a_key].dtype if a_key in client_sds[0] else pairs[0][1].dtype

        W_avg = None
        for (B_i, A_i), weight in zip(pairs, pair_weights):
            W_i = torch.matmul(B_i.double(), A_i.double())
            W_avg = float(weight) * W_i if W_avg is None else W_avg + float(weight) * W_i

        if W_avg is None or not torch.isfinite(W_avg).all():
            raise RuntimeError(f"[CRITICAL ERROR] Non-finite weighted LoRA product before SVD for {b_key}.")

        try:
            U, S, Vh = torch.linalg.svd(W_avg, full_matrices=False)
            if not torch.isfinite(S).all():
                raise RuntimeError("SVD returned non-finite singular values")
            keep = min(rank, S.numel())
            sqrt_S = torch.sqrt(S[:keep]).float()
            U_r = U[:, :keep].float()
            Vh_r = Vh[:keep, :].float()

            B_new = torch.zeros((out_features, rank), dtype=torch.float32, device=U_r.device)
            A_new = torch.zeros((rank, in_features), dtype=torch.float32, device=Vh_r.device)
            B_new[:, :keep] = U_r * sqrt_S.unsqueeze(0)
            A_new[:keep, :] = sqrt_S.unsqueeze(1) * Vh_r
        except Exception as exc:
            raise RuntimeError(f"SVD factorization failed for LoRA layer {b_key}: {exc}") from exc

        if not torch.isfinite(B_new).all() or not torch.isfinite(A_new).all():
            raise RuntimeError(f"[CRITICAL ERROR] SVD produced non-finite LoRA factors for {b_key}.")

        aggregated_sd[b_key] = B_new.to(original_dtype_B)
        aggregated_sd[a_key] = A_new.to(original_dtype_A)

    return aggregated_sd
