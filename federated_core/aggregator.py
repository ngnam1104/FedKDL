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
        safe_delta = torch.nan_to_num(delta_flat.to(device), nan=0.0, posinf=0.0, neginf=0.0)
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

    # Nếu là mô hình có LoRA, bắt buộc dùng SVD để tránh cross-terms phá hủy kiến thức
    # Hàm svd_lora_aggregate tự động áp dụng FedAvg chuẩn cho các layer không phải LoRA (ví dụ Detection Head)
    return svd_lora_aggregate(relay_state_dicts, weights)


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
                val = torch.nan_to_num(sd[k].float(), nan=0.0, posinf=0.0, neginf=0.0)
                weighted_sum += val * w
        if weighted_sum is not None:
            aggregated_sd[k] = weighted_sum.to(original_dtype)

    # SVD cho LoRA keys
    for b_key in lora_B_keys:
        a_key = b_key.replace('lora_B', 'lora_A')
        
        rank = None
        W_avg = None
        original_dtype = None
        list_B = []
        list_A = []
        
        for sd, w in zip(client_sds, weights):
            if b_key in sd and a_key in sd:
                B_i = torch.nan_to_num(sd[b_key].float(), nan=0.0, posinf=0.0, neginf=0.0)
                A_i = torch.nan_to_num(sd[a_key].float(), nan=0.0, posinf=0.0, neginf=0.0)
                if original_dtype is None:
                    original_dtype = sd[b_key].dtype
                    rank = B_i.shape[1]
                    out_features = B_i.shape[0]
                    in_features = A_i.shape[1]
                list_B.append(B_i)
                list_A.append(A_i)
        
        if list_B:
            # --- SAFETY CHECK: Kiểm tra từng client xem ai gửi NaN ---
            for idx, (B_client, A_client) in enumerate(zip(list_B, list_A)):
                if torch.isnan(B_client).any() or torch.isnan(A_client).any():
                    print(f"\n[CRITICAL ERROR] Client index {idx} in this cluster returned NaN/Inf for layer {b_key}!")
                    print(f"B_client has_nan: {torch.isnan(B_client).any().item()}, has_inf: {torch.isinf(B_client).any().item()}")
                    print(f"A_client has_nan: {torch.isnan(A_client).any().item()}, has_inf: {torch.isinf(A_client).any().item()}")
            
            W_avg = sum(w * torch.matmul(B_i, A_i) for w, B_i, A_i in zip(weights, list_B, list_A))
            
            try:
                # [CRITICAL FIX] Ép kiểu float64 (double) để tránh lỗi số học (tràn số, NaN) 
                # khi phân rã các ma trận low-rank có singular value cực nhỏ.
                U, S, Vh = torch.linalg.svd(W_avg.double(), full_matrices=False)
                U, S, Vh = U.float(), S.float(), Vh.float()
                
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
                import traceback
                tb_str = traceback.format_exc()
                has_nan = torch.isnan(W_avg).any().item()
                has_inf = torch.isinf(W_avg).any().item()
                max_val = W_avg.abs().max().item() if not has_nan else "NaN"
                
                print(f"\n{'='*60}")
                print(f"[FATAL SVD ERROR] Lỗi phân rã SVD tại {b_key}")
                print(f"Exception: {e}")
                print(f"[SVD DEBUG] W_avg shape={W_avg.shape}, dtype={W_avg.dtype}")
                print(f"[SVD DEBUG] has_nan={has_nan}, has_inf={has_inf}, max_abs={max_val}")
                print(f"[SVD DEBUG] Traceback:\n{tb_str}")
                
                # Dump ma trận lỗi ra đĩa để phân tích
                try:
                    safe_key = b_key.replace('.', '_')
                    torch.save(W_avg, f"error_W_avg_{safe_key}.pt")
                    print(f"[SVD DEBUG] Đã lưu ma trận lỗi ra file: error_W_avg_{safe_key}.pt")
                except Exception as save_e:
                    pass
                print(f"{'='*60}\n")
                print(f"[SVD Warning] Fallback to standard FedAvg.")
                B_sum = None
                A_sum = None
                for sd, w in zip(client_sds, weights):
                    if b_key in sd:
                        if B_sum is None: B_sum = torch.zeros_like(sd[b_key].float())
                        val_b = torch.nan_to_num(sd[b_key].float(), nan=0.0, posinf=0.0, neginf=0.0)
                        B_sum += val_b * w
                    if a_key in sd:
                        if A_sum is None: A_sum = torch.zeros_like(sd[a_key].float())
                        val_a = torch.nan_to_num(sd[a_key].float(), nan=0.0, posinf=0.0, neginf=0.0)
                        A_sum += val_a * w
                if B_sum is not None: aggregated_sd[b_key] = B_sum.to(original_dtype)
                if A_sum is not None: aggregated_sd[a_key] = A_sum.to(original_dtype)

    return aggregated_sd
