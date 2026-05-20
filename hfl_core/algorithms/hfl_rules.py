"""
hfl_rules.py
Quy tắc điều phối HFL-Selective và Flat-Topology cho Scenario 1.

Bao gồm:
    - nearest_feasible_fog:  Sensor → Fog gần nhất khả thi
    - should_cooperate:      Điều kiện HFL-Selective (Eq. 28)
    - find_coop_partner:     Tìm Fog láng giềng để "vay mượn tri thức" (Eq. 29)
    - blend_models:          Phép trộn α × self + (1-α) × neighbor (Eq. 29)
    - compute_q1_fog_distance: Q1 của fog-fog distances khả thi (Eq. 29 filter)
    - flat_feasible_sensors: Lọc sensors khả thi lên gateway (FedProx)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


def should_cooperate(cluster_size: int, mean_cluster_size: float) -> bool:
    """
    Điều kiện kích hoạt HFL-Selective cho Fog node m  —  Eq. 41.

    Fog m hợp tác khi cụm "đói tri thức":
        c_m ≤ max(2, 0.75 × c̄)

    Args:
        cluster_size:      c_m — số sensors trong cụm của Fog m.
        mean_cluster_size: c̄  — quy mô cụm trung bình toàn mạng.

    Returns:
        True nếu Fog m cần kích hoạt hợp tác liên cụm.
    """
    threshold = max(2, 0.75 * mean_cluster_size)
    return cluster_size <= threshold


def compute_q1_fog_distance(feasibility_graph: Dict) -> float:
    """
    Tính Q1 (25th percentile) của tất cả khoảng cách fog-fog khả thi trong mạng.

    Dùng để lọc ứng viên hợp tác trong HFL-Selective (paper Section V-B Eq. 29):
    Fog m chỉ xét láng giềng có distance < Q1 fog-fog distances.

    Args:
        feasibility_graph: Đồ thị khả thi G — output của build_feasibility_graph().

    Returns:
        q1 in metres. Nếu không có fog-fog link nào → float('inf').
    """
    import numpy as np
    distances = [
        info.distance
        for (type_u, _id_u, type_v, _id_v), info in feasibility_graph.items()
        if type_u == 'fog' and type_v == 'fog'
    ]
    if not distances:
        return float('inf')
    return float(np.percentile(distances, 25))


def find_coop_partner(
    fog_id: int,
    cluster_sizes: Dict[int, int],
    feasibility_graph: Dict,
    fog_positions=None,
    q1_distance: Optional[float] = None,
) -> Optional[int]:
    """
    Tìm Fog láng giềng j khả thi gần nhất có cụm lớn hơn cụm m.

    Theo paper Eq. 29 (HFL-Selective): chỉ xét láng giềng có
        - cluster_size > my_size
        - distance ≤ Q1 của tất cả fog-fog distances (khi q1_distance được truyền vào)

    HFL-Nearest không truyền q1_distance → xét tất cả feasible neighbors.

    Args:
        fog_id:            ID của Fog m đang tìm partner.
        cluster_sizes:     dict[fog_id → cluster_size].
        feasibility_graph: Đồ thị khả thi G.
        fog_positions:     (M, 3) array — không dùng (khoảng cách lấy từ graph).
        q1_distance:       Ngưỡng Q1 distance (chỉ HFL-Selective). None → bỏ qua filter.

    Returns:
        partner_fog_id nếu tìm thấy, else None.
    """
    my_size = cluster_sizes.get(fog_id, 0)
    candidates = []

    for other_id, other_size in cluster_sizes.items():
        if other_id == fog_id:
            continue
        if other_size <= my_size:
            continue
        # Kiểm tra link khả thi (theo cả hai chiều)
        key_fwd = ('fog', fog_id, 'fog', other_id)
        key_bwd = ('fog', other_id, 'fog', fog_id)
        if key_fwd in feasibility_graph or key_bwd in feasibility_graph:
            key = key_fwd if key_fwd in feasibility_graph else key_bwd
            dist = feasibility_graph[key].distance
            # Lọc Q1 distance (chỉ áp dụng cho HFL-Selective — Eq. 29)
            if q1_distance is not None and dist > q1_distance:
                continue
            candidates.append((other_id, dist))

    if not candidates:
        return None

    # Chọn partner gần nhất trong số ứng viên hợp lệ
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]


def blend_models(
    self_model: nn.Module,
    neighbor_model: nn.Module,
    alpha: float = 0.8,
) -> Dict[str, torch.Tensor]:
    """
    Phép trộn mô hình HFL-Selective  —  Eq. 42.

    θ̃_m = α × θ_m + (1-α) × θ_neighbor
    Mặc định: 0.8 × self + 0.2 × neighbor

    Returns:
        blended_state_dict: state dict đã trộn (chưa load vào model).
    """
    blended = {}
    self_sd = self_model.state_dict()
    nbr_sd = neighbor_model.state_dict()
    for key in self_sd:
        blended[key] = alpha * self_sd[key].float() + (1.0 - alpha) * nbr_sd[key].float()
    return blended


def blend_state_dicts(
    self_sd: Dict[str, torch.Tensor],
    neighbor_sd: Dict[str, torch.Tensor],
    alpha: float = 0.8,
) -> Dict[str, torch.Tensor]:
    """
    Phép trộn trực tiếp trên state dicts (không cần model object).
    """
    blended = {}
    for key in self_sd:
        blended[key] = alpha * self_sd[key].float() + (1.0 - alpha) * neighbor_sd[key].float()
    return blended


def compute_mean_cluster_size(cluster_sizes: Dict[int, int]) -> float:
    """Tính c̄ — quy mô cụm trung bình toàn mạng."""
    sizes = [s for s in cluster_sizes.values() if s > 0]
    return float(sum(sizes) / len(sizes)) if sizes else 1.0
