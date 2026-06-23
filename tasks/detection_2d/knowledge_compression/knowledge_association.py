"""
knowledge_association.py
Triển khai Knowledge-Aware Association cho mạng IoUT (FedKDL).
Theo Eq. 30-32 (Research Proposal).
"""
import numpy as np
from typing import Dict
from physics_models.topology import Topology3D

def compute_emd(
    label_hist_i: np.ndarray,
    label_hist_m: np.ndarray,
) -> float:
    """
    Tính Earth Mover's Distance (EMD) dưới dạng L2-norm của hiệu histogram nhãn.
    Eq. 31:  EMD(i, m) = || p^(i)(y=z) - p^(m)(y=z) ||_2

    Args:
        label_hist_i: Histogram xác suất nhãn cục bộ của auv i.
                      Shape (n_classes,), tổng = 1.
        label_hist_m: Histogram xác suất nhãn cục bộ của relay m (prototype).
                      Shape (n_classes,), tổng = 1.

    Returns:
        emd (float): Khoảng cách EMD ≥ 0.
    """
    return float(np.linalg.norm(label_hist_i - label_hist_m))

def compute_djoint_matrix(
    topology: Topology3D,
    G: Dict,
    auv_label_hists: np.ndarray,
    relay_label_hists: np.ndarray,
    beta: float = 0.5,
) -> np.ndarray:
    """
    Tính ma trận D_joint(i, m) cho mọi cặp auv-relay khả thi.
    Eq. 30:  D_joint(i,m) = β·EMD_norm(i,m) + (1-β)·Dist_norm(i,m)

    Cả hai thành phần được chuẩn hóa tuyến tính về [0,1] trước khi kết hợp.
    """
    N, M = topology.N, topology.M

    # ── Raw EMD matrix ────────────────────────────────────────────────────
    emd_raw = np.full((N, M), np.inf)
    for i in range(N):
        for m in range(M):
            if ('auv', i, 'relay', m) in G:
                emd_raw[i, m] = compute_emd(auv_label_hists[i],
                                            relay_label_hists[m])

    # ── Raw Distance matrix ───────────────────────────────────────────────
    dist_raw = np.full((N, M), np.inf)
    d_max = 0.0
    for i in range(N):
        for m in range(M):
            key = ('auv', i, 'relay', m)
            if key in G:
                dist_raw[i, m] = G[key].distance
                d_max = max(d_max, dist_raw[i, m])

    if d_max == 0.0:
        d_max = 1.0  # fallback

    # ── Min-max normalization (chỉ trên các giá trị hữu hạn) ─────────────
    finite_emd = emd_raw[np.isfinite(emd_raw)]
    emd_min = finite_emd.min() if len(finite_emd) > 0 else 0.0
    emd_max = finite_emd.max() if len(finite_emd) > 0 else 1.0
    emd_range = emd_max - emd_min
    if emd_range == 0:
        emd_range = 1e-9
    
    emd_norm = np.where(np.isfinite(emd_raw),
                        (emd_raw - emd_min) / emd_range,
                        np.inf)
    dist_norm = np.where(np.isfinite(dist_raw),
                         dist_raw / d_max,
                         np.inf)

    # ── D_joint = β·EMD_norm + (1-β)·Dist_norm ───────────────────────────
    # Avoid eager np.where evaluation on inf values (e.g. 0 * inf -> NaN warning).
    valid = np.isfinite(emd_norm) & np.isfinite(dist_norm)
    djoint = np.full_like(emd_norm, np.inf, dtype=float)
    djoint[valid] = beta * emd_norm[valid] + (1.0 - beta) * dist_norm[valid]
    return djoint

def knowledge_aware_association(
    topology: Topology3D,
    G: Dict,
    auv_label_hists: np.ndarray,
    relay_label_hists: np.ndarray,
    beta: float = 0.5,
) -> Dict[int, int]:
    """
    Knowledge-Aware Direct Association — Eq. 32 (Research Proposal).

    m* = argmin_{m: (i,m) ∈ G} D_joint(i, m)
    """
    import math
    djoint = compute_djoint_matrix(
        topology, G, auv_label_hists, relay_label_hists, beta)

    N, M = topology.N, topology.M
    max_capacity = math.ceil(N / M) + 2
    
    # 1. Thu thập tất cả các cặp (cost, auv_id, relay_id) khả thi
    valid_pairs = []
    for i in range(N):
        for m in range(M):
            cost = djoint[i, m]
            if np.isfinite(cost):
                valid_pairs.append((cost, i, m))
                
    # 2. Sắp xếp theo cost tăng dần (ưu tiên gán cặp tốt nhất trước)
    valid_pairs.sort(key=lambda x: x[0])
    
    association = {}
    relay_counts = {m: 0 for m in range(M)}
    
    # 3. Phân bổ có ràng buộc sức chứa (Capacity-Constrained Assignment)
    for cost, auv_id, relay_id in valid_pairs:
        if auv_id not in association and relay_counts[relay_id] < max_capacity:
            association[auv_id] = relay_id
            relay_counts[relay_id] += 1
            
    # 4. Fallback: Gán vét cho các AUV bị rớt lại (do các Relay tốt đều đã đầy)
    for i in range(N):
        if i not in association:
            row = djoint[i, :]
            feasible_relays = np.where(np.isfinite(row))[0]
            if len(feasible_relays) > 0:
                best_relay = int(feasible_relays[np.argmin(row[feasible_relays])])
                association[i] = best_relay
                relay_counts[best_relay] += 1

    return association
