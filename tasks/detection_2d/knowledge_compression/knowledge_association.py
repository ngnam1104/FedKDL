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
        label_hist_i: Histogram xác suất nhãn cục bộ của sensor i.
                      Shape (n_classes,), tổng = 1.
        label_hist_m: Histogram xác suất nhãn cục bộ của fog m (prototype).
                      Shape (n_classes,), tổng = 1.

    Returns:
        emd (float): Khoảng cách EMD ≥ 0.
    """
    return float(np.linalg.norm(label_hist_i - label_hist_m))

def compute_djoint_matrix(
    topology: Topology3D,
    G: Dict,
    sensor_label_hists: np.ndarray,
    fog_label_hists: np.ndarray,
    beta: float = 0.5,
) -> np.ndarray:
    """
    Tính ma trận D_joint(i, m) cho mọi cặp sensor-fog khả thi.
    Eq. 30:  D_joint(i,m) = β·EMD_norm(i,m) + (1-β)·Dist_norm(i,m)

    Cả hai thành phần được chuẩn hóa tuyến tính về [0,1] trước khi kết hợp.
    """
    N, M = topology.N, topology.M

    # ── Raw EMD matrix ────────────────────────────────────────────────────
    emd_raw = np.full((N, M), np.inf)
    for i in range(N):
        for m in range(M):
            if ('sensor', i, 'fog', m) in G:
                emd_raw[i, m] = compute_emd(sensor_label_hists[i],
                                            fog_label_hists[m])

    # ── Raw Distance matrix ───────────────────────────────────────────────
    dist_raw = np.full((N, M), np.inf)
    d_max = 0.0
    for i in range(N):
        for m in range(M):
            key = ('sensor', i, 'fog', m)
            if key in G:
                dist_raw[i, m] = G[key].distance
                d_max = max(d_max, dist_raw[i, m])

    if d_max == 0.0:
        d_max = 1.0  # fallback

    # ── Min-max normalization (chỉ trên các giá trị hữu hạn) ─────────────
    finite_emd = emd_raw[np.isfinite(emd_raw)]
    emd_min = finite_emd.min() if len(finite_emd) > 0 else 0.0
    emd_max = finite_emd.max() if len(finite_emd) > 0 else 1.0
    emd_range = emd_max - emd_min if emd_max > emd_min else 1.0

    emd_norm = np.where(np.isfinite(emd_raw),
                        (emd_raw - emd_min) / emd_range,
                        np.inf)
    dist_norm = np.where(np.isfinite(dist_raw),
                         dist_raw / d_max,
                         np.inf)

    # ── D_joint = β·EMD_norm + (1-β)·Dist_norm ───────────────────────────
    djoint = np.where(
        np.isfinite(emd_norm) & np.isfinite(dist_norm),
        beta * emd_norm + (1.0 - beta) * dist_norm,
        np.inf,
    )
    return djoint

def knowledge_aware_association(
    topology: Topology3D,
    G: Dict,
    sensor_label_hists: np.ndarray,
    fog_label_hists: np.ndarray,
    beta: float = 0.5,
) -> Dict[int, int]:
    """
    Knowledge-Aware Direct Association — Eq. 32 (Research Proposal).

    m* = argmin_{m: (i,m) ∈ G} D_joint(i, m)
    """
    djoint = compute_djoint_matrix(
        topology, G, sensor_label_hists, fog_label_hists, beta)

    association = {}
    for i in range(topology.N):
        row = djoint[i, :]
        feasible_fogs = np.where(np.isfinite(row))[0]
        if len(feasible_fogs) == 0:
            continue  # Sensor cô lập — không có link âm thanh khả thi
        best_fog = int(feasible_fogs[np.argmin(row[feasible_fogs])])
        association[i] = best_fog

    return association
