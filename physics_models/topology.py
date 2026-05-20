"""
topology.py
Khởi tạo Topology 3D Quasi-static và Đồ thị Khả thi (Feasibility Graph).
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from physics_models.communication import (
    is_link_feasible, transmission_loss, wenz_noise_level, shannon_capacity,
)


@dataclass
class LinkInfo:
    """Thông tin vật lý của một liên kết khả thi."""
    distance: float
    SL_min: float
    TL: float
    NL: float
    R_bps: float


class Topology3D:
    """Quản lý topology 3D quasi-static cho mạng IoUT."""

    def __init__(self, net_cfg, acoustic_cfg, seed: int = 42):
        self.net_cfg = net_cfg
        self.acoustic_cfg = acoustic_cfg
        self.rng = np.random.RandomState(seed)
        self.N = net_cfg.N_SENSORS
        self.M = net_cfg.M_FOGS
        self.sensor_positions = self._place_sensors()
        self.fog_positions = self._place_fogs()
        self.gateway_position = np.array([
            net_cfg.AREA_X / 2.0, net_cfg.AREA_Y / 2.0, net_cfg.SURFACE_Z,
        ])

    def _place_sensors(self) -> np.ndarray:
        cfg = self.net_cfg
        x = self.rng.uniform(0, cfg.AREA_X, self.N)
        y = self.rng.uniform(0, cfg.AREA_Y, self.N)
        z = self.rng.uniform(cfg.SENSOR_DEPTH[0], cfg.SENSOR_DEPTH[1], self.N)
        return np.column_stack([x, y, z])

    def _place_fogs(self) -> np.ndarray:
        cfg = self.net_cfg
        x = self.rng.uniform(0, cfg.AREA_X, self.M)
        y = self.rng.uniform(0, cfg.AREA_Y, self.M)
        z = self.rng.uniform(cfg.FOG_DEPTH[0], cfg.FOG_DEPTH[1], self.M)
        return np.column_stack([x, y, z])

    @staticmethod
    def euclidean_3d(pos_a: np.ndarray, pos_b: np.ndarray) -> float:
        return float(np.linalg.norm(pos_a - pos_b))


def build_feasibility_graph(topology: Topology3D, acoustic_cfg) -> Dict:
    """
    Xây dựng đồ thị khả thi G — tất cả liên kết có SL_min ≤ SL_max.
    Keys: (type_u, id_u, type_v, id_v) → LinkInfo
    """
    G = {}
    ac = acoustic_cfg

    def _check_and_add(type_u, id_u, pos_u, type_v, id_v, pos_v):
        d = Topology3D.euclidean_3d(pos_u, pos_v)
        if d <= 0:
            return
        feasible, SL_min = is_link_feasible(
            d, ac.CARRIER_FREQ, ac.BANDWIDTH, ac.TARGET_SNR, ac.SL_MAX,
            ac.IL_LOSS, ac.SPREADING_FACTOR, ac.WIND_SPEED, ac.SHIPPING_FACTOR)
        if feasible:
            tl = transmission_loss(d, ac.CARRIER_FREQ, ac.SPREADING_FACTOR)
            nl = wenz_noise_level(ac.CARRIER_FREQ, ac.BANDWIDTH,
                                  ac.WIND_SPEED, ac.SHIPPING_FACTOR)
            R = shannon_capacity(ac.BANDWIDTH, ac.TARGET_SNR)
            G[(type_u, id_u, type_v, id_v)] = LinkInfo(
                distance=d, SL_min=SL_min, TL=tl, NL=nl, R_bps=R)

    for i in range(topology.N):
        for m in range(topology.M):
            _check_and_add('sensor', i, topology.sensor_positions[i],
                           'fog', m, topology.fog_positions[m])
    for m in range(topology.M):
        for n in range(m + 1, topology.M):
            _check_and_add('fog', m, topology.fog_positions[m],
                           'fog', n, topology.fog_positions[n])
            _check_and_add('fog', n, topology.fog_positions[n],
                           'fog', m, topology.fog_positions[m])
    for m in range(topology.M):
        _check_and_add('fog', m, topology.fog_positions[m],
                       'gateway', 0, topology.gateway_position)
    for i in range(topology.N):
        _check_and_add('sensor', i, topology.sensor_positions[i],
                       'gateway', 0, topology.gateway_position)
    return G


def nearest_feasible_association(topology: Topology3D, G: Dict) -> Dict[int, int]:
    """Sensor bắt tay Fog gần nhất khả thi. Returns dict[sensor_id → fog_id]."""
    association = {}
    for i in range(topology.N):
        best_fog, best_dist = None, float('inf')
        for m in range(topology.M):
            key = ('sensor', i, 'fog', m)
            if key in G and G[key].distance < best_dist:
                best_dist = G[key].distance
                best_fog = m
        if best_fog is not None:
            association[i] = best_fog
    return association


def flat_topology_association(topology: Topology3D, G: Dict) -> Dict[int, int]:
    """FedAvg/FedProx: Sensor → Gateway. Infeasible sensors bị drop."""
    association = {}
    for i in range(topology.N):
        if ('sensor', i, 'gateway', 0) in G:
            association[i] = -1
    return association


def build_clusters(association: Dict[int, int], M: int) -> Dict[int, List[int]]:
    """Xây dựng danh sách cụm từ association map."""
    clusters = {m: [] for m in range(M)}
    for sensor_id, fog_id in association.items():
        if fog_id >= 0:
            clusters[fog_id].append(sensor_id)
    return clusters





def get_topology_stats(topology: Topology3D, G: Dict,
                       association: Dict[int, int]) -> dict:
    """Thống kê topology để debug."""
    flat_assoc = flat_topology_association(topology, G)
    clusters = build_clusters(association, topology.M)
    sizes = [len(v) for v in clusters.values()]
    return {
        'total_sensors': topology.N,
        'connected_hfl': len(association),
        'participation_hfl': len(association) / topology.N,
        'connected_flat': len(flat_assoc),
        'participation_flat': len(flat_assoc) / topology.N,
        'cluster_sizes': sizes,
        'mean_cluster_size': float(np.mean(sizes)) if sizes else 0,
        'total_feasible_links': len(G),
    }
