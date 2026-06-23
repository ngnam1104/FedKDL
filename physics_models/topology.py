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
        self.N = net_cfg.N_AUVS
        self.M = net_cfg.M_RELAYS
        self.auv_positions = self._place_auvs()
        self.relay_positions = self._place_relays()
        self.gateway_position = np.array([
            net_cfg.AREA_X / 2.0, net_cfg.AREA_Y / 2.0, net_cfg.SURFACE_Z,
        ])
        # [GM Mobility] Trạng thái Gauss-Markov hình cầu (Eq. 1a-1c): tốc độ, heading, pitch
        mean_speed = max(0.0, float(getattr(net_cfg, 'GM_MEAN_SPEED', 1.5)))
        self.auv_speeds   = self.rng.uniform(0.0, mean_speed, self.N)      # s_i (m/s)
        self.auv_headings = self.rng.uniform(-np.pi, np.pi, self.N)        # φ_i (rad)
        self.auv_pitches  = self.rng.uniform(-0.2, 0.2, self.N)            # ψ_i (rad)
        # Legacy — giữ lại để tương thích ngược với code cũ tham chiếu auv_velocities
        self.auv_velocities = np.zeros((self.N, 3))

    def step_mobile_auvs(
        self,
        mu: float = 0.7,
        mean_speed: float = 1.5,
        max_speed: float = 5.0,
        mean_heading: float = 0.0,
        mean_pitch: float = 0.0,
        sigma_speed: float = 0.5,
        sigma_heading: float = 0.3,
        sigma_pitch: float = 0.1,
        dt: float = 1.0,
    ):
        """
        Gauss-Markov 3D mobility cho AUVs — khớp chính xác với Eq. (1a-1c) và (2) trong paper.

        Các biến trạng thái: (s, φ, ψ) — tốc độ, heading ngang, pitch.
        Update rule (Eq. 1a-1c):
            s[t+1] = μ·s[t] + (1-μ)·s̄ + √(1-μ²)·ξ_s
            φ[t+1] = μ·φ[t] + (1-μ)·φ̄ + √(1-μ²)·ξ_φ
            ψ[t+1] = μ·ψ[t] + (1-μ)·ψ̄ + √(1-μ²)·ξ_ψ

        Chuyển sang Cartesian (Eq. 2):
            Δx = s·cos(ψ)·cos(φ)
            Δy = s·cos(ψ)·sin(φ)
            Δz = s·sin(ψ)

        Args:
            mu:           μ_GM — memory factor ∈ [0,1]
            mean_speed:   s̄ — tốc độ trung bình mục tiêu (m/s)
            max_speed:    Giới hạn trên tốc độ (đảm bảo Δt·s ≤ 5m)
            mean_heading: φ̄ — heading ngang trung bình (rad)
            mean_pitch:   ψ̄ — góc pitch trung bình (rad)
            sigma_speed:  σ_s — std của nhiễu tốc độ
            sigma_heading:σ_φ — std của nhiễu heading (rad)
            sigma_pitch:  σ_ψ — std của nhiễu pitch (rad)
        """
        noise_scale = np.sqrt(1.0 - mu ** 2)

        # Cập nhật tốc độ (Eq. 1a)
        xi_s = self.rng.normal(0.0, sigma_speed, self.N)
        self.auv_speeds = mu * self.auv_speeds + (1.0 - mu) * mean_speed + noise_scale * xi_s
        # Giới hạn tốc độ ≥ 0 và ≤ max_speed (đảm bảo Δt·s ≤ 5m)
        self.auv_speeds = np.clip(self.auv_speeds, 0.0, max_speed)

        # Cập nhật heading φ (Eq. 1b)
        xi_phi = self.rng.normal(0.0, sigma_heading, self.N)
        self.auv_headings = mu * self.auv_headings + (1.0 - mu) * mean_heading + noise_scale * xi_phi

        # Cập nhật pitch ψ (Eq. 1c) — giới hạn trong [-π/2, π/2] để tránh lật ngược
        xi_psi = self.rng.normal(0.0, sigma_pitch, self.N)
        self.auv_pitches = mu * self.auv_pitches + (1.0 - mu) * mean_pitch + noise_scale * xi_psi
        self.auv_pitches = np.clip(self.auv_pitches, -np.pi / 2.0, np.pi / 2.0)

        # Chuyển sang Cartesian (Eq. 2): Δc = Δt·s·[cos(ψ)cos(φ), cos(ψ)sin(φ), sin(ψ)]
        # dt is one normalized mobility slot by default; sweeps may override it.
        old_positions = self.auv_positions.copy()
        cos_psi = np.cos(self.auv_pitches)
        delta = np.column_stack([
            dt * self.auv_speeds * cos_psi * np.cos(self.auv_headings),   # Δx
            dt * self.auv_speeds * cos_psi * np.sin(self.auv_headings),   # Δy
            dt * self.auv_speeds * np.sin(self.auv_pitches),              # Δz
        ])
        self.auv_positions += delta

        # Boundary reflection — giữ AUV trong vùng triển khai
        cfg = self.net_cfg
        for i in range(self.N):
            x, y, z = self.auv_positions[i]
            if x < 0 or x > cfg.AREA_X:
                self.auv_headings[i] = np.pi - self.auv_headings[i]
                self.auv_positions[i, 0] = np.clip(x, 0.0, cfg.AREA_X)
            if y < 0 or y > cfg.AREA_Y:
                self.auv_headings[i] = -self.auv_headings[i]
                self.auv_positions[i, 1] = np.clip(y, 0.0, cfg.AREA_Y)
            if z < cfg.AUV_DEPTH[0] or z > cfg.AUV_DEPTH[1]:
                self.auv_pitches[i] = -self.auv_pitches[i]
                self.auv_positions[i, 2] = np.clip(z, cfg.AUV_DEPTH[0], cfg.AUV_DEPTH[1])

        actual_delta = self.auv_positions - old_positions
        distances = np.linalg.norm(actual_delta, axis=1)
        self.auv_velocities = actual_delta / max(float(dt), 1e-12)
        return {
            "delta": actual_delta,
            "distance_m": distances,
            "speed_mps": self.auv_speeds.copy(),
            "avg_move_m": float(np.mean(distances)) if len(distances) else 0.0,
            "max_move_m": float(np.max(distances)) if len(distances) else 0.0,
            "avg_speed_mps": float(np.mean(self.auv_speeds)) if len(self.auv_speeds) else 0.0,
        }

    def _place_auvs(self) -> np.ndarray:
        cfg = self.net_cfg
        x = self.rng.uniform(0, cfg.AREA_X, self.N)
        y = self.rng.uniform(0, cfg.AREA_Y, self.N)
        z = self.rng.uniform(cfg.AUV_DEPTH[0], cfg.AUV_DEPTH[1], self.N)
        return np.column_stack([x, y, z])

    def _place_relays(self) -> np.ndarray:
        """
        Đặt Relay node đều vào 4 vùng (quadrant XY) để đảm bảo phủ sóng toàn vùng biển.
        Nếu M không chia hết cho 4, các relay dư sẽ đặt ngẫu nhiên ở khu vực trung tâm.
        """
        cfg = self.net_cfg
        hx, hy = cfg.AREA_X / 2.0, cfg.AREA_Y / 2.0

        quadrants = [
            (0,   hx, 0,   hy),   # Q0: Tây-Bắc
            (hx, cfg.AREA_X, 0,   hy),   # Q1: Đông-Bắc
            (0,   hx, hy, cfg.AREA_Y),   # Q2: Tây-Nam
            (hx, cfg.AREA_X, hy, cfg.AREA_Y),   # Q3: Đông-Nam
        ]

        per_quad = self.M // 4
        remainder = self.M % 4

        positions = []
        for qx_min, qx_max, qy_min, qy_max in quadrants:
            for _ in range(per_quad):
                x = self.rng.uniform(qx_min, qx_max)
                y = self.rng.uniform(qy_min, qy_max)
                z = self.rng.uniform(cfg.RELAY_DEPTH[0], cfg.RELAY_DEPTH[1])
                positions.append([x, y, z])

        # Relay dư đặt ở vùng trung tâm (chiếm 20% diện tích giữa)
        for _ in range(remainder):
            x = self.rng.uniform(hx * 0.8, hx * 1.2)
            y = self.rng.uniform(hy * 0.8, hy * 1.2)
            z = self.rng.uniform(cfg.RELAY_DEPTH[0], cfg.RELAY_DEPTH[1])
            positions.append([x, y, z])

        return np.array(positions, dtype=float)

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
            _check_and_add('auv', i, topology.auv_positions[i],
                           'relay', m, topology.relay_positions[m])
    for m in range(topology.M):
        for n in range(m + 1, topology.M):
            _check_and_add('relay', m, topology.relay_positions[m],
                           'relay', n, topology.relay_positions[n])
            _check_and_add('relay', n, topology.relay_positions[n],
                           'relay', m, topology.relay_positions[m])
    for m in range(topology.M):
        _check_and_add('relay', m, topology.relay_positions[m],
                       'gateway', 0, topology.gateway_position)
    for i in range(topology.N):
        _check_and_add('auv', i, topology.auv_positions[i],
                       'gateway', 0, topology.gateway_position)
    return G


def nearest_feasible_association(topology: Topology3D, G: Dict) -> Dict[int, int]:
    """AUV bắt tay Relay gần nhất khả thi (có cân bằng tải). Returns dict[auv_id → relay_id]."""
    import math
    N, M = topology.N, topology.M
    max_capacity = math.ceil(N / M) + 3
    
    valid_pairs = []
    for i in range(N):
        for m in range(M):
            key = ('auv', i, 'relay', m)
            if key in G:
                valid_pairs.append((G[key].distance, i, m))
                
    valid_pairs.sort(key=lambda x: x[0])
    
    association = {}
    relay_counts = {m: 0 for m in range(M)}
    
    for dist, auv_id, relay_id in valid_pairs:
        if auv_id not in association and relay_counts[relay_id] < max_capacity:
            association[auv_id] = relay_id
            relay_counts[relay_id] += 1
            
    for i in range(N):
        if i not in association:
            best_relay, best_dist = None, float('inf')
            for m in range(M):
                key = ('auv', i, 'relay', m)
                if key in G and G[key].distance < best_dist:
                    best_dist = G[key].distance
                    best_relay = m
            if best_relay is not None:
                association[i] = best_relay
                relay_counts[best_relay] += 1
                
    return association


def flat_topology_association(topology: Topology3D, G: Dict) -> Dict[int, int]:
    """FedAvg/FedProx: AUV → Gateway. Infeasible auvs bị drop."""
    association = {}
    for i in range(topology.N):
        if ('auv', i, 'gateway', 0) in G:
            association[i] = -1
    return association


def gateway_disconnected_relays(topology: Topology3D, G: Dict) -> List[int]:
    """Return relay IDs without a feasible uplink to the gateway."""
    return [
        relay_id
        for relay_id in range(topology.M)
        if ('relay', relay_id, 'gateway', 0) not in G
    ]


def build_clusters(association: Dict[int, int], M: int) -> Dict[int, List[int]]:
    """Xây dựng danh sách cụm từ association map."""
    clusters = {m: [] for m in range(M)}
    for auv_id, relay_id in association.items():
        if relay_id >= 0:
            clusters[relay_id].append(auv_id)
    return clusters





def get_topology_stats(topology: Topology3D, G: Dict,
                       association: Dict[int, int]) -> dict:
    """Thống kê topology để debug."""
    flat_assoc = flat_topology_association(topology, G)
    clusters = build_clusters(association, topology.M)
    sizes = [len(v) for v in clusters.values()]
    return {
        'total_auvs': topology.N,
        'connected_hfl': len(association),
        'participation_hfl': len(association) / topology.N,
        'connected_flat': len(flat_assoc),
        'participation_flat': len(flat_assoc) / topology.N,
        'cluster_sizes': sizes,
        'mean_cluster_size': float(np.mean(sizes)) if sizes else 0,
        'total_feasible_links': len(G),
    }
