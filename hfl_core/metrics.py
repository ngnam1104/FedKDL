"""
metrics.py
Các hàm đánh giá và logging cho Scenario 1.
Bao gồm:
    - anomaly_threshold: Tính ngưỡng τ_A từ tập validation (phân vị 99).
    - point_adjusted_f1: Tính Point-Adjusted F1 (PA-F1) cho anomaly detection.
    - EnergyTracker:     Theo dõi và khấu hao năng lượng toàn mạng.
    - LatencyTracker:    Đo độ trễ vòng lặp τ_round theo Eq. 21 của paper.
    - MetricsLogger:     Ghi nhận metrics mỗi round.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def anomaly_threshold(val_errors: np.ndarray, percentile: float = 99.0) -> float:
    """
    Tính ngưỡng bất thường τ_A dựa trên lỗi tái tạo của tập validation.

    Args:
        val_errors: (N,) mảng lỗi tái tạo MSE của tập validation.
        percentile: Phân vị làm ngưỡng (thường 99.0 hoặc 99.5).

    Returns:
        Ngưỡng τ_A.
    """
    if len(val_errors) == 0:
        return 0.0
    return float(np.percentile(val_errors, percentile))


def point_adjusted_f1(y_true: np.ndarray, y_pred_scores: np.ndarray, threshold: float) -> Tuple[float, float, float]:
    """
    Tính Point-Adjusted F1 (PA-F1) score.
    Logic: Nếu một điểm trong segment anomaly được phát hiện (score > threshold),
    toàn bộ segment đó được coi là True Positive.

    Args:
        y_true:        (N,) ground truth (0: normal, 1: anomaly).
        y_pred_scores: (N,) reconstruction errors.
        threshold:     Ngưỡng τ_A.

    Returns:
        (pa_f1, precision, recall)
    """
    y_pred = (y_pred_scores > threshold).astype(int)
    
    # Point-Adjustment
    adjusted_pred = y_pred.copy()
    in_anomaly = False
    start_idx = -1
    
    for i, label in enumerate(y_true):
        if label == 1 and not in_anomaly:
            in_anomaly = True
            start_idx = i
        elif label == 0 and in_anomaly:
            in_anomaly = False
            end_idx = i
            # Nếu có bất kỳ dự đoán = 1 nào trong segment này, mark cả segment là 1
            if np.any(y_pred[start_idx:end_idx] == 1):
                adjusted_pred[start_idx:end_idx] = 1
                
    # Xử lý segment cuối cùng nếu kết thúc bằng anomaly
    if in_anomaly:
        if np.any(y_pred[start_idx:] == 1):
            adjusted_pred[start_idx:] = 1

    tp = np.sum((y_true == 1) & (adjusted_pred == 1))
    fp = np.sum((y_true == 0) & (adjusted_pred == 1))
    fn = np.sum((y_true == 1) & (adjusted_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return float(f1), float(precision), float(recall)


class EnergyTracker:
    """
    Theo dõi năng lượng tiêu thụ của mạng lưới theo từng round (Eq. 27).
    """
    def __init__(self):
        self.history = []
        self.cumulative_energy = 0.0

    def add_round(self, round_idx: int, e_s2f: float, e_f2f: float, e_f2g: float, e_comp: float):
        """Ghi nhận hóa đơn năng lượng của 1 round."""
        round_total = e_s2f + e_f2f + e_f2g + e_comp
        self.cumulative_energy += round_total
        
        self.history.append({
            'round': round_idx,
            'e_s2f': e_s2f,
            'e_f2f': e_f2f,
            'e_f2g': e_f2g,
            'e_comp': e_comp,
            'round_total': round_total,
            'cumulative_total': self.cumulative_energy,
        })

    def get_dataframe(self) -> pd.DataFrame:
        """Trả về lịch sử dưới dạng DataFrame."""
        return pd.DataFrame(self.history)


class LatencyTracker:
    """
    Đo độ trễ vòng lặp (round latency) theo Eq. 21 của paper.

    τ_round = max(
        max_{i → a_i} τ_{i→fog},
        max_{m → j} τ_{fog→fog},
        max_{m → g} τ_{fog→gateway}
    ) + τ_comp

    Công thức 1 link: τ_{u→v} = d_{uv}/c_s + L_{uv}/R_{uv}
    """

    def __init__(self, sound_speed: float = 1500.0, time_per_epoch: float = 0.1):
        """
        Args:
            sound_speed:     c_s (m/s) — vận tốc âm thanh trong nước.
            time_per_epoch:  Đƹn vị thời gian tính toán cục bộ (đủ cho AE nhỏ).
        """
        self.c_s = sound_speed
        self.time_per_epoch = time_per_epoch
        self.history: List[Dict] = []

    def _link_delay(self, S_bits: float, R_bps: float, d_m: float) -> float:
        """Tính trễ 1 link: tx delay + propagation delay."""
        tx = S_bits / R_bps if R_bps > 0 else 0.0
        prop = d_m / self.c_s
        return tx + prop

    def compute_round_latency(
        self,
        G: dict,
        association: Dict[int, int],
        cooperation_partners: Dict[int, int],
        n_local_epochs: int,
        sensor_payload_bits: float,
        fog_model_bits: float,
    ) -> float:
        """
        Tính τ_round theo Eq. 21.

        Args:
            G:                    Feasibility graph (output của build_feasibility_graph).
            association:          dict[sensor_id → fog_id].
            cooperation_partners: dict[fog_id → partner_fog_id] (rỗng nếu ko coop).
            n_local_epochs:       Số epoch cục bộ (E).
            sensor_payload_bits:  Kích thước payload cảm biến (bits).
            fog_model_bits:       Kích thước model fog full-precision (bits).

        Returns:
            tau_round in seconds.
        """
        from physics_models.latency import comm_delay, comp_delay_simple

        # 1. Sensor → Fog delays (mỗi fog lấy max trong cụm)
        s2f_per_fog: Dict[int, float] = {}
        for s_id, fog_id in association.items():
            key = ('sensor', s_id, 'fog', fog_id)
            if key in G:
                link = G[key]
                delay = comm_delay(sensor_payload_bits, link.R_bps, link.distance, self.c_s)
            else:
                delay = 0.0
            s2f_per_fog[fog_id] = max(s2f_per_fog.get(fog_id, 0.0), delay)

        # 2. Fog → Fog cooperation delays
        f2f_per_fog: Dict[int, float] = {}
        for fog_id, partner_id in cooperation_partners.items():
            key_fwd = ('fog', partner_id, 'fog', fog_id)  # partner phát → fog nhận
            key_bwd = ('fog', fog_id, 'fog', partner_id)
            key = key_fwd if key_fwd in G else (key_bwd if key_bwd in G else None)
            if key:
                link = G[key]
                delay = comm_delay(fog_model_bits, link.R_bps, link.distance, self.c_s)
            else:
                delay = 0.0
            f2f_per_fog[fog_id] = delay

        # 3. Fog → Gateway delays
        f2g_delays = []
        for m in set(association.values()):
            key = ('fog', m, 'gateway', 0)
            if key in G:
                link = G[key]
                f2g_delays.append(comm_delay(fog_model_bits, link.R_bps, link.distance, self.c_s))

        # 4. Tính toán cục bộ
        tau_comp = comp_delay_simple(n_local_epochs, self.time_per_epoch)

        # Bottleneck: max over all fogs of (s2f + f2f + f2g)
        all_fog_ids = set(association.values())
        per_fog_total = [
            s2f_per_fog.get(m, 0.0) +
            f2f_per_fog.get(m, 0.0) +
            (max(f2g_delays) if f2g_delays else 0.0)
            for m in all_fog_ids
        ]
        tau_round = (max(per_fog_total) if per_fog_total else 0.0) + tau_comp
        return tau_round

    def add_round(self, round_idx: int, tau_round: float):
        """Ghi nhận latency của 1 round."""
        self.history.append({'round': round_idx, 'tau_round_s': tau_round})

    def get_dataframe(self) -> pd.DataFrame:
        """Trả về lịch sử latency dưới dạng DataFrame."""
        return pd.DataFrame(self.history)


class MetricsLogger:
    """
    Trình quản lý log metrics cho vòng lặp mô phỏng.
    """
    def __init__(self):
        self.logs = []

    def log(self, round_idx: int, metrics: Dict[str, float]):
        """Ghi nhận metrics của round."""
        entry = {'round': round_idx}
        entry.update(metrics)
        self.logs.append(entry)

    def print_latest(self):
        """In log mới nhất ra console."""
        if not self.logs:
            return
        latest = self.logs[-1]
        msg = f"Round {latest['round']:3d} | "
        for k, v in latest.items():
            if k == 'round': continue
            msg += f"{k}: {v:.4f} | "
        print(msg)

    def get_dataframe(self) -> pd.DataFrame:
        """Trả về toàn bộ logs dưới dạng DataFrame."""
        return pd.DataFrame(self.logs)
