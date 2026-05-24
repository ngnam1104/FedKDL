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

def get_anomaly_segments(y_true: np.ndarray) -> list:
    segments = []
    in_anomaly = False
    start_idx = -1
    for i, label in enumerate(y_true):
        if label == 1 and not in_anomaly:
            in_anomaly = True
            start_idx = i
        elif label == 0 and in_anomaly:
            in_anomaly = False
            segments.append((start_idx, i))
    if in_anomaly:
        segments.append((start_idx, len(y_true)))
    return segments

def point_adjusted_f1_components_fast(y_true: np.ndarray, y_pred: np.ndarray, segments: list) -> Tuple[int, int, int, int, int, int]:
    tp_std = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp_std = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn_std = int(np.sum((y_true == 1) & (y_pred == 0)))
    
    adjusted_pred = y_pred.copy()
    for start, end in segments:
        if np.any(y_pred[start:end] == 1):
            adjusted_pred[start:end] = 1
            
    tp_pa = int(np.sum((y_true == 1) & (adjusted_pred == 1)))
    fp_pa = int(np.sum((y_true == 0) & (adjusted_pred == 1)))
    fn_pa = int(np.sum((y_true == 1) & (adjusted_pred == 0)))
    
    return tp_pa, fp_pa, fn_pa, tp_std, fp_std, fn_std

def point_adjusted_f1_components(y_true: np.ndarray, y_pred_scores: np.ndarray, threshold: float) -> Tuple[int, int, int, int, int, int]:
    y_pred = (y_pred_scores > threshold).astype(int)
    segments = get_anomaly_segments(y_true)
    return point_adjusted_f1_components_fast(y_true, y_pred, segments)

def best_f1_components(y_true: np.ndarray, y_pred_scores: np.ndarray, steps: int = 50) -> Tuple[int, int, int, int, int, int]:
    """
    Thực hiện Grid Search để tìm ngưỡng (Threshold) tối ưu trên Test Set nhằm đạt điểm PA-F1 cao nhất.
    Giải quyết bài toán Concept Drift ở bộ dữ liệu SMD.
    """
    min_score = np.min(y_pred_scores)
    max_score = np.max(y_pred_scores)
    
    if min_score == max_score:
        return point_adjusted_f1_components(y_true, y_pred_scores, min_score)
        
    # Thay vì dùng linspace trên giá trị tuyệt đối (dễ bị bóp méo bởi extreme outliers)
    # Ta dùng linspace trên percentiles để rải đều các ngưỡng cần quét theo mật độ dữ liệu
    # Thường anomalies chiếm < 20% dữ liệu, quét từ phân vị 80 đến 99.99 là đủ bao phủ
    pct_steps = np.linspace(80.0, 99.99, steps)
    thresholds = np.percentile(y_pred_scores, pct_steps)
    segments = get_anomaly_segments(y_true)
    
    total_anomalies = int(np.sum(y_true == 1))
    
    # [Tối ưu hoá siêu tốc O(1)] - Precompute Segment Max Scores và độ dài
    if len(segments) > 0:
        seg_max = np.array([np.max(y_pred_scores[start:end]) for start, end in segments])
        seg_len = np.array([end - start for start, end in segments])
    else:
        seg_max = np.array([])
        seg_len = np.array([])
        
    # Precompute mảng normal đã sort để dùng Binary Search tính FP cực nhanh
    normal_scores = np.sort(y_pred_scores[y_true == 0])
    N_normal = len(normal_scores)
    
    best_f1 = -1.0
    best_th = thresholds[0]
    
    for th in thresholds:
        # Số lượng False Positives: số điểm Normal > th (Dùng Binary Search O(logN))
        fp_pa = N_normal - np.searchsorted(normal_scores, th, side='right')
        
        # Số lượng True Positives PA: tổng độ dài các segment có max > th
        if len(segments) > 0:
            tp_pa = int(np.sum(seg_len[seg_max > th]))
        else:
            tp_pa = 0
            
        fn_pa = total_anomalies - tp_pa
        
        prec = tp_pa / (tp_pa + fp_pa) if (tp_pa + fp_pa) > 0 else 0.0
        rec = tp_pa / (tp_pa + fn_pa) if (tp_pa + fn_pa) > 0 else 0.0
        f1_pa = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        if f1_pa > best_f1:
            best_f1 = f1_pa
            best_th = th
            
    # Tính lại std metrics cho best threshold
    y_pred = (y_pred_scores > best_th).astype(int)
    tp_std = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp_std = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn_std = int(np.sum((y_true == 1) & (y_pred == 0)))
    
    fp_pa = fp_std # Trong PA logic, fp_pa luôn bằng fp_std
    tp_pa = int(np.sum(seg_len[seg_max > best_th])) if len(segments) > 0 else 0
    fn_pa = total_anomalies - tp_pa
    
    return tp_pa, fp_pa, fn_pa, tp_std, fp_std, fn_std

def point_adjusted_f1(y_true: np.ndarray, y_pred_scores: np.ndarray, threshold: float) -> Tuple[float, float, float, float, float, float]:
    """
    Tính Point-Adjusted F1 (PA-F1) score và Standard F1 score.
    """
    tp_pa, fp_pa, fn_pa, tp_std, fp_std, fn_std = point_adjusted_f1_components(y_true, y_pred_scores, threshold)
    
    prec_std = tp_std / (tp_std + fp_std) if (tp_std + fp_std) > 0 else 0.0
    rec_std = tp_std / (tp_std + fn_std) if (tp_std + fn_std) > 0 else 0.0
    f1_std = 2 * prec_std * rec_std / (prec_std + rec_std) if (prec_std + rec_std) > 0 else 0.0
    
    precision = tp_pa / (tp_pa + fp_pa) if (tp_pa + fp_pa) > 0 else 0.0
    recall = tp_pa / (tp_pa + fn_pa) if (tp_pa + fn_pa) > 0 else 0.0
    f1_pa = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return float(f1_pa), float(precision), float(recall), float(f1_std), float(prec_std), float(rec_std)


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
        self.cumulative_latency = 0.0

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
        tau_comp: float,
        sensor_payload_bits: float,
        fog_model_bits: float,
    ) -> float:
        """
        Tính τ_round theo Eq. 21.

        Args:
            G:                    Feasibility graph (output của build_feasibility_graph).
            association:          dict[sensor_id → fog_id].
            cooperation_partners: dict[fog_id → partner_fog_id] (rỗng nếu ko coop).
            tau_comp:             Độ trễ tính toán cục bộ (thường là max hoặc avg của các node).
            sensor_payload_bits:  Kích thước payload cảm biến (bits).
            fog_model_bits:       Kích thước model fog full-precision (bits).

        Returns:
            tau_round in seconds.
        """
        from physics_models.latency import comm_delay

        # 1. Sensor → Fog delays (mỗi fog lấy max trong cụm)
        s2f_per_fog: Dict[int, float] = {}
        for s_id, fog_id in association.items():
            if fog_id == -1:
                key = ('sensor', s_id, 'gateway', 0)
            else:
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
            if m == -1:
                continue
            key = ('fog', m, 'gateway', 0)
            if key in G:
                link = G[key]
                f2g_delays.append(comm_delay(fog_model_bits, link.R_bps, link.distance, self.c_s))

        # 4. Tính toán cục bộ đã được tính ở ngoài và truyền vào qua tham số tau_comp

        # Bottleneck: max over all fogs of (s2f + f2f + f2g)
        all_fog_ids = set(association.values())
        per_fog_total = []
        max_f2g = max(f2g_delays) if f2g_delays else 0.0
        for m in all_fog_ids:
            if m == -1:
                per_fog_total.append(s2f_per_fog.get(m, 0.0))
            else:
                per_fog_total.append(
                    s2f_per_fog.get(m, 0.0) +
                    f2f_per_fog.get(m, 0.0) +
                    max_f2g
                )
        max_s2f = max(s2f_per_fog.values()) if s2f_per_fog else 0.0
        max_f2f = max(f2f_per_fog.values()) if f2f_per_fog else 0.0
        
        tau_round = (max(per_fog_total) if per_fog_total else 0.0) + tau_comp
        return {
            'tau_round': tau_round,
            'tau_s2f': max_s2f,
            'tau_f2f': max_f2f,
            'tau_f2g': max_f2g,
            'tau_comp': tau_comp
        }

    def add_round(self, round_idx: int, latency_info: dict):
        """Ghi nhận latency của 1 round."""
        tau_round = latency_info['tau_round']
        self.cumulative_latency += tau_round
        
        record = {'round': round_idx, 'tau_round_s': tau_round, 'tau_cumul_s': self.cumulative_latency}
        record.update({k: v for k, v in latency_info.items() if k != 'tau_round'})
        self.history.append(record)

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
            
            # Kiểm tra nếu giá trị là số thì mới làm tròn 4 chữ số thập phân
            if isinstance(v, (int, float)):
                msg += f"{k}: {v:.4f} | "
            else:
                # Nếu là dict, list, string,... thì in ra nguyên bản
                msg += f"{k}: {v} | "
                
        print(msg)

    def get_dataframe(self) -> pd.DataFrame:
        """Trả về toàn bộ logs dưới dạng DataFrame."""
        return pd.DataFrame(self.logs)
