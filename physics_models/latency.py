"""
latency.py
Mô hình Độ trễ Bất đối xứng cho IoUT (Eq. 13-19 trong Research Proposal).
"""

import numpy as np
from typing import Dict, List


def comm_delay(S_bits: float, R_bps: float,
               d_m: float, c_s: float = 1500.0) -> float:
    """
    Độ trễ truyền thông giữa hai nút (u, v)  —  Eq. 15.

    τ_comm = S/R (trễ truyền dẫn) + d/c_s (trễ lan truyền)
    """
    tx_delay = S_bits / R_bps
    prop_delay = d_m / c_s
    return tx_delay + prop_delay


def comp_delay_simple(n_local_epochs: int,
                      time_per_epoch: float = 0.1) -> float:
    """
    Độ trễ tính toán cục bộ (đơn giản hóa cho Scenario 1).
    Ước tính ~0.1s/epoch cho autoencoder nhỏ trên ARM.
    """
    return n_local_epochs * time_per_epoch

def comp_delay_dynamic(n_samples: int,
                       n_local_epochs: int,
                       flops_per_sample: float,
                       flop_multiplier: float = 1.0,
                       f_cpu: float = 2.0e9) -> float:
    """
    Độ trễ tính toán cục bộ động.
    T_comp = (samples * epochs * flops_per_sample * multiplier) / f_cpu
    """
    total_flops = n_samples * n_local_epochs * flops_per_sample * flop_multiplier
    return total_flops / f_cpu



def round_delay(
    sensor_delays: List[float],
    fog_delays: List[float],
    coop_delays: List[float],
    gateway_delay: float,
    downlink_delay: float,
) -> float:
    """
    Tổng độ trễ vòng lặp toàn mạng  —  Eq. 19.

    τ_round = max_m(τ_intra_m + τ_coop_m + τ_inter_m) + τ_gateway + τ_down

    Args:
        sensor_delays:  Per-fog max(τ_comp_i + τ_comm(i→fog)) for each fog.
        fog_delays:     Per-fog τ_agg_m + τ_comm(fog→gateway).
        coop_delays:    Per-fog τ_coop_m (0 if no cooperation).
        gateway_delay:  τ_gateway processing time.
        downlink_delay: τ_down = max_i(τ_comm(gateway→i)).

    Returns:
        τ_round in seconds.
    """
    per_fog_total = [s + c + f for s, c, f in
                     zip(sensor_delays, coop_delays, fog_delays)]
    bottleneck = max(per_fog_total) if per_fog_total else 0.0
    return bottleneck + gateway_delay + downlink_delay
