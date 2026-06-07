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
                       f_cpu: float = 1.5e9,
                       n_cores: int = 6,
                       flops_per_cycle: float = 4.0) -> float:
    """
    Độ trễ tính toán cục bộ động.
    T_comp = FLOPs / (f_cpu * n_cores * flops_per_cycle)
    Nguồn: Jetson Orin Nano Datasheet r4 (F_CPU=1.5GHz, N_CORES=6)
    """
    total_flops = n_samples * n_local_epochs * flops_per_sample * flop_multiplier
    return total_flops / (f_cpu * n_cores * flops_per_cycle)


def relay_comp_delay(d_out: int = 256, d_in: int = 128,
                     n_svd_calls: int = 2,
                     f_cpu: float = 1.5e9,
                     n_cores: int = 6,
                     flops_per_cycle: float = 4.0) -> float:
    """
    Độ trễ tính toán tại trạm Relay  —  τ_comp,m = Φ_m / (f_cpu * n_cores * flops_per_cycle).

    Khối lượng công việc Phi_m bao gồm các bước tổng hợp mô hình
    (xấp xỉ bằng FLOPs của phân rã không gian con, thực hiện 2 lần mỗi vòng):
        Φ_m ≈ n_svd_calls × 6 × d_out × d_in × min(d_out, d_in)

    Args:
        d_out:       Chiều output của lớp đại diện (default: 256).
        d_in:        Chiều input của lớp đại diện (default: 128).
        n_svd_calls: Số lần xử lý mỗi vòng lặp (default: 2).
        f_cpu:       Xung nhịp CPU tại Relay. Nguồn: Jetson Orin Nano Datasheet r4.
        n_cores:     Số lõi CPU. Nguồn: Jetson Orin Nano Datasheet r4.

    Returns:
        τ_comp,m in seconds.
    """
    k = min(d_out, d_in)
    phi_m = n_svd_calls * 6 * d_out * d_in * k
    return phi_m / (f_cpu * n_cores * flops_per_cycle)



def round_delay(
    auv_delays: List[float],
    relay_delays: List[float],
    coop_delays: List[float],
    gateway_delay: float,
    downlink_delay: float,
) -> float:
    """
    Tổng độ trễ vòng lặp toàn mạng  —  Eq. 19.

    τ_round = max_m(τ_intra_m + τ_coop_m + τ_inter_m) + τ_gateway + τ_down

    Args:
        auv_delays:  Per-relay max(τ_comp_i + τ_comm(i→relay)) for each relay.
        relay_delays:     Per-relay τ_agg_m + τ_comm(relay→gateway).
        coop_delays:    Per-relay τ_coop_m (0 if no cooperation).
        gateway_delay:  τ_gateway processing time.
        downlink_delay: τ_down = max_i(τ_comm(gateway→i)).

    Returns:
        τ_round in seconds.
    """
    per_relay_total = [s + c + f for s, c, f in
                     zip(auv_delays, coop_delays, relay_delays)]
    bottleneck = max(per_relay_total) if per_relay_total else 0.0
    return bottleneck + gateway_delay + downlink_delay
