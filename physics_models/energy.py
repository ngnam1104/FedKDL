"""
energy.py
Mô hình Năng lượng Vật lý cho IoUT (Eq. 20-27 trong Research Proposal).
Sử dụng công thức công suất âm thanh chuẩn vật lý với ρ_w và c_s.
"""

import numpy as np


# ──────────────────────────────────────────────────────────────────────
#  Công suất bức xạ âm thanh chuẩn vật lý (Critical for correct scale)
# ──────────────────────────────────────────────────────────────────────

def acoustic_power_watts(SL_min_dB: float,
                         rho_w: float = 1025.0,
                         c_s: float = 1500.0) -> float:
    """
    Công suất bức xạ âm thanh P_ac từ Source Level (dB re 1µPa @ 1m).
    
    Công thức vật lý chuẩn:
        P_ac = (4π * p_ref²) / (ρ_w * c_s) * 10^(SL/10)
    
    Với p_ref = 1µPa = 10⁻⁶ Pa (reference pressure underwater acoustics).

    Args:
        SL_min_dB: Source Level tối thiểu (dB re 1µPa @ 1m).
        rho_w:     Mật độ nước biển (kg/m³), mặc định 1025.
        c_s:       Vận tốc âm thanh (m/s), mặc định 1500.

    Returns:
        P_ac in Watts.
    """
    p_ref = 1e-6  # 1 µPa in Pascals
    P_ac = (4.0 * np.pi * p_ref ** 2) / (rho_w * c_s) * 10.0 ** (SL_min_dB / 10.0)
    return P_ac


# ──────────────────────────────────────────────────────────────────────
#  Năng lượng truyền thông (Communication Energy)
# ──────────────────────────────────────────────────────────────────────

def e_tx(S_bits: float, R_bps: float,
         SL_min_dB: float,
         eta_ea: float = 0.25,
         P_c_tx: float = 0.05,
         rho_w: float = 1025.0,
         c_s: float = 1500.0) -> float:
    """
    Năng lượng phát trên liên kết (u, v)  —  Eq. 22.

    E_tx = (P_ac / η_ea + P_c_tx) × (S / R)

    Args:
        S_bits:    Kích thước gói tin (bits).
        R_bps:     Tốc độ Shannon (bps).
        SL_min_dB: Source Level tối thiểu cho liên kết (dB).
        eta_ea:    Hiệu suất chuyển đổi điện-âm.
        P_c_tx:    Công suất tĩnh mạch phát (W).
        rho_w:     Mật độ nước biển (kg/m³).
        c_s:       Vận tốc âm thanh (m/s).

    Returns:
        E_tx in Joules.
    """
    P_ac = acoustic_power_watts(SL_min_dB, rho_w, c_s)
    tx_duration = S_bits / R_bps  # seconds
    return (P_ac / eta_ea + P_c_tx) * tx_duration


def e_rx(S_bits: float, R_bps: float,
         P_c_rx: float = 0.03) -> float:
    """
    Năng lượng nhận trên liên kết  —  Eq. 23.

    E_rx = P_c_rx × (S / R)

    Returns:
        E_rx in Joules.
    """
    rx_duration = S_bits / R_bps
    return P_c_rx * rx_duration


# ──────────────────────────────────────────────────────────────────────
#  Năng lượng điện toán (Computation Energy)
# ──────────────────────────────────────────────────────────────────────

def e_comp_dynamic(n_samples: int,
                   n_local_epochs: int,
                   flops_per_sample: float,
                   epsilon_op: float,
                   flop_multiplier: float = 3.0) -> float:
    """
    Năng lượng tính toán cục bộ dựa trên số lượng FLOPs (tường minh).
    
    Phi_i = n_samples * n_local_epochs * flops_per_sample * flop_multiplier
    E_comp = epsilon_op * Phi_i
    
    Returns:
        E_comp in Joules.
    """
    total_flops = n_samples * n_local_epochs * flops_per_sample * flop_multiplier
    return total_flops * epsilon_op


def e_comp_full(n_samples: int, n_local_epochs: int,
                zeta: float, theta_size: int,
                f_cpu: float, epsilon_op: float) -> float:
    """
    Năng lượng tính toán cục bộ (công thức đầy đủ)  —  Eq. 20.

    Phi = n_samples × E_local × ζ × |Θ|
    E_comp = ε_op × Phi × f_cpu²

    Returns:
        E_comp in Joules.
    """
    Phi = n_samples * n_local_epochs * zeta * theta_size
    return epsilon_op * Phi * f_cpu ** 2


# ──────────────────────────────────────────────────────────────────────
#  Tổng hợp năng lượng theo tầng (Layer-based Energy Decomposition)
# ──────────────────────────────────────────────────────────────────────

def total_energy_round(e_a2r: float, e_r2r: float,
                       e_r2g: float, e_comp_total: float) -> float:
    """
    Tổng năng lượng tiêu thụ toàn mạng tại vòng t  —  Eq. 27.

    E_total = E_a2r + E_r2r + E_r2g + Σ E_comp
    """
    return e_a2r + e_r2r + e_r2g + e_comp_total
