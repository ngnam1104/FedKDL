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

def e_comp(n_samples: int,
           local_epochs: int,
           flops_per_sample: float,
           epsilon_op: float,
           flop_multiplier: float = 1.0,
           f_cpu: float = 1.5e9) -> float:
    """
    Năng lượng tính toán cục bộ chuẩn vật lý vi mạch (CMOS).
    
    Trong mô hình phân tán, khối lượng công việc và năng lượng yêu cầu:
    - Phi_i = n_samples * E_local * flops_per_sample * flop_multiplier
    - E_comp = epsilon_op * Phi_i * (f_cpu ** 2)
    
    Args:
        n_samples:        Số lượng mẫu dữ liệu nội bộ.
        local_epochs:     Số vòng lặp nội bộ.
        flops_per_sample: Chi phí tính toán (FLOPs) cho 1 mẫu dữ liệu.
        epsilon_op:       Hệ số điện dung hiệu dụng tĩnh.
        flop_multiplier:  Hệ số nhân thuật toán (bù trừ lan truyền ngược).
        f_cpu:            Xung nhịp CPU tĩnh.
        
    Returns:
        E_comp in Joules.
    """
    total_flops = n_samples * local_epochs * flops_per_sample * flop_multiplier
    return epsilon_op * total_flops * (f_cpu ** 2)


def e_svd(d_out: int, d_in: int,
           epsilon_op: float,
           n_svd_calls: int = 2,
           f_cpu: float = 1.5e9) -> float:
    """
    Năng lượng tính toán SVD tại Relay  —  Eq. bổ sung.

    FLOPs của Truncated SVD (full_matrices=False) xấp xỉ:
        Phi_svd ≈ 6 × d_out × d_in × min(d_out, d_in)

    Relay thực hiện 2 lần SVD mỗi vòng:
        - Lần 1 (Temp SVD): sau intra-cluster aggregation
        - Lần 2 (Final SVD): sau HFL-Nearest blending

    Args:
        d_out:       Chiều output của lớp LoRA (d_out).
        d_in:        Chiều input của lớp LoRA (d_in).
        epsilon_op:  Hệ số điện dung hiệu dụng.
        n_svd_calls: Số lần SVD thực hiện (mặc định 2).
        f_cpu:       Xung nhịp CPU.

    Returns:
        E_svd in Joules.
    """
    k = min(d_out, d_in)
    flops_per_svd = 6 * d_out * d_in * k
    total_flops = n_svd_calls * flops_per_svd
    return epsilon_op * total_flops * (f_cpu ** 2)


# ──────────────────────────────────────────────────────────────────────
#  Tổng hợp năng lượng theo tầng (Layer-based Energy Decomposition)
# ──────────────────────────────────────────────────────────────────────

def e_move_yang_surge(
    speed_mps,
    duration_s: float,
    rho_w: float = 1025.0,
    thruster_radius: float = 0.025,
    surge_drag_coeff: float = 48.17,
    n_horizontal_thrusters: int = 2,
    hotel_power: float = 0.0,
):
    """
    AUV mobility energy following Yang et al. CDC 2018.

    The paper models thruster power as
        P(T_i) = C_p * T_i^1.5,  C_p = sqrt(1 / (2*pi*rho_w)) / R,
    and surge drag as
        T_total = X_|u|u * |u| * u.

    For forward motion, u >= 0, so T_total = X_|u|u * u^2. We split this
    total horizontal thrust evenly over the horizontal thrusters and sum
    their power over one mobility slot.
    """
    speeds = np.asarray(speed_mps, dtype=np.float64)
    duration = max(float(duration_s), 0.0)
    radius = max(float(thruster_radius), 1e-12)
    n_thrusters = max(int(n_horizontal_thrusters), 1)
    cp = np.sqrt(1.0 / (2.0 * np.pi * float(rho_w))) / radius
    total_thrust = float(surge_drag_coeff) * np.maximum(speeds, 0.0) ** 2
    per_thruster_thrust = total_thrust / n_thrusters
    propulsion_power = n_thrusters * cp * per_thruster_thrust ** 1.5
    return (propulsion_power + float(hotel_power)) * duration


def total_energy_round(
    e_a2r: float,
    e_r2r: float,
    e_r2g: float,
    e_comp_total: float,
    e_svd_total: float = 0.0,
    e_move_total: float = 0.0,
) -> float:
    """
    Tổng năng lượng tiêu thụ toàn mạng tại vòng t  —  Eq. 27.

    E_total = E_a2r + E_r2r + E_r2g + Σ E_comp

    e_move_total is accepted for backward compatibility and logged by the
    simulator, but it is not included in the main objective/energy total yet.
    """
    return e_a2r + e_r2r + e_r2g + e_comp_total + e_svd_total
