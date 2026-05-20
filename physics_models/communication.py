"""
communication.py
Mô hình Kênh truyền Sóng âm Dưới nước (Underwater Acoustic Channel Model).
Triển khai các phương trình Thorp-Wenz từ Omeke et al. 2026 và Research Proposal Eq. 1-7.

References:
    - Thorp absorption model (Eq. 2)
    - Wenz ambient noise model (Eq. 3)
    - Passive Sonar equation (Eq. 5)
    - Shannon capacity (Eq. 4)
    - Minimum Source Level for feasibility (Eq. 6)
"""

import numpy as np
from typing import Tuple


# ──────────────────────────────────────────────────────────────────────
#  1. Thorp Absorption Coefficient  (Eq. 2)
# ──────────────────────────────────────────────────────────────────────

def thorp_absorption(f_khz: float) -> float:
    """
    Hệ số hấp thụ Thorp α(f) theo tần số sóng mang.

    Args:
        f_khz: Tần số sóng mang (kHz).

    Returns:
        α(f) in dB/km.
    """
    f2 = f_khz ** 2
    alpha = (0.11 * f2 / (1.0 + f2)
             + 44.0 * f2 / (4100.0 + f2)
             + 2.75e-4 * f2
             + 0.003)
    return alpha


# ──────────────────────────────────────────────────────────────────────
#  2. Transmission Loss  (Eq. 1)
# ──────────────────────────────────────────────────────────────────────

def transmission_loss(d_m: float, f_khz: float, k: float = 1.5) -> float:
    """
    Suy hao truyền dẫn TL(d, f) kết hợp lan truyền hình học + hấp thụ nhiệt.

    Args:
        d_m:    Khoảng cách Euclidean 3D giữa phát và thu (m). Phải > 0.
        f_khz:  Tần số sóng mang (kHz).
        k:      Hệ số lan truyền thực tế (1.0 = cylindrical, 2.0 = spherical).

    Returns:
        TL in dB.
    """
    if d_m <= 0:
        return 0.0
    alpha = thorp_absorption(f_khz)
    tl = 10.0 * k * np.log10(d_m) + alpha * d_m / 1000.0
    return tl


# ──────────────────────────────────────────────────────────────────────
#  3. Wenz Ambient Noise Model  (Eq. 3)
# ──────────────────────────────────────────────────────────────────────

def wenz_noise_components(f_khz: float,
                          wind_speed: float = 5.0,
                          shipping_factor: float = 0.5) -> dict:
    """
    Tính 4 thành phần nhiễu phổ mật độ Wenz (dB re 1µPa per Hz).

    Args:
        f_khz:           Tần số (kHz).
        wind_speed:      Vận tốc gió (m/s).
        shipping_factor: Hệ số hoạt động tàu bè s ∈ [0, 1].

    Returns:
        Dict chứa N_turb, N_ship, N_wind, N_thermal (dB re 1µPa/Hz).
    """
    f = f_khz  # kHz

    # Nhiễu nhiễu loạn (Turbulence)
    N_turb = 17.0 - 30.0 * np.log10(f)

    # Nhiễu tàu bè (Shipping)
    N_ship = (40.0 + 20.0 * (shipping_factor - 0.5)
              + 26.0 * np.log10(f)
              - 60.0 * np.log10(f + 0.03))

    # Nhiễu gió (Wind)
    N_wind = (50.0 + 7.5 * np.sqrt(wind_speed)
              + 20.0 * np.log10(f)
              - 40.0 * np.log10(f + 0.4))

    # Nhiễu nhiệt (Thermal)
    N_thermal = -15.0 + 20.0 * np.log10(f)

    return {
        'turbulence': N_turb,
        'shipping': N_ship,
        'wind': N_wind,
        'thermal': N_thermal,
    }


def wenz_noise_level(f_khz: float, B_hz: float,
                     wind_speed: float = 5.0,
                     shipping_factor: float = 0.5) -> float:
    """
    Mức nhiễu nền tổng hợp NL(f, B) trong băng thông B (Eq. 3).

    Args:
        f_khz:           Tần số sóng mang (kHz).
        B_hz:            Băng thông máy thu (Hz).
        wind_speed:      Vận tốc gió (m/s).
        shipping_factor: Hệ số hoạt động tàu bè s ∈ [0, 1].

    Returns:
        NL in dB re 1µPa.
    """
    components = wenz_noise_components(f_khz, wind_speed, shipping_factor)

    # Tổng hợp công suất tuyến tính (dB → linear → sum → dB)
    total_linear = sum(10.0 ** (n / 10.0) for n in components.values())
    NL = 10.0 * np.log10(total_linear) + 10.0 * np.log10(B_hz)
    return NL


# ──────────────────────────────────────────────────────────────────────
#  4. Passive Sonar Equation — SNR  (Eq. 5)
# ──────────────────────────────────────────────────────────────────────

def snr_passive(SL: float, TL: float, NL: float, IL: float = 2.0) -> float:
    """
    SNR thực tế trên liên kết (u, v) theo phương trình Sonar thụ động.

    Args:
        SL: Source Level (dB re 1µPa @ 1m).
        TL: Transmission Loss (dB).
        NL: Noise Level (dB re 1µPa).
        IL: Implementation Loss — tổn hao triển khai phần cứng (dB).

    Returns:
        SNR in dB.
    """
    return SL - TL - NL - IL


# ──────────────────────────────────────────────────────────────────────
#  5. Shannon Capacity  (Eq. 4)
# ──────────────────────────────────────────────────────────────────────

def shannon_capacity(B_hz: float, snr_target_dB: float) -> float:
    """
    Tốc độ Shannon tĩnh trên liên kết (bps).

    Args:
        B_hz:          Băng thông (Hz).
        snr_target_dB: SNR mục tiêu vận hành (dB).

    Returns:
        R in bps.
    """
    snr_linear = 10.0 ** (snr_target_dB / 10.0)
    R = B_hz * np.log2(1.0 + snr_linear)
    return R


# ──────────────────────────────────────────────────────────────────────
#  6. Minimum Source Level for Feasibility  (Eq. 6)
# ──────────────────────────────────────────────────────────────────────

def min_source_level(d_m: float, f_khz: float, B_hz: float,
                     snr_target_dB: float, IL: float = 2.0,
                     k: float = 1.5,
                     wind_speed: float = 5.0,
                     shipping_factor: float = 0.5) -> float:
    """
    Mức nguồn phát tối thiểu SL_min để đạt SNR mục tiêu trên liên kết (u, v).

    Args:
        d_m:            Khoảng cách 3D (m).
        f_khz:          Tần số sóng mang (kHz).
        B_hz:           Băng thông (Hz).
        snr_target_dB:  Ngưỡng SNR (dB).
        IL:             Implementation Loss (dB).
        k:              Hệ số lan truyền.
        wind_speed:     Vận tốc gió (m/s).
        shipping_factor: Hệ số tàu bè.

    Returns:
        SL_min in dB re 1µPa @ 1m.
    """
    TL = transmission_loss(d_m, f_khz, k)
    NL = wenz_noise_level(f_khz, B_hz, wind_speed, shipping_factor)
    SL_min = snr_target_dB + TL + NL + IL
    return SL_min


# ──────────────────────────────────────────────────────────────────────
#  7. Feasibility Check  (Eq. 7)
# ──────────────────────────────────────────────────────────────────────

def is_link_feasible(d_m: float, f_khz: float, B_hz: float,
                     snr_target_dB: float, SL_max: float,
                     IL: float = 2.0, k: float = 1.5,
                     wind_speed: float = 5.0,
                     shipping_factor: float = 0.5) -> Tuple[bool, float]:
    """
    Kiểm tra khả thi liên kết: SL_min ≤ SL_max?

    Returns:
        (feasible: bool, SL_min: float)
    """
    SL_min = min_source_level(d_m, f_khz, B_hz, snr_target_dB,
                              IL, k, wind_speed, shipping_factor)
    return SL_min <= SL_max, SL_min
