"""
settings.py — Cấu hình toàn cục IoUT-FedKDL.

Import singleton:
    from config.settings import network_cfg, acoustic_cfg, energy_cfg, fed_cfg

Non-IID α và dataset partition: lấy từ file .pkl (generate_all_envs / run_kdl_experiments),
không hard-code trong fed_cfg.
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class NetworkConfig:
    """Topology — dùng bởi physics_models/topology.py, env_manager, generate_all_envs."""
    N_AUVS: int = 50
    M_RELAYS: int = 10           # Ghi đè runtime từ topo .pkl (base_simulator)
    M_RELAYS_1D: int = 10        # generate_all_envs (1D)
    M_RELAYS_2D: int = 10        # generate_all_envs (2D URPC, run_kdl_experiments.sh)
    AREA_X: float = 2000.0
    AREA_Y: float = 2000.0
    AUV_DEPTH: Tuple[float, float] = (500.0, 1000.0)
    RELAY_DEPTH: Tuple[float, float] = (100.0, 400.0)
    SURFACE_Z: float = 0.0


@dataclass
class AcousticChannelConfig:
    """Kênh sóng âm (Thorp-Wenz) — dùng bởi physics_models/topology.py."""
    SOUND_SPEED: float = 1500.0
    CARRIER_FREQ: float = 12.0       # kHz
    BANDWIDTH: float = 4000.0        # Hz
    TARGET_SNR: float = 10.0         # dB
    SL_MAX: float = 140.0            # dB re 1µPa @ 1m
    SPREADING_FACTOR: float = 1.5
    WIND_SPEED: float = 5.0          # m/s
    SHIPPING_FACTOR: float = 0.5
    IL_LOSS: float = 2.0             # dB


@dataclass
class EnergyConfig:
    """Năng lượng — dùng bởi simulator 1D/2D, base_simulator, main_trainer*."""
    E_INIT: float = 1500.0           # J — pin khởi tạo AUV
    E_MIN: float = 50.0              # J — ngưỡng dự trữ khẩn cấp
    EPSILON_OP: dict = field(default_factory=lambda: {"1D": 1.0e-11, "2D": 2.0e-12})
    F_CPU: float = 2.0e9             # Hz — CPU AUV (Gateway nhân ×5 trong main_trainer)
    P_C_TX: float = 0.05             # W — mạch phát (e_tx)
    P_C_RX: float = 0.03             # W — mạch thu (e_rx, chưa dùng trong FL loop)
    ETA_EA: float = 0.25             # Hiệu suất điện-âm


@dataclass
class FedKDLConfig:
    """Thuật toán FL / FedKDL — dùng bởi trainer, simulator, verify scripts."""

    # ── Vòng FL & local training ────────────────────────────────────────────
    GLOBAL_ROUNDS: dict = field(default_factory=lambda: {"1D": 50, "2D": 100})
    LOCAL_EPOCHS: int = 2
    LOCAL_BATCH_SIZE: int = 16
    LOCAL_LR: float = 2e-4
    DATALOADER_WORKERS: int = 0      # trainer.py (LoRA/KD: giữ 0)
    CACHE_DATASET: bool = True       # trainer.py, main_trainer_od.py

    # ── FLOPs / năng lượng tính toán ────────────────────────────────────────
    MODEL_FLOPS_PER_SAMPLE: dict = field(
        default_factory=lambda: {"1D": 108000.0, "2D": 2.175e9}
    )
    FLOP_MULTIPLIER: dict = field(default_factory=lambda: {"1D": 3.0, "2D": 1.2})

    # ── LoRA + INT8 payload ───────────────────────────────────────────────────
    LORA_RANK: int = 8               # FlexLoRA gửi A+B → ~146KB INT8 @ rank 8
    QUANTIZATION_BITS: int = 8       # int8_quantization.py
    TARGET_PAYLOAD_KB: float = 200.0
    DELTA_SKIP: float = 0.01         # Lazy communication filter

    # ── HFL inter-relay (hfl_rules.should_cooperate) ────────────────────────
    COOP_THRESHOLD_MULTIPLIER: float = 0.75  # Eq. 41: c_m ≤ max(2, mult × c̄)

    # ── Knowledge-aware re-clustering (knowledge_association) ───────────────
    BETA_EMD: float = 0.0            # 0 = chỉ khoảng cách; 0.5 = EMD + địa lý

    # ── 1D anomaly detection only ───────────────────────────────────────────
    RHO_S: float = 0.05              # Top-K sparsity (AUVWorker1D)
    ANOMALY_EVAL_MODE: str = "best_f1"
    ANOMALY_PERCENTILE: float = 99.8

    # ── Joint optimisation / latency budget (base_simulator logs) ───────────
    LAMBDA_E: float = 1e-3
    LAMBDA_TAU: float = 1e-3
    TAU_MAX: float = 1800.0          # s — giới hạn độ trễ vòng FL


network_cfg = NetworkConfig()
acoustic_cfg = AcousticChannelConfig()
energy_cfg = EnergyConfig()
fed_cfg = FedKDLConfig()
