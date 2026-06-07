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
    N_AUVS: int = 30
    M_RELAYS: int = 5           # Ghi đè runtime từ topo .pkl (base_simulator)
    M_RELAYS_1D: int = 5        # generate_all_envs (1D)
    M_RELAYS_2D: int = 5        # generate_all_envs (2D URPC, run_kdl_experiments.sh)
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
    E_INIT: float = 250000.0         # J — pin AUV (tương đương ~70 Wh, đủ cho ~60 vòng)
    RELAY_E_INIT: float = 200000.0   # J — pin Relay
    E_MIN: float = 5000.0            # J — ngưỡng dự trữ khẩn cấp (AUV)
    RELAY_E_MIN: float = 5000.0      # J — ngưỡng dự trữ khẩn cấp (Relay)
    EPSILON_OP: dict = field(default_factory=lambda: {"1D": 1.0e-28, "2D": 1.0e-28})
    F_CPU: float = 1.5e9             # Hz — CPU Max Freq. (Jetson Orin Nano Datasheet r4)
    N_CORES: int = 6                 # Số lõi (Jetson Orin Nano Datasheet r4: 6-core ARM Cortex-A78AE)
    FLOPS_PER_CYCLE: float = 4.0     # Số FLOPs/chu kỳ/lõi (ARM Cortex-A78AE NEON SIMD, ước tính)
    # T_comp = FLOPs / (F_CPU × N_CORES × FLOPS_PER_CYCLE)
    P_C_TX: float = 10.0             # W — mạch phát Acoustic Modem (VD: Evologics S2C)
    P_C_RX: float = 1.0              # W — mạch thu
    ETA_EA: float = 0.25             # Hiệu suất điện-âm


@dataclass
class FedKDLConfig:
    """Thuật toán FL / FedKDL — dùng bởi trainer, simulator, verify scripts."""

    # ── Vòng FL & local training ────────────────────────────────────────────
    GLOBAL_ROUNDS: dict = field(default_factory=lambda: {"1D": 50, "2D": 60})
    LOCAL_EPOCHS: int = 3
    LOCAL_BATCH_SIZE: int = 16
    LOCAL_LR: float = 5e-4
    LOCAL_HEAD_LR_MULT: float = 4.0   # FL Local SGD: Head LR = LOCAL_LR × multiplier
    LOCAL_LORA_LR_MULT: float = 1.0   # FL Local SGD: LoRA LR = LOCAL_LR × multiplier
    DATALOADER_WORKERS: int = 0      # trainer.py (LoRA/KD: giữ 0)
    CACHE_DATASET: bool = True       # trainer.py, main_trainer_od.py

    # ── FLOPs / năng lượng tính toán ────────────────────────────────────────
    MODEL_FLOPS_PER_SAMPLE: dict = field(
        default_factory=lambda: {"1D": 108000.0, "2D": 2.175e9}
    )
    FLOP_MULTIPLIER: dict = field(default_factory=lambda: {"1D": 3.0, "2D": 1.5})

    # ── LoRA + INT8 payload ───────────────────────────────────────────────────
    LORA_RANK: int = 8               # Phải khớp Teacher (rank=8); payload ~127KB INT8 @ adaptive
    QUANTIZATION_BITS: int = 8       # int8_quantization.py
    TARGET_PAYLOAD_KB: float = 300.0
    DELTA_SKIP: float = 0.01         # Lazy communication filter

    # ── HFL inter-relay (hfl_rules.should_cooperate) ────────────────────────
    COOP_THRESHOLD_MULTIPLIER: float = 0.75  # Eq. 41: c_m ≤ max(2, mult × c̄)

    # ── Knowledge-aware re-clustering (knowledge_association) ───────────────
    BETA_EMD: float = 0.0            # 0 = chỉ khoảng cách; 0.5 = EMD + địa lý

    # ── 1D anomaly detection only ───────────────────────────────────────────
    RHO_S: float = 0.05              # Top-K sparsity (AUVWorker1D)
    ANOMALY_EVAL_MODE: str = "best_f1"
    ANOMALY_PERCENTILE: float = 99.8

    # ── Gateway Knowledge Distillation ──────────────────────────────────────
    KD_ACTIVE: bool = True           # Bật/tắt Gateway KD (Teacher distills global model)
    KD_STU_LAMBDA: float = 0.50     # Trọng số Supervised Loss trong KD (0.5 = cân bằng GT/KD)
    KD_HEAD_LR_MULT: float = 8.0    # Head LR = LoRA LR × multiplier trong Gateway KD
    WARMUP_HEAD_LR_MULT: float = 2.5 # Warmup LoRA: Head LR = lr0 × multiplier
    WARMUP_LORA_LR_MULT: float = 0.5 # Warmup LoRA: LoRA LR = lr0 × multiplier
    CENTRAL_HEAD_LR_MULT: float = 2.5 # Centralized LoRA: Head LR = lr0 × multiplier
    CENTRAL_LORA_LR_MULT: float = 0.5 # Centralized LoRA: LoRA LR = lr0 × multiplier

    # ── Joint optimisation / latency budget (base_simulator logs) ───────────
    LAMBDA_E: float = 1e-3
    LAMBDA_TAU: float = 1e-3
    TAU_MAX: float = 1800.0          # s — giới hạn độ trễ vòng FL


network_cfg = NetworkConfig()
acoustic_cfg = AcousticChannelConfig()
energy_cfg = EnergyConfig()
fed_cfg = FedKDLConfig()
