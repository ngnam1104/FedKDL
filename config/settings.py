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
    M_RELAYS: int = 8           # Ghi đè runtime từ topo .pkl (base_simulator)
    M_RELAYS_1D: int = 8        # generate_all_envs (1D)
    M_RELAYS_2D: int = 8        # generate_all_envs (2D URPC, run_kdl_experiments.sh)
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
    E_INIT: float = float('inf')     # Bài toán P1 nới lỏng giới hạn (chạy đủ vòng)
    E_INIT_REF: float = 250000.0     # J — Ngưỡng dung lượng tham chiếu (Lớp 3)
    RELAY_E_INIT: float = float('inf')
    E_MIN: float = 5000.0            # J — ngưỡng dự trữ khẩn cấp (AUV)
    RELAY_E_MIN: float = 5000.0      # J — ngưỡng dự trữ khẩn cấp (Relay)
    EPSILON_OP: dict = field(default_factory=lambda: {"1D": 1.0e-28, "2D": 5.0e-30})
    F_CPU: float = 1.5e9             # Hz — CPU Max Freq. (Jetson Orin Nano Datasheet r4)
    N_CORES: int = 6                 # Số lõi (Jetson Orin Nano Datasheet r4: 6-core ARM Cortex-A78AE)
    FLOPS_PER_CYCLE: float = 4.0     # Số FLOPs/chu kỳ/lõi (ARM Cortex-A78AE NEON SIMD, ước tính)
    # T_comp = FLOPs / (F_CPU × N_CORES × FLOPS_PER_CYCLE)
    P_C_TX: float = 0.1              # W — mạch phát Acoustic Modem
    P_C_RX: float = 0.05             # W — mạch thu
    ETA_EA: float = 0.25             # Hiệu suất điện-âm


@dataclass
class FedKDLConfig:
    """Thuật toán FL / FedKDL — dùng bởi trainer, simulator, verify scripts."""

    # ── Vòng FL & local training ────────────────────────────────────────────
    GLOBAL_ROUNDS: dict = field(default_factory=lambda: {"1D": 50, "2D": 60})
    LOCAL_EPOCHS: int = 3
    LOCAL_BATCH_SIZE: int = 8        # Trả về 8 theo yêu cầu để giảm tải GPU/VRAM cho AUV
    LOCAL_LR: float = 8e-4
    LOCAL_HEAD_LR_MULT: float = 5.0   # Head LR
    LOCAL_LORA_LR_MULT: float = 1.0   # LoRA LR
    DATALOADER_WORKERS: int = 0      # trainer.py (LoRA/KD: giữ 0)
    LOCAL_DATALOADER_WORKERS: int = 0 # FL local YOLO dataloader workers (0 để tránh overhead spawn process chậm 20s)
    CACHE_DATASET: bool = True       # trainer.py, main_trainer_od.py
    LOCAL_CACHE_DATASET: bool = True # Bật cache RAM cho AUV để vượt qua nút thắt I/O mạng chậm của server
    LOCAL_AMP: bool = True           # Keep AMP on after lowering FL LR; set False if non-finite grads persist
    LOCAL_AUGMENT: bool = True       # Mild augmentation; mosaic/mixup remain disabled for small non-IID clients
    LOCAL_HSV_H: float = 0.01
    LOCAL_HSV_S: float = 0.30
    LOCAL_HSV_V: float = 0.20
    LOCAL_TRANSLATE: float = 0.05
    LOCAL_SCALE: float = 0.15
    LOCAL_FLIPLR: float = 0.50
    GRAD_DIAGNOSTICS: bool = False   # Expensive per-batch GPU sync; enable only when debugging NaN/Inf
    CLEAR_CUDA_CACHE_PER_AUV: bool = False
    LOG_ROUND_TOPOLOGY: bool = False
    LOG_TRAJECTORIES: bool = False
    PREWARM_YOLO_LABEL_CACHE: bool = False # Tắt tính năng tự động tạo cache song song lúc đầu để tránh rác màn hình

    # ── FLOPs / năng lượng tính toán ────────────────────────────────────────
    MODEL_FLOPS_PER_SAMPLE: dict = field(
        default_factory=lambda: {"1D": 108000.0, "2D": 2.175e9}
    )
    FLOP_MULTIPLIER: dict = field(default_factory=lambda: {"1D": 3.0, "2D": 1.5})

    # ── LoRA + INT8 payload ───────────────────────────────────────────────────
    LORA_RANK: int = 8               # Adaptive rank: backbone r=2, neck r=8
    MODEL_TOTAL_PARAMS_2D: int = 2_731_912
    LORA_TRAINABLE_PARAMS_2D: int = 356_312
    # Measured from pack_payload() for the current LoRA+Head+BN state:
    # INT8 tensors carry an 8-byte header; BN tensors remain FP32.
    LORA_INT8_PAYLOAD_BYTES_2D: int = 517_988
    QUANTIZATION_BITS: int = 8       # int8_quantization.py
    INT8_DELTA_PAYLOAD: bool = True  # Quantize updates instead of repeatedly quantizing absolute weights
    TARGET_PAYLOAD_KB: float = 300.0
    LAZY_FILTER_ENABLED: bool = False
    DELTA_SKIP: float = 0.01         # Lazy communication filter

    # ── HFL inter-relay (hfl_rules.should_cooperate) ────────────────────────
    COOP_THRESHOLD_MULTIPLIER: float = 0.75  # Eq. 41: c_m ≤ max(2, mult × c̄)
    COOP_NEIGHBOR_WEIGHT_NEAREST: float = 0.30
    COOP_NEIGHBOR_WEIGHT_SELECTIVE: float = 0.20

    # ── Knowledge-aware re-clustering (knowledge_association) ───────────────
    BETA_EMD: float = 0.0            # 0 = chỉ khoảng cách; 0.5 = EMD + địa lý

    # ── 1D anomaly detection only ───────────────────────────────────────────
    RHO_S: float = 0.05              # Top-K sparsity (AUVWorker1D)
    ANOMALY_EVAL_MODE: str = "best_f1"
    ANOMALY_PERCENTILE: float = 99.8
    FEDPROX_MU: float = 0.01
    LORA_R4_RANK: int = 4

    # ── Gateway Knowledge Distillation ──────────────────────────────────────
    KD_ACTIVE: bool = True           # Bật/tắt Gateway KD (Teacher distills global model)
    KD_STU_LAMBDA: float = 0.20     # Absolute supervised-loss scale during gateway KD
    KD_HEAD_LR_MULT: float = 4.0    # Gateway KD Head LR = 4e-3 when KD_LR=1e-3
    KD_LORA_LR_MULT: float = 1.0    # Gateway KD LoRA LR = KD_LR
    KD_EPOCHS: int = 1
    KD_BATCH_SIZE: int = 4
    KD_WORKERS: int = 0
    KD_AMP: bool = True
    KD_LR: float = 1e-3
    KD_TEMPERATURE: float = 4.0
    KD_LAMBDA_START: float = 1.00  # KD equals weighted supervised contribution at phase start
    KD_LAMBDA_FLOOR: float = 0.20  # KD remains meaningful before the pure-FL phase
    KD_BALANCE_BY_SUPERVISED: bool = True
    KD_BALANCE_SCALE_MIN: float = 0.001
    KD_BALANCE_SCALE_MAX: float = 20.0  # Allows weak KL/projection branches to reach their target share
    KD_CLS_WEIGHT: float = 0.45
    KD_BOX_WEIGHT: float = 0.35
    KD_PROJ_WEIGHT: float = 0.20
    KD_CONF_THRESHOLD: float = 0.10
    KD_CONF_GAMMA: float = 2.0
    KD_DFL_WEIGHT: float = 1.0
    KD_CIOU_WEIGHT: float = 0.5
    KD_PHASE1_END_FRAC: float = 0.5        # KD every round in phase 1
    KD_STOP_FRAC: float = 1.0        # Phase 2: every 2 rounds; then pure FL
    KD_ADAPTIVE_DROPOUT_ENABLED: bool = False
    KD_ADAPTIVE_DROP_THRESHOLD: int = 5
    LOCAL_KD_STU_LAMBDA: float = 0.20
    LOCAL_KD_LAMBDA: float = 1.0
    LOCAL_KD_HEAD_LR_MULT: float = 3.0
    PROXY_FT_EPOCHS: int = 1
    PROXY_FT_BATCH_SIZE: int = 4
    PROXY_FT_WORKERS: int = 0
    PROXY_FT_LR: float = 1e-3
    PROXY_FT_HEAD_LR_MULT: float = 4.0  # Match KD optimizer for a loss-only ablation
    PROXY_FT_LORA_LR_MULT: float = 1.0
    WARMUP_HEAD_LR_MULT: float = 2.5 # Warmup LoRA: Head LR = lr0 × multiplier
    WARMUP_LORA_LR_MULT: float = 0.5 # Warmup LoRA: LoRA LR = lr0 × multiplier
    STUDENT_WARMUP_EPOCHS: int = 5   # One shared warmup; non-LoRA baselines use its baked copy
    CENTRAL_HEAD_LR_MULT: float = 2.5 # Centralized LoRA: Head LR = lr0 × multiplier
    CENTRAL_LORA_LR_MULT: float = 0.5 # Centralized LoRA: LoRA LR = lr0 × multiplier

    # ── Joint optimisation / latency budget (base_simulator logs) ───────────
    LAMBDA_E: float = 0.005
    LAMBDA_TAU: float = 0.01
    TAU_MAX: float = float('inf')    # Bài toán P1 nới lỏng giới hạn
    TAU_MAX_REF: float = 1800.0      # s — Ngưỡng tham chiếu (Lớp 3)
    SERVER_MIX_BETA: float = 0.90    # FedKDL-only: old global 0.10 + new aggregate 0.90


network_cfg = NetworkConfig()
acoustic_cfg = AcousticChannelConfig()
energy_cfg = EnergyConfig()
fed_cfg = FedKDLConfig()
