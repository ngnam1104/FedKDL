"""
settings.py — Cấu hình toàn cục IoUT-FedKDL.

Import singleton:
    from config.settings import network_cfg, acoustic_cfg, energy_cfg, fed_cfg

Non-IID α và dataset partition: lấy từ file .pkl (generate_all_envs / run_kdl_experiments),
không hard-code trong fed_cfg.
"""
import os
from dataclasses import dataclass, field
from typing import Tuple


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


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
    MOBILITY_ENABLED: bool = _env_bool("FEDKDL_MOBILITY_ENABLED", True)
    MOBILITY_DT_PER_ROUND: float = _env_float("FEDKDL_MOBILITY_DT", 1.0)

    # ── Gauss-Markov AUV Mobility (Eq. 1a-1c in paper) ─────────────────────
    # Khớp chính xác ký hiệu trong paper:
    #   s[t+1] = μ*s[t] + (1-μ)*s̄ + sqrt(1-μ²)*ξ_s
    #   φ[t+1] = μ*φ[t] + (1-μ)*φ̄ + sqrt(1-μ²)*ξ_φ
    #   ψ[t+1] = μ*ψ[t] + (1-μ)*ψ̄ + sqrt(1-μ²)*ξ_ψ
    GM_ALPHA: float = 0.7           # μ_GM ∈ [0,1]: memory factor (0=random walk, 1=linear)
    GM_MEAN_SPEED: float = _env_float("FEDKDL_GM_MEAN_SPEED", 1.5)      # s̄ (m/s): tốc độ trung bình mục tiêu
    GM_MAX_SPEED: float = _env_float("FEDKDL_GM_MAX_SPEED", 5.0)       # Tốc độ tối đa: Δt·s ≤ 5m/round
    GM_MEAN_HEADING: float = 0.0    # φ̄ (rad): hướng ngang trung bình (0=East)
    GM_MEAN_PITCH: float = 0.0      # ψ̄ (rad): góc pitch trung bình (0=horizontal)
    GM_SIGMA_SPEED: float = _env_float("FEDKDL_GM_SIGMA_SPEED", 0.5)     # σ_s: độ lệch chuẩn nhiễu tốc độ
    GM_SIGMA_HEADING: float = _env_float("FEDKDL_GM_SIGMA_HEADING", 0.3) # σ_φ: độ lệch chuẩn nhiễu hướng (rad)
    GM_SIGMA_PITCH: float = _env_float("FEDKDL_GM_SIGMA_PITCH", 0.1)     # σ_ψ: độ lệch chuẩn nhiễu pitch (rad)


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
    EPSILON_OP: dict = field(default_factory=lambda: {"1D": 1.20e-28, "2D": 1.20e-28})
    F_CPU: float = 1.5e9             # Hz — CPU Max Freq. (Jetson Orin Nano Datasheet r4)
    N_CORES: int = 6                 # Số lõi (Jetson Orin Nano Datasheet r4: 6-core ARM Cortex-A78AE)
    FLOPS_PER_CYCLE: float = 4.0     # Số FLOPs/chu kỳ/lõi (ARM Cortex-A78AE NEON SIMD, ước tính)
    # T_comp = FLOPs / (F_CPU × N_CORES × FLOPS_PER_CYCLE)
    P_C_TX: float = 0.1              # W — mạch phát Acoustic Modem
    P_C_RX: float = 0.05             # W — mạch thu
    ETA_EA: float = 0.25             # Hiệu suất điện-âm
    MOVE_ENERGY_ENABLED: bool = _env_bool("FEDKDL_MOVE_ENERGY_ENABLED", False)
    AUV_WATER_DENSITY: float = 1025.0
    # Yang et al. CDC 2018 DROP-Sphere appendix: R=0.025m, Xu|u|=48.17 kg/m.
    AUV_THRUSTER_RADIUS: float = 0.025
    AUV_SURGE_DRAG_COEFF: float = 48.17
    AUV_HORIZONTAL_THRUSTERS: int = 2
    AUV_HOTEL_POWER: float = 0.0


@dataclass
class FedKDLConfig:
    """Thuật toán FL / FedKDL — dùng bởi trainer, simulator, verify scripts."""

    # ── Vòng FL & local training ────────────────────────────────────────────
    GLOBAL_ROUNDS: dict = field(default_factory=lambda: {"1D": 40, "2D": 40})
    NONIID_ALPHA: float = 1.0
    LOCAL_EPOCHS: int = 3
    LOCAL_BATCH_SIZE: int = 8        # Trả về 8 theo yêu cầu để giảm tải GPU/VRAM cho AUV
    LOCAL_LR: float = 1.0e-3
    LOCAL_HEAD_LR_MULT: float = 1.0
    LOCAL_LORA_LR_MULT: float = 1.0
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
    LOG_ROUND_TOPOLOGY: bool = True
    LOG_TRAJECTORIES: bool = False
    PREWARM_YOLO_LABEL_CACHE: bool = False # Tắt tính năng tự động tạo cache song song lúc đầu để tránh rác màn hình
    SCAFFOLD_OPTIMIZER: str = "AdamW" # "AdamW" for YOLO parity; set "SGD" for paper-like SCAFFOLD

    # ── FLOPs / năng lượng tính toán ────────────────────────────────────────
    MODEL_FLOPS_PER_SAMPLE: dict = field(
        default_factory=lambda: {"1D": 108000.0, "2D": 2.175e9}
    )
    FLOP_MULTIPLIER: dict = field(default_factory=lambda: {"1D": 3.0, "2D": 1.5})

    # ── LoRA + INT8 payload ───────────────────────────────────────────────────
    LORA_RANK: int = _env_int("FEDKDL_LORA_RANK", 8)
    LORA_BACKBONE_RANK: int = _env_int("FEDKDL_LORA_BACKBONE_RANK", 4)
    LORA_NECK_RANK: int = _env_int("FEDKDL_LORA_NECK_RANK", 8)
    LORA_ALPHA: float = _env_float("FEDKDL_LORA_ALPHA", 8.0)
    LORA_STRATEGY: str = os.getenv("FEDKDL_LORA_STRATEGY", "adaptive")  # backbone rank 4, neck rank 8
    LORA_TARGETS: tuple = ("Conv",)  # fixed target set for student and LoRA teacher pretraining
    MODEL_TOTAL_PARAMS_2D: int = 2_731_912
    LORA_TRAINABLE_PARAMS_2D: int = 356_312
    # Measured from pack_payload() for the current LoRA+Head+BN state:
    # INT8 tensors carry an 8-byte header; BN tensors remain FP32.
    LORA_INT8_PAYLOAD_BYTES_2D: int = 517_988
    QUANTIZATION_BITS: int = 8       # int8_quantization.py
    INT8_DELTA_PAYLOAD: bool = True  # Quantize updates instead of repeatedly quantizing absolute weights
    INT8_CLIP_PERCENTILE: float = 99.9  # Robust INT8 scale; 100 disables clipping
    INT8_CLIP_MIN_NUMEL: int = 256      # Avoid clipping tiny tensors where extrema are meaningful
    TARGET_PAYLOAD_KB: float = 500.0
    RESET_LORA_OPTIMIZER_STATE: bool = True  # Aggregation changes LoRA basis; keep Head/BN moments only
    LAZY_FILTER_ENABLED: bool = False
    DELTA_SKIP: float = 0.01         # Lazy communication filter

    # ── HFL inter-relay (hfl_rules.should_cooperate) ────────────────────────
    COOP_THRESHOLD_MULTIPLIER: float = 0.75  # Eq. 41: c_m ≤ max(2, mult × c̄)
    COOP_NEIGHBOR_WEIGHT_NEAREST: float = 0.30
    COOP_NEIGHBOR_WEIGHT_SELECTIVE: float = 0.20

    # ── Knowledge-aware re-clustering (knowledge_association) ───────────────
    BETA_EMD: float = 0.0            # 0 = chỉ khoảng cách; 0.5 = EMD + địa lý

    # ── 1D anomaly detection only ───────────────────────────────────────────
    RHO_S: float = 0.05              # Top-K sparsity
    ANOMALY_EVAL_MODE: str = "best_f1"
    ANOMALY_PERCENTILE: float = 99.8
    FEDPROX_MU: float = 0.02
    LORA_R4_RANK: int = _env_int("FEDKDL_LORA_R4_RANK", 4)

    # ── Gateway Knowledge Distillation ──────────────────────────────────────
    KD_ACTIVE: bool = True           # Bật/tắt Gateway KD (Teacher distills global model)
    TEACHER_CKPT: str = "teacher_lora_best.pt"  # YOLO12l LoRA teacher aligned with projection KD
    KD_OPTIMIZER: str = "AdamW"
    KD_STU_LAMBDA: float = 0.70
    KD_HEAD_LR_MULT: float = 1.0
    KD_LORA_LR_MULT: float = 1.0
    KD_EPOCHS: int = 1
    KD_BATCH_SIZE: int = 4
    KD_WORKERS: int = 0
    KD_AMP: bool = True
    KD_LR: float = 2.5e-4           # All-or-nothing KD: keep the accepted step conservative
    KD_LRF: float = 1.0             # LR final fraction: 1.0 = flat LR (no cosine decay)
    KD_WARMUP_EPOCHS: float = 0.1   # Warmup 10% to prevent AdamW cold-start shock
    KD_TEMPERATURE: float = 4.0
    KD_LAMBDA: float = 0.30
    KD_LAMBDA_START: float = 0.30
    KD_LAMBDA_FLOOR: float = 0.30
    KD_BALANCE_BY_SUPERVISED: bool = False  # Disabled: old double-scaling caused harm
    KD_BALANCE_SCALE_MIN: float = 0.001
    KD_BALANCE_SCALE_MAX: float = 4.0
    KD_CLS_WEIGHT: float = 0.50
    KD_BOX_WEIGHT: float = 0.50
    KD_PROJ_WEIGHT: float = _env_float("FEDKDL_KD_PROJ_WEIGHT", 0.0)  # Default simple KD; LoRA-proj stays optional
    KD_PROJ_MODE: str = "lora_spatial_proj"  # Safer default: rank-invariant spatial attention over h=A*x
    KD_PROJ_ANCHOR_MATCH: bool = True        # Match first/last LoRA op inside each YOLO block exactly
    KD_CONF_THRESHOLD: float = 0.20
    KD_CONF_GAMMA: float = 2.0
    KD_DFL_WEIGHT: float = 1.0
    KD_CIOU_WEIGHT: float = 0.5
    KD_PHASE1_END_FRAC: float = 0.35        # KD only in early/mid rounds, then pure FL
    KD_STOP_FRAC: float = 0.50              # Continue KD into mid rounds, then let pure FL polish
    KD_ADAPTIVE_DROPOUT_ENABLED: bool = True # Safety net: auto-stop if KD still harms
    KD_ADAPTIVE_DROP_THRESHOLD: int = 2
    KD_ACCEPTANCE_GATE: bool = True
    KD_WISE_ALPHA_CANDIDATES: tuple = (1.00,)  # No KD weight mixing: accept full KD step or rollback
    KD_ACCEPT_TOL: float = 0.0
    KD_MIN_MAP5095_DELTA: float = -5e-4
    KD_MIN_MAP50_DELTA: float = -5e-4
    KD_MIN_REC_DELTA: float = -1e-3
    BOX_LOSS_WEIGHT: float = 7.5
    CLS_LOSS_WEIGHT: float = 0.5
    DFL_LOSS_WEIGHT: float = 1.5
    LOCAL_KD_STU_LAMBDA: float = 0.70
    LOCAL_KD_LAMBDA: float = 0.30
    LOCAL_KD_HEAD_LR_MULT: float = 1.0
    PROXY_FT_EPOCHS: int = 2
    PROXY_FT_BATCH_SIZE: int = 4
    PROXY_FT_WORKERS: int = 0
    PROXY_FT_OPTIMIZER: str = "AdamW"
    PROXY_FT_REUSE_OPTIMIZER: bool = False
    PROXY_FT_LR: float = 2e-4
    PROXY_FT_LRF_GLOBAL: float = 0.35
    PROXY_FT_LRF: float = 1.0       # LR final fraction: 1.0 = flat LR (no cosine decay)
    PROXY_FT_WARMUP_EPOCHS: float = 0.0 # Proxy FT is a one-epoch nudge; avoid warmup shock/extra bias LR
    # [WiSE-FT] Weight-Space Ensemble: w_final = alpha * w_post + (1-alpha) * w_pre
    # Hard-label proxy data can over-specialize the detector, so default to a
    # small weight-space nudge and let the acceptance gate reject harmful steps.
    PROXY_FT_WISE_ALPHA: float = 1.0
    PROXY_FT_WISE_ALPHA_MIN: float = 1.0
    PROXY_FT_WISE_ALPHA_CANDIDATES: tuple = (1.0,)
    PROXY_FT_BLEND_REFERENCE: str = "pure_aggregated"  # "pre_gateway" or "pure_aggregated"
    PROXY_FT_ACCEPTANCE_GATE: bool = True
    PROXY_FT_ACCEPT_TOL: float = 0.0
    PROXY_FT_MIN_MAP5095_DELTA: float = 0.0
    PROXY_FT_MIN_MAP50_DELTA: float = -5e-4
    PROXY_FT_MIN_REC_DELTA: float = -5e-4
    PROXY_FT_HEAD_LR_MULT: float = 1.5
    PROXY_FT_LORA_LR_MULT: float = 1.0
    WARMUP_HEAD_LR_MULT: float = 2.5 # Warmup LoRA: Head LR = lr0 × multiplier
    WARMUP_LORA_LR_MULT: float = 0.5 # Warmup LoRA: LoRA LR = lr0 × multiplier
    STUDENT_WARMUP_EPOCHS: int = _env_int("FEDKDL_STUDENT_WARMUP_EPOCHS", 2)   # One shared warmup; non-LoRA baselines use its baked copy
    STUDENT_WARMUP_SUFFIX: str = os.getenv("FEDKDL_WARMUP_SUFFIX", "")
    CENTRAL_HEAD_LR_MULT: float = 2.5 # Centralized LoRA: Head LR = lr0 × multiplier
    CENTRAL_LORA_LR_MULT: float = 0.5 # Centralized LoRA: LoRA LR = lr0 × multiplier

    # ── Joint optimisation / latency budget (base_simulator logs) ───────────
    LAMBDA_E: float = 0.0005
    LAMBDA_TAU: float = 0.001
    TAU_MAX: float = float('inf')    # Bài toán P1 nới lỏng giới hạn
    TAU_MAX_REF: float = 1800.0      # s — Ngưỡng tham chiếu (Lớp 3)
    SERVER_MIX_BETA: float = 0.90    # FedKDL-only: old global 0.10 + new aggregate 0.90


network_cfg = NetworkConfig()
acoustic_cfg = AcousticChannelConfig()
energy_cfg = EnergyConfig()
fed_cfg = FedKDLConfig()
