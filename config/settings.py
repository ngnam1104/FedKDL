"""
settings.py
Cấu hình tham số toàn cục cho hệ thống IoUT-FedKDL.
Tích hợp tham số vật lý từ Omeke et al. 2026 và kiến trúc phân cấp của FedKDL.
"""
from dataclasses import dataclass, field
from typing import Tuple, List

@dataclass
class NetworkConfig:
    """Cấu hình Topology và quy mô bầy đàn"""
    N_SENSORS: int = 100         # Số lượng AUV cảm biến (Tầng Deep)
    M_FOGS: int = 10             # Số lượng Trạm Fog trung gian (hiện tại - có thể bị override)
    M_FOGS_1D: int = 10          # Số lượng Trạm Fog trung gian cho tác vụ 1D
    M_FOGS_2D: int = 4           # Số lượng Trạm Fog trung gian cho tác vụ 2D
    AREA_X: float = 2000.0       # Không gian X (m)
    AREA_Y: float = 2000.0       # Không gian Y (m)
    MAX_DEPTH: float = 1000.0    # Độ sâu tối đa (m)
    SENSOR_DEPTH: Tuple[float, float] = (500.0, 1000.0)
    FOG_DEPTH: Tuple[float, float] = (100.0, 400.0)
    SURFACE_Z: float = 0.0

@dataclass
class AcousticChannelConfig:
    """Tham số Vật lý Kênh truyền Sóng âm (Thorp-Wenz Model)"""
    SOUND_SPEED: float = 1500.0       # Vận tốc âm thanh dưới nước (m/s) 
    CARRIER_FREQ: float = 12.0        # Tần số sóng mang (kHz) 
    BANDWIDTH: float = 4000.0         # Băng thông máy thu (Hz) -> ~15 kbps 
    TARGET_SNR: float = 10.0          # Ngưỡng SNR vận hành (dB) 
    SL_MAX: float = 140.0             # Công suất phát tối đa phần cứng (dB re 1μPa @ 1m) 
    SPREADING_FACTOR: float = 1.5     # Hệ số lan truyền thực tế (k) 
    WIND_SPEED: float = 5.0           # Tốc độ gió chuẩn hóa (m/s) 
    SHIPPING_FACTOR: float = 0.5      # Hệ số hoạt động tàu bè (0 - 1) 
    IL_LOSS: float = 2.0              # Tổn hao triển khai (dB) 

@dataclass
class EnergyConfig:
    """Ngân sách Sinh tồn và Tiêu hao Năng lượng"""
    E_INIT: float = 2500.0            # Nâng lên 2500 Joules để gánh được Payload của Rank 12 trong 100 vòng
    E_MIN: float = 50.0               # Ngưỡng pin dự trữ khẩn cấp để ngoi lên mặt nước (Joules)
    EPSILON_OP: dict = field(default_factory=lambda: {"1D": 1.0e-11, "2D": 2.0e-12}) # Tiêu hao năng lượng trên mỗi FLOP (1D: FP32, 2D: INT8)
    F_CPU: float = 2.0e9              # Tần số CPU của AUV (Cycles/s hoặc FLOPs/s), ví dụ 2 GHz
    P_C_TX: float = 0.05              # Công suất tĩnh mạch phát vô tuyến (Watts) 
    P_C_RX: float = 0.03              # Công suất tĩnh mạch thu (Watts) 
    ETA_EA: float = 0.25              # Hiệu suất chuyển đổi điện-âm 


@dataclass
class FedKDLConfig:
    """Tham số Thuật toán Học liên kết & Đề xuất FedKDL"""
    # Baseline Parameters
    GLOBAL_ROUNDS: dict = field(default_factory=lambda: {"1D": 50, "2D": 100}) # Chu kỳ sống dự kiến cho từng tác vụ
    MODEL_FLOPS_PER_SAMPLE: dict = field(default_factory=lambda: {"1D": 108000.0, "2D": 2.175e9}) # 1D: Autoencoder ~54k params | 2D: YOLOv8n ở 320x320
    FLOP_MULTIPLIER: dict = field(default_factory=lambda: {"1D": 3.0, "2D": 1.2}) # Hệ số nhân: 1D (Full fine-tuning), 2D (LoRA)
    LOCAL_EPOCHS: int = 2             # Giảm xuống 2 để giảm Client Drift
    LOCAL_BATCH_SIZE: int = 16        # Trả về 16 vì batch 64 làm training bị nghẽn (2.9s/it)
    DATALOADER_WORKERS: int = 0       # Giữ 0 để an toàn tuyệt đối cho logic LoRA/KD
    CACHE_DATASET: bool = True        # Đưa toàn bộ dataset vào RAM để tăng tốc thay vì dùng đa luồng
    LOCAL_LR: float = 0.002           # Giảm xuống 0.002 để chống Client Drift / Overfitting cục bộ
    NON_IID_ALPHA: float = 0.1        # Phân phối Dirichlet cho Concept Drift/Data Skew
    DATASET_2D: str = "URPC_2020"     # Kịch bản 2 & 3
    DATASETS_1D: List[str] = None
    
    # Sensor sparsity ratio
    RHO_S: float = 0.05
    
    # HFL-Selective Parameters
    # Cụm chỉ hợp tác nếu size <= max(2, 0.75 * mean_size)
    COOP_THRESHOLD_MULTIPLIER: float = 0.75 
    
    # KD-LoRA-INT8 Parameters
    # Kịch bản 1: LORA_RANK=4 → payload ~74KB  (LoRA 72KB + Head partial 2KB)
    # Kịch bản 2: LORA_RANK=8 → payload ~146KB (LoRA 144KB + Head partial 2KB) ≈ 150KB target
    # Kịch bản 3: LORA_RANK=12 → payload ~196KB -> tau_round ~1650s (<1800s limit)
    LORA_RANK: int = 12               # Nâng lên 12 để não to hơn
    QUANTIZATION_BITS: int = 8        # Affine Quantization từng tensor riêng biệt (INT8)
    TARGET_PAYLOAD_KB: float = 200.0  # Target payload: 200KB (LoRA+Head partial INT8)
    
    # Deterministic Rules Thresholds
    BETA_EMD: float = 0.5             # Trọng số lai D_joint giữa Tri thức (EMD) và Địa lý
    EPSILON_DRIFT: float = 0.05       # Ngưỡng dung sai báo động Concept Drift
    DELTA_SKIP: float = 0.01          # Ngưỡng kích hoạt Bộ lọc truyền thông lười (Lazy Filter)
    
    # Anomaly Evaluation Parameters
    ANOMALY_EVAL_MODE: str = "best_f1"     # Options: "percentile", "best_f1"
    ANOMALY_PERCENTILE: float = 99.8       # Ngưỡng cắt phân vị nếu dùng ANOMALY_EVAL_MODE = "percentile"

    # Joint Optimisation Cost Coefficients  ──  Eq. 22 trong bài báo
    # min  F(θ^T) + λ_E · Σ E_round^t  +  λ_τ · Σ τ_round^t
    #
    # Đơn vị thô rất lệch nhau (Joules vs. Giây vs. dimensionless loss).
    # Các hệ số λ dưới đây được chọn để chuẩn hóa tương đối:
    #   λ_E  ×  300 J  ≈ 0.3  (cùng bậc với loss 1D ~0.07, loss 2D ~8)
    #   λ_τ  ×  300 s  ≈ 0.3  (tương tự trên)
    # Người dùng có thể ghi đè khi cần thực nghiệm sensitivity analysis.
    LAMBDA_E: float = 1e-3   # Trọng số năng lượng  (J⁻¹ — "per Joule cost")
    LAMBDA_TAU: float = 1e-3 # Trọng số độ trễ     (s⁻¹ — "per second cost")
    
    # System Constraints (từ bài báo gốc Omeke 2026)
    TAU_MAX: float = 1800.0   # Trễ tối đa cho phép 1 vòng FL (30 phút)

    def __post_init__(self):
        self.DATASETS_1D = ["SMD", "SMAP", "MSL"]

# Khởi tạo Singleton để import vào các module khác
network_cfg = NetworkConfig()
acoustic_cfg = AcousticChannelConfig()
energy_cfg = EnergyConfig()
fed_cfg = FedKDLConfig()
