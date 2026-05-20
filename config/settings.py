"""
settings.py
Cấu hình tham số toàn cục cho hệ thống IoUT-FedKDL.
Tích hợp tham số vật lý từ Omeke et al. 2026 và kiến trúc phân cấp của FedKDL.
"""
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class NetworkConfig:
    """Cấu hình Topology và quy mô bầy đàn"""
    N_SENSORS: int = 100         # Số lượng AUV cảm biến (Tầng Deep)
    M_FOGS: int = 10             # Số lượng Trạm Fog trung gian
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
    SL_MAX: float = 140.0             # Công suất phát tối đa phần cứng (dB re 1µPa @ 1m) 
    SPREADING_FACTOR: float = 1.5     # Hệ số lan truyền thực tế (k) 
    WIND_SPEED: float = 5.0           # Tốc độ gió chuẩn hóa (m/s) 
    SHIPPING_FACTOR: float = 0.5      # Hệ số hoạt động tàu bè (0 - 1) 
    IL_LOSS: float = 2.0              # Tổn hao triển khai (dB) 

@dataclass
class EnergyConfig:
    """Ngân sách Sinh tồn và Tiêu hao Năng lượng"""
    E_INIT: float = 500.0             # Pin khởi tạo của mỗi thiết bị (Joules) 
    E_COMP_EPOCH: float = 0.5         # Tiêu hao điện toán nội bộ (J/epoch - ước tính cho ARM)
    P_C_TX: float = 0.05              # Công suất tĩnh mạch phát vô tuyến (Watts) 
    P_C_RX: float = 0.03              # Công suất tĩnh mạch thu (Watts) 
    ETA_EA: float = 0.25              # Hiệu suất chuyển đổi điện-âm 

@dataclass
class FedKDLConfig:
    """Tham số Thuật toán Học liên kết & Đề xuất FedKDL"""
    # Baseline Parameters
    GLOBAL_ROUNDS: int = 150          # Chu kỳ sống dự kiến cho ảnh 2D
    LOCAL_EPOCHS: int = 5             # Số vòng lặp SGD cục bộ 
    LOCAL_LR: float = 0.01            # Learning rate (Thêm vào cho Centralised & Worker)
    NON_IID_ALPHA: float = 0.1        # Phân phối Dirichlet cho Concept Drift/Data Skew
    DATASET_2D: str = "URPC_2020"     # Kịch bản 2 & 3
    DATASETS_1D: List[str] = None
    
    # HFL-Selective Parameters
    # Cụm chỉ hợp tác nếu size <= max(2, 0.75 * mean_size)
    COOP_THRESHOLD_MULTIPLIER: float = 0.75 
    
    # KD-LoRA-INT8 Parameters
    LORA_RANK: int = 4                # Cấu trúc Low-rank cho 2 C2f blocks cuối
    QUANTIZATION_BITS: int = 8        # Affine Quantization từng tensor riêng biệt
    TARGET_PAYLOAD_KB: float = 11.0   # S ≈ 11 KB (Khóa cứng kích thước gói tin)
    
    # Deterministic Rules Thresholds
    BETA_EMD: float = 0.5             # Trọng số lai D_joint giữa Tri thức (EMD) và Địa lý
    EPSILON_DRIFT: float = 0.05       # Ngưỡng dung sai báo động Concept Drift
    DELTA_SKIP: float = 0.01          # Ngưỡng kích hoạt Bộ lọc truyền thông lười (Lazy Filter)

    def __post_init__(self):
        self.DATASETS_1D = ["SMD", "SMAP", "MSL"]

# Khởi tạo Singleton để import vào các module khác
network_cfg = NetworkConfig()
acoustic_cfg = AcousticChannelConfig()
energy_cfg = EnergyConfig()
fed_cfg = FedKDLConfig()
