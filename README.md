# FedKDL

Mô phỏng **Federated Learning** trên mạng **Internet of Underwater Things (IoUT)** — hai kịch bản:

| Kịch bản | Mô tả | Entrypoint |
|----------|--------|------------|
| **1D — HFL** | Phát hiện bất thường (Autoencoder), Top-K + INT8 | `main_trainer.py` |
| **2D — FedKDL** | Phát hiện vật thể (YOLOv26n + LoRA + INT8), Gateway-side KD | `main_trainer_od.py` |

Kiến trúc phân cấp 3 tầng: **Sensor → Fog → Gateway**. Mô hình vật lý kênh âm dưới nước sử dụng Thorp-Wenz, vùng mô phỏng 2000 × 2000 × 1000 m.

---

## Yêu cầu

- Python 3.10+
- Linux server khuyến nghị cho train dài hơi
- **HFL (1D):** CPU đủ dùng (Autoencoder ~54k params)
- **KDL (2D):** GPU + CUDA (YOLOv26n + LoRA rank-8)
- **Dataset URPC2020:** cần Kaggle API token (xem bên dưới)

---

## Quick Start (Server Linux)

```bash
git clone <repo-url> FedKDL && cd FedKDL

# Đặt Kaggle token — bắt buộc nếu tải URPC qua script
export KAGGLE_API_TOKEN="your_token_from_kaggle_settings"

chmod +x quick_start.sh run_hfl_experiments.sh run_kdl_experiments.sh
./quick_start.sh
```

`quick_start.sh` thực hiện tuần tự: tạo venv → `pip install` → tải dataset → sinh môi trường → chạy toàn bộ thực nghiệm HFL và KDL.

Chạy trong **tmux** để tránh mất session khi SSH đứt:

```bash
tmux new -s fedkdl
./quick_start.sh
```

### Tùy chọn `quick_start.sh`

```bash
./quick_start.sh --help
./quick_start.sh --setup-only      # chỉ tạo venv + cài thư viện
./quick_start.sh --train-only      # bỏ qua tải data và sinh môi trường
./quick_start.sh --hfl-only        # chỉ chạy kịch bản 1D (CPU)
./quick_start.sh --kdl-only        # chỉ chạy kịch bản 2D (GPU)
```

---

## Kaggle API Token (URPC2020 / SMAP)

1. Vào [Kaggle Settings → API](https://www.kaggle.com/settings) → **Generate New Token**.
2. Trên server, **một trong hai cách** (không đưa token vào git):

```bash
# Cách 1 — export trong shell / tmux
export KAGGLE_API_TOKEN=KGAT_xxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Cách 2 — file local (đã khai báo trong .gitignore)
cp .env.example .env
# Sửa .env: KAGGLE_API_TOKEN=...
```

3. Kiểm tra tải dataset URPC:

```bash
source .venv/bin/activate
python utils/download_datasets.py --urpc
```

> **Bảo mật:** Không commit `.env`, không ghi token vào README hay mã nguồn. Nếu token đã lộ, hãy **thu hồi và tạo token mới** trên Kaggle.

---

## Chạy thủ công từng bước

```bash
# 1. Môi trường Python
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Tải dataset (cần KAGGLE_API_TOKEN)
export KAGGLE_API_TOKEN=...
python utils/download_datasets.py --all

# 3. Sinh Topology và Data Partition
python utils/generate_all_envs.py              # 1D (SMD, SMAP, MSL)
python utils/generate_all_envs.py --dataset URPC  # 2D (URPC2020)

# 4. Pre-train Teacher model (chỉ cần làm 1 lần cho 2D)
python scripts/fedkdl/pretrain.py

# 5. Chạy thực nghiệm
./run_hfl_experiments.sh   # ~432 run, 1D, CPU
./run_kdl_experiments.sh   # ~37 run,  2D, GPU
```

---

## Cấu hình thực nghiệm

Chỉnh sửa các biến đầu tập lệnh, không cần sửa code Python:

### `run_kdl_experiments.sh` (2D)

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `ROUNDS` | `1` (test) / `60` (full) | Số vòng liên kết toàn cầu |
| `M_FOGS_2D` | `5` | Số Fog node cho kịch bản 2D |
| `DS` | `URPC` | Tên dataset (hiện tại chỉ URPC) |
| `SEED` | `42` | Seed ngẫu nhiên |

Ví dụ chạy trên server với ít fog hơn:
```bash
ROUNDS=60 M_FOGS_2D=4 ./run_kdl_experiments.sh
```

### `run_hfl_experiments.sh` (1D)

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `ROUNDS` | `30` | Số vòng liên kết |
| `M_FOGS_1D` | `10` | Số Fog node cho kịch bản 1D |
| `N_LIST` | `50 100 150 200` | Danh sách số lượng sensor |
| `DATASETS` | `SMD SMAP MSL` | Các dataset 1D |
| `ALPHAS` | `1.0 10000.0` | Mức độ non-IID (Dirichlet α) |
| `SEEDS` | `42 123 2024` | Seeds thực nghiệm |

---

## Các Baseline

### Kịch bản 2D — `run_kdl_experiments.sh`

| Group | Baseline | Mô tả |
|-------|----------|-------|
| **A1** | `fedkdl` | **Đề xuất**: HFL-Selective + LoRA + INT8 + Gateway KD |
| A1 | `fedavg_kdl`, `fedprox_kdl` | FedAvg/FedProx với nén KDL |
| A1 | `hfl_nocoop_kdl`, `hfl_nearest_kdl` | HFL không hợp tác / hợp tác gần nhất + KDL |
| **A2** | `fedkdl_r4` | Ablation: LoRA rank-4 |
| A2 | `full_param_kd`, `full_param_nokd` | Không LoRA/INT8 ± KD |
| A2 | `lora_head_kd_noint8`, `head_kd_int8_nolora`, `lora_head_int8_nokd` | Các tổ hợp ablation |
| **A3** | `centralized`, `fedavg`, `fedprox` | Flat baselines không nén |
| A3 | `fedkd`, `hfl_nocoop`, `hfl_nearest`, `hfl_selective` | Local KD / HFL không nén |
| **B** | N=40,50 | Scalability: MAIN_BASELINES |
| **C** | α=10000 | Heterogeneity: MAIN_BASELINES |

### Kịch bản 1D — `run_hfl_experiments.sh`

`hfl_selective`, `hfl_nearest`, `hfl_nocoop`, `fedprox`, `fedavg`, `centralized`

---

## Kết quả train (JSON + Stdout log)

Mỗi run lưu **hai file** cùng tên gốc:

| Loại | HFL (1D) | KDL (2D) |
|------|----------|----------|
| **JSON** (cho plot) | `results/logs/` | `results/logs_kdl/` |
| **Stdout log** | `results/train_logs/hfl/` | `results/train_logs/kdl/` |

Ví dụ tên file 2D:
```
results/logs_kdl/log_N10_URPC_a2p0_fedkdl_seed42.json
results/train_logs/kdl/log_N10_URPC_a2p0_fedkdl_seed42.stdout.log
```

Cấu trúc JSON: `metadata` → `metrics` (theo round) → `energy_consumption` → `latency_history`.

> Các thư mục `results/` và `environments/` bị gitignore — chỉ tồn tại trên máy chạy thực nghiệm.

---

## Vẽ đồ thị (sau khi có JSON)

### 1D (HFL)
```bash
python scripts/hfl/plot_convergence.py
python scripts/hfl/plot_scalability.py
python scripts/hfl/plot_heterogeneity.py
python scripts/hfl/plot_real_benchmark.py
python scripts/hfl/plot_tradeoff.py
```

### 2D (KDL)
```bash
python scripts/fedkdl/plot_od_comparison.py
python scripts/fedkdl/plot_od_scalability.py
python scripts/fedkdl/plot_heterogeneity.py
python scripts/fedkdl/eval_baselines.py --results-dir results/logs_kdl
```

---

## Cấu trúc thư mục

```
FedKDL/
├── config/
│   └── settings.py              # Cấu hình vật lý toàn cục (NetworkConfig, AcousticChannelConfig, ...)
├── federated_core/              # FL chung cho cả 2 kịch bản
│   ├── base_simulator.py        # Vòng lặp chính, phân phối flat vs. HFL
│   ├── aggregator.py            # FedAvg / FedProx global aggregation
│   ├── hfl_rules.py             # Luật hợp tác inter-cluster (selective / nearest / nocoop)
│   ├── metrics.py               # Thu thập, format và lưu metric
│   └── workers.py               # SensorWorker, FogWorker abstraction
├── tasks/
│   ├── anomaly_1d/              # Kịch bản 1D
│   │   ├── autoencoder.py       # Student Autoencoder model
│   │   ├── dataloader.py        # Load SMD/SMAP/MSL + non-IID partition
│   │   ├── simulator.py         # Simulator1D
│   │   └── trainer.py           # Training loop 1D
│   └── detection_2d/            # Kịch bản 2D
│       ├── simulator.py         # Simulator2D (FedKDL, ablation, baselines)
│       ├── trainer.py           # KDDetectionTrainer (LoRA, INT8, KD loss)
│       ├── models/              # StudentModel, TeacherModel, LoRA adapters
│       └── knowledge_compression/  # Quantization INT8, payload encoding
├── physics_models/              # Mô hình vật lý kênh âm dưới nước
│   ├── communication.py         # Thorp-Wenz TL, Wenz NL, Shannon capacity, feasibility
│   ├── energy.py                # Tiêu hao năng lượng phát/thu/tính toán
│   ├── latency.py               # Mô hình độ trễ truyền dẫn
│   └── topology.py              # Topology3D, feasibility graph, association
├── utils/
│   ├── download_datasets.py     # Tải SMD, SMAP/MSL, URPC2020 qua Kaggle
│   ├── env_manager.py           # Sinh/lưu/tải Topology và Data Partition (pkl)
│   ├── generate_all_envs.py     # CLI wrapper: sinh môi trường theo grid N × seed × dataset
│   ├── kaggle_auth.py           # Xác thực Kaggle API token
│   ├── log_export.py            # Export JSON artifact sau mỗi run
│   ├── plot_styles.py           # Màu sắc, font chung cho các script vẽ
│   └── train_io.py              # Quản lý stdout log + artifact JSON
├── scripts/
│   ├── hfl/                     # 5 script vẽ đồ thị cho kịch bản 1D
│   ├── fedkdl/                  # 5 script vẽ + pretrain Teacher cho kịch bản 2D
│   └── od/                      # plot_ablation.py
├── main_trainer.py              # CLI entrypoint cho kịch bản 1D
├── main_trainer_od.py           # CLI entrypoint cho kịch bản 2D
├── run_hfl_experiments.sh       # Grid runner 1D (432 run)
├── run_kdl_experiments.sh       # Grid runner 2D (37 run)
├── quick_start.sh               # One-shot setup + train
├── requirements.txt
└── .env.example                 # Mẫu khai báo KAGGLE_API_TOKEN
```

---

## Thông số vật lý mặc định

| Tham số | Giá trị | Mô tả |
|---------|---------|-------|
| Vùng mô phỏng | 2000 × 2000 m | Mặt phẳng XY |
| Độ sâu Sensor | 500 – 1000 m | Tầng Deep |
| Độ sâu Fog | 100 – 400 m | Tầng Middle |
| Tần số sóng mang | 12 kHz | Acoustic |
| Băng thông | 4000 Hz | ~15 kbps |
| SL_MAX | 140 dB re 1µPa@1m | Giới hạn công suất phát |
| Bán kính tối đa | ~1.09 km | Tính theo Thorp-Wenz |
| Fog nodes (1D) | 10 | Mặc định, đổi qua `M_FOGS_1D` |
| Fog nodes (2D) | 5 | Mặc định, đổi qua `M_FOGS_2D` |
| Pin mỗi node | 500 J | `EnergyConfig.E_INIT` |
