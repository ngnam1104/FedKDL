# FedKDL

**Federated Knowledge Distillation Learning** trên mạng **Internet of Underwater Things (IoUT)** — phát hiện vật thể (Object Detection) dưới nước với mô hình nén LoRA + INT8, truyền thông qua kênh âm thanh dưới nước.

Kiến trúc phân cấp 3 tầng: **AUV → Relay → Gateway**. Mô hình vật lý kênh âm sử dụng Thorp-Wenz, vùng mô phỏng 2000 × 2000 m.

---

## Yêu cầu

- Python 3.10+
- GPU + CUDA (YOLOv12n + LoRA rank-8)
- Dataset URPC2020 (xem hướng dẫn bên dưới)

```bash
git clone <repo-url> FedKDL && cd FedKDL
python -m venv .venv && source .venv/bin/activate   # Linux
# hoặc: .venv\Scripts\activate                       # Windows
pip install -r requirements.txt
```

---

## Dataset URPC2020

Tải qua Kaggle API:

```bash
# 1. Vào https://www.kaggle.com/settings → API → Generate New Token
export KAGGLE_API_TOKEN=KGAT_xxxxxxxxxxxxxxxxxxxxxxxxxxxx

# 2. Tải dataset
python utils/download_datasets.py --urpc
```

> **Bảo mật:** Không commit token vào git. Nếu token đã lộ, thu hồi và tạo mới trên Kaggle.

---

## Quick Start

### 1. Sinh môi trường Topology & Data Partition

```bash
python utils/generate_all_envs.py --dataset URPC
```

Sinh ra các file `.pkl` trong `environments/`:
- `topo_N<n>_URPC_a<alpha>_seed<s>.pkl` — topology AUV/Relay
- `data_N<n>_URPC_a<alpha>_seed<s>.pkl` — phân phối dữ liệu Non-IID

### 2. Pre-train Teacher (chỉ cần 1 lần)

```bash
python scripts/fedkdl/pretrain.py
```

Sinh ra `yolo12l_lora_pretrained.pt` — Teacher model cho Gateway-side KD.

### 3. Train

Entrypoint chính là `main_trainer_od.py`:

```bash
python main_trainer_od.py \
    --topo environments/topo_N10_URPC_a2p0_seed42.pkl \
    --data environments/data_N10_URPC_a2p0_seed42.pkl \
    --baseline fedkdl \
    --rounds 60
```

**Các tham số:**

| Tham số | Bắt buộc | Mặc định | Mô tả |
|---------|----------|----------|-------|
| `--topo` | ✅ | — | Đường dẫn file topology `.pkl` |
| `--data` | ✅ | — | Đường dẫn file data partition `.pkl` |
| `--baseline` | ✅ | — | Tên baseline (xem danh sách bên dưới) |
| `--rounds` | ❌ | config | Ghi đè số vòng `GLOBAL_ROUNDS` |
| `--lora-rank` | ❌ | config | Ghi đè LoRA rank (4 hoặc 8) |
| `--out-dir` | ❌ | `results/logs_kdl` | Thư mục lưu JSON metrics |
| `--log-dir` | ❌ | `results/train_logs/kdl` | Thư mục lưu stdout `.log` |

### 4. Chạy toàn bộ grid thực nghiệm

```bash
chmod +x run_kdl_experiments.sh
./run_kdl_experiments.sh
```

Có thể chạy trong **tmux** để tránh mất session khi SSH đứt:

```bash
tmux new -s fedkdl
./run_kdl_experiments.sh
```

---

## Các Baseline

Truyền vào `--baseline`:

| Baseline | Mô tả |
|----------|-------|
| `fedkdl` | **Đề xuất**: HFL-Selective + LoRA + INT8 + Gateway KD |
| `fedavg_hfl` | FedAvg với HFL, không nén |
| `fedprox_hfl` | FedProx với HFL, không nén |
| `topk_grad` | Top-K Gradient Sparsification |
| `naive_lora` | LoRA thuần, không có Gateway KD |
| `flora` | SVD-based LoRA aggregation (FLORA), không KD |
| `fedkdl_nokd` | FedKDL không có Knowledge Distillation (ablation) |
| `fedkdl_nocoop` | FedKDL không có Relay Cooperation (ablation) |
| `fedkdl_nolora` | FedKDL không nén LoRA — full params + KD (ablation) |
| `logit_kd` | Gateway KD chỉ dùng Logit Matching |
| `centralized` | Huấn luyện tập trung tại Gateway (upper bound) |

---

## Kết quả

Mỗi run lưu hai loại artifact:

| Loại | Đường dẫn | Nội dung |
|------|-----------|----------|
| **JSON metrics** | `results/logs_kdl/log_N<n>_URPC_a<alpha>_<baseline>_seed<s>.json` | mAP, loss, energy, latency theo từng round |
| **Stdout log** | `results/train_logs/kdl/log_...stdout.log` | Log huấn luyện chi tiết |

Cấu trúc JSON: `metadata` → `metrics` (theo round) → `energy_consumption` → `latency_history`.

> `results/` bị gitignore — chỉ tồn tại trên máy chạy thực nghiệm.

---

## Vẽ đồ thị (sau khi có kết quả)

```bash
# Vẽ toàn bộ các hình
PYTHONPATH="." python scripts/fedkdl/plot_all_figures.py

# Hoặc từng hình riêng
PYTHONPATH="." python scripts/fedkdl/K1_connectivity_scalability.py
PYTHONPATH="." python scripts/fedkdl/K2_rq2_learning.py
PYTHONPATH="." python scripts/fedkdl/K2_payload_comparison.py
PYTHONPATH="." python scripts/fedkdl/K2_joint_cost.py
PYTHONPATH="." python scripts/fedkdl/K3_rq3_learning.py
PYTHONPATH="." python scripts/fedkdl/K3_rq3_ablation.py
PYTHONPATH="." python scripts/fedkdl/K4_rq4_learning.py
PYTHONPATH="." python scripts/fedkdl/K4_detection_quality.py
```

Output PDF/PNG lưu vào `results/plots/`.

---

## Cấu trúc thư mục

```
FedKDL/
├── config/
│   └── settings.py              # Cấu hình vật lý toàn cục (NetworkConfig, AcousticChannelConfig, ...)
├── federated_core/              # FL core
│   ├── base_simulator.py        # Vòng lặp chính, phân phối HFL
│   ├── aggregator.py            # FedAvg / FedProx global aggregation
│   ├── hfl_rules.py             # Luật hợp tác inter-cluster (selective / nearest / nocoop)
│   ├── metrics.py               # Thu thập, format và lưu metric
│   └── workers.py               # AUVWorker, RelayWorker abstraction
├── tasks/
│   └── detection_2d/            # Kịch bản 2D (Object Detection)
│       ├── simulator.py         # Simulator2D (FedKDL, ablation, baselines)
│       ├── trainer.py           # KDDetectionTrainer (LoRA, INT8, KD loss)
│       ├── baselines.py         # BASELINE_CONFIGS registry
│       ├── models/              # StudentModel, TeacherModel, LoRA adapters
│       └── knowledge_compression/  # Quantization INT8, payload encoding
├── physics_models/              # Mô hình vật lý kênh âm dưới nước
│   ├── communication.py         # Thorp-Wenz TL, Wenz NL, Shannon capacity
│   ├── energy.py                # Tiêu hao năng lượng phát/thu/tính toán
│   ├── latency.py               # Mô hình độ trễ truyền dẫn
│   └── topology.py              # Topology2D, feasibility graph, association
├── utils/
│   ├── download_datasets.py     # Tải URPC2020 qua Kaggle
│   ├── env_manager.py           # Sinh/lưu/tải Topology và Data Partition (.pkl)
│   ├── generate_all_envs.py     # CLI wrapper: sinh môi trường theo grid N × seed
│   └── train_io.py              # Quản lý stdout log + artifact JSON
├── scripts/
│   └── fedkdl/                  # Script vẽ đồ thị (K1–K4) + pretrain Teacher
├── main_trainer_od.py           # ← Entrypoint chính để train
├── run_kdl_experiments.sh       # Grid runner tự động
└── requirements.txt
```

---

## Thông số vật lý mặc định

| Tham số | Giá trị | Mô tả |
|---------|---------|-------|
| Vùng mô phỏng | 2000 × 2000 m | Mặt phẳng XY |
| Độ sâu AUV | 500 – 1000 m | Tầng Deep |
| Độ sâu Relay | 100 – 400 m | Tầng Middle |
| Tần số sóng mang | 12 kHz | Acoustic |
| Băng thông | 4000 Hz | ~15 kbps |
| SL_MAX | 140 dB re 1µPa@1m | Giới hạn công suất phát |
| Bán kính tối đa | ~1.09 km | Tính theo Thorp-Wenz |
| Relay nodes | 5 | Mặc định, đổi qua `M_RELAYS_2D` trong shell |
| Pin mỗi node | 500 J | `EnergyConfig.E_INIT` |
