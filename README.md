# FedKDL

Mô phỏng **Federated Learning** trên mạng **Internet of Underwater Things (IoUT)** — hai kịch bản:

| Kịch bản | Mô tả | Entrypoint |
|----------|--------|------------|
| **1D — HFL** | Anomaly detection (Autoencoder), Top-K + INT8 | `main_trainer.py` |
| **2D — FedKDL** | Object detection (YOLO26n + LoRA), Gateway-side KD | `main_trainer_od.py` |

Kiến trúc: `federated_core/` (chung) + `tasks/anomaly_1d/` + `tasks/detection_2d/`.

---

## Yêu cầu

- Python 3.10+
- Linux server khuyến nghị cho train dài
- **HFL (1D):** CPU đủ (model ~54k params)
- **KDL (2D):** GPU + CUDA khuyến nghị
- **URPC2020:** token Kaggle (xem bên dưới)

---

## Quick start (server)

```bash
git clone <repo-url> FedKDL && cd FedKDL

# Token Kaggle — bắt buộc nếu tải URPC qua script (KHÔNG commit token vào git)
export KAGGLE_API_TOKEN="your_token_from_kaggle_settings"
# hoặc: cp .env.example .env && nano .env

chmod +x quick_start.sh run_hfl_experiments.sh run_kdl_experiments.sh
./quick_start.sh
```

`quick_start.sh` lần lượt: venv → `pip install` → tải dataset → `generate_all_envs.py` → chạy grid HFL + KDL.

Chạy trong **tmux** nếu SSH có thể đứt:

```bash
tmux new -s fedkdl
./quick_start.sh
```

### Tùy chọn `quick_start.sh`

```bash
./quick_start.sh --help
./quick_start.sh --setup-only      # chỉ venv
./quick_start.sh --train-only      # bỏ qua tải data và setup môi trường
./quick_start.sh --hfl-only        # chỉ 1D (CPU)
./quick_start.sh --kdl-only        # chỉ 2D (GPU)
```

---

## Kaggle API token (URPC / SMAP)

1. Vào [Kaggle Settings → API](https://www.kaggle.com/settings) → **Generate New Token**.
2. Trên server, **một trong hai cách** (không đưa token vào git):

```bash
# Cách 1 — export trong shell / tmux
export KAGGLE_API_TOKEN=KGAT_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Cách 2 — file local (đã được khai báo trong .gitignore)
cp .env.example .env
# Sửa .env: export KAGGLE_API_TOKEN=...
```

3. Kiểm tra tải URPC:

```bash
source .venv/bin/activate
export KAGGLE_API_TOKEN=...   # hoặc source .env
python utils/download_datasets.py --urpc
```

> **Bảo mật:** Không commit `.env`, không ghi token vào README hay mã nguồn. Nếu token đã lộ (chat, log), hãy **thu hồi và tạo token mới** trên Kaggle.

---

## Chạy thủ công từng bước

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install kagglehub

export KAGGLE_API_TOKEN=...    # nếu cần Kaggle
python utils/download_datasets.py --all
python utils/generate_all_envs.py

./run_hfl_experiments.sh       # ~360 run, CPU
./run_kdl_experiments.sh       # ~48 run, GPU
```

---

## Kết quả train (JSON + log)

Mỗi run lưu **hai file** cùng tên gốc:

| Loại | HFL (1D) | KDL (2D) |
|------|----------|----------|
| **JSON** (cho plot) | `results/logs/` | `results/logs_kdl/` |
| **Stdout log** | `results/train_logs/hfl/` | `results/train_logs/kdl/` |

Ví dụ 2D:

```text
results/logs_kdl/log_N50_URPC_a0p1_fedkdl_seed42.json
results/train_logs/kdl/log_N50_URPC_a0p1_fedkdl_seed42.stdout.log
```

Vẽ đồ thị (sau khi có JSON):

```bash
python scripts/hfl/plot_convergence.py
python scripts/fedkdl/plot_od_comparison.py
```

---

## Cấu trúc thư mục (rút gọn)

```text
FedKDL/
├── config/settings.py
├── federated_core/          # FL chung (aggregator, HFL rules, metrics, base_simulator)
├── tasks/
│   ├── anomaly_1d/        # Kịch bản 1D
│   └── detection_2d/      # Kịch bản 2D (YOLO + LoRA)
├── physics_models/          # Kênh âm, năng lượng, độ trễ
├── utils/
│   ├── download_datasets.py
│   ├── generate_all_envs.py
│   ├── log_export.py
│   └── train_io.py
├── scripts/hfl/             # Plot 1D
├── scripts/fedkdl/          # Plot 2D
├── quick_start.sh
├── run_hfl_experiments.sh
└── run_kdl_experiments.ps1 / .sh
```

## Windows (dev)

```powershell
.\run_hfl_experiments.ps1
.\run_kdl_experiments.ps1
```

Token Kaggle: `$env:KAGGLE_API_TOKEN = "..."` trước khi tải data.


