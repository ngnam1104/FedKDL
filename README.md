# FedKDL

FedKDL là mã nguồn thực nghiệm cho bài toán **Federated Object Detection trong Internet of Underwater Things (IoUT)**. Hệ thống mô phỏng mạng phân cấp **AUV -> Relay -> Gateway**, kênh truyền âm dưới nước, năng lượng/độ trễ truyền thông, và các baseline học liên kết cho YOLO object detection trên URPC2020.

Repo hiện tập trung vào bài toán 2D object detection:

- YOLO12n student.
- LoRA/INT8 để giảm payload truyền thông.
- HFL qua relay.
- Gateway KD/proxy fine-tuning cho các biến thể FedKDL.
- Các baseline: FedAvg, FedProx, Scaffold, FLoRA, Top-K, Naive LoRA, FedKDL ablations, centralized LoRA.

## 1. Cài Đặt

Yêu cầu khuyến nghị:

- Python 3.10+.
- GPU NVIDIA có CUDA.
- Dung lượng trống tối thiểu khoảng 20-30 GB nếu tải URPC2020 và chạy nhiều log.

```bash
git clone https://github.com/ngnam1104/FedKDL.git
cd FedKDL

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Trên Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Tải Dataset URPC2020

Không commit Kaggle token vào git. Trên server/máy chạy:

```bash
mkdir -p ~/.kaggle
echo 'YOUR_KAGGLE_TOKEN_HERE' > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
export KAGGLE_API_TOKEN='YOUR_KAGGLE_TOKEN_HERE'
```

Tải dataset:

```bash
python utils/download_datasets.py --urpc
```

Kiểm tra:

```bash
ls datasets/URPC2020/URPC2020/train/images | head
ls datasets/URPC2020/URPC2020/valid/images | head
```

Nếu cần tạo lại YAML dataset:

```bash
cat > datasets/URPC2020.yaml <<'YAML'
path: URPC2020/URPC2020
train: train/images
val: valid/images

nc: 4
names: ['holothurian', 'echinus', 'scallop', 'starfish']
YAML
```

## 3. Sinh Topology Và Data Partition

Case mặc định hay dùng trong các thí nghiệm gần đây:

- `N = 30` AUV.
- `M = 8` relay.
- `alpha = 1.0`.
- `seed = 1109`.
- Dataset `URPC`.

```bash
python utils/generate_all_envs.py \
  --dataset URPC \
  --n 30 \
  --m-relays 8 \
  --alphas 1.0 \
  --seeds 1109
```

Kiểm tra file đầu ra:

```bash
ls environments/2d/topo/N_30/
ls environments/2d/data/URPC/N_30/
```

Cần có:

```text
environments/2d/topo/N_30/topo_N30_seed1109.pkl
environments/2d/data/URPC/N_30/data_N30_URPC_a1p0_seed1109.pkl
```

## 4. Tạo Proxy Data Cho Gateway

Proxy data dùng cho warmup, proxy fine-tuning, và gateway KD.

```bash
python scripts/archive_tests/proxy.py
```

Kiểm tra:

```bash
ls datasets/URPC2020_proxy.yaml
wc -l datasets/URPC2020/proxy_train.txt
head datasets/URPC2020/proxy_train.txt
```

Kỳ vọng khoảng 830 ảnh proxy nếu dùng tỷ lệ 15%.

## 5. Entrypoint Train Chính

Mọi thực nghiệm chạy trực tiếp bằng `python main_trainer_od.py`. Repo không còn dùng file `.sh` làm runner chính.

Mẫu lệnh:

```bash
PYTHONUNBUFFERED=1 WANDB_MODE=disabled python -u main_trainer_od.py \
  --topo environments/2d/topo/N_30/topo_N30_seed1109.pkl \
  --data environments/2d/data/URPC/N_30/data_N30_URPC_a1p0_seed1109.pkl \
  --baseline fedkdl \
  --rounds 40 \
  --out-dir results/logs/N_30/M_8 \
  --log-dir results/train_logs/N_30/M_8
```

Tham số chính:

| Tham số | Ý nghĩa |
| --- | --- |
| `--topo` | File topology `.pkl`. |
| `--data` | File data partition `.pkl`. |
| `--baseline` | Tên baseline trong `tasks/detection_2d/baselines.py`. |
| `--rounds` | Số vòng FL. Với `centralized`, tham số này được dùng như số epoch. |
| `--lora-rank` | Ghi đè LoRA rank nếu cần ablation. |
| `--out-dir` | Thư mục lưu artifact/JSON metrics. |
| `--log-dir` | Thư mục lưu stdout log. |

## 6. Baseline Hỗ Trợ

Các baseline hiện có:

| Baseline | Mô tả ngắn |
| --- | --- |
| `fedavg` | Flat FedAvg full-parameter. |
| `fedprox` | Flat FedProx full-parameter. |
| `fedavg_hfl` | Hierarchical FedAvg full-parameter. |
| `fedprox_hfl` | Hierarchical FedProx full-parameter. |
| `scaffold` | SCAFFOLD-like HFL cho YOLO/AdamW. |
| `topk_grad` | Top-K gradient/state compression. |
| `flora` | SVD-based LoRA aggregation, không INT8/KD. |
| `naive_lora` | LoRA aggregation naive, không SVD/KD. |
| `fedkdl_nokd` | FedKDL transport LoRA/INT8 nhưng không gateway KD. |
| `fedkdl_proxy_ft` | Gateway proxy fine-tuning. |
| `logit_kd` | Gateway KD chỉ với logit KD. |
| `logit_box_kd` | Gateway KD với logit + box KD. |
| `fedkdl` | FedKDL chính. |
| `fedkdl_nocoop` | FedKDL không relay cooperation. |
| `fedkdl_selective` | FedKDL selective relay cooperation. |
| `fedkdl_32bit` | Biến thể không INT8 để so sánh truyền thông. |
| `centralized` | Centralized LoRA training tại gateway, dùng `--rounds` như epoch. |

## 7. Lệnh Chạy Nhanh Các Nhóm Chính

Flat baselines:

```bash
for b in fedavg fedprox; do
  PYTHONUNBUFFERED=1 WANDB_MODE=disabled python -u main_trainer_od.py \
    --topo environments/2d/topo/N_30/topo_N30_seed1109.pkl \
    --data environments/2d/data/URPC/N_30/data_N30_URPC_a1p0_seed1109.pkl \
    --baseline "$b" \
    --rounds 40 \
    --out-dir results/logs/N_30/M_8 \
    --log-dir results/train_logs/N_30/M_8
done
```

HFL full-model:

```bash
for b in fedavg_hfl fedprox_hfl scaffold; do
  PYTHONUNBUFFERED=1 WANDB_MODE=disabled python -u main_trainer_od.py \
    --topo environments/2d/topo/N_30/topo_N30_seed1109.pkl \
    --data environments/2d/data/URPC/N_30/data_N30_URPC_a1p0_seed1109.pkl \
    --baseline "$b" \
    --rounds 40 \
    --out-dir results/logs/N_30/M_8 \
    --log-dir results/train_logs/N_30/M_8
done
```

LoRA/compression baselines:

```bash
for b in flora naive_lora topk_grad fedkdl_nokd; do
  PYTHONUNBUFFERED=1 WANDB_MODE=disabled python -u main_trainer_od.py \
    --topo environments/2d/topo/N_30/topo_N30_seed1109.pkl \
    --data environments/2d/data/URPC/N_30/data_N30_URPC_a1p0_seed1109.pkl \
    --baseline "$b" \
    --rounds 40 \
    --out-dir results/logs/N_30/M_8 \
    --log-dir results/train_logs/N_30/M_8
done
```

FedKDL/KD family:

```bash
for b in logit_kd logit_box_kd fedkdl_proxy_ft fedkdl fedkdl_nocoop fedkdl_selective fedkdl_32bit; do
  PYTHONUNBUFFERED=1 WANDB_MODE=disabled python -u main_trainer_od.py \
    --topo environments/2d/topo/N_30/topo_N30_seed1109.pkl \
    --data environments/2d/data/URPC/N_30/data_N30_URPC_a1p0_seed1109.pkl \
    --baseline "$b" \
    --rounds 40 \
    --out-dir results/logs/N_30/M_8 \
    --log-dir results/train_logs/N_30/M_8
done
```

Centralized LoRA upper bound:

```bash
PYTHONUNBUFFERED=1 WANDB_MODE=disabled python -u main_trainer_od.py \
  --topo environments/2d/topo/N_30/topo_N30_seed1109.pkl \
  --data environments/2d/data/URPC/N_30/data_N30_URPC_a1p0_seed1109.pkl \
  --baseline centralized \
  --rounds 100 \
  --out-dir results/logs/N_30/M_8 \
  --log-dir results/train_logs/N_30/M_8
```

## 8. Mobility Sweep Bằng Python

Mobility energy hiện được log riêng và mặc định không cộng vào objective. Để khảo sát tái phân cụm do vận tốc, bật mobility và đặt quãng đường mỗi round qua `speed * dt`.

Ví dụ 50 m/round:

```bash
FEDKDL_MOBILITY_ENABLED=1 \
FEDKDL_MOVE_ENERGY_ENABLED=0 \
FEDKDL_MOBILITY_DT=30 \
FEDKDL_GM_MEAN_SPEED=1.6667 \
FEDKDL_GM_MAX_SPEED=1.6667 \
PYTHONUNBUFFERED=1 WANDB_MODE=disabled python -u main_trainer_od.py \
  --topo environments/2d/topo/N_30/topo_N30_seed1109.pkl \
  --data environments/2d/data/URPC/N_30/data_N30_URPC_a1p0_seed1109.pkl \
  --baseline fedkdl \
  --rounds 40 \
  --out-dir results/mobility/50m \
  --log-dir results/train_logs/mobility/50m
```

Ví dụ 100 m/round:

```bash
FEDKDL_MOBILITY_ENABLED=1 \
FEDKDL_MOVE_ENERGY_ENABLED=0 \
FEDKDL_MOBILITY_DT=30 \
FEDKDL_GM_MEAN_SPEED=3.3333 \
FEDKDL_GM_MAX_SPEED=3.3333 \
PYTHONUNBUFFERED=1 WANDB_MODE=disabled python -u main_trainer_od.py \
  --topo environments/2d/topo/N_30/topo_N30_seed1109.pkl \
  --data environments/2d/data/URPC/N_30/data_N30_URPC_a1p0_seed1109.pkl \
  --baseline fedkdl \
  --rounds 40 \
  --out-dir results/mobility/100m \
  --log-dir results/train_logs/mobility/100m
```

## 9. LoRA Rank Ablation Bằng Python

Rank mặc định nằm trong `config/settings.py`. Có thể ghi đè bằng CLI hoặc biến môi trường.

Rank chung:

```bash
PYTHONUNBUFFERED=1 WANDB_MODE=disabled python -u main_trainer_od.py \
  --topo environments/2d/topo/N_30/topo_N30_seed1109.pkl \
  --data environments/2d/data/URPC/N_30/data_N30_URPC_a1p0_seed1109.pkl \
  --baseline fedkdl \
  --rounds 40 \
  --lora-rank 4 \
  --out-dir results/rank/r4 \
  --log-dir results/train_logs/rank/r4
```

Backbone/neck rank riêng:

```bash
FEDKDL_LORA_RANK=4 \
FEDKDL_LORA_BACKBONE_RANK=2 \
FEDKDL_LORA_NECK_RANK=4 \
PYTHONUNBUFFERED=1 WANDB_MODE=disabled python -u main_trainer_od.py \
  --topo environments/2d/topo/N_30/topo_N30_seed1109.pkl \
  --data environments/2d/data/URPC/N_30/data_N30_URPC_a1p0_seed1109.pkl \
  --baseline fedkdl \
  --rounds 40 \
  --out-dir results/rank/r2_4 \
  --log-dir results/train_logs/rank/r2_4
```

## 10. Vẽ Đồ Thị Và Bảng Kết Quả

Sau khi chạy xong các thực nghiệm và có đầy đủ log dưới thư mục `results/`, bạn có thể vẽ toàn bộ đồ thị (tiếng Anh & tiếng Việt) và sinh các bảng LaTeX/Markdown/PDF.

Vẽ toàn bộ 8 hình và 3 bảng:

```bash
python scripts/fedkdl/plot/plot_all.py
```

Hoặc chạy riêng lẻ từng hình/bảng:

```bash
# Vẽ hình 1
python scripts/fedkdl/plot/fig1.py

# Vẽ hình 2
python scripts/fedkdl/plot/fig2.py

# Sinh bảng 2
python scripts/fedkdl/plot/table2.py
```

Các hình ảnh đầu ra được lưu vào `.images/en/` và `.images/vi/`. Các bảng kết quả được lưu dưới `results/metrics_final/tables_paper/`.

## 11. Demo UI

Demo nằm trong `demo/`.

Chạy backend trên server GPU:

```bash
python demo/app.py
```

Mở tunnel từ máy local:

```bash
ssh -p <SSH_PORT> root@<PUBLIC_IP> -L 5000:localhost:5000
```

Sau đó mở:

```text
http://localhost:5000
```

Nếu mở frontend local nhưng backend ở server/tunnel, nhập API URL:

```text
http://localhost:5000/api
```

## 12. Kết Quả Và Git Ignore

Toàn bộ kết quả thực nghiệm nằm dưới `results/` và **không được commit vào git**. README chỉ mô tả cách chạy; log, JSON, CSV, ảnh plot và model output cần lưu ngoài git hoặc tải riêng khi phân tích.

Các thư mục/file sinh ra khi chạy thường gồm:

```text
results/logs/...
results/train_logs/...
results/mobility/...
results/rank/...
runs/...
```

`results/` đã nằm trong `.gitignore`, nên sau khi train xong có thể nén/tải về để phân tích nhưng không đưa trực tiếp lên repository.

## 13. Cấu Trúc Repo

```text
FedKDL/
├── config/
│   └── settings.py
├── federated_core/
│   ├── base_simulator.py
│   ├── aggregator.py
│   ├── hfl_rules.py
│   ├── metrics.py
│   └── workers.py
├── physics_models/
│   ├── communication.py
│   ├── energy.py
│   ├── latency.py
│   └── topology.py
├── tasks/
│   └── detection_2d/
│       ├── baselines.py
│       ├── simulator.py
│       ├── trainer.py
│       ├── models/
│       └── knowledge_compression/
├── scripts/
│   ├── fedkdl/
│   │   └── plot/                # Module chứa toàn bộ script vẽ đồ thị và sinh bảng
│   └── archive_tests/
├── utils/
│   ├── download_datasets.py
│   ├── generate_all_envs.py
│   └── train_io.py
├── demo/
├── main_trainer_od.py
└── requirements.txt
```

## 14. Ghi Chú Thực Nghiệm

- `centralized` là centralized LoRA upper bound; `--rounds` được hiểu như số epoch.
- `fedkdl_nokd` dùng cùng đường truyền LoRA/INT8 nhưng tắt gateway KD.
- `flora` dùng LoRA không INT8, thường là mốc quan trọng để đo khoảng cách do nén INT8.
- Mobility energy hiện được log riêng qua `e_move`, không cộng vào `e_total`/joint cost mặc định.
- Khi chạy trên server yếu CPU/I/O, `LOCAL_DATALOADER_WORKERS=0` and `LOCAL_CACHE_DATASET=True` thường ổn định hơn.
