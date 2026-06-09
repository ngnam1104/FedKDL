# Metrics Tính Toán Từ Mô Hình Hệ Thống (Section 3 / 5.2–5.5)

- **Model**: YOLOv12-N (Student) — 2,695,948 total params
- **Topology**: N=30 AUVs, M=4 Relays (from `topo_N30_seed1104.pkl`)
- **Shannon Capacity** R = 13838 bps
- **Avg samples/AUV** ≈ 100 (URPC ~3000 train ÷ 30 AUVs)

## 5.2 Dual Compression

Payload được tính dựa trên cấu trúc nén thực tế.
Top-K truyền INT8 values **và** INT32 indices.

| Method | Payload (KB) | vs Full |
|---|---|---|
| Full Parameter FL (FP32) | 10531.0 | 1.0× |
| Top-K Compression (5%, INT8 + idx) | 658.2 | 16.0× |
| Top-K Compression (1%, INT8 + idx) | 131.6 | 80.0× |
| LoRA Only (FP32) | 1522.5 | 6.9× |
| LoRA + INT8 (FedKDL Tier-1) | 380.6 | 27.7× |

## 5.3 Relay Aggregation

Tất cả baselines đều dùng LoRA+INT8 payload để so sánh thuần tuý chiến lược tập hợp.

| Method | Payload A→R (KB) | Relay Op | tau_svd (s) | tau_round (s) | E_total (J) | Joint Cost |
|---|---|---|---|---|---|---|
| FedAvg-LoRA (no relay) | 380.6 | Average | 0.0000 | 506.5 | 89843.2 | 0.7761 |
| Naive SVD-LoRA | 380.6 | SVD (no coop) | 0.0014 | 479.4 | 83236.7 | 0.7291 |
| FedKDL Relay (SVD+Coop) | 380.6 | SVD + Coop | 0.0014 | 479.4 | 83236.7 | 0.7291 |

## 5.4 Knowledge Distillation

Payload truyền không đổi; overhead KD nằm ở phía Gateway (không ảnh hưởng comm energy).

| Method | Payload (KB) | Comm overhead | tau_round (s) | E_total (J) | Joint Cost |
|---|---|---|---|---|---|
| No KD | 380.6 | None | 479.4 | 83236.7 | 0.7291 |
| Logit KD | 380.6 | + Output logits | 479.4 | 83236.7 | 0.7291 |
| Feature KD | 380.6 | + Dense features | 479.4 | 83236.7 | 0.7291 |
| LoRA-Proj KD | 380.6 | None (proj only) | 479.4 | 83236.7 | 0.7291 |

## 5.5 Latency-Energy-Accuracy Tradeoff

- **tau_comp** in tau_round: YES (local training bottleneck modeled).
- **Lambda_tau** = 0.001 (s⁻¹), **Lambda_E** = 3e-06 (J⁻¹) — scaled to balance contributions.

| Method | Payload (KB) | tau_round (s) | E_total (J) | Joint Cost | Survival (rounds) |
|---|---|---|---|---|---|
| Flat FL (Full FP32) | 10531.0 | 6289.7 | 1884167.8 | 11.9422 | 4.0 |
| HFL (Full FP32, no compress) | 10531.0 | 12524.7 | 2133332.6 | 18.9247 | 4.0 |
| LoRA FL (FP32, flat) | 1522.5 | 929.5 | 277103.5 | 1.7608 | 27.1 |
| HFL LoRA (FP32, no SVD) | 1522.5 | 1831.4 | 313127.0 | 2.7708 | 27.1 |
| HFL Top-K 5% | 658.2 | 835.2 | 145720.2 | 1.2723 | 57.6 |
| FedKDL (LoRA+INT8+SVD+KD) | 380.6 | 479.4 | 83236.7 | 0.7291 | 101.0 |
