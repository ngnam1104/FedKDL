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
| FedAvg-LoRA (no relay) | 380.6 | Average | 0.0000 | 506.5 | 1824.8 | 23.3137 |
| Naive SVD-LoRA | 380.6 | SVD (no coop) | 0.0014 | 479.4 | 1494.5 | 19.7386 |
| FedKDL Relay (SVD+Coop) | 380.6 | SVD + Coop | 0.0014 | 479.4 | 1494.5 | 19.7386 |

## 5.4 Knowledge Distillation

Payload truyền không đổi; overhead KD nằm ở phía Gateway (không ảnh hưởng comm energy).

| Method | Payload (KB) | Comm overhead | tau_round (s) | E_total (J) | Joint Cost |
|---|---|---|---|---|---|
| No KD | 380.6 | None | 479.4 | 1494.5 | 19.7386 |
| Logit KD | 380.6 | + Output logits | 479.4 | 1494.5 | 19.7386 |
| Feature KD | 380.6 | + Dense features | 479.4 | 1494.5 | 19.7386 |
| LoRA-Proj KD | 380.6 | None (proj only) | 479.4 | 1494.5 | 19.7386 |

## 5.5 Latency-Energy-Accuracy Tradeoff

- **tau_comp** in tau_round: YES (local training bottleneck modeled).
- **Lambda_tau** = 0.01 (s⁻¹), **Lambda_E** = 0.01 (J⁻¹) — scaled to balance contributions.

| Method | Payload (KB) | tau_round (s) | E_total (J) | Joint Cost | Survival (rounds) |
|---|---|---|---|---|---|
| Flat FL (Full FP32) | 10531.0 | 6289.7 | 29341.2 | 356.3098 | 368.2 |
| HFL (Full FP32, no compress) | 10531.0 | 12524.7 | 32869.6 | 453.9433 | 66.1 |
| LoRA FL (FP32, flat) | 1522.5 | 929.5 | 4476.9 | 54.0636 | 2358.8 |
| HFL LoRA (FP32, no SVD) | 1522.5 | 1831.4 | 4987.0 | 68.1839 | 450.9 |
| HFL Top-K 5% | 658.2 | 835.2 | 2673.7 | 35.0888 | 973.0 |
| FedKDL (LoRA+INT8+SVD+KD) | 380.6 | 479.4 | 1494.5 | 19.7386 | 1702.2 |

## 5.6 Per-Device Energy & Latency Breakdown (FedKDL)

Tính toán chi tiết cho AUV 0 và Relay phụ trách (Sử dụng đúng các hàm vật lý gốc).

| Thiết bị | Chặng | Trễ (s) | Năng lượng (J) | Chi tiết |
|---|---|---|---|---|
| **AUV 0** | Huấn luyện cục bộ (LoRA) | 27.19 | 11.01 | Local training |
| **AUV 0** | Truyền AUV -> Relay | 226.01 | 23.17 | Khoảng cách: 1006m |
| **Relay 1** | Nhận từ AUV 0 | - | 11.27 | Mạch thu: 0.05W |
| **Relay 1** | Tổng hợp SVD | 0.0007 | 0.000283 | D_out=256, D_in=128 |
| **Relay 1** | Truyền Relay -> Gateway | 226.01 | 23.17 | Khoảng cách: 1011m |

