# Báo cáo Hiện trạng Mã nguồn — FedKDL Repository

**Dự án:** FedHKL / FedKDL — Hierarchical Federated Learning for IoUT  
**Mục tiêu bài báo tái hiện:** Omeke et al. (2026) *"Energy-Efficient Hierarchical Federated Anomaly Detection for the Internet of Underwater Things via Selective Cooperative Aggregation"*  
**Ngôn ngữ lập trình:** Python 3.11  
**Deep Learning Framework:** PyTorch  
**Trạng thái:** Đang tích cực phát triển & chạy thử nghiệm thực  

---

## 1. Cây Thư mục Hiện tại

```
FedKDL/
├── config/
│   └── settings.py              # Dataclass configs (NetworkConfig, AcousticChannelConfig,
│                                #   EnergyConfig, FedKDLConfig) — cấu hình toàn cục, import singleton
│
├── physics_models/              # Mô hình vật lý âm học dưới nước
│   ├── topology.py              # Topology3D, build_feasibility_graph, nearest_feasible_association
│   ├── communication.py         # Thorp-Wenz TL model, Shannon capacity, feasibility check
│   ├── energy.py                # e_tx, e_rx, e_comp_simple/full — Eq. 20-27 paper
│   └── latency.py               # comm_delay, comp_delay_simple — Eq. 21 paper
│
├── hfl_core/                    # Baseline HFL (tái hiện Omeke et al. 2026)
│   ├── simulator.py             # Scenario1Simulator: vòng lặp mô phỏng chính (1D AnomalyDet)
│   ├── metrics.py               # EnergyTracker, LatencyTracker, MetricsLogger, anomaly_threshold, PA-F1
│   ├── algorithms/
│   │   ├── local_trainer.py     # local_sgd() — trả về (Δθ, avg_loss), hỗ trợ FedProx proximal term
│   │   ├── worker.py            # SensorWorker, FogNode, SurfaceGateway — thực thể mô phỏng OOP
│   │   ├── aggregator.py        # fedavg_intra_cluster (Eq. 40), fedavg_global (Eq. 43)
│   │   └── hfl_rules.py         # should_cooperate (Eq. 41), find_coop_partner, blend_state_dicts (Eq. 42)
│   ├── knowledge_compression/
│   │   ├── topk_sparsification.py  # TopKCompressor: Top-K + Error Feedback, rho_s=0.05
│   │   └── int8_quantization.py    # quantize_tensor, SparseINT8Payload, pack_payload/unpack_payload
│   ├── models/
│   │   └── autoencoder.py       # SmallAutoencoder: D→32→16→8→16→32→D (~1350 params)
│   └── data/
│       └── dataloader_1d.py     # SlidingWindowDataset, non_iid_partition (Dirichlet), load_real_smd
│
├── kdl_core/                    # [Đang xây dựng] FedKDL nâng cao — cùng cấu trúc với hfl_core
│   ├── simulator.py             # Kế thừa + mở rộng hfl_core simulator với KD-LoRA
│   └── simulator_od.py          # Simulator cho Kịch bản 2&3 (Object Detection, YOLO)
│
├── scripts_baseline/            # Các kịch bản chạy đánh giá (tái hiện paper)
│   ├── run_real_benchmarks.py   # Figure 8: SMD/SMAP/MSL — 30 rounds, 6 baselines
│   ├── run_scalability.py       # Figure 5: N=50,100,150,200 — 20 rounds, 3 seeds
│   ├── run_heterogeneity.py     # Figure 7: alpha=0.1 vs alpha=10000 — Non-IID sensitivity
│   ├── run_convergence.py       # Figure 4: Loss curve N=150,200 — fill_between shaded std
│   ├── run_engineering_effects.py # Figure 6: Compressed vs Uncompressed energy savings
│   ├── run_scenario2_bottleneck.py # Kịch bản 2 (Object Detection bandwidth test)
│   ├── eval_baselines.py        # Hàm đánh giá chung (PA-F1, Energy per round)
│   └── plot_metrics.py          # Generic JSON→plot utility (cho OD scenarios)
│
├── scripts_fedkdl/
│   └── run_scenario3_fedkdl.py  # Kịch bản 3: FedKDL với KD-LoRA + YOLO
│
├── datasets/                    # Dữ liệu thực tế (download thủ công)
│   ├── SMD/train/, test/, test_label/   # Server Machine Dataset (38 features)
│   ├── SMAP/ & MSL/             # NASA Telemetry (25 & 55 features)
│   └── URPC2020.yaml            # YOLO format config cho Object Detection
│
├── results/                     # Output của các kịch bản (tự động tạo)
│   ├── real_benchmarks/         # fig8_real_benchmarks.png + summary.csv
│   ├── scalability/             # fig5_scalability.png
│   ├── heterogeneity/           # fig7_heterogeneity.png
│   ├── convergence/             # fig4_convergence.png
│   └── engineering/             # fig6_engineering.png
│
├── run_all_experiments.ps1      # PowerShell script chạy toàn bộ pipeline một lệnh
├── requirements.txt             # torch, numpy, pandas, matplotlib, scikit-learn, PyMuPDF
└── TODO.md                      # Backlog nợ kỹ thuật
```

---

## 2. Quản lý Môi trường & Topology

### Cách Topology được khởi tạo

Topology 3D quasi-static được khởi tạo **mỗi lần chạy** từ `Topology3D(net_cfg, ac_cfg, seed)`, với seed xác định hoàn toàn vị trí của cảm biến và fog. Không có cơ chế lưu/nạp file (`pickle`, `json`).

```python
# physics_models/topology.py — Luồng khởi tạo
class Topology3D:
    def __init__(self, net_cfg, acoustic_cfg, seed: int = 42):
        self.rng = np.random.RandomState(seed)  # Seed-deterministic
        self.sensor_positions = self._place_sensors()   # (N, 3) uniform random
        self.fog_positions    = self._place_fogs()      # (M, 3) shallower depth
        self.gateway_position = np.array([AREA_X/2, AREA_Y/2, 0.0])  # Surface, centered

# Sau đó build feasibility graph O(N×M + M² + N)
G = build_feasibility_graph(topology, acoustic_cfg)
# G: dict[(type_u, id_u, type_v, id_v) → LinkInfo(distance, SL_min, TL, NL, R_bps)]
```

**Phân chia Non-IID Dirichlet:**

```python
# hfl_core/data/dataloader_1d.py
client_indices = non_iid_partition(train_ds, n_sensors, alpha=0.1, seed=seed)
# Dirichlet(α=0.1): mỗi sensor chỉ thấy ~1-2 class → data rất skewed
# Dirichlet(α=1e4): gần IID
client_loaders = make_client_loaders(train_ds, client_indices)
```

**⚠️ Hiện trạng:** Không có cơ chế persist/cache topology hay partition. Mỗi lần khởi tạo `Scenario1Simulator`, toàn bộ `build_feasibility_graph` (O(N×M + M²)) được tính lại từ đầu. Với N=200, M=20, đây là ~4200 cặp kiểm tra, có thể mất vài giây.

---

## 3. Lõi Mô phỏng (Simulator & Algorithms)

### 3.1 Khởi tạo Scenario1Simulator

```python
class Scenario1Simulator:
    def __init__(
        self,
        net_cfg: NetworkConfig,        # N_SENSORS, M_FOGS, AREA_X/Y, depths
        ac_cfg: AcousticChannelConfig, # SOUND_SPEED, CARRIER_FREQ, BANDWIDTH, SL_MAX, ...
        en_cfg: EnergyConfig,          # E_INIT (500J), E_COMP_EPOCH, P_C_TX, ETA_EA
        fed_cfg: FedKDLConfig,         # LOCAL_EPOCHS, LOCAL_LR, NON_IID_ALPHA, DATASETS_1D
        baseline: str = 'hfl_selective', # 'fedavg'|'fedprox'|'hfl_nocoop'|'hfl_nearest'|'hfl_selective'|'centralised'
        seed: int = 42
    )
```

Trong `__init__`, Simulator thực hiện theo thứ tự:
1. `Topology3D` + `build_feasibility_graph` → `self.G`
2. `load_dataset(DATASETS_1D[0])` → train/val split (70:30) → `SlidingWindowDataset`
3. `non_iid_partition` → `make_client_loaders`
4. Association: `flat_topology_association` (FedAvg/FedProx) hoặc `nearest_feasible_association` (HFL)
5. Khởi tạo `SmallAutoencoder(input_dim)`, `SensorWorker` × N, `FogNode` × M, `SurfaceGateway` × 1

### 3.2 Luồng thực thi mỗi Round

```
for t in range(T_rounds):
  │
  ├─ Phase 1: Local Training + Compression [ThreadPoolExecutor, max_workers=8 (CPU)]
  │    └─ sensor.train_and_compress()
  │         ├─ local_sgd(epochs=E, lr=0.01, mu=0.01 nếu FedProx)
  │         │   └─ returns: (Δθ: Tensor, avg_loss: float)
  │         ├─ TopKCompressor.compress(Δθ) → Error Feedback → (topk_indices, topk_values)
  │         └─ SparseINT8Payload(indices, quantize(values)) → payload_bits computed
  │
  ├─ Phase 2: Fog Aggregation & HFL Cooperation [Sequential]
  │    ├─ fog.aggregate_intra_cluster() → fedavg_intra_cluster (Eq. 40)
  │    └─ fog.cooperate(rule='selective'|'nearest'|'nocoop')
  │         ├─ should_cooperate(c_m, c̄) [Eq. 41]
  │         ├─ find_coop_partner(fog_id, cluster_sizes, G, q1_distance) [Eq. 29]
  │         └─ blend_state_dicts(self_sd, neighbor_sd, alpha=0.8) [Eq. 42]
  │
  ├─ Phase 3: Global Aggregation [Gateway]
  │    └─ gateway.aggregate_global() → fedavg_global (Eq. 43)
  │
  └─ Phase 4: Logging & Evaluation
       ├─ EnergyTracker.add_round(e_s2f, e_f2f, e_f2g, e_comp)
       ├─ LatencyTracker.compute_round_latency() → max() qua mọi bottleneck link [Eq. 21]
       ├─ evaluate_global_model() → PA-F1, Precision, Recall
       └─ MetricsLogger.log(t, {PA-F1, Precision, Recall, Participation, Cumul_Energy, Tau_Round_s, Train_Loss})
```

### 3.3 6 Baselines được hỗ trợ

| Baseline | Association | Cooperation | Đặc điểm |
|---|---|---|---|
| `centralised` | N/A | N/A | Huấn luyện tập trung, chuẩn oracle |
| `fedavg` | flat (gateway trực tiếp) | None | Chỉ sensors nổi lên mặt nước (20-30% tham gia) |
| `fedprox` | flat (gateway trực tiếp) | None | μ=0.01 proximal term |
| `hfl_nocoop` | HFL (fog gần nhất) | None | ~100% sensors qua fog |
| `hfl_nearest` | HFL | Always-on, α=0.7 | Tất cả fog đều hợp tác nếu có link |
| `hfl_selective` | HFL | Selective, α=0.8 | Chỉ fog "đói tri thức" hợp tác — Eq. 41+42 |

### 3.4 Pipeline Nén Top-K + INT8

```
Δθ (Float32, ~1350 params)
    ↓ TopKCompressor.compress()       [rho_s=0.05 → K=68 entries]
    ↓ Error Feedback: compensated = Δθ + error_buffer
    ↓ TopK Selection: indices (K,), values (K,)
    ↓ quantize_tensor(values) → INT8  [Δ=(max-min)/255, Z=-128..127]
    ↓ SparseINT8Payload
         - payload_bits = K×8 + K×ceil(log2(P)) + 40 header ≈ 1.3 kbits
         - payload_bytes ≈ 163 Bytes
    ↓ e_tx(payload_bits, R_bps, SL_min, eta_ea=0.25, P_c_tx=0.05)  [Eq. 22]
```

---

## 4. Cấu trúc Experiment Scripts

### Tổ chức

Mỗi `run_*.py` là một file self-contained chứa:
- Hàm `run_X(dry_run=False)` — chạy simulation và thu kết quả
- Hàm `plot_X(df)` — vẽ biểu đồ matplotlib từ DataFrame kết quả
- Block `if __name__ == '__main__': run_X('--dry-run' in sys.argv)`

| Script | Figure | Chi tiết |
|---|---|---|
| `run_convergence.py` | Fig. 4 | N=150,200; T=20 rounds; 3 seeds; Loss curve with shaded std |
| `run_scalability.py` | Fig. 5 | N=[50,100,150,200]; M=N//10; 3 seeds; 3 subplots |
| `run_engineering_effects.py` | Fig. 6 | rho_s=0.05 vs 1.0; Energy Savings bar chart |
| `run_heterogeneity.py` | Fig. 7 | α=0.1 vs α=10000; F1 + Energy bar charts |
| `run_real_benchmarks.py` | Fig. 8 | SMD/SMAP/MSL; T=30; PA-F1 + Energy (log scale) |

### DRY / Code Reuse

**Chưa tốt.** Mỗi script tự import và khởi tạo `Scenario1Simulator` trực tiếp. Các hàm vẽ biểu đồ (`style_map`, `bar plot` pattern, auto-zoom F1 axis) bị lặp lại trong 4-5 file khác nhau. Có file `eval_baselines.py` và `plot_metrics.py` nhưng không được dùng bởi các `run_*.py` mới — chỉ dành cho kịch bản OD cũ.

**Hàm dùng chung nằm ở đâu:**
- Logging: `hfl_core/metrics.py` (`MetricsLogger`)
- Vẽ hình: **Không tập trung** — mỗi file tự định nghĩa hàm `plot_X(df)`

---

## 5. Điểm Mù & Nợ Kỹ thuật

### ⚠️ Nghiêm trọng

**1. `rho_s` (sparsity ratio) bị hardcode trong Simulator**
```python
# hfl_core/simulator.py, line 95
self.sensors[i] = SensorWorker(sensor_id=i, ..., rho_s=0.05)  # HARDCODED
```
Script `run_engineering_effects.py` đang thay đổi `fed_cfg.RHO_S` nhưng Simulator **không đọc giá trị này** — `rho_s=0.05` luôn bất biến. Kết quả biểu đồ Figure 6b sẽ sai (hai chế độ compressed/uncompressed thực ra cho cùng kết quả).

**2. Không persist topology** — Mỗi lần chạy thử nghiệm, `build_feasibility_graph` phải xây lại toàn bộ. Khi chạy 3 seeds × 4 N-values × 6 baselines = 72 lần khởi tạo Simulator, đây là overhead đáng kể.

**3. `load_dataset` chỉ tải 1 trong 3 dataset thực tế** — SMAP và MSL vẫn dùng dữ liệu tổng hợp (synthetic fallback), chỉ SMD được load thật từ file. Hàm `load_real_smd()` tồn tại, nhưng không có `load_real_smap()` hay `load_real_msl()`.

### ⚡ Hiệu năng

**4. ThreadPoolExecutor trong vòng lặp for** — `import concurrent.futures` và định nghĩa `def process_sensor()` được khai báo **bên trong** `for t in range(T_rounds)`, tức là tái định nghĩa function object mỗi round. Ít tốn kém về semantics, nhưng là code smell.

**5. Không có gradient clipping** — `local_sgd()` chạy SGD không có `clip_grad_norm_`, có nguy cơ gradient explosion nếu thay đổi kiến trúc model hay tăng learning rate.

**6. Memory leak tiềm năng trong FogNode cooperation** — `all_intra` dict tích lũy `copy.deepcopy` của state_dict từ mọi Fog trong mỗi round, không được giải phóng tường minh.

### 📝 Nợ Thiết kế

**7. `fed_cfg.DATASETS_1D[0]` hardcode vào index 0** — Simulator chỉ tải tập đầu tiên trong danh sách. Để chạy qua 3 tập SMD/SMAP/MSL, `run_real_benchmarks.py` phải khởi tạo Simulator với `fed_cfg.DATASETS_1D = [dataset_name]` — thay đổi singleton toàn cục, có thể gây side effects nếu chạy đa luồng.

**8. PA-F1 plateau** — Mô hình `SmallAutoencoder` (~1350 params, kiến trúc cố định) đang cho PA-F1 gần như bằng nhau giữa mọi baselines do model quá đơn giản để thể hiện sự khác biệt sau aggregation. Cần tăng số rounds (20 → 100+) hoặc dùng dữ liệu thực đủ lớn.

**9. Không có checkpoint/resume** — Không có cơ chế lưu trạng thái giữa chừng. Nếu chương trình bị gián đoạn sau 15/20 rounds, toàn bộ phải chạy lại từ đầu.

---

## Tóm tắt Kiến trúc (Architecture Snapshot)

```
┌──────────────────────────────────────────────────────────┐
│                     SurfaceGateway                        │
│  global_state_dict ─── fedavg_global(Eq.43) ──► Θ^(t+1) │
└───────────────┬──────────────────────────────────────────┘
                │  fog_final_state_dicts × M
┌───────────────▼──────────────────────────────────────────┐
│  FogNode × M    (Tầng Fog / AUV Relay)                   │
│  aggregate_intra_cluster → fedavg_intra (Eq.40)          │
│  cooperate() → should_cooperate (Eq.41)                  │
│              → find_coop_partner (Q1 filter)             │
│              → blend_state_dicts α=0.8 (Eq.42)          │
└───────────────┬──────────────────────────────────────────┘
                │  SparseINT8Payload (K×8b + K×logP bits)
┌───────────────▼──────────────────────────────────────────┐
│  SensorWorker × N   (Tầng Cảm biến / Deep Layer)        │
│  train_and_compress()                                    │
│    → local_sgd (E=5 epochs, μ=0.01 FedProx)             │
│    → TopKCompressor(rho_s=0.05) + Error Feedback        │
│    → quantize_tensor() → SparseINT8Payload              │
│    → deduct_battery(e_tx + e_comp)                      │
│    [Battery: 500J init, dies if ≤ 0J]                  │
└──────────────────────────────────────────────────────────┘
         ▲
         │  SlidingWindowDataset (window=10, stride=1)
         │  Dirichlet(α=0.1) Non-IID partition
         │  Real data: SMD (38-dim), SMAP (25-dim), MSL (55-dim)
         │  Synthetic fallback cho SMAP/MSL
```
