# Báo cáo Hiện trạng Mã nguồn — FedKDL Repository

**Dự án:** FedHKL / FedKDL — Hierarchical Federated Learning for IoUT  
**Mục tiêu bài báo tái hiện:** Omeke et al. (2026) *"Energy-Efficient Hierarchical Federated Anomaly Detection for the Internet of Underwater Things via Selective Cooperative Aggregation"*  
**Ngôn ngữ lập trình:** Python 3.11  
**Deep Learning Framework:** PyTorch + Ultralytics YOLO  
**Cập nhật lần cuối:** 2026-05-21  
**Trạng thái:** Mã nguồn hoàn chỉnh, sẵn sàng chạy thực nghiệm

---

## 1. Cây Thư mục Hiện tại

```
FedKDL/
│
├── config/
│   └── settings.py              # Dataclass configs: NetworkConfig, AcousticChannelConfig,
│                                #   EnergyConfig, FedKDLConfig — singleton, import toàn cục
│
├── physics_models/              # Mô hình vật lý âm học dưới nước (KHÔNG thay đổi)
│   ├── topology.py              # Topology3D, build_feasibility_graph,
│   │                            #   nearest_feasible_association, flat_topology_association
│   ├── communication.py         # Thorp-Wenz TL model, Shannon capacity, feasibility check
│   ├── energy.py                # e_tx, e_rx, e_comp_simple — Eq. 20-27 paper
│   └── latency.py               # comm_delay, comp_delay_simple — Eq. 21 paper
│
├── utils/                       # Tiện ích dùng chung
│   ├── env_manager.py           # EnvironmentManager: sinh/lưu/tải Topology & DataPartition
│   │                            #   [NEW] Cấu trúc path: topo/N_{N}/ và data/{DS}/N_{N}/
│   ├── generate_all_envs.py     # Script sinh 12 topo × 4 datasets × 2 alpha = 96 file pkl
│   ├── plot_styles.py           # setup_global_plot_style(), get_style() — dùng chung cho plot
│   └── download_datasets.py     # Helper tải SMD/SMAP/MSL/URPC về datasets/
│
├── environments/                # [RESTRUCTURED] File pkl môi trường pre-generated
│   ├── topo/
│   │   ├── N_50/  topo_N50_seed{42,123,2024}.pkl
│   │   ├── N_100/ topo_N100_seed{42,123,2024}.pkl
│   │   ├── N_150/ topo_N150_seed{42,123,2024}.pkl
│   │   └── N_200/ topo_N200_seed{42,123,2024}.pkl
│   └── data/
│       ├── SMD/  N_50/ … N_200/   data_N*_SMD_a{0p1,10000p0}_seed*.pkl
│       ├── SMAP/ N_50/ … N_200/   data_N*_SMAP_a{0p1,10000p0}_seed*.pkl
│       ├── MSL/  N_50/ … N_200/   data_N*_MSL_a{0p1,10000p0}_seed*.pkl
│       └── URPC/ N_50/ … N_200/   data_N*_URPC_a{0p1,10000p0}_seed*.pkl
│                                  (mỗi Dataset × 2 alpha × 3 seed = 6 file/N = 24 file/DS)
│
├── hfl_core/                    # Kịch bản 1: HFL Anomaly Detection (stable, reference impl.)
│   ├── simulator.py             # Scenario1Simulator: vòng lặp chính,
│   │                            #   ThreadPoolExecutor(max_workers=8), gc.collect()
│   ├── metrics.py               # EnergyTracker, LatencyTracker, MetricsLogger, PA-F1
│   ├── algorithms/
│   │   ├── local_trainer.py     # local_sgd() → (Δθ, avg_loss), FedProx proximal term
│   │   ├── worker.py            # SensorWorker, FogNode (gc.collect()), SurfaceGateway
│   │   ├── aggregator.py        # fedavg_intra_cluster (Eq. 40), fedavg_global (Eq. 43)
│   │   └── hfl_rules.py         # should_cooperate (Eq. 41), find_coop_partner, blend_state_dicts
│   ├── knowledge_compression/
│   │   ├── topk_sparsification.py  # TopKCompressor: Top-K + Error Feedback, rho_s=0.05
│   │   └── int8_quantization.py    # quantize_tensor, SparseINT8Payload, pack_payload
│   ├── models/
│   │   └── autoencoder.py       # SmallAutoencoder: D→32→16→8→16→32→D (~1350 params)
│   └── data/
│       └── dataloader_1d.py     # SlidingWindowDataset, non_iid_partition (Dirichlet), load_real_smd
│
├── kdl_core/                    # Kịch bản 2 & 3: FedKDL Object Detection (YOLO)
│   │                            #   [SYNCED] Mọi bugfix từ hfl_core đã được áp dụng
│   ├── simulator.py             # [SYNCED] Kịch bản 1 bản kdl — giống hfl_core/simulator.py,
│   │                            #   chỉ khác prefix import (kdl_core.*)
│   ├── simulator_od.py          # [UPDATED] Kịch bản 2 & 3: ODSimulator
│   │                            #   ✓ ThreadPoolExecutor(max_workers=2) cho YOLO
│   │                            #   ✓ gc.collect() + cuda.empty_cache() sau mỗi round
│   │                            #   ✓ e_f2f và e_f2g được tính chính xác
│   │                            #   ✓ Tải Topology & DataPartition từ EnvironmentManager
│   ├── algorithms/
│   │   ├── local_trainer.py     # [SYNCED] local_sgd() — giống hfl_core
│   │   ├── local_trainer_od.py  # local_sgd_od() — YOLO training loop, evaluate_od()
│   │   ├── worker.py            # [SYNCED] gc.collect() — giống hfl_core
│   │   ├── aggregator.py        # fedavg_intra_cluster, fedavg_global
│   │   ├── hfl_rules.py         # should_cooperate, find_coop_partner, blend_state_dicts
│   │   ├── concept_drift.py     # ConceptDriftMonitor (theo dõi drift norm per sensor)
│   │   ├── knowledge_association.py  # Mapping student↔teacher layer
│   │   └── lazy_filter.py       # Bộ lọc lazy sensor (bỏ qua update nhỏ)
│   ├── knowledge_compression/
│   │   ├── topk_sparsification.py  # [SYNCED]
│   │   ├── int8_quantization.py    # [SYNCED] + pack_payload dùng cho OD state dict
│   │   └── knowledge_distillation.py  # KD loss: cross-entropy với teacher soft labels
│   ├── models/
│   │   ├── autoencoder.py       # [SYNCED] SmallAutoencoder
│   │   ├── lora.py              # LoRA adapter: thêm rank-r perturbation vào YOLO layers
│   │   └── yolo_wrapper.py      # StudentModel (YOLO26n + LoRA), TeacherModel (YOLO12l)
│   └── data/
│       ├── dataloader_1d.py     # [SYNCED] — giống hfl_core
│       └── dataloader_2d.py     # create_client_datasets_yolo() cho URPC YOLO format
│
├── scripts/
│   ├── hfl/                     # Plot-only scripts cho Kịch bản 1 (HFL)
│   │   ├── plot_convergence.py  # Fig. 4: Loss curve N=150,200 — shaded std
│   │   ├── plot_scalability.py  # Fig. 5: Participation/Energy/Latency vs N
│   │   ├── plot_heterogeneity.py # Fig. 7: alpha=0.1 vs 10000 — Bar charts
│   │   └── plot_real_benchmark.py # Fig. 8: SMD/SMAP/MSL PA-F1 + Energy
│   └── fedkdl/                  # Plot-only scripts cho Kịch bản 2 & 3 (OD)
│       ├── plot_od_comparison.py  # Trace so sánh baseline_od vs fedkdl (mAP, alive, energy)
│       ├── plot_od_scalability.py # Bar chart mAP/alive/energy vs N cho Scenario 3
│       ├── plot_heterogeneity.py  # Bar chart alpha heterogeneity cho OD
│       └── eval_baselines.py     # Print bảng tổng kết từ results/logs_kdl/*.json
│
├── datasets/                    # Dữ liệu thực tế
│   ├── SMD/train/, test/, test_label/   # Server Machine Dataset (38 features)
│   ├── SMAP/ & MSL/             # NASA Telemetry (25 & 55 features)
│   └── URPC2020/                # URPC2020 dataset (images + labels, YOLO format)
│       └── URPC2020.yaml        # Config YOLO: train/val/test paths, classes
│
├── results/                     # Output (tự động tạo)
│   ├── logs/                    # HFL JSON logs: log_N{N}_{DS}_a{alpha}_{bl}_rho{r}_seed{s}.json
│   ├── logs_kdl/                # KDL JSON logs: log_N{N}_{DS}_a{alpha}_{bl}_seed{s}.json
│   ├── convergence/             # fig4_convergence.png (✓ có sẵn)
│   ├── real_benchmarks/         # fig8_real_benchmarks.png
│   ├── scalability/             # fig5_*.png
│   ├── heterogeneity/           # fig7_heterogeneity.png
│   └── scenario3/               # fedkdl_comparison.png, fedkdl_scalability.png
│
├── main_trainer.py              # CLI entry-point cho HFL: --topo --data --baseline --rounds
├── main_trainer_od.py           # CLI entry-point cho OD: --topo --data --baseline --rounds
│
├── run_hfl_experiments.ps1      # [UPDATED] PowerShell: train toàn bộ HFL → gọi 4 plot scripts
├── run_kdl_experiments.ps1      # [UPDATED] PowerShell: train toàn bộ KDL → gọi plot scripts
│
├── requirements.txt             # torch, numpy, pandas, matplotlib, scikit-learn, ultralytics
├── TODO.md                      # Backlog nợ kỹ thuật
└── fix_sync.py                  # [Temp] Script tạo khi fix encoding — có thể xóa
```

---

## 2. Kiến trúc Tách biệt Topology / Data

Từ Phase 5, Topology và Data Partition được tách hoàn toàn và **pre-generated** một lần duy nhất:

```
generate_all_envs.py  →  EnvironmentManager.generate_topology()  →  environments/topo/N_{N}/
                      →  EnvironmentManager.generate_data_partition()  →  environments/data/{DS}/N_{N}/
```

Simulator chỉ `load()` từ file, không bao giờ tính lại:

```python
topo = EnvironmentManager.load_topology("environments/topo/N_50/topo_N50_seed42.pkl")
data = EnvironmentManager.load_data_partition("environments/data/SMD/N_50/data_N50_SMD_a0p1_seed42.pkl")
sim  = Scenario1Simulator(topo, data, baseline="hfl_selective")
```

**Cấu trúc path API:**
```python
EnvironmentManager.topo_path(N=50, seed=42)
  → environments/topo/N_50/topo_N50_seed42.pkl

EnvironmentManager.data_path(N=50, dataset="SMD", alpha=0.1, seed=42)
  → environments/data/SMD/N_50/data_N50_SMD_a0p1_seed42.pkl
```

---

## 3. Lõi Mô phỏng

### 3.1 Luồng thực thi mỗi Round (chung cho HFL & KDL Scenario 1)

```
for t in range(T_rounds):
  │
  ├─ Phase 1: Local Training + Compression [ThreadPoolExecutor, max_workers=8]
  │    └─ sensor.train_and_compress()
  │         ├─ local_sgd(epochs=E, lr=0.01, mu=0.01 nếu FedProx)  → (Δθ, avg_loss)
  │         ├─ TopKCompressor.compress(Δθ)  → Error Feedback → (topk_indices, topk_values)
  │         └─ SparseINT8Payload(indices, quantize(values))
  │
  ├─ Phase 2: Fog Aggregation & HFL Cooperation [Sequential]
  │    ├─ fog.aggregate_intra_cluster()  → fedavg_intra_cluster (Eq. 40)
  │    └─ fog.cooperate(rule)
  │         ├─ should_cooperate(c_m, c̄) [Eq. 41]
  │         ├─ find_coop_partner(fog_id, cluster_sizes, G, q1_distance)
  │         ├─ blend_state_dicts(self, neighbor, alpha=0.8) [Eq. 42]
  │         └─ gc.collect()  ← memory cleanup
  │
  ├─ Phase 3: Global Aggregation [Gateway]
  │    └─ gateway.aggregate_global()  → fedavg_global (Eq. 43)
  │
  └─ Phase 4: Logging & Evaluation
       ├─ EnergyTracker.add_round(e_s2f, e_f2f, e_f2g, e_comp)
       ├─ LatencyTracker.compute_round_latency()
       ├─ evaluate_global_model()  → PA-F1, Precision, Recall
       └─ Ghi JSON: results/logs/log_*.json
```

### 3.2 Luồng thực thi mỗi Round (ODSimulator — Kịch bản 2 & 3)

```
for rnd in range(GLOBAL_ROUNDS):
  │
  ├─ Phase 1: YOLO Local Training [ThreadPoolExecutor, max_workers=2]
  │    └─ local_sgd_od(student, teacher, client_yaml)
  │         ├─ Kịch bản 2 (baseline_od): kd_lambda=0.0 → chỉ train thuần tuý
  │         └─ Kịch bản 3 (fedkdl):      kd_lambda=1.0 → KD loss với teacher
  │
  ├─ Phase 2: Aggregation (giống HFL nhưng state_dict là LoRA weights)
  │
  ├─ Phase 3: gc.collect() + cuda.empty_cache()  ← bắt buộc sau YOLO
  │
  └─ Phase 4: evaluate_od() → mAP@0.5:0.95
       └─ Ghi JSON: results/logs_kdl/log_*.json
```

### 3.3 Các Baselines được hỗ trợ

**Kịch bản 1 (HFL — 1D Anomaly Detection):**

| Baseline | Association | Cooperation | Đặc điểm |
|---|---|---|---|
| `centralised` | N/A | N/A | Oracle — chuẩn trên |
| `fedavg` | flat → gateway | None | ~20-30% tham gia (AUV không lên surface được) |
| `fedprox` | flat → gateway | None | μ=0.01 proximal term chống client drift |
| `hfl_nocoop` | HFL → fog gần nhất | None | ~100% tham gia qua fog relay |
| `hfl_nearest` | HFL | Always-on, α=0.7 | Mọi fog đều hợp tác nếu có link |
| `hfl_selective` | HFL | Eq.41 + Eq.42, α=0.8 | Chỉ fog "đói tri thức" hợp tác |

**Kịch bản 2 & 3 (OD — Object Detection):**

| Baseline | Mô tả |
|---|---|
| `baseline_od` | YOLO không nén, flat aggregation — bottleneck bandwidth |
| `fedkdl` | YOLO + KD-LoRA-INT8 + HFL-Selective — FedKDL đầy đủ |

---

## 4. Automation Pipeline

Toàn bộ thực nghiệm được điều phối qua 2 file PowerShell:

```
.\run_hfl_experiments.ps1         .\run_kdl_experiments.ps1
│                                  │
├─ Loop: N × DS × alpha × seed × baseline   (Kịch bản 1 / 2&3)
├─ Check skip nếu log JSON đã tồn tại        (resumable)
├─ Check warning nếu pkl env bị thiếu
├─ Gọi: python main_trainer.py --topo --data --baseline ...
│         └─ Ghi results/logs/*.json         results/logs_kdl/*.json
│
└─ Sau khi train xong → tự động gọi plot scripts:
   HFL: plot_convergence, plot_scalability, plot_heterogeneity, plot_real_benchmark
   KDL: plot_od_comparison, plot_od_scalability
```

**Tổng số runs cho HFL:** 4N × 3DS × 2alpha × 3seed × 5baselines = **360 runs**  
**Tổng số runs cho KDL:** 4N × 1DS × 2alpha × 3seed × 2baselines = **48 runs**

---

## 5. Điểm Mù & Nợ Kỹ thuật Còn lại

### ✅ Đã giải quyết (so với report cũ)
- ~~Không persist topology~~ → **EnvironmentManager** với pkl pre-generated
- ~~ThreadPoolExecutor bên trong vòng lặp~~ → **đã chuyển ra ngoài**
- ~~Memory leak FogNode~~ → **gc.collect() sau cooperate()**
- ~~Plot bị trộn với run logic~~ → **scripts/hfl/ và scripts/fedkdl/ chỉ plot-only**

### ⚠️ Còn tồn đọng

**1. `ultralytics` chưa cài** — `kdl_core` yêu cầu `ultralytics>=8.3.0` nhưng môi trường chưa xác nhận. Chạy `pip install ultralytics` trước khi dùng `run_kdl_experiments.ps1`.

**2. `yolo_wrapper.py` dùng tên checkpoint hardcode** — `yolo26n.pt` và `yolo12l.pt` phải tồn tại trong thư mục gốc. Nếu chưa có, YOLO sẽ tự download từ Ultralytics Hub.

**3. `load_dataset` chỉ tải thật cho SMD** — SMAP và MSL vẫn có thể dùng synthetic fallback nếu file không tồn tại trong `datasets/SMAP/` hay `datasets/MSL/`.

**4. Không có checkpoint/resume cho training mid-round** — Nếu bị ngắt giữa chừng trong 1 run, toàn bộ round đó phải chạy lại. Tuy nhiên, cơ chế skip-by-log-file đã đảm bảo không bị chạy lại các run đã hoàn thành.

**5. `fix_sync.py`** — File temp sinh ra khi fix encoding, không cần thiết. Có thể xóa.

---

## 6. Kết quả Có sẵn

| File | Nội dung |
|---|---|
| `results/convergence/fig4_convergence.png` | ✅ Đã có |
| `results/logs/log_N50_SMD_a0p1_fedavg_rho0p05_seed42.json` | ✅ Đã có |
| `results/logs/log_N50_SMD_a0p1_hfl_selective_rho0p05_seed42.json` | ✅ Đã có |
| Còn lại (360 - 2 = 358 HFL runs, 48 KDL runs) | ⏳ Chờ chạy |
