# Báo cáo Kỹ thuật Hệ thống FedKDL

> [!NOTE]
> Tài liệu này mô tả kiến trúc **sau lần tái cấu trúc lớn (Major Refactoring)** theo nguyên tắc OOP phân tầng và DRY.
> Phiên bản trước dùng hai lõi song song `hfl_core/` và `kdl_core/` — phiên bản này đã xoá cả hai và thay bằng `federated_core/` (lõi chung) + `tasks/` (các nhiệm vụ cụ thể).
> Cập nhật lần cuối: **2026-05-22**.

---

## 1. Tổng quan Kiến trúc

Hệ thống mô phỏng mạng **Internet of Underwater Things (IoUT)** theo kiến trúc HFL 3 tầng:

| Tầng | Thực thể | Vai trò |
|---|---|---|
| **Sensor** | AUV / Cảm biến ngầm | Thu dữ liệu, huấn luyện cục bộ, nén và truyền |
| **Fog** | Trạm sương mù | Tổng hợp nội cụm + hợp tác liên cụm |
| **Gateway** | Trạm mặt nước | Tổng hợp toàn cục (FedAvg global) |

Hệ thống hỗ trợ **hai kịch bản nghiên cứu** hoàn toàn tách biệt về logic AI, nhưng dùng chung toàn bộ lõi viễn thông và vật lý:

- **Kịch bản 1D** — Time-series Anomaly Detection: Autoencoder nhỏ, nén Top-K + INT8. Entry: `main_trainer.py`.
- **Kịch bản 2D** — Underwater Object Detection: YOLO26n + LoRA (INT8), partial Detect head, **Gateway-side KD** (Teacher YOLO12l). Entry: `main_trainer_od.py`.

---

## 2. Cấu trúc Mã nguồn (Directory Structure)

```text
FedKDL/
├── config/
│   └── settings.py               # Cấu hình trung tâm (Network, Acoustic, Energy, FedKDL)
│                                 # Các tham số phân chia "1D"/"2D" qua dict
│
├── physics_models/               # Lõi vật lý kênh sóng âm — KHÔNG SỬA
│   ├── communication.py          # Suy hao, nhiễu Wenz, SNR, Shannon capacity
│   ├── topology.py               # Đồ thị khả thi 3D Quasi-static
│   ├── energy.py                 # Công suất âm thanh, e_tx, e_comp_dynamic
│   └── latency.py                # comp_delay_dynamic, tau_round
│
├── federated_core/               # [NEW] Lõi Liên kết Học Liên Hợp — CHUNG cho 1D và 2D
│   ├── aggregator.py             # fedavg_intra_cluster, fedavg_global
│   ├── hfl_rules.py              # should_cooperate, find_coop_partner, blend_state_dicts
│   ├── metrics.py                # EnergyTracker, LatencyTracker, MetricsLogger
│   │                             # + anomaly_threshold, point_adjusted_f1, compute_round_latency
│   ├── workers.py                # BaseWorker, BaseFogNode, BaseGateway (base classes)
│   └── base_simulator.py         # BaseSimulator — Template Method Pattern
│
├── tasks/
│   ├── anomaly_1d/               # [NEW] Kịch bản 1D — Autoencoder + HFL
│   │   ├── autoencoder.py        # SmallAutoencoder, get_model_flat_params, ...
│   │   ├── dataloader.py         # Dataloader, cửa sổ trượt, phân chia Dirichlet
│   │   ├── trainer.py            # local_sgd cho time-series
│   │   ├── knowledge_compression/
│   │   │   ├── topk_sparsification.py  # TopKCompressor + Error Feedback
│   │   │   └── int8_quantization.py    # quantize, SparseINT8Payload, pack/unpack
│   │   └── simulator.py          # Simulator1D (kế thừa BaseSimulator)
│   │                             # + SensorWorker1D, FogNode1D
│   │
│   └── detection_2d/             # [NEW] Kịch bản 2D — Ultra-low payload FL
│       ├── models/
│       │   ├── lora.py           # LoRAConv2d, inject_lora (r=4 hoặc r=8)
│       │   └── yolo_wrapper.py   # trainable_state_dict: lora_ + cv3.x.2 only
│       ├── knowledge_compression/
│       │   ├── knowledge_distillation.py  # KDDetectionTrainer — chỉ Gateway
│       │   ├── int8_quantization.py       # pack_payload / unpack_payload (dense INT8)
│       │   ├── lazy_filter.py
│       │   ├── knowledge_association.py
│       │   └── concept_drift.py
│       ├── trainer.py            # local_sgd_od (không KD tại AUV)
│       └── simulator.py          # Simulator2D + _gateway_knowledge_distillation()
│
├── utils/
│   ├── env_manager.py            # Sinh & tải môi trường (Topology, DataPartition)
│   ├── generate_all_envs.py      # Script tự động sinh tất cả môi trường thực nghiệm
│   └── download_datasets.py      # Tải SMAP/MSL/SMD/URPC2020
│
├── main_trainer.py               # Entrypoint 1D: khởi tạo Simulator1D và chạy
└── main_trainer_od.py            # Entrypoint 2D: khởi tạo Simulator2D và chạy
```

---

## 3. Thiết kế OOP Phân tầng (Kiến trúc mới)

### 3.1. Template Method Pattern — `BaseSimulator`

`federated_core/base_simulator.py` định nghĩa **khung vòng lặp FL cố định**, các class con chỉ cần override các hook:

```text
BaseSimulator.run(T_rounds)
  ├── Phase 1: for each alive sensor → _process_sensor(s_id)   ← Override
  ├── Phase 2: for each fog → _aggregate_intra_fog(...)         ← Override
  │            → fog.cooperate() [dùng hfl_rules chung]
  ├── Phase 3: gateway.aggregate_global() [dùng aggregator chung]
  ├── Phase 3b: _gateway_knowledge_distillation()              ← Override (2D fedkdl)
  └── Phase 4: Logging → evaluate()                             ← Override
                       → _compute_payload_bits()                ← Override
                       → _compute_fog_model_bits()              ← Override
```

| Phương thức Override | `Simulator1D` | `Simulator2D` |
|---|---|---|
| `_process_sensor` | TopK + INT8 → `SparseINT8Payload` | `local_sgd_od` (no KD) → `pack_payload` INT8 |
| `_aggregate_intra_fog` | `fedavg_intra_cluster` từ delta | FedAvg sau `unpack_payload` |
| `_gateway_knowledge_distillation` | no-op | `KDDetectionTrainer` + Teacher (baseline `fedkdl`) |
| `evaluate` | `point_adjusted_f1` (PA-F1) | `evaluate_od` → mAP |
| `_compute_payload_bits` | `payload.payload_bits` trung bình | `len(bytes)*8` trung bình |

### 3.2. Chuỗi Kế thừa Workers

```
BaseWorker ──────────────┬─→ SensorWorker1D  (tasks/anomaly_1d/simulator.py)
                         └─→ SensorWorker2D  (tasks/detection_2d/simulator.py)

BaseFogNode ─────────────┬─→ FogNode1D       (tasks/anomaly_1d/simulator.py)
                         └─→ FogNode2D       (tasks/detection_2d/simulator.py)

BaseGateway  (dùng chung — không cần kế thừa thêm)
```

---

## 4. Logic Pipeline Chi tiết

### 4.1. Khởi tạo Môi trường (Decoupled Environment)

Môi trường được sinh **một lần duy nhất** (offline) qua `generate_all_envs.py` để đảm bảo công bằng giữa các thuật toán:

1. **Topology Generation** → `environments/topo/*.pkl`: Tọa độ 3D ngẫu nhiên → Feasibility Graph (SNR ≥ TARGET_SNR) → lưu `TopologySnapshot`.
2. **Data Partition** → `environments/data/*.pkl`: Đọc dataset → Dirichlet Non-IID split → lưu `DataPartitionSnapshot`.

### 4.2. Pipeline Kịch bản 1D (`Simulator1D`)

```
main_trainer.py
  └─ Simulator1D.run(T_rounds, baseline)
       ├─ SensorWorker1D.train_and_get_payload(global_state)
       │    ├─ local_sgd(model, dataloader, epochs)   [tasks/anomaly_1d/trainer.py]
       │    ├─ TopKCompressor.compress(Δθ)             [Error Feedback, rho_s=5%]
       │    └─ SparseINT8Payload(indices, values)      [INT8 quantize]
       ├─ FogNode1D.aggregate_intra_cluster()
       │    └─ fedavg_intra_cluster(delta_decompress)  [federated_core/aggregator.py]
       ├─ BaseFogNode.cooperate(rule)                  [hfl_rules.py]
       │    └─ should_cooperate / find_coop_partner / blend_state_dicts
       ├─ BaseGateway.aggregate_global()               [fedavg_global]
       └─ evaluate() → anomaly_threshold + point_adjusted_f1 (PA-F1)
```

**Chu trình nén hoàn chỉnh (1D):**
> $\Delta\theta$ (float32) → `TopKCompressor` (K = 5% params, + Error Feedback) → `SparseINT8Payload` → **~1.3 kbit** payload → Truyền qua kênh sóng âm → Fog decompress → FedAvg.

### 4.3. Pipeline Kịch bản 2D (`Simulator2D`) — Ultra-Low Payload (cập nhật 2026-05)

Kiến trúc **3 tầng** sau refactor 2D (xem `update_2dFL.md`):

| Tầng | Vai trò 2D | Ghi chú |
|---|---|---|
| **Tier 1 — Sensor (AUV)** | `local_sgd_od` thuần, **không KD** | Chỉ upload LoRA + lớp `cv3.x.2` / `one2one_cv3.x.2` |
| **Tier 2 — Fog** | FedAvg nội cụm + HFL liên cụm | Không chạy KD |
| **Tier 3 — Gateway** | FedAvg toàn cục + **Gateway-side KD** | Teacher `YOLO12l` trên proxy (`coco8.yaml`) |

```
main_trainer_od.py
  └─ Simulator2D.run(T_rounds, baseline)
       ├─ SensorWorker2D.train_and_get_payload(global_state)
       │    ├─ local_sgd_od(student, yaml)        [không Teacher tại AUV]
       │    └─ pack_payload(trainable_state_dict)  [INT8 dense, ~74KB (r=4) / ~146KB (r=8)]
       ├─ FogNode2D.aggregate_intra_cluster()
       │    ├─ unpack_payload(bytes) → state_dict
       │    └─ FedAvg mean trên LoRA + partial head
       ├─ BaseFogNode.cooperate(rule)              [hfl_rules.py]
       ├─ BaseGateway.aggregate_global()
       ├─ _gateway_knowledge_distillation()        [chỉ baseline fedkdl]
       │    └─ KDDetectionTrainer + Teacher tại Gateway
       └─ evaluate() → evaluate_od → mAP
```

**Payload (đo thực tế URPC, nc=4):**

| Thành phần | r=4 | r=8 |
|---|---|---|
| LoRA INT8 | ~72 KB | ~144 KB |
| `cv3.x.2` classifier INT8 | ~2 KB | ~2 KB |
| **Tổng / round** | **~74 KB** | **~146 KB** |

**Baselines 2D:** `baseline_od` (không KD Gateway), `fedkdl` (có Gateway KD).

**Export log cho scripts:** `main_trainer_od.py` → `results/logs_kdl/log_*.json` qua `utils/log_export.py` (keys: `map`, `alive`, `energy_cumul_J`, `avg_payload_kb`, `latency_history`, `energy_consumption`).

---

## 5. Mô hình Vật lý — Nguồn Chân lý

Toàn bộ công thức vật lý được tập trung tại `physics_models/`. **Không file nào trong `tasks/` hay `federated_core/` được phép hardcode hằng số vật lý.**

| Hàm | File | Phương trình |
|---|---|---|
| `path_loss` | `communication.py` | $PL = 20\log_{10}(d) + \alpha(f) \cdot d$ |
| `wenz_noise_db` | `communication.py` | Mô hình Wenz (4 thành phần) |
| `shannon_capacity` | `communication.py` | $C = B \log_2(1 + \text{SNR})$ |
| `e_tx` | `energy.py` | $E_{tx} = \frac{S}{R} \cdot P_{tx} + P_{c,tx}$ |
| `e_comp_dynamic` | `energy.py` | $E_{comp} = \epsilon_{op} \cdot F \cdot N \cdot E_{local}$ |
| `comp_delay_dynamic` | `latency.py` | $\tau_{comp} = \frac{F_{mul} \cdot FLOPs \cdot N}{f_{CPU}}$ |
| `compute_round_latency` | `metrics.py` | $\tau_{round} = \tau_{comp} + \tau_{s2f} + \tau_{f2f} + \tau_{f2g}$ |

---

## 6. Hướng dẫn Gỡ lỗi (Debugging Guide)

> [!WARNING]
> **ModuleNotFoundError sau Refactoring**
> Toàn bộ thư mục `hfl_core/` và `kdl_core/` đã bị xoá. Nếu gặp lỗi `No module named 'hfl_core'` hoặc `'kdl_core'`, kiểm tra ngay file đang báo lỗi và cập nhật import theo bảng dưới:
>
> | Import cũ (Đã xoá) | Import mới (Hiện tại) |
> |---|---|
> | `hfl_core.simulator` | `tasks.anomaly_1d.simulator` |
> | `hfl_core.algorithms.aggregator` | `federated_core.aggregator` |
> | `hfl_core.algorithms.hfl_rules` | `federated_core.hfl_rules` |
> | `hfl_core.models.autoencoder` | `tasks.anomaly_1d.autoencoder` |
> | `hfl_core.data.dataloader_1d` | `tasks.anomaly_1d.dataloader` |
> | `hfl_core.knowledge_compression.topk_sparsification` | `tasks.anomaly_1d.knowledge_compression.topk_sparsification` |
> | `kdl_core.simulator_od` | `tasks.detection_2d.simulator` |
> | `kdl_core.models.yolo_wrapper` | `tasks.detection_2d.models.yolo_wrapper` |
> | `kdl_core.algorithms.lazy_filter` | `tasks.detection_2d.knowledge_compression.lazy_filter` |
> | `scripts_baseline/*` | `scripts/hfl/*` (1D) hoặc `scripts/fedkdl/*` (2D) |

> [!CAUTION]
> **Shape Mismatch trong KDL 2D (Feature Distillation)**
> - **Vị trí:** `tasks/detection_2d/knowledge_compression/knowledge_distillation.py` (KDDetectionTrainer).
> - Teacher (YOLOv12l) và Student (YOLO26n) có số block khác nhau.
> - Hooks dùng `min(len(s_feats), len(t_feats))` để so khớp.
> - `adaptive_avg_pool2d` căn chỉnh (H, W); conv 1×1 căn chỉnh kênh C.
> - Nếu gặp `RuntimeError` liên quan `conv2d`, kiểm tra thứ tự lấy ra từ hooks.

> [!TIP]
> **Năng lượng tính sai đơn vị (Energy Unit Bug)**
> - `EPSILON_OP`, `MODEL_FLOPS_PER_SAMPLE`, `FLOP_MULTIPLIER` đều là **dict** với key `"1D"` / `"2D"`.
> - Khi gọi `e_comp_dynamic(...)`, luôn truyền `energy_cfg.EPSILON_OP[self.task_key]` chứ không truyền cả dict.
> - `self.task_key` được đặt trong constructor: `"1D"` cho `Simulator1D`, `"2D"` cho `Simulator2D`.

> [!NOTE]
> **Sensor bị cô lập (Isolated Sensor)**
> - **Vị trí:** `physics_models/topology.py` → `build_feasibility_graph`.
> - Xảy ra khi không tìm được link nào có SNR ≥ `TARGET_SNR` giữa sensor và fog.
> - Hệ thống sẽ bỏ qua sensor này (không có link trong `self.G`).
> - Cách gỡ: Tăng `N_FOGS`, giảm `MAX_DEPTH`, hoặc hạ `TARGET_SNR` trong `settings.py`.

> [!WARNING]
> **DataLoader rỗng (0 samples)**
> - Xảy ra khi Dirichlet phân rã tạo ra partition có 0 mẫu cho 1 sensor.
> - Hệ thống đã xử lý: `if len(idx_list) == 0: skip`. Nhưng nếu thêm dataset mới, luôn kiểm tra partition output trước khi build DataLoader.

---

## 7. Chạy thực nghiệm — Hai loại artifact (JSON + Stdout log)

Mỗi lần train (`main_trainer.py` / `main_trainer_od.py`, hoặc qua runner bash/ps1) **luôn ghi 2 file** cùng stem tên, qua `utils/train_io.py`:

| Loại | Mục đích | Thư mục mặc định | Đuôi file | Ai đọc |
|---|---|---|---|---|
| **JSON metrics** | Số liệu theo round → vẽ đồ thị | 1D: `results/logs/`<br>2D: `results/logs_kdl/` | `.json` | `scripts/hfl/plot_*.py`, `scripts/fedkdl/plot_*.py` |
| **Stdout log** | Console replay, debug, tư liệu báo cáo | 1D: `results/train_logs/hfl/`<br>2D: `results/train_logs/kdl/` | `.stdout.log` | Người vận hành / grep lỗi |

**Quy tắc đặt tên (cùng stem cho cặp JSON + log):**

- **1D:** `log_N{N}_{dataset}_a{alpha}_{baseline}_rho{rho}_seed{seed}`
- **2D:** `log_N{N}_{dataset}_a{alpha}_{baseline}_seed{seed}`

Ví dụ 2D:

```text
results/logs_kdl/log_N50_URPC_a0p1_fedkdl_seed42.json
results/train_logs/kdl/log_N50_URPC_a0p1_fedkdl_seed42.stdout.log
```

**Nội dung JSON** (`utils/log_export.build_experiment_bundle`):

```json
{
  "metadata": { "task", "baseline", "N", "dataset", "alpha", "seed", "artifacts": { "json_path", "stdout_log_path" } },
  "metrics": { "round": [...], "map" | "PA-F1": [...], "alive": [...], "energy_cumul_J": [...], ... },
  "energy_consumption": { "e_s2f", "e_f2f", "e_f2g", "e_comp", "cumulative_total" },
  "latency_history": { "round", "tau_round_s" }
}
```

**Nội dung stdout log:** mọi `print()` khi train (round metrics, `[Sensor …] Payload: … KB`, `[Gateway KD]`, lỗi Ultralytics, …) — mirror console qua `tee_stdout_to_file`.

**CLI:**

```bash
python main_trainer.py --topo ... --data ... --baseline hfl_selective \
  --out-dir results/logs --log-dir results/train_logs/hfl

python main_trainer_od.py --topo ... --data ... --baseline fedkdl \
  --out-dir results/logs_kdl --log-dir results/train_logs/kdl
```

Sau mỗi run, trainer in block `[Artifacts]` với đường dẫn tuyệt đối cả hai file.

| Runner grid | Bash | PowerShell |
|---|---|---|
| **Full server setup** | `quick_start.sh` | (dùng WSL hoặc từng bước tay) |
| 1D HFL | `run_hfl_experiments.sh` | `run_hfl_experiments.ps1` |
| 2D FedKDL | `run_kdl_experiments.sh` | `run_kdl_experiments.ps1` |

**`quick_start.sh`** (Linux server): venv → `pip install -r requirements.txt` → `download_datasets.py --all` → `generate_all_envs.py` → `run_hfl_experiments.sh` + `run_kdl_experiments.sh`. Log tổng: `results/quick_start/master.log`.

---

## 8. Kết luận và Điểm mới

Sau lần tái cấu trúc, hệ thống **FedKDL** đạt được:

1. **Tuân thủ DRY**: Logic viễn thông (pin, năng lượng, HFL cooperation) chỉ còn tồn tại ở **một nơi duy nhất** trong `federated_core/`.
2. **Template Method Pattern**: `BaseSimulator.run()` là khung cố định; `Simulator1D` và `Simulator2D` chỉ override 4 phương thức hook.
3. **Cấu trúc phẳng**: Thư mục chứa 1 file được kéo ra cùng cấp (ví dụ `tasks/anomaly_1d/autoencoder.py`).
4. **Import rõ ràng**: Không còn circular import hay import chéo giữa 2 lõi cũ.
5. **Mở rộng dễ dàng**: Thêm kịch bản 4D chỉ cần tạo `tasks/detection_4d/` và kế thừa `BaseSimulator`.
