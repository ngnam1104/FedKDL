# Báo Cáo Kiến Trúc Và Hướng Tái Cấu Trúc Repo FedKDL

Tài liệu này được viết để bạn có một “bản đồ repo” dễ đọc hơn khi debug, chạy thí nghiệm, hoặc chuẩn bị refactor. Trọng tâm hiện tại của repo là **object detection 2D trên URPC2020** với YOLO + LoRA + INT8 + HFL + Gateway KD/Proxy-FT. Phần bài toán 1D đã gần như không còn là nhánh chính, nhưng vẫn để lại dấu vết trong config, base class, metrics và một số thư mục môi trường.

> Cập nhật gần nhất: **2026-07-07**. Runner hiện hành cho nhóm FedKDL/KD là
> `run_fedkdl.sh`, `run_kd_logit.sh` và `run_kd_logit_proj.sh`.
> Package detection hiện nằm ở `detection_2d/` tại root repo; `tasks/` không còn là package runtime.

## 1. Đọc Nhanh Trong 5 Phút

Nếu chỉ cần hiểu repo chạy thế nào, đọc theo thứ tự này:

1. [detection_2d/baselines.py](D:/Documents/HUST/2022-2026/Research_Thesis/FedKDL/detection_2d/baselines.py): registry ánh xạ tên baseline sang các cờ thuật toán.
2. [main_trainer_od.py](D:/Documents/HUST/2022-2026/Research_Thesis/FedKDL/main_trainer_od.py): entrypoint chạy thí nghiệm.
3. [detection_2d/simulator.py](D:/Documents/HUST/2022-2026/Research_Thesis/FedKDL/detection_2d/simulator.py): logic 2D chính: AUV train, Relay aggregate, Gateway KD/proxy-FT.
4. [federated_core/base_simulator.py](D:/Documents/HUST/2022-2026/Research_Thesis/FedKDL/federated_core/base_simulator.py): vòng lặp FL tổng quát.
5. [detection_2d/trainer.py](D:/Documents/HUST/2022-2026/Research_Thesis/FedKDL/detection_2d/trainer.py): train local YOLO/LoRA và evaluate.
6. [detection_2d/knowledge_compression/knowledge_distillation.py](D:/Documents/HUST/2022-2026/Research_Thesis/FedKDL/detection_2d/knowledge_compression/knowledge_distillation.py): loss và trainer Gateway KD.
7. [detection_2d/models/yolo_wrapper.py](D:/Documents/HUST/2022-2026/Research_Thesis/FedKDL/detection_2d/models/yolo_wrapper.py): StudentModel, TeacherModel, payload keys.
8. [detection_2d/knowledge_compression/int8_quantization.py](D:/Documents/HUST/2022-2026/Research_Thesis/FedKDL/detection_2d/knowledge_compression/int8_quantization.py): pack/unpack INT8 payload.
9. [federated_core/aggregator.py](D:/Documents/HUST/2022-2026/Research_Thesis/FedKDL/federated_core/aggregator.py): FedAvg và SVD-LoRA aggregation.
10. [detection_2d/compat.py](D:/Documents/HUST/2022-2026/Research_Thesis/FedKDL/detection_2d/compat.py): shim để load checkpoint cũ từng lưu dưới namespace `tasks.detection_2d.*`.

Ý chính:

- `main_trainer_od.py` chỉ chuẩn bị args, checkpoint, simulator, artifact.
- `Simulator2D` mới là “bộ não” của bài toán 2D.
- `BaseSimulator.run()` vẫn giữ vòng lặp FL, nhưng đây là abstraction cũ từ thời còn 1D/2D song song.
- `detection_2d.compat` là ngoại lệ có chủ đích: nó giữ alias `tasks.detection_2d.*` để unpickle checkpoint cũ sau khi package đã được chuyển ra root.
- Nếu refactor, hướng hợp lý là kéo `run()` về `Simulator2D` và làm phẳng inheritance.

## 2. Sơ Đồ Luồng Chạy

```text
main_trainer_od.py
  |
  |-- parse args: topo, data, baseline, rounds
  |-- ensure_warmup_checkpoints()
  |-- tạo Simulator2D
  |
  `-- sim.run()
        |
        |-- Phase 1: AUV local training
        |      AUVWorker2D.train_and_get_payload()
        |      -> local_sgd_od()
        |      -> pack payload INT8 / Top-K / Float32
        |
        |-- Phase 2: Relay tier
        |      RelayNode2D.aggregate_intra_cluster()
        |      -> unpack payload
        |      -> SVD-LoRA hoặc naive FedAvg
        |      -> optional relay cooperation
        |
        |-- Phase 3: Gateway aggregation
        |      BaseGateway.aggregate_global()
        |      -> fedavg_global()
        |      -> server_mix nếu baseline bật
        |
        |-- Phase 3b: Gateway update
        |      nếu fedkdl: Gateway KD
        |      nếu fedkdl_proxy_ft: supervised proxy-FT
        |
        |-- Phase 4: evaluate + log metrics
        |
        `-- Mobility/re-clustering cho round sau
```

Ba runner mới chỉ là lớp orchestration mỏng đứng trước entrypoint:

```text
run_fedkdl.sh / run_kd_logit.sh / run_kd_logit_proj.sh
  -> kiểm tra topology + data partition
  -> chọn GPU bằng CUDA_VISIBLE_DEVICES
  -> gọi main_trainer_od.py với baseline tương ứng
  -> utils/train_io.py ghi JSON metrics + stdout log
```

## 3. Cấu Trúc Thư Mục Hiện Tại

Sau refactor package, `tasks/` đã được bỏ khỏi đường runtime. Module detection nằm trực tiếp ở root tại `detection_2d/`; mọi import mới nên dùng `detection_2d...`, không dùng `tasks.detection_2d...` nữa. Cây dưới đây cũng bổ sung các phần hiện có nhưng report cũ chưa phản ánh: `utils/image_payload.py`, `demo/live_jobs.py`, và `server/entrypoint.sh`.

```text
FedKDL/
|-- config/
|   `-- settings.py
|-- detection_2d/
|   |-- baselines.py
|   |-- compat.py
|   |-- simulator.py
|   |-- trainer.py
|   |-- models/
|   |   |-- yolo_wrapper.py
|   |   `-- lora.py
|   `-- knowledge_compression/
|       |-- int8_quantization.py
|       |-- knowledge_distillation.py
|       |-- knowledge_association.py
|       |-- lazy_filter.py
|       `-- topk_sparsification.py
|-- federated_core/
|   |-- base_simulator.py
|   |-- workers.py
|   |-- aggregator.py
|   |-- hfl_rules.py
|   `-- metrics.py
|-- physics_models/
|   |-- topology.py
|   |-- communication.py
|   |-- latency.py
|   `-- energy.py
|-- utils/
|   |-- download_datasets.py
|   |-- env_manager.py
|   |-- generate_all_envs.py
|   |-- image_payload.py
|   |-- kaggle_auth.py
|   |-- plot_styles.py
|   |-- train_io.py
|   `-- log_export.py
|-- scripts/
|   |-- calc_all.py
|   |-- fedkdl/
|   |   |-- train_student_warmup.py
|   |   |-- train_teacher_lora.py
|   |   |-- bake_teacher_lora.py
|   |   |-- eval_baselines.py
|   |   |-- measure_payloads.py
|   |   |-- payload_breakdown.py
|   |   |-- summarize_multiseed.py
|   |   `-- plot/
|   `-- archive_tests/
|-- demo/
|   |-- app.py
|   |-- live_jobs.py
|   `-- static/
|-- server/
|   `-- entrypoint.sh
|-- run_fedkdl.sh
|-- run_kd_logit.sh
|-- run_kd_logit_proj.sh
|-- main_trainer_od.py
`-- README.md
```

### Kiểm tra path sau refactor

`tasks/` không còn là package code chính. Nếu quét chuỗi cũ, match hợp lệ hiện tại chỉ nên nằm ở shim tương thích checkpoint hoặc comment giải thích shim:

```bash
rg "tasks\\.detection_2d" -g "*.py"
```

## 4. Các Khái Niệm Cần Nắm

### Global state

`gateway.global_state_dict` là trọng số hiện tại của global model. Với LoRA baseline, state này không nhất thiết là full YOLO model, mà là tập tensor được truyền trong FL: LoRA A/B, head, BN/running stats.

### Uplink và downlink payload

- **Uplink**: AUV gửi lên Relay/Gateway. Để tiết kiệm năng lượng, code chỉ gửi một phần head.
- **Downlink**: Gateway gửi state hiện hành xuống AUV theo contract payload của `StudentModel`.

Quyết định tensor nào được gửi nằm ở:

```python
StudentModel._is_payload_key(k, downlink=False/True)
```

### INT8 delta payload

Mặc định AUV không gửi absolute weight, mà gửi delta:

```text
delta = local_state - global_state
```

Sau đó delta được lượng tử INT8. Relay giải nén bằng đúng template global state của link đó.

### SVD-LoRA aggregation

Không average trực tiếp `lora_A` và `lora_B`, vì như vậy dễ tạo cross-term sai. Code làm:

```text
W_i = B_i @ A_i
W_avg = weighted_avg(W_i)
W_avg ≈ B_new @ A_new  bằng truncated SVD
```

File chính: [federated_core/aggregator.py](D:/Documents/HUST/2022-2026/Research_Thesis/FedKDL/federated_core/aggregator.py).

## 5. Entry Point

### `main_trainer_od.py`

#### `NumpyEncoder`

Dùng khi lưu JSON metrics. Nó đổi kiểu numpy sang kiểu JSON native:

- `np.integer` -> `int`
- `np.floating` -> `float`
- `np.ndarray` -> `list`

#### `parse_args()`

Đọc CLI:

- `--topo`: file topology `.pkl`
- `--data`: file partition `.pkl`
- `--baseline`: tên baseline trong `BASELINE_CONFIGS`
- `--rounds`: override số round
- `--lora-rank`: override LoRA rank
- `--out-dir`, `--log-dir`: nơi lưu artifact

#### `main()`

Các việc chính:

1. Kiểm tra topo/data tồn tại.
2. Override config nếu CLI truyền `--rounds` hoặc `--lora-rank`.
3. Parse metadata từ tên file data: `N`, dataset, alpha, seed.
4. Tạo `ExperimentPaths`.
5. Gọi `_train()` bên trong `run_trainer_with_artifacts()`.

Trước khi load checkpoint, file này import [detection_2d/compat.py](D:/Documents/HUST/2022-2026/Research_Thesis/FedKDL/detection_2d/compat.py) bằng side effect. Đây là để các checkpoint cũ từng pickle class dưới path `tasks.detection_2d.*` vẫn load được sau khi package đã chuyển thành `detection_2d.*`.

#### `_train()`

Đây là phần thực sự khởi tạo thí nghiệm:

1. Lấy `baseline_cfg`.
2. Chuẩn bị warmup checkpoints bằng `ensure_warmup_checkpoints`.
3. Chọn checkpoint student:
   - LoRA baseline dùng `yolo12n_warmup.pt`.
   - Full/head-only baseline dùng `yolo12n_head_warmup.pt`.
4. Tạo `Simulator2D`.
5. Nếu baseline là `centralized`, chạy flow centralized riêng.
6. Nếu không, gọi:

```python
history = sim.run(T_rounds=T_rounds, baseline=args.baseline)
```

Điểm cần refactor:

- Logic centralized quá dài, nên tách ra file riêng.
- Parse metadata từ filename hơi fragile, nên tách helper.

## 6. Config

File: [config/settings.py](D:/Documents/HUST/2022-2026/Research_Thesis/FedKDL/config/settings.py)

### `NetworkConfig`

Chứa topology:

- số AUV/Relay
- kích thước vùng mô phỏng
- độ sâu AUV/Relay
- tham số Gauss-Markov mobility

Dấu vết 1D còn ở:

- `M_RELAYS_1D`
- `GLOBAL_ROUNDS["1D"]` trong `FedKDLConfig`

### `AcousticChannelConfig`

Chứa tham số kênh âm:

- sound speed
- carrier frequency
- bandwidth
- target SNR
- source level max
- spreading factor
- Wenz noise parameters

### `EnergyConfig`

Chứa tham số năng lượng:

- pin AUV/Relay
- ngưỡng pin
- năng lượng tính toán
- CPU frequency
- TX/RX circuit power
- hiệu suất điện-âm

Hiện tại `E_INIT` và `RELAY_E_INIT` đang là `inf` để chạy đủ round, nhưng code vẫn có deduct pin và check threshold.

### `FedKDLConfig`

Đây là config lớn nhất, gồm:

- local training
- LoRA rank
- INT8 payload
- HFL cooperation
- EMD association
- KD, gồm logit/box KD và LoRA projection KD (`KD_PROJ_MODE`, `KD_PROJ_ANCHOR_MATCH`)
- proxy-FT
- joint objective cost

Cách đọc hiện tại: `settings.py` vẫn gom nhiều nhóm cấu hình trong một file để các runner, simulator và script phụ cùng dùng chung singleton config. Một số key `"1D"` còn xuất hiện vì `BaseSimulator` và metrics vẫn giữ lớp tương thích cũ, nhưng đường chạy chính của report này là 2D.

## 7. Environment Và Data Partition

File: [utils/env_manager.py](D:/Documents/HUST/2022-2026/Research_Thesis/FedKDL/utils/env_manager.py)

Ngoài `env_manager.py`, repo hiện có [utils/image_payload.py](D:/Documents/HUST/2022-2026/Research_Thesis/FedKDL/utils/image_payload.py). Helper này liệt kê ảnh duy nhất và tính encoded image bytes theo owner/AUV. `main_trainer_od.py` dùng nó để tính payload raw-data cho centralized baseline; `detection_2d/simulator.py` dùng nó khi dựng danh sách ảnh/proxy/AUV YAML để tránh duplicate file ảnh.

[utils/download_datasets.py](D:/Documents/HUST/2022-2026/Research_Thesis/FedKDL/utils/download_datasets.py) là script tải dataset. Đường hiện dùng là `--urpc` qua Kaggle/kagglehub; các hàm SMD/SMAP/MSL còn tồn tại để tương thích lịch sử nhưng không nằm trong luồng 2D chính.

[utils/generate_all_envs.py](D:/Documents/HUST/2022-2026/Research_Thesis/FedKDL/utils/generate_all_envs.py) là CLI sinh topology/data partition. Script hiện ưu tiên URPC/2D, hỗ trợ `--n`, `--n-list`, `--m-relays`, `--alphas`, `--seeds`, `--topology-view`, `--topology-only`, `--force-topo` và `--find-gateway-seed`.

### `TopologySnapshot`

Lưu trạng thái topology đã sinh:

- số AUV/Relay
- seed
- vị trí AUV/Relay/Gateway
- feasibility graph
- association cho HFL
- association cho flat baseline
- clusters
- `topology_view`: `shared`, `flat`, hoặc `hfl`

### `DataPartitionSnapshot`

Lưu cách chia data:

- dataset name
- N, alpha, seed
- `auv_data_indices`: AUV nào nhận ảnh nào
- `public_data_indices`: proxy/public data cho Gateway KD/proxy-FT
- `log_text`: mô tả partition để debug

### `EnvironmentManager.generate_topology()`

Tạo topology:

1. Tạo `Topology3D`.
2. Build feasibility graph.
3. Tạo HFL association.
4. Tạo flat association.
5. Build clusters.
6. Ghi nhận `topology_view` để có thể lưu shared topology hoặc view riêng cho flat/HFL.
7. Đóng gói thành `TopologySnapshot`.

### `EnvironmentManager.generate_data_partition_2d()`

Đây là phần chia URPC data:

1. Đọc YAML dataset.
2. Tìm toàn bộ ảnh train.
3. Đọc label YOLO.
4. Gom ảnh theo habitat/class.
5. Tách 15% public/proxy data.
6. Dựa vào độ sâu AUV để tạo affinity với habitat.
7. Dùng Dirichlet alpha để tạo non-IID quantity skew.
8. Trả về `DataPartitionSnapshot`.

Ngoài ra, `shrink_image_pool()` dùng cho dry/smoke test: nó remap pool ảnh xuống kích thước nhỏ hơn nhưng vẫn giữ public split và mapping AUV hợp lệ.

Điểm cần refactor:

- Hàm này dài và nhiều trách nhiệm. Nên tách thành:
  - `resolve_yolo_images()`
  - `read_label_buckets()`
  - `split_proxy_public_data()`
  - `assign_auv_images_by_depth()`

## 8. Physics Models

### `physics_models/topology.py`

#### `LinkInfo`

Lưu thông tin một link khả thi:

- distance
- required source level
- transmission loss
- noise level
- data rate

#### `Topology3D`

Tạo và giữ vị trí:

- AUV
- Relay
- Gateway

Nó cũng giữ trạng thái mobility:

- speed
- heading
- pitch

#### `Topology3D.step_mobile_auvs()`

Cập nhật vị trí AUV theo Gauss-Markov:

```text
speed[t+1]   = μ speed[t]   + (1-μ) mean_speed   + noise
heading[t+1] = μ heading[t] + (1-μ) mean_heading + noise
pitch[t+1]   = μ pitch[t]   + (1-μ) mean_pitch   + noise
```

Sau đó đổi sang delta x/y/z và reflect nếu vượt biên.

#### `build_feasibility_graph()`

Tạo graph link khả thi cho:

- AUV -> Relay
- Relay -> Relay
- Relay -> Gateway
- AUV -> Gateway

Một link khả thi nếu source level cần thiết không vượt `SL_MAX`.

#### `nearest_feasible_association()`

Gán AUV vào relay khả thi gần nhất, có cân bằng tải nhẹ:

```python
max_capacity = ceil(N / M) + 3
```

Nếu AUV chưa được gán do capacity, fallback về relay gần nhất khả thi.

#### `flat_topology_association()`

Dùng cho flat baseline như FedAvg/FedProx: AUV nối trực tiếp Gateway nếu link khả thi.

### `physics_models/communication.py`

Chứa công thức kênh âm:

- `thorp_absorption`
- `transmission_loss`
- `wenz_noise_components`
- `wenz_noise_level`
- `snr_passive`
- `shannon_capacity`
- `min_source_level`
- `is_link_feasible`

### `physics_models/latency.py`

Chứa công thức latency:

- `comm_delay`: truyền payload qua link.
- `comp_delay_dynamic`: tính delay train local.
- `relay_comp_delay`: tính delay SVD ở relay.
- `round_delay`: tổng hợp latency một round.

### `physics_models/energy.py`

Chứa công thức năng lượng:

- `e_tx`: năng lượng truyền.
- `e_rx`: năng lượng nhận.
- `e_comp`: năng lượng tính toán.
- `e_svd`: năng lượng relay SVD.
- `total_energy_round`: tổng hợp.

Nhóm physics tương đối sạch, nên giữ độc lập sau refactor.

## 9. Federated Core

### `federated_core/workers.py`

#### `BaseWorker`

Base class cũ cho AUV:

- giữ `auv_id`
- giữ pin
- `deduct_battery()`
- placeholder `train_and_get_payload()`

Vì giờ chỉ còn 2D, class này có thể được inline vào `AUVWorker2D` trong tương lai.

#### `BaseRelayNode`

Base class cũ cho Relay:

- giữ relay id
- giữ cluster members
- giữ state sau intra aggregate và final state
- `cooperate()` để hợp tác relay-relay

Tương lai có thể đổi thành `RelayNode` cụ thể, không cần base.

#### `BaseGateway`

Gateway state holder:

- `global_state_dict`
- `proxy_ft_optimizer_state`
- `kd_optimizer_state`
- `pure_aggregated_state`

`pure_aggregated_state` quan trọng cho WiSE-FT, vì nó lưu FedAvg thuần trước server_mix.

### `federated_core/aggregator.py`

#### `fedavg_global()`

Aggregate các relay state tại Gateway. Nếu `lora_aggregation="svd"` thì gọi `svd_lora_aggregate`.

#### `weighted_state_dict_average()`

FedAvg thường cho state dict.

#### `svd_lora_aggregate()`

Đây là logic LoRA aggregation chính:

1. Với non-LoRA keys: weighted average.
2. Với LoRA pair:
   - lấy `B_i`, `A_i`
   - tính `W_i = B_i @ A_i`
   - weighted average `W_avg`
   - SVD `W_avg`
   - factorize lại thành `B_new`, `A_new`
3. Log reconstruction error.

Debug note:

- Nếu metric sập sau aggregation, xem log `[Debug-SVD]`.
- Nếu reconstruction error quá cao, nghi ngờ rank quá thấp hoặc scale LoRA bất thường.

### `federated_core/base_simulator.py`

#### `BaseSimulator.__init__()`

Load config singleton, topology, trackers, logger. Sau đó subclass sẽ tạo AUV/Relay/Gateway.

#### `_load_environment()`

Load `TopologySnapshot`, restore:

- `Topology3D`
- vị trí AUV/Relay/Gateway
- feasibility graph
- association
- clusters

#### `run()`

Đây là vòng lặp FL lớn nhất repo. Mỗi round:

1. Cập nhật LR cosine.
2. Tìm AUV active.
3. AUV local train.
4. Tính AUV TX/RX energy.
5. Relay intra aggregate.
6. Relay cooperation.
7. Relay gửi Gateway.
8. Gateway aggregate.
9. Nếu KD/proxy-FT active thì evaluate pre-gateway.
10. Chạy Gateway KD hoặc proxy-FT.
11. Evaluate post-gateway.
12. Tính latency, energy, objective cost.
13. Log metrics.
14. Move AUV và re-cluster.

`pre_kd_*` trong log là metric trước bước Gateway training, còn metric chính của round là sau Gateway KD/proxy-FT.

Điểm cần refactor:

- `run()` quá dài.
- Nó chứa cả branch 1D lẫn 2D.
- Nên tách thành:
  - `_run_round()`
  - `_run_auv_tier()`
  - `_run_relay_tier()`
  - `_run_gateway_tier()`
  - `_log_round_metrics()`
  - `_update_mobile_topology()`

Nếu bỏ hẳn 1D, nên kéo `run()` vào `Simulator2D` và bỏ abstract base.

### `federated_core/hfl_rules.py`

Chứa rule relay cooperation:

- `should_cooperate()`: selective cooperation theo cluster size.
- `compute_q1_relay_distance()`: lấy Q1 khoảng cách relay-relay.
- `find_coop_partner()`: tìm relay partner.
- `blend_state_dicts()`: trộn state hai relay.

### `federated_core/metrics.py`

Có hai nhóm:

1. Legacy anomaly detection 1D:
   - `anomaly_threshold`
   - `point_adjusted_f1`
   - `best_f1_components`
2. Physics/FL metrics còn dùng:
   - `EnergyTracker`
   - `LatencyTracker`
   - `MetricsLogger`
   - `physical_joint_cost`

Nên tách anomaly functions ra `legacy_1d_metrics.py` khi refactor.

## 10. Detection 2D

### `detection_2d/baselines.py`

#### `BaselineConfig`

Mỗi baseline là một tập flag:

- có full params không
- có LoRA không
- có INT8 không
- có Gateway KD không
- có proxy-FT không
- có HFL không
- rule relay cooperation
- LoRA aggregation strategy
- server_mix
- cờ ablation KD: logit-only, logit+box, logit+projection

#### `BASELINE_CONFIGS`

Registry baseline. Ví dụ:

- `fedkdl`: LoRA + INT8 + Gateway KD + HFL nearest + server_mix.
- `fedkdl_nokd`: giống FedKDL nhưng không KD.
- `fedkdl_proxy_ft`: không KD, nhưng bật Gateway proxy-FT.
- `logit_kd`: chỉ classification/logit KD.
- `logit_box_kd`: classification/logit + box KD, tắt projection KD.
- `logit_proj_kd`: classification/logit + LoRA projection KD, tắt box KD.
- `fedkdl_nocoop`, `fedkdl_selective`: ablation luật hợp tác relay.
- `fedkdl_32bit`: ablation bỏ LoRA/INT8, dùng full params nhưng vẫn bật Gateway KD.
- `fedprox_kdl`, `fedkd`: các đối chứng có KD.
- `fedavg`, `fedprox`: flat full-param baselines.
- `topk_grad`, `topk_grad_10`, `topk_grad_20`: sparse-gradient baselines, trong đó hai biến thể sau override tỷ lệ Top-K.
- `centralized`: upper bound train tập trung bằng LoRA.

Lưu ý quan trọng khi đọc kết quả: `KD_PROJ_WEIGHT` hiện mặc định bằng `0.0`.
Vì vậy `fedkdl` mặc định dùng logit + box KD; loss KD của `logit_box_kd`
trùng với `fedkdl` ở cấu hình mặc định. `logit_proj_kd` xử lý trường hợp
projection weight không dương bằng cách đặt nó thành `0.10`, nên đây mới là
nhánh projection KD khác biệt để chạy song song.

#### `parse_baseline_config()`

Validate baseline name và trả config.

### `detection_2d/simulator.py`

Đây là file quan trọng nhất.

#### `AUVWorker2D`

Đại diện một AUV.

`__init__()`:

- giữ đường dẫn YAML riêng của AUV
- đọc số ảnh train
- tạo cache optimizer/dataloader/topk

`train_and_get_payload()`:

1. Nếu AUV chết hoặc không có data thì bỏ qua.
2. Tạo local `StudentModel`.
3. Load global state vào local model.
4. Nếu baseline local KD thì tạo teacher local.
5. Gọi `local_sgd_od()`.
6. Lưu optimizer state cho round sau.
7. Pack payload:
   - Top-K sparse nếu baseline topk.
   - INT8 delta payload nếu baseline INT8.
   - Float32 state dict nếu không nén.

#### `RelayNode2D`

Đại diện một relay.

`aggregate_intra_cluster()`:

1. Duyệt các AUV trong cluster.
2. Giải nén payload:
   - Top-K -> dense delta.
   - INT8 delta -> reconstructed state.
   - Float32 -> dùng trực tiếp.
3. Weighted aggregate theo số sample.
4. Nếu LoRA SVD thì dùng `svd_lora_aggregate`.
5. Gán `intra_state_dict` và `final_state_dict`.

#### `Simulator2D.__init__()`

1. Gọi `BaseSimulator.__init__`.
2. Set task key = `"2D"`.
3. Parse baseline config.
4. Set `KD_ACTIVE` và `GLOBAL_FT`.
5. Load data partition.
6. Tạo YAML riêng cho từng AUV.
7. Tạo `proxy_kd_train.txt` từ public data.
8. Tạo proxy/eval YAML.
9. Tạo teacher nếu baseline cần KD.
10. Tạo global student.
11. Tạo gateway.
12. Tính payload budget.
13. Gọi `_init_network()`.

#### `_init_network()`

1. Nếu `BETA_EMD > 0`, đọc label histogram và association theo EMD + distance.
2. Nếu không, dùng physical association.
3. Build clusters.
4. Tạo `AUVWorker2D`.
5. Tạo `RelayNode2D`.

#### `_process_auv()`

Gọi AUV train, tính:

- payload
- train loss
- TX energy
- compute energy
- battery feasibility

#### `_aggregate_intra_relay()`

Gọi relay aggregate. Đây là wrapper để base simulator gọi được.

#### `_transport_relay_state()`

Mô phỏng việc relay state đi qua link:

- nếu không INT8: trả state nguyên.
- nếu INT8: pack delta rồi unpack lại.

#### `_build_gateway_proxy_yaml()`

Tạo YAML cho Gateway KD/proxy-FT:

- `train` = public/proxy data.
- `val` = validation set gốc.

#### `_gateway_supervised_finetune()`

Proxy-FT không dùng teacher. Đây là bước supervised tuning ở Gateway trên proxy/public data để tinh chỉnh global student sau aggregation.

Luồng đọc code:

1. Build proxy YAML.
2. Tính LR/epoch theo round.
3. Tạo `CustomDetectionTrainer`.
4. Load gateway state vào global student.
5. Fine-tune trên proxy data.
6. Cập nhật `gateway.global_state_dict`.
7. Ghi metric proxy-FT vào `_last_kd_metrics` để logger dùng chung format với KD.

Trong report này chỉ cần hiểu proxy-FT như một bước tune supervised ở Gateway, không phải knowledge distillation.

#### `_gateway_knowledge_distillation()`

Gateway KD với teacher. Luồng:

1. Nếu baseline là proxy-FT thì chuyển qua `_gateway_supervised_finetune`.
2. Nếu baseline không KD thì return metrics 0.
3. Với các baseline KD, coi như Gateway KD chạy ở mọi round.
4. Tạo `KDDetectionTrainer` với optimizer/LR/warmup riêng của KD.
5. Set teacher, KD weights, KD lambda và mode ablation (`logit_kd`, `logit_box_kd`, `logit_proj_kd`).
6. Load gateway state vào student.
7. Train KD trên proxy data.
8. Lưu `kd_optimizer_state`, cập nhật `gateway.global_state_dict` và ghi KD summary.

#### `evaluate()`

1. Strip inference tensors.
2. Load gateway state vào global student.
3. Gọi `evaluate_od()`.
4. Clear cache.

## 11. Trainer Và Model

### `detection_2d/trainer.py`

#### `CustomDetectionTrainer`

Subclass của Ultralytics `DetectionTrainer`, dùng cho local SGD và proxy-FT.

Các phần quan trọng:

- `get_dataloader()`: reuse cached dataloader.
- `_setup_train()`: inject lại model, restore optimizer state.
- `_restore_optimizer_state()`: restore AdamW/SGD state theo param name.
- `get_named_optimizer_state()`: lưu optimizer state theo param name.
- `build_optimizer()`: freeze non-payload params và tạo differential LR cho head/LoRA.
- `optimizer_step()`: guard NaN/Inf và gradient diagnostics.

Lưu ý: trong file hiện có dấu hiệu duplicate method `validate()`. Python chỉ dùng method định nghĩa sau cùng. Khi refactor nên dọn lại cho rõ.

#### `local_sgd_od()`

Train local model trên YAML của AUV:

1. Build YOLO overrides.
2. Tạo trainer.
3. Set LR multiplier.
4. Train.
5. Lấy trainable state.
6. Tính delta norm.
7. Xử lý FedProx/SCAFFOLD.
8. Return state, loss, optimizer state.

#### `evaluate_od()`

Evaluate global model:

1. Deepcopy model chưa fuse.
2. Bake LoRA vào Conv.
3. Gọi `YOLO.val`.
4. Restore model chưa fuse.
5. Return mAP/precision/recall/val loss nếu có.

### `detection_2d/models/yolo_wrapper.py`

#### `StudentModel`

Wrapper quanh YOLOv12n.

`__init__()`:

- load checkpoint
- rebuild head nếu số class khác
- inject LoRA nếu cần
- freeze non-payload params

`_is_payload_key()`:

Đây là contract truyền thông. Nó quyết định tensor nào được gửi:

- BN/running stats luôn gửi.
- LoRA A/B gửi nếu dùng LoRA.
- Uplink gửi một phần head.
- Downlink dùng contract payload riêng qua `downlink=True`.

`trainable_state_dict()`:

Return state dict đúng key payload.

`load_trainable_state_dict()`:

Load state vào model bằng `.data`, tránh lỗi inference tensor.

`bake_lora()`:

Gộp LoRA vào Conv trước khi evaluate.

#### `TeacherModel`

Wrapper YOLOv12l frozen, dùng cho Gateway KD.

### `detection_2d/models/lora.py`

#### `LoRAConv2d`

Bọc một Conv2d bằng low-rank adapter:

```text
output = Conv_base(x) + LoRA_B(LoRA_A(x))
```

#### `inject_lora()`

Duyệt model và thay module target bằng `LoRAConv2d`.

Các chi tiết đang có trong code:

- Strategy `adaptive` dùng rank nhỏ hơn cho backbone và rank chuẩn cho neck, đọc override từ env `FEDKDL_LORA_BACKBONE_RANK` và `FEDKDL_LORA_NECK_RANK`.
- Detection head bị skip khi inject LoRA; head được train trực tiếp bằng payload keys để tránh thêm low-rank path thứ hai vào logits/box.
- Target mặc định tránh các attention/MLP có hidden dim đặc biệt; với YOLO nano, `StudentModel` thường truyền target `Conv`.

## 12. Compression Và KD

### `int8_quantization.py`

#### `quantize_tensor()`

Lượng tử affine sang int8. Có guard NaN/Inf.

#### `dequantize_tensor()`

Khôi phục float tensor.

#### `pack_payload()`

Pack state dict thành bytes:

- sort key để đồng bộ pack/unpack.
- BN giữ float32, không quantize.
- tensor khác quantize int8.

#### `unpack_payload()`

Giải nén bytes theo template.

#### `pack_delta_payload()`

Pack delta:

```text
state_dict - reference_state
```

#### `unpack_delta_payload()`

Unpack delta rồi cộng lại reference.

Debug quan trọng:

- Nếu mAP sập về 0, kiểm tra BN có bị quantize không.
- Nếu dequant ra NaN/Inf, kiểm tra scale/zero point và tensor trước pack.
- Nếu unpack lệch key, kiểm tra sort key/template.

### `knowledge_distillation.py`

#### `_compose_balanced_kd()`

Combine KD components. Logic double-scaling cũ đã bị bỏ.

#### `_LoRAProjectionHook`

Hook lấy LoRA projection activation để làm projection KD.

#### `_lora_projection_kl_loss()`

KL theo LoRA projection activation. Hiện có 2 mode:

- `lora_spatial_proj` (mặc định): lấy `h = A*x`, collapse rank bằng `mean(h^2)` rồi KL trên spatial attention map. Mode này giảm phụ thuộc vào việc rank axis của teacher/student có cùng ý nghĩa.
- `lora_rank_proj`: giữ alignment theo từng rank dimension để chạy ablation.

Hàm dùng Structural Anchor Matching khi `KD_PROJ_ANCHOR_MATCH=True`: projection đầu stage map với đầu stage teacher, projection cuối stage map với cuối stage teacher, phần bottleneck giữa mới nội suy. Mục tiêu là tránh lỗi layer cuối student bị học nhầm bottleneck gần cuối của teacher.

#### `KDDetectionTrainer`

Trainer cho Gateway KD.

Các method chính:

- `_setup_train()`: strip inference tensors, patch criterion, restore optimizer.
- `set_teacher()`: gắn teacher frozen.
- `_kd_criterion_wrapper()`: supervised loss + KD losses.
- `get_kd_summary()`: trả metrics KD cho logging.

KD loss gồm:

- classification/logit KD.
- box KD, nếu output format hợp lệ.
- LoRA projection KD, nếu bật.

`logit_kd_only`, `logit_box_kd_only`, `logit_proj_kd_only` không phải baseline riêng trong trainer; chúng là cờ được `Simulator2D` set lên `KDDetectionTrainer`. Classification/logit KD luôn là nền, box KD tắt khi chạy `logit_kd` hoặc `logit_proj_kd`, projection KD chỉ chạy khi không bị cờ ablation chặn và `kd_proj_weight > 0`.

## 13. Proxy-FT Hiện Tại

Proxy-FT là supervised fine-tune ở Gateway trên public/proxy data, không dùng teacher.

Mục tiêu:

- Sau FedAvg/SVD aggregation, dùng proxy data để tune global model.
- Là nhánh đối chứng không dùng teacher, không tính KD loss.

Logic hiện tại:

1. Build proxy YAML.
2. Tạo `CustomDetectionTrainer`.
3. Tune global student trên proxy/public data.
4. Cập nhật `gateway.global_state_dict`.
5. Ghi proxy-FT metrics vào log.

Vì vậy khi đọc kết quả, chỉ cần hiểu Proxy-FT là nhánh “Gateway tự tune bằng nhãn proxy”.

## 13.5. Ba Runner FedKDL/KD Chạy Song Song


Trên server Vast.ai/Linux, [server/entrypoint.sh](D:/Documents/HUST/2022-2026/Research_Thesis/FedKDL/server/entrypoint.sh) là script bootstrap phục vụ chạy các runner: cài SSH/system packages, bật `sshd`, clone/pull repo, tạo `.venv`, cài `requirements.txt`, `chmod +x` các runner, tạo thư mục `results`, rồi giữ container sống bằng `tail -f /dev/null`. Script này không thay đổi luồng train; nó chỉ chuẩn bị môi trường máy chạy.
Ba file ở thư mục gốc:

| Script | Baseline | KD components |
| --- | --- | --- |
| `run_fedkdl.sh` | `fedkdl` | classification/logit + box KD theo config mặc định |
| `run_kd_logit.sh` | `logit_kd` | classification/logit KD |
| `run_kd_logit_proj.sh` | `logit_proj_kd` | classification/logit + LoRA projection KD |

Mỗi runner độc lập, dùng cùng quy ước biến môi trường:

```text
PYTHON, GPU, DS, N, M, ALPHA, SEED, ROUNDS,
ENVS_DIR, OUT_DIR, LOG_DIR, WANDB_MODE
```

Giá trị mặc định:

```text
DS=URPC
N=30
M=8
ALPHA=1.0
SEED=1109
ROUNDS=40
GPU=0 / 1 / 2 tương ứng ba file
```

### Chuẩn bị một lần

Không để ba process cùng sinh topology hoặc warmup checkpoint. Chuẩn bị trước:

```bash
python utils/download_datasets.py --urpc

python utils/generate_all_envs.py \
  --dataset URPC --n 30 --m-relays 8 --alphas 1.0 --seeds 1109

python scripts/fedkdl/train_student_warmup.py --mode warmup
chmod +x run_fedkdl.sh run_kd_logit.sh run_kd_logit_proj.sh
mkdir -p results
```

### Chạy trên ba GPU

```bash
GPU=0 ./run_fedkdl.sh > results/fedkdl.runner.log 2>&1 &
GPU=1 ./run_kd_logit.sh > results/logit_kd.runner.log 2>&1 &
GPU=2 ./run_kd_logit_proj.sh > results/logit_proj_kd.runner.log 2>&1 &
wait
```

Ghi đè cùng một cấu hình:

```bash
GPU=0 ROUNDS=60 SEED=1104 ./run_fedkdl.sh
GPU=1 ROUNDS=60 SEED=1104 ./run_kd_logit.sh
GPU=2 ROUNDS=60 SEED=1104 ./run_kd_logit_proj.sh
```

Nếu chỉ có một GPU, chạy tuần tự với `GPU=0`; không chạy ba job đồng thời vì
mỗi process giữ riêng Student, Teacher và optimizer KD trong VRAM.

Các runner dừng sớm với exit code `2` nếu thiếu topology/data và in chính xác
lệnh `generate_all_envs.py` cần chạy. Chúng không tự sinh môi trường để tránh
race condition khi launch song song.

Artifact:

```text
results/logs/N_30/M_8/
  log_N30_URPC_a1p0_<baseline>_seed1109.json

results/train_logs/N_30/M_8/
  log_N30_URPC_a1p0_<baseline>_seed1109.stdout.log
```

### Demo runtime

Thư mục [demo](D:/Documents/HUST/2022-2026/Research_Thesis/FedKDL/demo) là lớp runtime/UI riêng với FastAPI backend, static frontend, model inference và quản lý job live qua `demo/live_jobs.py`. Phần này không nằm trong đường train chính `main_trainer_od.py -> Simulator2D -> BaseSimulator.run()`, nên khi refactor training core cần tránh trộn logic demo vào simulator.

`demo/app.py` cung cấp API summary/topology/replay, upload ảnh để detect bằng model demo, và endpoint live-round. `demo/live_jobs.py` giới hạn mỗi lần chỉ một job training thật, đọc stdout để cập nhật AUV hiện tại, hỗ trợ cancel process và giữ log gần nhất trong bộ nhớ.

## 14. Những Nơi Dễ Bug Nhất

### Payload/state dict

Dễ lỗi nhất vì liên quan nhiều nơi:

- `StudentModel._is_payload_key`
- `pack_delta_payload`
- `unpack_delta_payload`
- relay aggregate
- gateway downlink state

Invariant cần giữ:

- pack/unpack luôn sort key.
- BN không quantize.
- delta template phải đúng reference state.
- uplink/downlink key set khác nhau có chủ đích.

### Top-K sparse payload

File: [detection_2d/knowledge_compression/topk_sparsification.py](D:/Documents/HUST/2022-2026/Research_Thesis/FedKDL/detection_2d/knowledge_compression/topk_sparsification.py).

Top-K gửi `indices`, `values` và metadata `shapes`. Index là vị trí trong vector flatten toàn cục của payload, nên thứ tự flatten phải ổn định qua các round. Hiện `flatten_state_dict()` đã sort key trước khi nối tensor:

```text
for key in sorted(state_dict.keys())
```

Invariant cần giữ:

- cùng một index luôn map về cùng tensor/key/vị trí.
- `SparseFloatPayload.decompress()` chỉ tạo dense delta.
- Relay phải `unflatten_state_dict()` rồi cộng delta vào đúng `global_state_dict`.
- Error feedback của Top-K là stateful theo từng AUV, nên lỗi thứ tự key thường chỉ lộ từ round 2 trở đi.

### SVD-LoRA

Rủi ro:

- SVD sign ambiguity.
- reconstruction error cao.
- optimizer momentum LoRA reuse sau khi SVD xoay basis.

Đã có logic bỏ LoRA momentum local khi dùng SVD.

### Evaluate YOLO + LoRA

YOLO `.val()` có thể fuse model và làm mất LoRA. Vì vậy `evaluate_od()` phải:

1. deepcopy model.
2. bake LoRA.
3. val.
4. restore model.

### Gateway KD

Rủi ro:

- teacher/student output format mismatch.
- KD ratio quá cao.
- confidence mask không lọc background tốt.
- hook đăng ký sai thời điểm.
- LoRA projection KD có thể lệch semantic nếu teacher checkpoint không cùng LoRA strategy/rank, hoặc nếu tắt anchor matching.

### Dummy test archive

File: [scripts/archive_tests/test_all_scenarios.py](D:/Documents/HUST/2022-2026/Research_Thesis/FedKDL/scripts/archive_tests/test_all_scenarios.py).

File này không cần YOLO/URPC/GPU; nó dùng tensor nhỏ để test contract thuật toán:

- baseline config và gateway mode.
- FedAvg/FedProx flat.
- HFL FedAvg/FedProx.
- LoRA naive và SVD aggregation.
- INT8 delta payload.
- Top-K sparse payload.
- SCAFFOLD/KD/proxy metadata contract.
- latency/energy accounting.

Lệnh test nhóm baseline debug:

```bash
python scripts/archive_tests/test_all_scenarios.py --baselines fedavg fedprox fedavg_hfl naive_lora flora topk_grad fedprox_hfl
```

Mỗi baseline hiện được test cả 1 round và 2 round. Test 2 round dùng global state sau round 1 làm input round 2; riêng Top-K giữ compressor/error buffer theo từng client để bắt lỗi “round 1 pass nhưng round 2 sập”.

## 15. Kế Hoạch Refactor Không Xóa Code Ngay

### Phase 0: Đóng băng hiểu biết

- Giữ report này.
- Test hiện có trong `scripts/archive_tests/test_all_scenarios.py` đã cover:
  - pack/unpack INT8 delta.
  - SVD-LoRA aggregation.
  - Top-K sparse payload và error feedback 2 round.
  - baseline config.
  - gateway mode contract.
- Test nên bổ sung tiếp:
  - payload key set thật từ `StudentModel._is_payload_key`.
  - evaluate YOLO + LoRA bake/restore.
  - proxy-FT metrics/logging bằng metrics giả lập.

### Phase 1: Tách file, không đổi behavior

Tách `detection_2d/simulator.py` thành:

```text
detection_2d/
|-- simulator.py            # chi orchestration
|-- workers.py              # AUVWorker2D, RelayNode2D
|-- gateway_training.py     # proxy-FT va KD
|-- dataset_builder.py      # tao AUV YAML/proxy YAML
`-- evaluation.py           # wrapper evaluate
```

### Phase 2: Làm phẳng inheritance

Vì không còn 1D là nhánh chính:

- chuyển `BaseSimulator.run()` vào simulator 2D.
- bỏ abstract hooks.
- đổi tên `Simulator2D` thành `FedKDLSimulator` hoặc `DetectionFedKDLSimulator`.
- `BaseGateway` có thể đổi thành `GatewayState`.

### Phase 3: Tách config

Tách `FedKDLConfig` lớn thành:

```text
TrainingConfig
CompressionConfig
GatewayKDConfig
ProxyFTConfig
ObjectiveConfig
```

### Phase 4: Đưa legacy 1D vào archive

Không xóa ngay, chỉ chuyển:

- anomaly metrics.
- SMD/SMAP/MSL scripts.
- config 1D.
- môi trường 1D.

## 16. Checklist Debug

### Nếu metric giảm sau INT8

- Xem BN có bị quantize không.
- Xem `INT8_DELTA_PAYLOAD`.
- Xem unpack template đúng không.
- Xem warning non-finite sau dequant.

### Nếu metric giảm sau SVD

- Xem `[Debug-SVD] Reconstruction Error`.
- Xem LoRA optimizer momentum có bị reuse không.
- Xem rank LoRA có quá thấp không.

### Nếu Top-K làm model hỏng từ round 2

- Chạy dummy test:

```bash
python scripts/archive_tests/test_all_scenarios.py --baselines topk_grad
```

- Kiểm tra `flatten_state_dict()` còn sort key không.
- Kiểm tra payload có mang đúng `shapes` không.
- Kiểm tra relay đang cộng dense delta vào đúng `global_state_dict`.
- Kiểm tra mỗi AUV giữ Top-K compressor/error buffer riêng, không dùng chung compressor giữa client.

### Nếu muốn smoke test FedKDL/KD bằng train thật

- Chuẩn bị environment và warmup theo mục 13.5.
- Chạy tuần tự một GPU với ít round:

```bash
GPU=0 ROUNDS=2 bash run_fedkdl.sh
GPU=0 ROUNDS=2 bash run_kd_logit.sh
GPU=0 ROUNDS=2 bash run_kd_logit_proj.sh
```

- Xem stdout tại `results/train_logs/N_30/M_8/`.
- Xem JSON metrics tại `results/logs/N_30/M_8/`.

### Nếu metric giảm sau proxy-FT

- Xem LR/epoch/augment của round đó.
- Nếu precision tăng nhưng recall giảm: LR/epoch/augment quá mạnh hoặc proxy lệch distribution.
- Nếu proxy-FT gây hại đều: giảm LR/epoch hoặc xem lại proxy split.

### Nếu metric giảm sau KD

- Xem `kd_ratio`, `kd_contrib`.
- Xem warning shape mismatch trong KD.
- Xem teacher checkpoint đúng không.

## 17. Kết Luận

Repo hiện chạy được bài toán FedKDL 2D, nhưng kiến trúc vẫn mang dấu vết thời còn 1D. Việc nên làm không phải xóa ngay, mà là:

1. Tạo test quanh payload/SVD/eval.
2. Tách `simulator.py` thành các module nhỏ.
3. Làm phẳng `BaseSimulator` khi đã chắc không cần 1D.
4. Tách config theo nhóm.
5. Đưa legacy 1D vào archive.

Sau refactor, repo sẽ dễ debug hơn rất nhiều: khi metric tụt, ta biết phải nhìn vào payload, SVD, proxy-FT, KD, hay physics cost, thay vì phải đọc xuyên qua một vòng lặp hơn nghìn dòng.
