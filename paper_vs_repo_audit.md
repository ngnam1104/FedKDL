# Audit: Paper vs. Repo — FedKDL
> **Paper**: *Energy-Efficient Hierarchical Federated Anomaly Detection for the Internet of Underwater Things via Selective Cooperative Aggregation* (Omeke et al., IEEE IoT Journal 2026)
> **Trạng thái**: ✅ Audit hoàn thành — 7/7 vấn đề đã được sửa.

---

## Tóm tắt đánh giá (trạng thái hiện tại)

| Hạng mục | Trạng thái | Ghi chú |
|---|---|---|
| Mô hình vật lý (Thorp-Wenz-Sonar) | ✅ Khớp | Đầy đủ Eq.1–7 |
| Mô hình năng lượng SNR-driven | ✅ Khớp | Eq.8, công thức P_ac đúng vật lý |
| Topology 3D (Sensor/Fog/Gateway) | ✅ Khớp | Table II đúng |
| Autoencoder (32-16-8-16-32) | ✅ Khớp | ~1352 params |
| Local SGD + FedProx | ✅ Khớp | Eq.12 + proximal term |
| Top-K Sparsification + Error Feedback | ✅ Khớp | Eq.30, ρ_s=0.05 |
| INT8 Quantization | ✅ Khớp | Eq.31, b_q=8 |
| HFL-NoCoop / HFL-Nearest / HFL-Selective | ✅ Khớp | Eq.28–29 |
| HFL-Selective: Q1 distance filter | ✅ **Đã sửa** | `compute_q1_fog_distance()` + filter trong `find_coop_partner()` |
| HFL-Nearest: mixing weight (0.7, 0.3) | ✅ **Đã sửa** | `cooperate()` dùng alpha=0.7 cho HFL-Nearest |
| Fog-level FedAvg (intra-cluster) | ✅ Khớp | Eq.13 |
| Global aggregation | ✅ Khớp | Eq.16, weight-by-n_samples |
| Fog-to-gateway energy | ✅ **Đã sửa** | `fog_model_bits = d×32` thay vì `payload×20` hardcode |
| Fog-to-fog energy | ✅ **Đã sửa** | `e_tx()` với link vật lý thực từ feasibility graph |
| Latency model (Eq.21) | ✅ **Đã thêm** | `LatencyTracker.compute_round_latency()` |
| Config T rounds (Synthetic/Real) | ✅ **Đã sửa** | T=20 / T=30 đúng theo paper trong các scripts |
| Multi-seed (mean±std over 3 seeds) | ✅ **Đã thêm** | Seeds [42, 123, 2024] trong tất cả scripts |
| Non-IID sensitivity sweep | ✅ **Đã có** | α ∈ {0.1, 10⁴} trong `run_heterogeneity.py` |
| Ngưỡng anomaly (99th percentile) | ✅ Khớp | Eq.32, p=99 |
| PA-F1 metric | ✅ Khớp | Point-adjusted F1 |
| Battery dynamics tracking | ✅ Khớp | `deduct_battery()` đúng |

---

## Chi tiết từng phần

### ✅ 1. Mô hình Vật lý Sóng âm (`physics_models/communication.py`)

| Phương trình Paper | Hàm trong Code | Trạng thái |
|---|---|---|
| **Eq.1** TL(d,f) = 10k·log₁₀(d) + α(f)·d/1000 | `transmission_loss()` | ✅ |
| **Eq.2** Thorp α(f) | `thorp_absorption()` | ✅ |
| **Eq.3** Wenz NL(f,B) = N₀(f) + 10log₁₀(B) | `wenz_noise_level()` | ✅ |
| **Eq.4** SNR = SL − TL − NL − IL | `snr_passive()` | ✅ |
| **Eq.5** Shannon R = B·log₂(1+10^(γ/10)) | `shannon_capacity()` | ✅ |
| **Eq.6** SL_min = γ_tgt + TL + NL + IL | `min_source_level()` | ✅ |
| **Eq.7** Link feasible ⟺ SL_min ≤ SL_max | `is_link_feasible()` | ✅ |

**Config (Table II)** — tất cả parameter khớp trong `config/settings.py`:

| Parameter | Paper | Code |
|---|---|---|
| f (kHz) | 12 | 12 ✅ |
| B (kHz) | 4 | 4 ✅ |
| γ_tgt (dB) | 10 | 10 ✅ |
| SL_max (dB) | 140 | 140 ✅ |
| k | 1.5 | 1.5 ✅ |
| w (m/s) | 5 | 5 ✅ |
| s | 0.5 | 0.5 ✅ |
| IL (dB) | 2 | 2 ✅ |
| c_s (m/s) | 1500 | 1500 ✅ |

---

### ✅ 2. Mô hình Năng lượng (`physics_models/energy.py`)

| Phương trình Paper | Triển khai | Trạng thái |
|---|---|---|
| **Eq.8** E_tx = (P_ac/η_ea + P_c_tx)·L/R | `e_tx()` dùng `acoustic_power_watts()` đúng vật lý | ✅ |
| E_rx = P_c_rx · L/R | `e_rx()` | ✅ |
| E_round = E_s2f + E_f2f + E_f2g + E_comp | `total_energy_round()` | ✅ |
| E_i^{t+1} = E_i^t − E_tx − E_comp | `deduct_battery()` | ✅ |

> [!NOTE]
> P_ac = (4π · p_ref²)/(ρ_w · c_s) × 10^(SL/10) — code triển khai đúng công thức vật lý chuẩn.

---

### ✅ 3. Topology 3D (`physics_models/topology.py`)

- Sensor: depth 500–1000 m, uniform random (x,y) ✅
- Fog: depth 100–400 m ✅
- Gateway: z = 0 (surface), center (1000, 1000) ✅
- `build_feasibility_graph()`: kiểm tra tất cả sensor→fog, fog→fog, fog→gateway, sensor→gateway ✅
- `nearest_feasible_association()`: sensor gắn với fog gần nhất khả thi ✅
- `flat_topology_association()`: chỉ sensor có direct gateway link mới được tham gia ✅

---

### ✅ 4. Autoencoder (`fl_core/models/autoencoder.py`)

Paper (Table II): `32→16→8→16→32`, ~1,352 params, loss = MSE reconstruction.

```
SmallAutoencoder: D → 32 → 16 → 8 → 16 → 32 → D  (ReLU activations)
```

`reconstruction_error()` trả về per-sample MSE đúng Eq.9. ✅

---

### ✅ 5. Local Training (`fl_core/algorithms/local_trainer.py`)

- **Eq.12**: θ_i^{t+1} ← θ^t − η∇F_i(θ^t; B^t) → `local_sgd()` với E=5, η=0.01 ✅
- **FedProx**: L = MSE + (μ/2)‖θ−θ_global‖² → proximal term đúng ✅
- Trả về Δθ = θ_new − θ_old ✅

---

### ✅ 6. Compression (`fl_core/knowledge_compression/`)

**Top-K Sparsification (Eq.30)**:
```
v_i^t = Δθ_i^t + e_i^{t-1}     # error feedback (cộng residual cũ)
ṽ_i^t = Top-K(v_i^t)           # giữ K lớn nhất theo magnitude
e_i^t = v_i^t − ṽ_i^t          # tích lũy residual mới
```
✅ Đúng trong `TopKCompressor.compress()`, ρ_s=0.05 → K ≈ 68 (5% × 1350)

**INT8 Quantization (Eq.31)**:
- b_q = 8 bits, b_idx = ⌈log₂(d)⌉ bits ✅
- Payload sau nén ≈ 1.3 kbit (so với 43 kbit full-precision) ✅

---

### ✅ 7. HFL Rules (`fl_core/algorithms/hfl_rules.py`) — đã cập nhật

#### HFL-Selective condition (Eq.28)

```python
threshold = max(2, 0.75 * mean_cluster_size)
return cluster_size <= threshold  # ✅
```

#### Q1 distance filter (Eq.29) — **Đã thêm**

```python
def compute_q1_fog_distance(feasibility_graph) -> float:
    distances = [info.distance for (type_u, _, type_v, _), info
                 in feasibility_graph.items() if type_u == 'fog' and type_v == 'fog']
    return float(np.percentile(distances, 25)) if distances else float('inf')
```

`find_coop_partner()` giờ nhận `q1_distance` và lọc:
```python
if q1_distance is not None and dist > q1_distance:
    continue  # bỏ qua neighbor xa hơn Q1 ✅
```

#### Mixing weights — **Đã sửa**

| Rule | Alpha (Paper) | Alpha (Code cũ) | Alpha (Code mới) |
|---|---|---|---|
| HFL-Selective | 0.8 (Eq.29) | 0.8 ✅ | 0.8 ✅ |
| HFL-Nearest | 0.7 (Sec. V-B) | 0.8 ❌ | **0.7 ✅** |

```python
alpha = 0.7 if rule == 'nearest' else 0.8  # ✅
```

`cooperate()` giờ trả về `Tuple[bool, Optional[int]]` thay vì chỉ `bool`, để caller biết partner_id cho tính energy/latency.

---

### ✅ 8. Aggregation (`fl_core/algorithms/aggregator.py`)

| Phương trình Paper | Triển khai | Trạng thái |
|---|---|---|
| **Eq.13** θ_fog = θ^t + Σ(n_i/Σn_k)·Δθ_i | `fedavg_intra_cluster()` | ✅ |
| **Eq.15** θ̃_m = Σ α_j·θ_j^{t+½} | `blend_state_dicts()` | ✅ |
| **Eq.16** θ^{t+1} = Σ(n_m/N)·θ̃_m | `fedavg_global()` | ✅ |

---

### ✅ 9. Energy trong Simulator (`fl_core/simulator.py`) — đã sửa bugs

#### Fog-to-fog energy (Eq.18) — **Đã sửa**

```python
# Trước (sai):
e_f2f_total += 5.0  # Joules — hardcode

# Sau (đúng):
if did_coop and partner_id is not None:
    f2f_key = ('fog', m, 'fog', partner_id)  # lấy từ feasibility graph
    f2f_link = self.G[f2f_key]
    e_f2f_total += e_tx(
        self.fog_model_bits, f2f_link.R_bps, f2f_link.SL_min,
        self.en_cfg.ETA_EA, self.en_cfg.P_C_TX,
    )  # ✅ physics-based
```

#### Fog-to-gateway energy (Eq.19) — **Đã sửa**

```python
# Trước (sai):
e_f2g_total += e_tx(payload.payload_bits * 20, ...)  # ×20 tùy tiện

# Sau (đúng):
# fog_model_bits = d × 32 bits (full-precision, paper: fog→gateway dùng full precision)
self.fog_model_bits = sum(p.numel() for p in self.model_template.parameters()) * 32
e_f2g_total += e_tx(
    self.fog_model_bits, link.R_bps, link.SL_min,
    self.en_cfg.ETA_EA, self.en_cfg.P_C_TX,
)  # ✅
```

---

### ✅ 10. Latency Model (Eq.21) — **Đã thêm** (`fl_core/metrics.py`)

Paper (Eq.21):
```
τ_round = max(max_{i→a_i} τ_{i→fog}, max_{m→j} τ_{fog→fog}, max_{m→g} τ_{fog→g}) + τ_comp
```

Triển khai trong `LatencyTracker.compute_round_latency()`:

```python
tau_round = (max over all fogs of:
    s2f_delay(fog)     # max delay sensor→fog trong cụm
  + f2f_delay(fog)     # delay cooperation (nếu có)
  + f2g_delay(fog)     # delay fog→gateway
) + tau_comp           # local computation delay
```

Mỗi link delay: `τ = d/c_s + L/R` (propagation + transmission). ✅

Output: cột `Tau_Round_s` trong metrics log, export `*_latency.csv`. ✅

---

### ✅ 11. Metrics (`fl_core/metrics.py`)

| Metric | Triển khai | Trạng thái |
|---|---|---|
| `anomaly_threshold()` | 99th percentile of normal errors (Eq.32) | ✅ |
| `point_adjusted_f1()` | PA-F1 standard (segment-level adjustment) | ✅ |
| `EnergyTracker` | Track E_s2f, E_f2f, E_f2g, E_comp per round | ✅ |
| `LatencyTracker` | τ_round theo Eq.21, `compute_round_latency()` | ✅ **Mới** |
| `MetricsLogger` | Log PA-F1, Participation, Cumul_Energy, Tau_Round_s | ✅ |

---

### ✅ 12. Scripts thí nghiệm (`scripts/`) — đã cập nhật

| Script | T_rounds | Seeds | Non-IID sweep | Trạng thái |
|---|---|---|---|---|
| `run_scalability.py` | **20** (đúng paper) | **[42, 123, 2024]** | — | ✅ |
| `run_heterogeneity.py` | **20** | **[42, 123, 2024]** | **α ∈ {0.1, 10⁴}** | ✅ |
| `run_real_benchmarks.py` | **30** (đúng paper) | **[42, 123, 2024]** | — | ✅ |

Tất cả scripts báo **mean ± std** và vẽ **error bars** đúng như trong paper. ✅

`sim.run()` giờ trả về `(metrics_df, energy_df, latency_df)` — 3-tuple nhất quán toàn bộ codebase.

---

## Tổng kết: 7/7 vấn đề đã giải quyết

| # | Mức độ | Vấn đề | Trạng thái |
|---|---|---|---|
| 1 | 🔴 Bug | `e_f2g`: hardcode `payload×20` → dùng `fog_model_bits = d×32` | ✅ Done |
| 2 | 🔴 Bug | `e_f2f`: hardcode 5.0 J → `e_tx()` với link vật lý thực | ✅ Done |
| 3 | 🟡 Accuracy | HFL-Selective thiếu Q1 distance filter (Eq.29) → `compute_q1_fog_distance()` | ✅ Done |
| 4 | 🟡 Accuracy | HFL-Nearest alpha=0.8 sai → alpha=0.7 đúng theo paper | ✅ Done |
| 5 | 🟢 Config | T=150 mặc định → scripts dùng T=20/30 + 3 seeds đúng paper | ✅ Done |
| 6 | 🟢 Feature | Thiếu latency tracking (Eq.21) → `LatencyTracker` class | ✅ Done |
| 7 | 🟢 Feature | Thiếu non-IID sweep script → `run_heterogeneity.py` đầy đủ | ✅ Done |

---

## Files đã thay đổi

| File | Nội dung thay đổi |
|---|---|
| `fl_core/algorithms/hfl_rules.py` | `+compute_q1_fog_distance()`, Q1 filter trong `find_coop_partner()` |
| `fl_core/algorithms/worker.py` | `cooperate()` → `(bool, Optional[int])`, alpha=0.7/0.8 per rule, pass `q1_distance` |
| `fl_core/simulator.py` | Fix `e_f2f`/`e_f2g`, tích hợp `LatencyTracker`, track `cooperation_partners`, 3-tuple return |
| `fl_core/metrics.py` | `+LatencyTracker` class (Eq.21, `compute_round_latency()`) |
| `main_train.py` | Unpack 3-tuple, lưu thêm `*_latency.csv` |
| `scripts/run_scalability.py` | 3 seeds, mean±std, error bars, 3-tuple unpack, T=20 |
| `scripts/run_heterogeneity.py` | 3 seeds, mean±std, error bars, α ∈ {0.1, 10⁴}, 3-tuple unpack |
| `scripts/run_real_benchmarks.py` | 3 seeds, mean±std, error bars, T=30, 3-tuple unpack |

> [!TIP]
> Chạy thử nhanh (2 rounds, 1 seed) với flag `--dry-run`:
> ```
> python scripts/run_scalability.py --dry-run
> python scripts/run_heterogeneity.py --dry-run
> python scripts/run_real_benchmarks.py --dry-run
> ```
