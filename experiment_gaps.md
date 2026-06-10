# Phân tích Gap: Experiment Design vs. Code hiện tại

## Baselines đã có trong `parse_baseline_config`

| Baseline ID | Mô tả | RQ liên quan |
|---|---|---|
| `fedkdl` | HFL + LoRA + INT8 + Gateway KD | RQ1, RQ2, RQ3, RQ4 |
| `fedkdl_selective` | Như fedkdl nhưng selective cooperation | RQ3 |
| `fedprox_kdl` | fedkdl + FedProx proximal term | RQ1 |
| `fedkdl_nokd` | fedkdl không có Gateway KD | RQ4 |
| `topk_grad` | Full-param + Top-K gradient sparsification | RQ2 |
| `centralized` | Huấn luyện tập trung tại Gateway | RQ4 |
| `fedkd` | Flat + full-param + Gateway KD (FedKD reference) | — |
| `fedkdl_nolora` | HFL + full-param + KD (không LoRA) | — |

---

## Gap: Baselines cần implement thêm

### RQ1 — Kết nối và ổn định: Flat ở Fedavg và  prox

| Baseline cần | Trạng thái | Việc cần làm |
|---|---|---|
| **FedAvg** (flat, full-param, không KD, không HFL) | ❌ Thiếu | Thêm `'fedavg'` vào `parse_baseline_config` → `(True, False, False, False, False)`. Thêm xử lý flat topology trong `base_simulator.py` (bỏ qua relay tier). |
| **FedProx** (flat, full-param) | ❌ Thiếu | Thêm `'fedprox'` → `(True, False, False, False, False)` với `fedprox_mu > 0`. Cần đảm bảo `is_flat=True` để bỏ qua relay. |

> **Lưu ý:** `fedprox_kdl` hiện có nhưng là HFL version. Cần FedProx dạng flat (không relay) để so sánh đúng.

---

### Từ các RQ sau, tất cả đều được train phân cấp, bất kể là Avg hay Prox, sẽ tạo 1 version mới cho hfl

### RQ2 — Nén truyền thông

| Baseline cần | Trạng thái | Việc cần làm |
|---|---|---|
| **FedAvg** | ❌ Thiếu | Như trên |
| **Top-K Sparsification** | ✅ Có (`topk_grad`) | Chạy được ngay |
| **FLORA** (LoRA, không INT8, không KD) | ⚠️ Cần kiểm tra | Thêm `'flora'` → `(False, True, False, False, False)`. Cần topology flat (không relay) để khớp nghĩa gốc của FLORA. |
| **FedKDL** | ✅ Có | — |

---

### RQ3 — Non-IID và Relay

| Baseline cần | Trạng thái | Việc cần làm |
|---|---|---|
| **FedAvg** | ❌ Thiếu | Như trên |
| **SCAFFOLD** | ❌ Thiếu hoàn toàn | **Phức tạp nhất.** Cần implement control variates: (1) Server lưu `c` (control variate tổng hợp), (2) Client cập nhật `c_i` và `y_i` mỗi vòng, (3) Gradient bổ chính bởi `c - c_i` trong mỗi bước SGD. Ước tính: 1-2 ngày lập trình. |
| **FLORA** (flat) | ⚠️ Cần kiểm tra | Như RQ2 |
| **FedKDL không Relay Coop** | ⚠️ Gần có | Thêm `'fedkdl_nocoop'` → giống `fedkdl` nhưng `coop_rule = 'nocoop'`. Logic `nocoop` đã tồn tại trong code. |
| **FedKDL đầy đủ** | ✅ Có | — |

---

### RQ4 — Gateway KD

| Baseline cần | Trạng thái | Việc cần làm |
|---|---|---|
| **No KD** (`fedkdl_nokd`) | ✅ Có | — |
| **Logit KD** | ❌ Thiếu | Cần implement: Thay thế `_lora_projection_mse_loss` bằng `KL-Divergence` trên soft predictions của Teacher/Student. Thêm `'logit_kd'` baseline. Ước tính: nửa ngày. |
| **LoRA-Projection KD** (`fedkdl`) | ✅ Có | — |
| **Centralized** | ✅ Có | — |

---

## Tóm tắt công việc cần làm

### Ưu tiên cao (dễ, nhanh)

1. **Thêm `fedavg` baseline** (~30 phút):
   - Thêm vào `parse_baseline_config`: `'fedavg': (True, False, False, False, False)`
   - Đảm bảo flat topology (bỏ qua relay phase) khi `is_flat=True`

2. **Thêm `fedprox` (flat)** (~30 phút):
   - Thêm `'fedprox': (True, False, False, False, False)` với `fedprox_mu = 0.01`

3. **Thêm `flora` (flat, LoRA, no INT8)** (~1 giờ):
   - Thêm `'flora': (False, True, False, False, False)`
   - Kiểm tra relay aggregation có hoạt động đúng với float32 payload không

4. **Thêm `fedkdl_nocoop`** (~30 phút):
   - Copy `fedkdl` config, force `coop_rule = 'nocoop'`

5. **Thêm `logit_kd` baseline** (~4 giờ):
   - Thêm một nhánh KD mới trong `knowledge_distillation.py` dùng KL-Divergence thay vì LoRA projection

### Ưu tiên thấp (phức tạp)

1. **SCAFFOLD** (~1-2 ngày):
   - Server lưu global control variate `c`
   - Client cập nhật local `c_i` và hiệu chỉnh gradient
   - Cần sửa `workers.py`, `aggregator.py`, và `trainer.py`

---

## Kiến nghị thứ tự chạy thực nghiệm trên Kaggle

Mỗi baseline chạy độc lập với `main_trainer_od.py --baseline <id>`. Thứ tự đề xuất:

```
# Đã có sẵn:
fedkdl        → RQ1, RQ2, RQ3, RQ4
fedkdl_nokd   → RQ4
topk_grad     → RQ2
centralized   → RQ4

# Cần thêm (dễ):
fedavg        → RQ1, RQ2, RQ3
fedprox       → RQ1
flora         → RQ2, RQ3
fedkdl_nocoop → RQ3
logit_kd      → RQ4

# Cần thêm (khó):
scaffold      → RQ3
```
