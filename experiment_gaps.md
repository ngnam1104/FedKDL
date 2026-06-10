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
| **FedAvg** (flat, full-param, không KD, không HFL) | ✅ Đã có | `fedavg`; dùng virtual aggregator tại Gateway, không tính relay/SVD cost. |
| **FedProx** (flat, full-param) | ✅ Đã có | `fedprox`; dùng proximal term và flat topology. |

> **Lưu ý:** `fedprox_kdl` hiện có nhưng là HFL version. Cần FedProx dạng flat (không relay) để so sánh đúng.

---

### Từ RQ2 đến RQ4, các baseline FL đều train phân cấp

`centralized` và `fedkd` là reference riêng, không dùng để thay đổi nguyên tắc topology của các baseline FL trong RQ2/RQ3/RQ4.

### RQ2 — Nén truyền thông

| Baseline cần | Trạng thái | Việc cần làm |
|---|---|---|
| **FedAvg** | ✅ Đã có | Dùng `fedavg_hfl` để giữ cùng topology phân cấp. |
| **Top-K Sparsification** | ✅ Có (`topk_grad`) | Nén model delta bằng Top-K + error feedback. |
| **FLORA** (HFL + LoRA, không INT8, không KD) | ✅ Đã có | Dùng naive weighted average độc lập cho LoRA A/B. |
| **FedKDL** | ✅ Có | — |

---

### RQ3 — Non-IID và Relay

| Baseline cần | Trạng thái | Việc cần làm |
|---|---|---|
| **FedAvg** | ✅ Đã có | Dùng `fedavg_hfl`. |
| **SCAFFOLD** | ✅ Đã có | Có global/local control variates, gradient correction và client-count aggregation. |
| **FLORA** (HFL) | ✅ Đã có | Như RQ2 |
| **FedKDL không Relay Coop** | ✅ Đã có | `fedkdl_nocoop`. |
| **FedKDL đầy đủ** | ✅ Có | — |

---

### RQ4 — Gateway KD

| Baseline cần | Trạng thái | Việc cần làm |
|---|---|---|
| **No KD** (`fedkdl_nokd`) | ✅ Có | — |
| **Logit KD** | ✅ Đã có | Logit-only soft-target KD; dùng foreground-masked sigmoid BCE cho YOLO logits. |
| **LoRA-Projection KD** (`fedkdl`) | ✅ Có | — |
| **Centralized** | ✅ Có | — |

---

## Trạng thái hiện tại

Các baseline bắt buộc trong `experiment_design.md` đã được nối vào pipeline.
Bảng xác minh chi tiết, các điểm còn cần paper và đề xuất biểu đồ nằm trong
`baseline_audit.md`.

---

## Kiến nghị thứ tự chạy thực nghiệm trên Kaggle

Chạy suite chuẩn bằng `run_kdl_experiments.sh`. Script có 16 baseline duy nhất,
hỗ trợ grid qua biến môi trường `ALPHAS` và `SEEDS`, đồng thời tự bỏ qua log đã
hoàn thành.

```bash
ALPHAS="0.1 0.5 1.0 10000.0" SEEDS="42 1104 2024" \
  bash run_kdl_experiments.sh
```
