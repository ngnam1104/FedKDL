# Implementation Plan: Ultra-Low Payload FL cho 2D (YOLO + LoRA)

## Quyết định Kiến trúc Cuối cùng

Dựa trên đo đạc payload thực tế từ YOLO26n + URPC2020 (nc=4):

| Thành phần | INT8 Size | Quyết định |
|---|---|---|
| LoRA r=4 (62 layers) | 72 KB | ✅ Giữ làm Kịch bản 1 (74 KB) |
| LoRA r=8 (62 layers) | 144 KB | ✅ Giữ làm Kịch bản 2 (146 KB) |
| Full Head (nc=4) | 264 KB | ❌ Quá lớn |
| cv3.x.2 output conv only (nc=4) | 2 KB | ✅ Dùng partial head |

### Quyết định đã duyệt

- ✅ **Cấu hình tùy chọn Rank**: Hệ thống hỗ trợ cả `LORA_RANK = 4` (siêu nhẹ ~74KB) và `LORA_RANK = 8` (~146KB) để chạy thực nghiệm so sánh.
- ✅ **Truyền LoRA INT8** (dense, không sparse).
- ✅ **Truyền partial head**: Chỉ truyền `cv3.x.2` + `one2one_cv3.x.2` (lớp Conv cuối cùng xuất ra bounding box class, nc=4). KHÔNG truyền hidden layers của Head.
- ✅ **Bỏ hoàn toàn** Delta Encoding và Sparsification/Top-K.
- ✅ **Gateway-side KD (Surface Server)**: Di chuyển Knowledge Distillation lên **Gateway** (Trạm mặt nước), KHÔNG phải Fog.
  - *Lý do*: Fog node vẫn nằm dưới nước, chạy bằng pin và vi xử lý nhúng. Chạy suy luận Teacher (YOLO12 Large) sẽ vắt kiệt năng lượng Fog. Trạm mặt nước (Gateway) trên tàu/buoy có cắm nguồn, GPU mạnh, phù hợp nhất để chạy KD. AUV và Fog chỉ thuần túy học và tổng hợp LoRA.

> [!NOTE]
> **Lý do chỉ truyền cv3.x.2:** Đây là lớp output của class predictions. Khi domain shift (COCO→URPC), số class thay đổi (80→4) và class embeddings cần cập nhật. Các hidden layers cv3.x.0, cv3.x.1 là feature extractor chung — cập nhật LoRA ở backbone/neck đã đủ để adapt chúng.

---

## Audit: Trạng thái hiện tại vs Mục tiêu

### Nhóm 1: Payload (LoRA + Head)

| Item | Trạng thái | Việc cần làm |
|---|---|---|
| LoRA r=4 inject C2f/C3k2 backbone+neck | ✅ Đã có | Tăng lên r=8 trong settings |
| `trainable_state_dict()` lọc lora_ + detect | ✅ Đã có | Sửa: chỉ giữ `lora_` + `cv3.x.2` + `one2one_cv3.x.2` |
| INT8 pack_payload | ✅ Đã có | Giữ nguyên |
| Delta Encoding | ⚠️ Nửa vời | ❌ Bỏ hoàn toàn |
| Sparsification Top-K | ⚠️ Không kết nối | ❌ Bỏ hoàn toàn |

### Nhóm 2: Gateway-side KD (Server-side)

| Item | Trạng thái | Việc cần làm |
|---|---|---|
| KD tại AUV (hiện tại) | ✅ Đang chạy | Gỡ bỏ khỏi AUV |
| Gateway chạy KD sau Global Aggregation | ❌ Chưa có | Thêm vào vòng lặp chính (Gateway/BaseSimulator) |
| AUV train không cần Teacher | ❌ Chưa có | Sửa `SensorWorker2D` |

---

## Proposed Changes

### A. `config/settings.py`

#### [MODIFY] [settings.py](file:///d:/Documents/HUST/2022-2026/Research_Thesis/FedKDL/config/settings.py)

- `LORA_RANK: int = 8` (từ 4)

---

### B. `tasks/detection_2d/models/yolo_wrapper.py`

#### [MODIFY] [yolo_wrapper.py](file:///d:/Documents/HUST/2022-2026/Research_Thesis/FedKDL/tasks/detection_2d/models/yolo_wrapper.py)

**`trainable_state_dict()`** — thay vì `'lora_' in k or 'detect' in k.lower()`, chỉ giữ:

```python
# Truyền: lora_A, lora_B + cv3.x.2 + one2one_cv3.x.2 (output classifier layer)
'lora_' in k or k.endswith('.cv3.0.2.weight') or k.endswith('.cv3.1.2.weight') 
    or k.endswith('.cv3.2.2.weight') or k.endswith('.one2one_cv3.0.2.weight')
    or k.endswith('.one2one_cv3.1.2.weight') or k.endswith('.one2one_cv3.2.2.weight')
```

**`load_trainable_state_dict()`** — giữ nguyên (load mọi key match với model state dict).

---

### C. `tasks/detection_2d/trainer.py`

#### [MODIFY] [trainer.py](file:///d:/Documents/HUST/2022-2026/Research_Thesis/FedKDL/tasks/detection_2d/trainer.py)

- Bỏ tham số `use_kd` và `teacher_model` trong `local_sgd_od()`.
- Chuyển thành train chuẩn với data cục bộ: sử dụng `DetectionTrainer` gốc của Ultralytics thay vì `KDDetectionTrainer` (do không chạy KD tại AUV nữa).
- Trả về `(new_state, delta_norm_float)` — `new_state` là absolute (không phải delta).

---

### D. `tasks/detection_2d/simulator.py` và `federated_core/base_simulator.py`

#### [MODIFY] [simulator.py](file:///d:/Documents/HUST/2022-2026/Research_Thesis/FedKDL/tasks/detection_2d/simulator.py)

**`SensorWorker2D.train_and_get_payload()`**:

- Bỏ tham số `teacher` và `use_kd_lora_int8`.
- Gọi `local_sgd_od()`.
- Payload luôn là `pack_payload(new_state)` (INT8 dense).

**`Simulator2D._process_sensor()`**:

- Bỏ tham số `teacher` khi gọi `sensor.train_and_get_payload()`.

#### [MODIFY] [base_simulator.py](file:///d:/Documents/HUST/2022-2026/Research_Thesis/FedKDL/federated_core/base_simulator.py)

**Tại `run()` - Phase 4 (Gateway Global Aggregation)**:

- Sau khi nhận và aggregate từ các Fogs, **Gateway** sẽ chạy KD.
- Chèn hàm hook: `self._gateway_knowledge_distillation()`
- Trong `Simulator2D` (override lại hook này), nó sẽ dùng `KDDetectionTrainer` để fine-tune `self.global_student` với `self.teacher` trên tập dữ liệu Proxy của Server.

---

## Ước tính Payload

| Thành phần | Kịch bản 1 (r=4) | Kịch bản 2 (r=8) |
|---|---|---|
| LoRA INT8 | 72KB | 144KB |
| cv3.x.2 classifier (nc=4), INT8 | 2KB | 2KB |
| **Tổng** | **~74KB** | **~146KB** |
| Thời gian truyền (15kbps) | ~40s/round | ~78s/round |

> [!TIP]
> Target 146KB qua kênh âm thanh 15kbps (config `BANDWIDTH=4000Hz`) ≈ 78 giây truyền dữ liệu/vòng. Với round dài 10-30 phút thì hoàn toàn khả thi.

---

## Verification Plan

1. Print `payload_kb` sau mỗi call `train_and_get_payload()` — xác nhận ≤ 150KB.
2. Smoke test `--rounds 1` với `--baseline fedkdl` — kiểm tra KD chạy tại Fog (không tại AUV).
3. Kiểm tra `FogNode2D.server_side_kd()` được gọi 1 lần/round sau FedAvg.

> [!IMPORTANT]
> **Quyết định kiến trúc quan trọng cần bạn xác nhận trước khi triển khai:**
>
> Đề xuất **Nhóm 3 (Server-side KD)** xung đột trực tiếp với kiến trúc hiện tại (KD tại AUV). Đây là thay đổi kiến trúc cốt lõi. Nếu triển khai, toàn bộ luồng `SensorWorker2D` → `FogNode2D` phải được viết lại một phần lớn. Bạn có muốn thực hiện không?

---

## Open Questions

> [!NOTE]
> **Q1 — Nhóm 3 (Server-side KD):** Bạn có muốn tôi di chuyển hoàn toàn KD từ AUV lên Fog/Server không? Hay giữ nguyên KD tại AUV (đơn giản hơn, tính toán nặng hơn tại thiết bị)?

> [!NOTE]
> **Q2 — Nhóm 1 (Cắt tỉa Head):** Bạn có muốn chỉ truyền lớp Conv cuối của Head không? Điều này cần phân tích cụ thể kiến trúc YOLO26 head để xác định đúng tên layer (cv3 hay Detect.cv2, cv3).

> [!NOTE]
> **Q3 — Nhóm 1 (Neck-only LoRA):** Bạn có muốn giới hạn LoRA injection chỉ vào phần Neck (layer 12-21 của YOLO26) thay vì toàn backbone + neck? Điều này giúp tiết kiệm thêm ~30% payload nhưng cần biết layer index chính xác của YOLO26.

---

## Proposed Changes (Nếu được duyệt)

### Fix Nhóm 2: Delta Encoding thực sự + Sparse INT8 Pipeline

#### [MODIFY] [trainer.py](file:///d:/Documents/HUST/2022-2026/Research_Thesis/FedKDL/tasks/detection_2d/trainer.py)

- Thay vì trả `new_state` (absolute), tính `delta_state = new_state - state_before` rồi truyền delta đó.
- Kết nối `SparseINT8Payload` vào pipeline để sparse + quantize trước khi trả payload.

#### [MODIFY] [int8_quantization.py](file:///d:/Documents/HUST/2022-2026/Research_Thesis/FedKDL/tasks/detection_2d/knowledge_compression/int8_quantization.py)

- Bổ sung hàm `pack_sparse_payload(state_dict, rho_s)` dùng Top-K thực sự (không phải dense).
- `SparseINT8Payload` đã có sẵn, chỉ cần kết nối vào `SensorWorker2D`.

#### [MODIFY] [simulator.py (2D)](file:///d:/Documents/HUST/2022-2026/Research_Thesis/FedKDL/tasks/detection_2d/simulator.py)

- `SensorWorker2D.train_and_get_payload()` gọi `pack_sparse_payload()` thay vì `pack_payload()`.

---

### Fix Nhóm 3: Server-side KD (Nếu Q1 = Có)

#### [MODIFY] [simulator.py (2D)](file:///d:/Documents/HUST/2022-2026/Research_Thesis/FedKDL/tasks/detection_2d/simulator.py)

- `SensorWorker2D.train_and_get_payload()` gọi `local_sgd_od()` với `use_kd=False` (không truyền Teacher).
- Teacher chỉ được dùng trong `FogNode2D.aggregate_intra_cluster()`.

#### [MODIFY] [simulator.py (2D) — FogNode2D](file:///d:/Documents/HUST/2022-2026/Research_Thesis/FedKDL/tasks/detection_2d/simulator.py)

- Sau FedAvg, `FogNode2D` nhận Teacher từ `Simulator2D`, chạy `KDDetectionTrainer` để distill trên tập synthetic/coco.
- Trả về model đã được distill.

---

## Ước tính Payload sau khi fix đủ

| Thành phần | Hiện tại | Sau fix |
|---|---|---|
| LoRA A+B (r=4, toàn Neck) | ~200KB (FP32) | ~50KB (INT8 sparse, Top-5%) |
| Detection Head Conv cuối | ~120KB (FP32) | ~30KB (INT8) |
| **Tổng** | **~320KB** | **~80KB** |
| Qua kênh âm thanh 15kbps | ~170s/vòng ❌ | ~43s/vòng ✅ |

> [!TIP]
> Với băng thông 15kbps thực tế (config `BANDWIDTH=4000 Hz` → Shannon capacity ~15kbps), target 80KB/vòng ≈ 43 giây là hoàn toàn khả thi cho một vòng FL dài 10-30 phút.

## Verification Plan

1. In ra `payload_kb` sau mỗi vòng train — xác nhận ≤ 100KB.
2. Kiểm tra `delta_norm` có thực sự là delta (state_after - state_before) không phải absolute state.
3. Chạy smoke test `--rounds 1` và so sánh payload size trước/sau fix.

Viewed base_simulator.py:54-119
Ran command: `.venv\Scripts\python.exe main_trainer_od.py --rounds 1 --baseline fedkdl`
Ran command: `.venv\Scripts\python.exe -c "
import sys
import py_compile

files = [
    'config/settings.py',
    'tasks/detection_2d/models/yolo_wrapper.py',
    'tasks/detection_2d/trainer.py',
    'tasks/detection_2d/simulator.py',
    'federated_core/base_simulator.py',
    'main_trainer_od.py'
]

for f in files:
    try:
        py_compile.compile(f, doraise=True)
        print(f'Syntax OK: {f}')
    except Exception as e:
        print(f'Syntax Error in {f}: {e}')
"`
Edited task.md

Tôi đã xác nhận lại toàn bộ cú pháp, import, và logic thực thi theo kiến trúc 3 tầng:

1. **Tier 1 (Sensor AUV - YOLO26n)**:
   - Mã nguồn: `tasks/detection_2d/trainer.py` (Local SGD).
   - Đã gỡ bỏ sạch sẽ module Knowledge Distillation (KD) vì quá nặng.
   - Train YOLO26n thuần, sau đó chỉ cắt đúng các ma trận LoRA và lớp `cv3.x.2` (layer sinh bounding box, kích thước tùy biến theo số lượng class) -> Tổng cộng **~74KB (với rank=4)** hoặc **~146KB (với rank=8)**.
   - Các hidden layers khác của Head và phần Backbone tĩnh bị bỏ qua hoàn toàn để tiết kiệm băng thông tối đa.

2. **Tier 2 (Fog Node/Cụm AUV - Trung gian)**:
   - Mã nguồn: `federated_core/base_simulator.py` và `tasks/detection_2d/simulator.py`.
   - Fog thực hiện `FedAvg` nội bộ (Intra-cluster) và chạy HFL-Selective để vay mượn tri thức liên cụm (Inter-cluster) khi bị đói dữ liệu.
   - Không còn chạy KD ở đây nữa.

3. **Tier 3 (Gateway / Trạm mặt nước - YOLO12l Teacher)**:
   - Tại `base_simulator.py`, sau bước `Phase 3: Global Aggregation` tổng hợp LoRA từ tất cả Fog, tôi đã kích hoạt hook `_gateway_knowledge_distillation()`.
   - Trạm mặt nước với GPU mạnh sẽ nạp trọng số LoRA đó vào Student (`YOLO26n`), sau đó lấy Teacher khổng lồ (`YOLO12l`) chạy `KDDetectionTrainer` (chạy thêm 1 epoch trên proxy data) nhằm uốn nắn mô hình toàn cục tốt hơn trước khi truyền ngược xuống Fog và Sensor.

**Hoàn thiện!**
Mọi thứ đều đã sẵn sàng, và code pass toàn bộ các bài kiểm tra import. Nếu cần thêm gì để phục vụ việc viết báo cáo, bạn hãy cho tôi biết nhé!
