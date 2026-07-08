# FedKDL Demo Implementation Plan

## Mục tiêu demo

Demo hiện tại gồm bốn luồng chính:

1. **Detection**
   - Upload ảnh hoặc chọn ảnh mẫu URPC có sẵn trên server.
   - Chạy YOLO inference thật bằng checkpoint trong `demo/`.
   - Hiển thị ảnh đã vẽ bounding box, nhãn, confidence và latency.

2. **Centralized**
   - Minh họa kịch bản AUV gửi ảnh thô về Gateway qua relay.
   - Hiển thị payload ảnh thô, latency vật lý và upper-bound mAP50.

3. **FedAvg/FedAvg-HFL/FedKDL Round Replay**
   - Replay topology, phase truyền thông và metric theo round từ CSV.
   - Gắn log train thật vào phase `train` để local loss của từng AUV thay đổi theo batch.
   - Với `Replay 40`, log train được minh họa cho 10 round đầu; các round sau tiếp tục replay theo CSV/phase.

4. **Training Log Replay**
   - Nút `Replay log` tua riêng 10 round đầu từ log train thật.
   - Không chạy training mới; chỉ đọc log đã train để minh họa tiến trình local training nhanh và ổn định.

## Nguồn dữ liệu

- Metrics theo round:
  - `demo/fedavg_metrics.csv`
  - `demo/fedavg_hfl_results.csv`
  - `demo/fedkdl_metrics.csv`
- AUV loss cuối round:
  - `demo/fedavg_loss_matrix.csv`
  - `demo/fedavg_hfl_loss_matrix.csv`
  - `demo/fedkdl_loss_matrix.csv`
- Batch-level training log:
  - `demo/fedavg_flat_train.log`
  - `demo/fedavg_hfl_train.log`
  - `demo/fedkdl_train.log`
- Checkpoint inference ưu tiên:
  - `demo/student_lora_best.pt`
  - các checkpoint YOLO fallback trong `demo/` hoặc repo root.

## Backend API

- `GET /api/demo/summary`
- `GET /api/demo/scenario/{case_name}/{round_id}`
- `GET /api/demo/training-log/{case_name}?max_rounds=10`
- `GET /api/demo/sample-images`
- `GET /api/demo/sample-image/{image_id}`
- `POST /api/detect`
- `POST /api/detect/{auv_id}`

`case_name` hỗ trợ:

- `fedavg_flat`
- `fedavg_hfl`
- `fedkdl`
- `centralized` ở endpoint scenario.

## Frontend

Tab hiện tại:

- `Detection`
- `Centralized`
- `FedAvg`
- `FedAvg-HFL`
- `FedKDL`

Trong Detection, người xem có thể chọn ảnh mẫu để chạy inference ngay, không cần upload thủ công.

Trong các tab FL, phase `train` đọc log thật và hiển thị local loss từng AUV thay đổi theo batch. Telemetry bên phải hiển thị payload, latency, global loss, pre-Gateway mAP50, mAP50, mAP50-95 và energy.

## Ghi chú vận hành

Các file `.log` bị ignore bởi rule `*.log`. Khi cần đưa ba log demo lên git/server, dùng:

```bash
git add -f demo/fedavg_flat_train.log demo/fedavg_hfl_train.log demo/fedkdl_train.log
```
