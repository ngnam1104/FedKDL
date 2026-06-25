# FedKDL Demo Implementation Plan

## Mục tiêu demo

Demo được chia thành 4 phần để trình bày đủ câu chuyện với giảng viên:

1. **Detection Demo**
   - Upload ảnh dưới nước.
   - Chạy YOLO inference thật bằng checkpoint trong thư mục `demo/` nếu có.
   - Hiển thị bounding box, nhãn, confidence.

2. **Centralized Raw-Data Demo**
   - Mô phỏng kịch bản AUV gửi ảnh thô về Gateway.
   - Nhấn mạnh đây là upper-bound trực quan nhưng payload dữ liệu thô rất lớn.
   - Không train live; chỉ hiển thị pipeline và thống kê payload ước lượng.

3. **FedAvg Round Replay**
   - Replay 3 vòng FL từ CSV:
     - `demo/fedavg_hfl_results.csv`
     - `demo/fedavg_hfl_loss_matrix.csv`
   - Hiển thị AUV local loss, payload full-model, latency, energy, mAP.
   - Mô phỏng luồng AUV gửi full model update lên Gateway.

4. **FedKDL Round Replay**
   - Replay 3 vòng FL từ CSV:
     - `demo/fedkdl_metrics.csv`
     - `demo/fedkdl_loss_matrix.csv`
   - Hiển thị AUV local loss, LoRA/INT8 payload, relay aggregation, Gateway KD.
   - Mô phỏng luồng AUV -> Relay -> Gateway.

## Nguồn dữ liệu

- Backend đọc CSV bằng thư viện chuẩn `csv`, không phụ thuộc pandas.
- Round demo mặc định là 1, 2, 3. Round 0 trong CSV được xem như pre-round/eval ban đầu và không dùng cho replay chính.
- Nếu file CSV thiếu cột nào, API trả giá trị `0` hoặc `None` để UI không sập.

## Backend API

Các endpoint cần có:

- `GET /api/auvs`
- `POST /api/detect/{auv_id}`
- `GET /api/demo/summary`
- `GET /api/demo/round/{case_name}/{round_id}`
- `GET /api/demo/centralized`

`case_name` hỗ trợ:

- `fedavg`
- `fedkdl`

## Frontend

Frontend giữ phong cách dashboard hiện tại nhưng thêm tab:

- `Detection`
- `Centralized`
- `FedAvg`
- `FedKDL`

Mỗi tab có nội dung riêng:

- Detection: upload ảnh và kết quả infer.
- Centralized: pipeline gửi ảnh thô và payload ước lượng.
- FedAvg/FedKDL: selector round, flow diagram, metrics cards, AUV loss matrix.

## Checkpoint inference

Backend ưu tiên tìm model trong `demo/`:

1. `demo/yolo12n_lora_centralized.pt`
2. `demo/yolo12n_centralized.pt`
3. `demo/best.pt`
4. `demo/yolo12n_warmup.pt`
5. repo root `yolo12n_warmup.pt`
6. fallback `yolo12n.pt`

Nhờ vậy sau này chỉ cần thả file `.pt` vào `demo/` là demo tự dùng.

## Không train live

Demo không gọi training trong UI. Các vòng FL được replay từ log/CSV thật để tránh rủi ro thời gian và môi trường GPU khi bảo vệ.
