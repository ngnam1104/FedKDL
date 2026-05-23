# 🗺️ Hướng dẫn Khám phá Codebase FedKDL (Advanced Reading Roadmap)

Codebase của dự án **FedKDL** kết hợp mô phỏng vật lý đại dương (IoUT) và Học máy phân tán (Federated Learning). Để hiểu trọn vẹn, hãy đọc theo lộ trình từ vĩ mô đến vi mô dưới đây:

---

## 📍 Bước 1: Khởi tạo Vũ trụ và Luật lệ (Configurations & Entrypoints)

1. **`config/settings.py`**: Trái tim của cấu hình.
   - **Vật lý**: `AcousticChannelConfig` (Tốc độ âm thanh 1500m/s, suy hao Thorp, SNR). `EnergyConfig` định nghĩa tiêu hao pin trên mỗi FLOP và mỗi byte gửi đi.
   - **Học máy**: `FedKDLConfig` cấu hình tỷ lệ LORA_RANK (r=4 hoặc r=8), payload mục tiêu, số lượng Epoch.
2. **`main_trainer_od.py`** (2D Object Detection): Kịch bản chính của hệ thống. File này đóng vai trò Orchestrator, kết nối lưới vật lý và bộ mô phỏng FL.

---

## 🌊 Bước 2: Nền tảng Vật lý Đại dương (Physics Tier)

Bỏ qua các framework FL thông thường, FedKDL ép hệ thống hoạt động dưới ràng buộc vật lý. Đọc thư mục **`physics_models/`**:
- **`topology.py`**: Sinh tọa độ 3D. AUV lặn sâu (500m-1000m), Cluster Head lơ lửng (100m-400m), Gateway nằm trên mặt nước (z=0).
- **`communication.py`**: Tính toán `link_budget` và suy hao âm thanh (Thorp-Wenz model). Quyết định xem thiết bị có đủ công suất để gửi tin đi không.
- **`energy.py`**: Trừ pin. Năng lượng tiêu hao được tính chi tiết cho từng byte gửi đi (truyền thông) và từng dấu phẩy động (FLOPs) khi vi xử lý ARM nội suy gradient.

---

## 🧠 Bước 3: Động cơ Học liên kết (Federated Core)

Đọc thư mục **`federated_core/`** để hiểu cách các nút hợp tác:
- **`base_simulator.py`**: Vòng lặp thời gian thực (Round 1 to T).
   - *Phase 1 (Tier 1)*: Khởi chạy huấn luyện local tại các AUV, tính toán Payload và trừ Pin.
   - *Phase 2 (Tier 2)*: Khởi chạy `aggregate_intra_fog` để Cluster Head tổng hợp `payloads` (chỉ dùng các phép cộng và nhân vô hướng nhẹ bén). Ở đây có áp dụng **Selective Cooperation** (Hợp tác chọn lọc) để các Cluster Head chia sẻ tải cho nhau.
   - *Phase 3 (Tier 3)*: Gửi trọng số lên Gateway và kích hoạt `_gateway_knowledge_distillation`.

---

## 🚀 Bước 4: Ứng dụng AI đột phá (Detection 2D Task)

Đây là nơi chứa đóng góp khoa học chính. Đọc thư mục **`tasks/detection_2d/`**:

### 1. Phía Thiết bị biên (Sensor-side)
- **`models/yolo_wrapper.py`**: File này bọc mạng YOLO. Ở đây, một mạng Student (YOLO26n) được đóng băng (`freeze`) toàn bộ trọng số gốc, chỉ chèn thêm các ma trận **LoRA**. Việc này giúp triệt tiêu hàng tỷ FLOPs backward pass.
- **`simulator.py -> train_and_get_payload()`**: Gọi quá trình Local SGD. Sau khi hội tụ, gradient của các module LoRA được lượng tử hóa thành số nguyên 8-bit bằng cơ chế trong `knowledge_compression/int8_quantization.py`.

### 2. Phía Trạm nổi Mặt nước (Gateway-side KD)
Điểm nhấn lớn nhất là việc Gateway phải gánh vác toàn bộ quá trình Knowledge Distillation (KD). 
- **`knowledge_compression/knowledge_distillation.py`**: Lớp `KDDetectionTrainer` kế thừa Ultralytics.
  - **Logic hoạt động**:
    1. Gateway nhận một tập dữ liệu đại diện $\mathcal{D}_{proxy}$.
    2. Chạy quá trình **Lan truyền xuôi (Forward pass) trên mạng Student** (vừa được cập nhật từ biển sâu) để trích xuất *soft logits* và các biểu diễn đặc trưng (hidden features, attention maps).
    3. Chạy quá trình **Lan truyền xuôi trên mạng Oracle Teacher** (YOLO12-Large) đóng băng, để lấy câu trả lời "mẫu" xuất sắc nhất.
    4. Tính toán hàm mất mát Eq.37 bao gồm: KL Divergence (khoảng cách phân phối dự đoán), MSE của Hidden layers, và MSE của Attention layers.
    5. Cập nhật trọng số mạng Student, ép mô hình nhỏ này tiệm cận năng lực của Oracle, sau đó gửi ngược mô hình này xuống đáy biển.
- **`knowledge_association.py`**: Nơi tính toán khoảng cách EMD (Earth Mover's Distance) để đánh giá độ chệch dữ liệu.

---

## 🛠️ Bước 5: Tiện ích & Triển khai (Utils & Scripts)

- **`utils/generate_all_envs.py`**: Sinh sẵn các kịch bản môi trường (Non-IID $\alpha=0.1$, $\alpha=1.0$) ra file YAML để chạy đối chứng đồng loạt.
- **`run_kdl_experiments.sh`**: Kịch bản tự động hóa Bash để vứt lên server chạy hằng ngày mà không cần tương tác.
- **`scripts/fedkdl/plot_od_comparison.py`**: Script thu thập file log JSON và chuyển đổi thành các biểu đồ đẹp mắt cho báo cáo khoa học.
