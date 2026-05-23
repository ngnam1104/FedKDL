# 🗺️ Hướng dẫn Khám phá Codebase FedKDL (Codebase Reading Roadmap)

Codebase của dự án **FedKDL** được thiết kế để kết hợp giữa **mô phỏng vật lý mạng viễn thông dưới nước (IoUT)** và **Học máy phân tán (Federated Learning)**. 
Vì hệ thống trải dài từ việc tính toán năng lượng sóng âm đến tinh chỉnh mạng YOLO khổng lồ, việc đọc code cần có một trình tự nhất định để không bị "ngợp". Dưới đây là lộ trình đọc code chi tiết:

---

## 📍 Bước 1: Luật lệ của Vũ trụ (Configurations & Entrypoints)

Hãy bắt đầu bằng việc hiểu các biến số chi phối toàn bộ hệ thống.
1. **`config/settings.py`**: Đây là nơi chứa toàn bộ "hiến pháp" của mạng lưới.
   - Các biến số về năng lượng, tiêu hao tính toán (FLOPs).
   - Mô hình vật lý (Tốc độ truyền sóng, băng thông, suy hao Thorp-Wenz).
   - Thiết lập các thuật toán (Số round, Epoch, số Rank của LoRA, INT8, v.v).
2. **`main_trainer_od.py`** và **`main_trainer.py`**: Đây là 2 kịch bản chạy chính. Hãy đọc lướt qua hàm `run()` để hình dung vòng đời của một quy trình mô phỏng từ lúc khởi tạo đến lúc trả về kết quả (metrics).

---

## 🌊 Bước 2: Nền tảng Vật lý Đại dương (Physics Models)

Thay vì chạy mô phỏng AI chay, dự án này ép AI phải chạy trên một môi trường vật lý cực kỳ khắc nghiệt.
- Thư mục **`physics_models/`**:
  - Đọc `topology.py`: Cách tạo ra các tọa độ 3D của AUV và Cluster Head.
  - Đọc `communication.py` & `energy.py`: Để hiểu tại sao một payload vài MB lại có thể làm "cháy" pin, và cách tính toán năng lượng gửi đi/nhận về dựa trên SNR, băng thông và khoảng cách.
  - Đọc `latency.py`: Công thức tính độ trễ lan truyền và độ trễ điện toán.

---

## 🧠 Bước 3: Trái tim Học liên kết (Federated Core)

Sau khi hiểu rào cản vật lý, hãy xem cách các trạm giao tiếp với nhau.
- Thư mục **`federated_core/`**:
  - Khám phá `base_simulator.py` / `aggregator.py`: Cách một trạm Gateway điều phối các vòng lặp (Rounds), cách nó quyết định chọn AUV nào tham gia (Lazy Filter, Concept Drift).
  - Đọc cách thuật toán `FedAvg` được tùy biến lại (hợp tác chọn lọc - Selective Cooperation) giữa các trạm sương mù (Cluster Head) để tiết kiệm năng lượng.

---

## 🚀 Bước 4: Ứng dụng AI cốt lõi (Tasks)

Đây là nơi chứa các khối logic Học máy. Bạn sẽ thấy 2 nhánh rõ rệt:
1. **`tasks/anomaly_1d/`**: (Nhánh baseline dễ)
   - Xem cách dùng mạng `Autoencoder` (khoảng 54k params) để quét dữ liệu 1D.
   - Xem logic nén Top-K Sparsification kết hợp lượng tử hóa ở `knowledge_compression/`.
   
2. **`tasks/detection_2d/`**: (Nhánh **Đột phá** của bài báo)
   - **`yolo_student.py`**: Cách mạng nơ-ron sinh viên bị "khóa" (freeze) toàn bộ trọng số gốc và chỉ cấy thêm ma trận LoRA.
   - **`gateway_kd.py`**: Xem cách máy chủ Oracle (Gateway) chạy một giáo viên (Teacher) khổng lồ YOLOv12-Large và truyền lại tri thức (Knowledge Distillation) cho mô hình Global.
   - Khám phá `knowledge_association.py`: Xem cách tính khoảng cách EMD (Earth Mover's Distance) để giải quyết bài toán dữ liệu phân mảnh (Non-IID).

---

## 🛠️ Bước 5: Tự động hóa & Tiện ích (Utils & Scripts)

Cuối cùng, sau khi nắm vững logic, hãy xem cách chạy hàng trăm thực nghiệm tự động:
- **`utils/`**: Các script hỗ trợ tải dữ liệu từ Kaggle (`download_datasets.py`), sinh môi trường giả lập, lưu log JSON.
- **Bash Scripts (`quick_start.sh`, `run_kdl_experiments.sh`)**: Cách setup môi trường `.venv` và khởi chạy vòng lặp cho các kịch bản đối chứng (baselines).
- **`scripts/fedkdl/plot_od_comparison.py`**: Script bóc tách log JSON và vẽ đồ thị Matplotlib cực đẹp cho bài báo.

---

### 💡 Lời khuyên (Pro-tips) khi đọc code:
- Đừng đi quá sâu vào các hàm toán học lượng tử INT8 hay EMD ngay từ đầu. Hãy hiểu Interface (Đầu vào - Đầu ra) của chúng trước.
- **Flow tổng quát để nắm bài:** Môi trường (`settings.py`) $\rightarrow$ Khởi tạo (`topology`) $\rightarrow$ Vòng lặp Học máy (`main_trainer`) $\rightarrow$ Cập nhật AI ở biên (`LoRA`, `Sensor`) $\rightarrow$ Thu thập ở trạm nổi (`Gateway KD`).
