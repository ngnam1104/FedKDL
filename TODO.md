## Mục tiêu cốt lõi: Tối thiểu hóa Năng Lượng + Độ Trễ trong khi vẫn Đảm bảo Chất lượng Mô hình (FedKDL)

### 1. Nén kép để giảm thiểu chi phí giao tiếp

Phần này xác minh hiệu quả của cơ chế kết hợp tinh chỉnh LoRA và lượng tử hóa INT8 trong việc giải quyết bài toán nghẽn băng thông của kênh truyền thủy âm.

* **Phương pháp đối chiếu:** FedAvg \cite{mcmahan2017communication}: Không có lớp trung gian, truyền toàn bộ weights, Top-K Sparsification \cite{lin2017deep}: Truyền toàn bộ weights sau khi nén top-k, FLORA \cite{wang2024flora}: , FedKDL: Truyền trọng số LoRA
* **Tác dụng chính:** Đánh giá trực tiếp khả năng **giảm dung lượng gói tin (payload), tối ưu độ trễ và tiết kiệm năng lượng** trung bình trên mỗi vòng giao tiếp, đồng thời kiểm tra chéo xem mức độ **duy trì độ chính xác (mAP@0.5)** có bị ảnh hưởng khi ép nén hay không.

---

### 2. Tổng hợp cấp độ Relay xử lý Dữ liệu Non-IID

Kiểm tra tính bền bỉ của hệ thống khi dữ liệu tại các AUV bị phân mảnh và mất cân bằng nghiêm trọng (do sinh vật phân bố theo độ sâu). Bao gồm một nghiên cứu cắt bỏ (ablation study) đánh giá 3 phiên bản: Không SVD, Không Relay và Full FedKDL.

* **Phương pháp đối chiếu:** FedAvg \cite{mcmahan2017communication}, FedProx \cite{li2020federated}, FLORA \cite{wang2024flora}, FedKDL.
* **Tác dụng chính:** Kiểm tra khả năng **duy trì độ chính xác hội tụ của mô hình**, theo dõi hàm loss và tỷ lệ tham gia (participation rate) để chứng minh trạm trung chuyển giúp hệ thống không bị "đứt gãy" khi gặp dữ liệu Non-IID khắt khe.

---

### 3. Chưng cất tri thức (KD) tại Gateway bù đắp hao hụt chính xác

Xác minh xem việc áp dụng chưng cất tri thức ở cụm Gateway có vớt vát lại được lượng mAP bị mất đi do quá trình nén mô hình (ở phần 1) hay không, trong khi vẫn giữ mô hình đủ nhẹ.

* **Phương pháp đối chiếu:** No KD, Logit KD \cite{hinton2015distilling}, SPKD \cite{tung2019similarity}, LoRA-Projection KD (FedKDL).
* **Tác dụng chính:** Trực tiếp **kiểm tra việc duy trì và phục hồi độ chính xác (mAP)**, đồng thời đánh giá **chi phí tài nguyên (mức tiêu thụ bộ nhớ)** khi chạy các thuật toán KD khác nhau.

---

### 4. Phân tích đánh đổi Hiệu năng Hệ thống tổng thể

Đặt FedKDL vào môi trường mô phỏng dưới nước thực tế với các ràng buộc về phạm vi giao tiếp khác nhau, tạo ra các kịch bản AUV bị mất kết nối hoặc cô lập.

* **Phương pháp đối chiếu:** FedAvg, Top-K Sparsification, FLORA, FedKDL.
* **Tác dụng chính:** Đánh giá toàn diện sự đánh đổi (trade-off) giữa **Độ trễ - Năng lượng tiêu thụ - Độ chính xác nhận diện**. Đo lường tỷ lệ rớt mạng để khẳng định tính thực tiễn của framework trong điều kiện kết nối chập chờn.
