# Đề xuất kịch bản đánh giá FedKDL

Mục tiêu của FedKDL không phải tối đa hóa độ chính xác bằng mọi giá, mà là giảm chi phí vận hành của hệ thống IoUT dưới nước thông qua việc tối thiểu hóa năng lượng tiêu thụ và độ trễ truyền thông, trong khi vẫn duy trì chất lượng nhận diện mục tiêu ở mức độ cạnh tranh. Do đó, thực nghiệm được xây dựng theo bốn câu hỏi nghiên cứu chính (RQ).

## RQ1. Kiến trúc phân cấp (HFL) có giúp hệ thống duy trì kết nối ổn định khi AUV bị cô lập không?

### Bài toán

Trong môi trường dưới nước, AUV có thể mất kết nối tạm thời với Gateway do khoảng cách truyền dẫn xa hoặc rào cản vật lý. Ở các hệ thống phẳng truyền thống (Star Topology), điều này dẫn đến số lượng client tham gia FL giảm sút đáng kể.

### Đối thủ so sánh

* **FedAvg** (Star topology: AUV kết nối trực tiếp với Gateway)
* **FedKDL** (Hierarchical topology: AUV kết nối qua các trạm Relay)

### Chỉ số đánh giá

* Tỷ lệ tham gia (Participation Rate)
* Số lượng AUV tham gia huấn luyện thành công

### Kỳ vọng

Hệ thống phẳng (FedAvg) suy giảm mạnh về tỷ lệ tham gia khi mở rộng số lượng AUV trên một vùng địa lý rộng do giới hạn cự ly của sóng âm. FedKDL cải thiện đáng kể khả năng kết nối nhờ sử dụng kiến trúc Relay phân cấp.

---

## RQ2. Cơ chế nén kép (LoRA + INT8) có thực sự mang lại lợi thế về chi phí hệ thống (System Cost) không?

### Bài toán

Kênh truyền thủy âm có băng thông cực kỳ thấp (~15 kbps) và độ trễ lớn, khiến việc truyền toàn bộ mô hình là bất khả thi.

### Đối thủ so sánh

* **FedAvg-HFL**: Truyền toàn bộ mô hình (Full parameter).
* **Top-K Sparsification**: Truyền các gradient lớn nhất.
* **Naive LoRA**: Áp dụng LoRA tiêu chuẩn.
* **FLORA**: Khắc phục nhược điểm của LoRA tiêu chuẩn bằng cách dùng SVD.
* **FedKDL**: Kết hợp LoRA, lượng tử hóa INT8 và mã hóa Huffman.

### Chỉ số đánh giá

* Payload trên mỗi AUV (MB)
* Năng lượng tiêu thụ mỗi vòng (Energy / round)
* Độ trễ truyền tải mỗi vòng (Latency / round)
* Chi phí tích hợp (Joint Objective Cost)
* Best mAP@0.5 (để đối chiếu Trade-off)

### Kỳ vọng

FedAvg-HFL có payload và chi phí hệ thống quá lớn. Các phương pháp như Top-K, Naive LoRA và FLORA giúp giảm tải truyền thông nhưng hiệu năng học (mAP) có thể bị ảnh hưởng. FedKDL đạt payload, năng lượng và độ trễ thấp nhất trong khi vẫn giữ mAP ở mức rất tốt nhờ cơ chế nén kép và Knowledge Distillation.

---

## RQ3. Tầng Relay có giúp xử lý vấn đề Dữ liệu mất cân bằng (Non-IID) không?

### Bài toán

Các AUV thu thập dữ liệu ở các vùng sinh thái biển khác nhau, dẫn đến hiện tượng lệch phân phối (Non-IID) cực kỳ nghiêm trọng, gây ra client-drift.

### Đối thủ so sánh

* **FedAvg-HFL**
* **FedProx-HFL**: Dùng regularization parameter (mu) để chống drift.
* **FLORA**
* **SCAFFOLD**: Dùng Control Variates (Đạt accuracy cao nhất nhưng chi phí payload gấp đôi).
* **FedKDL-NoCoop**: FedKDL nhưng không có hợp tác giữa các Relay.
* **FedKDL**: Tích hợp đầy đủ Relay Cooperation (HFL-Nearest).

### Chỉ số đánh giá

* Training Loss
* Độ trễ mAP@0.5 qua các vòng hội tụ (Learning Curves)

### Kỳ vọng

SCAFFOLD đạt độ chính xác (mAP) cao nhất nhờ xử lý client drift triệt để bằng control variates, tuy nhiên điều này phải đánh đổi bằng payload không tưởng đối với môi trường thủy âm (chứng minh ở RQ2). FedKDL (đầy đủ) là giải pháp thực tiễn nhất, khắc phục đáng kể Non-IID nhờ kết nối Relay hợp tác, có độ chính xác bám sát SCAFFOLD nhưng tiết kiệm dung lượng truyền thông hơn hàng chục lần.

---

## RQ4. Gateway Knowledge Distillation (KD) có bù đắp được sự suy giảm chất lượng do nén cực hạn không?

### Bài toán

Nén mô hình quá mạnh (LoRA + INT8) sẽ gây ra quantization loss và mất mát thông tin đáng kể.

### Đối thủ so sánh

* **No KD** (tương đương với chạy FLORA tiêu chuẩn).
* **Logit KD**: Khôi phục kiến thức dựa trên soft labels từ Teacher.
* **FedKDL (LoRA-Projection KD + Logit KD)**: Đề xuất tổng hợp.
* **Centralized Training**: Mức tiệm cận trên (Upper bound), thu thập toàn bộ dữ liệu về một nơi.

### Chỉ số đánh giá

* mAP@0.5
* Chất lượng phát hiện vật thể (Bouding Box Predictions)

### Kỳ vọng

No KD cho độ chính xác thấp nhất do mất mát khi nén. Logit KD có bù đắp nhưng chưa đủ sâu do tính chất truyền thừa chỉ ở tầng cuối. FedKDL kết hợp LoRA-Projection KD đạt độ chính xác cao nhất (tiến gần hơn tới Centralized), tối ưu hóa được cả việc giữ gìn cấu trúc nội tại của mô hình và nhãn mềm, bù đắp thành công sự hao hụt do INT8 gây ra.
