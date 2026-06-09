# Đề xuất kịch bản đánh giá FedKDL

Mục tiêu của FedKDL không phải tối đa hóa độ chính xác bằng mọi giá, mà là giảm chi phí vận hành của hệ thống IoUT dưới nước thông qua việc tối thiểu hóa năng lượng tiêu thụ và độ trễ truyền thông, trong khi vẫn duy trì chất lượng nhận diện mục tiêu ở mức chấp nhận được. Do đó, thực nghiệm được xây dựng theo bốn câu hỏi nghiên cứu chính (RQ).

## RQ1. FedKDL có giúp hệ thống hoạt động ổn định hơn khi AUV bị cô lập không?

### Bài toán

Trong môi trường dưới nước, AUV có thể mất kết nối tạm thời do khoảng cách truyền dẫn, suy hao sóng âm hoặc thay đổi vị trí. Khi đó số lượng client tham gia FL giảm và chất lượng mô hình có thể suy giảm.

### Đối thủ so sánh

* FedAvg
* FedProx
* FedKDL

### Chỉ số đánh giá

* mAP@0.5
* Training Loss
* Participation Rate
* Số lượng AUV bị cô lập
* Năng lượng tiêu thụ
* Độ trễ

### Kỳ vọng

FedAvg suy giảm mạnh khi xuất hiện client bị cô lập. FedProx cải thiện hội tụ nhưng vẫn chịu giới hạn của kiến trúc phẳng. FedKDL đạt kết quả tốt nhất nhờ tầng Relay giúp duy trì kết nối và tăng tỷ lệ tham gia huấn luyện.

## RQ2. Cơ chế nén kép có thực sự giảm chi phí truyền thông không?

### Bài toán

Kênh truyền thủy âm có băng thông thấp và độ trễ lớn, khiến việc truyền toàn bộ mô hình trở nên tốn kém.

### Đối thủ so sánh

* FedAvg
* Top-K Sparsification
* FLORA
* FedKDL

### Chỉ số đánh giá

* Payload
* Năng lượng tiêu thụ
* Độ trễ
* mAP@0.5

### Kỳ vọng

FedAvg có chi phí truyền thông cao nhất. Top-K giảm payload nhưng vẫn phải tính toàn bộ gradient. FLORA giảm đáng kể chi phí nhờ LoRA. FedKDL đạt payload, năng lượng và độ trễ thấp nhất nhờ kết hợp LoRA và INT8.

## RQ3. Tầng Relay có giúp xử lý dữ liệu Non-IID không?

### Bài toán

Các AUV hoạt động ở những vùng sinh thái khác nhau nên dữ liệu thu được bị lệch phân phối nghiêm trọng, gây hiện tượng client drift.

### Đối thủ so sánh

* FedAvg
* SCAFFOLD: cải thiện hội tụ nhờ control variates, giảm drift tốt hơn FedAvg.
* FLORA
* FedKDL không Relay Cooperation
* FedKDL

### Chỉ số đánh giá

* Training Loss
* mAP@0.5

### Kỳ vọng

FedAvg hội tụ chậm và dao động mạnh. FLORA cải thiện chi phí truyền thông nhưng chưa xử lý trực tiếp Non-IID. FedKDL cho kết quả tốt nhất nhờ SVD aggregation và relay cooperation.

## RQ4. Gateway Knowledge Distillation có bù được phần accuracy bị mất do nén mô hình không?

### Bài toán

LoRA và INT8 giúp giảm chi phí truyền thông nhưng có thể làm suy giảm chất lượng mô hình.

### Đối thủ so sánh

* No KD
* Logit KD
* LoRA-Projection KD (FedKDL)
* Centralized training: Huấn luyện tập trung tại Gateway với toàn bộ dữ liệu

### Chỉ số đánh giá

* mAP@0.5
* Training Loss

### Kỳ vọng

No KD cho độ chính xác thấp nhất. Logit KD cải thiện chất lượng mô hình. LoRA-Projection KD đạt độ chính xác cao nhất trong khi vẫn duy trì chi phí bộ nhớ thấp. Centralized training cho độ chính xác cao nhất nhưng chi phí cao nhất.
