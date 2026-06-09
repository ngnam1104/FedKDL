# Đề xuất kịch bản đánh giá FedKDL

## Mục tiêu tổng thể

Mục tiêu của FedKDL không phải tối đa hóa độ chính xác bằng mọi giá, mà là giảm chi phí vận hành của hệ thống IoUT dưới nước thông qua việc tối thiểu hóa năng lượng tiêu thụ và độ trễ truyền thông, trong khi vẫn duy trì chất lượng mô hình nhận diện mục tiêu ở mức chấp nhận được.

Do đó, toàn bộ thực nghiệm được xây dựng xoay quanh bốn câu hỏi nghiên cứu chính tương ứng với bốn thách thức đã được phát biểu trong phần Problem Statement.

# Kịch bản 1. Giảm nghẽn truyền thông bằng cơ chế nén kép

## Bài toán

Trong môi trường truyền thông thủy âm, việc truyền toàn bộ tham số mô hình giữa các AUV và Gateway tạo ra tải lượng dữ liệu rất lớn, kéo theo độ trễ cao và tiêu hao năng lượng đáng kể. Đây là rào cản lớn nhất đối với việc triển khai Federated Learning cho các mô hình thị giác máy tính dưới nước.

## Mục tiêu đánh giá

Kiểm tra liệu việc kết hợp LoRA và lượng tử hóa INT8 có thực sự làm giảm chi phí truyền thông mà không gây suy giảm đáng kể độ chính xác của mô hình hay không.

## Các phương pháp đối chiếu

* FedAvg: truyền toàn bộ tham số mô hình.
* Top-K Sparsification: chỉ truyền các gradient quan trọng nhất.
* FLORA: chỉ truyền tham số LoRA.
* FedKDL: truyền tham số LoRA đã được lượng tử hóa INT8.

## Chỉ số đánh giá

* Payload trung bình mỗi vòng (KB/round).
* Độ trễ trung bình mỗi vòng (s/round).
* Năng lượng tiêu thụ trung bình mỗi vòng (J/round).
* mAP@0.5.

## Kết quả kỳ vọng

FedAvg được kỳ vọng tạo ra payload lớn nhất do phải truyền toàn bộ trọng số mô hình. Điều này dẫn đến độ trễ và năng lượng tiêu thụ cao nhất.

Top-K Sparsification giúp giảm lượng dữ liệu truyền đi nhưng vẫn phải tính toán gradient đầy đủ trước khi lựa chọn các phần tử quan trọng. Vì vậy chi phí tính toán và năng lượng vẫn tương đối lớn.

FLORA giảm mạnh payload nhờ chỉ truyền các ma trận LoRA hạng thấp, từ đó giảm đáng kể độ trễ và năng lượng.

FedKDL tiếp tục giảm chi phí truyền thông thông qua lượng tử hóa INT8, đạt payload thấp nhất trong khi vẫn duy trì độ chính xác gần tương đương FLORA.

Mục tiêu cuối cùng của kịch bản này là chứng minh rằng nút thắt truyền thông của FL dưới nước có thể được giải quyết thông qua cơ chế nén kép LoRA + INT8.

# Kịch bản 2. Xử lý dữ liệu Non-IID bằng tổng hợp phân cấp tại Relay

## Bài toán

Trong môi trường biển sâu, các loài sinh vật phân bố theo từng vùng sinh thái khác nhau. Các AUV hoạt động ở các độ sâu khác nhau sẽ quan sát các phân bố dữ liệu khác nhau, dẫn đến hiện tượng Non-IID nghiêm trọng.

Dữ liệu lệch phân bố thường làm giảm chất lượng tổng hợp toàn cục và gây ra hiện tượng client drift.

## Mục tiêu đánh giá

Kiểm tra khả năng duy trì hội tụ của FedKDL khi dữ liệu phân tán theo đặc điểm sinh thái biển sâu.

## Các phương pháp đối chiếu

* FedAvg.
* FLORA.
* FedKDL.

## Nghiên cứu cắt bỏ (Ablation)

* FedKDL không sử dụng SVD Aggregation.
* FedKDL không sử dụng Relay Cooperation.
* FedKDL đầy đủ.

## Chỉ số đánh giá

* mAP@0.5.
* Training Loss.
* Participation Rate.

## Kết quả kỳ vọng

FedAvg được kỳ vọng hội tụ chậm và dao động mạnh do ảnh hưởng trực tiếp của dữ liệu Non-IID.

FLORA giảm chi phí truyền thông nhưng vẫn chịu ảnh hưởng của sai lệch giữa các không gian con LoRA khi tổng hợp trực tiếp.

FedKDL cải thiện độ ổn định nhờ quá trình tái cấu trúc SVD trước khi tổng hợp và nhờ trao đổi mô hình giữa các Relay lân cận.

Trong nghiên cứu cắt bỏ, phiên bản không có SVD được kỳ vọng suy giảm độ chính xác do mất tính nhất quán của cấu trúc hạng thấp. Phiên bản không có Relay Cooperation vẫn hội tụ nhưng kém hơn Full FedKDL do thiếu cơ chế trao đổi tri thức giữa các vùng sinh thái khác nhau.

Mục tiêu của kịch bản này là chứng minh rằng kiến trúc Relay không chỉ đóng vai trò truyền dẫn mà còn giúp xử lý dữ liệu Non-IID hiệu quả hơn.

# Kịch bản 3. Bù đắp suy giảm độ chính xác bằng Knowledge Distillation

## Bài toán

Các cơ chế nén mô hình thường giúp giảm chi phí truyền thông nhưng đồng thời làm mất một phần năng lực biểu diễn của mô hình, dẫn đến suy giảm độ chính xác.

## Mục tiêu đánh giá

Kiểm tra khả năng phục hồi hiệu năng của mô hình sau khi áp dụng các kỹ thuật nén mạnh.

## Các phương pháp đối chiếu

* No KD.
* Logit KD.
* Similarity Preserving KD (SPKD).
* LoRA-Projection KD (FedKDL).

## Chỉ số đánh giá

* mAP@0.5.
* Training Loss.
* Memory Consumption.

## Kết quả kỳ vọng

No KD có độ chính xác thấp nhất do không có cơ chế truyền tri thức bổ sung.

Logit KD cải thiện độ chính xác thông qua việc học từ đầu ra của Teacher.

SPKD đạt độ chính xác cao nhờ sử dụng thông tin đặc trưng trung gian, tuy nhiên tiêu tốn nhiều bộ nhớ do phải lưu trữ các ma trận tương quan.

LoRA-Projection KD được kỳ vọng đạt độ chính xác gần tương đương SPKD nhưng với chi phí bộ nhớ thấp hơn đáng kể.

Kịch bản này nhằm chứng minh rằng Knowledge Distillation tại Gateway có thể phục hồi phần lớn độ chính xác bị mất do quá trình nén mô hình.

# Kịch bản 4. Đánh giá đánh đổi hiệu năng toàn hệ thống

## Bài toán

Một hệ thống IoUT thực tế không thể chỉ tối ưu độ chính xác. Các ràng buộc về năng lượng, độ trễ và khả năng kết nối phải được xem xét đồng thời.

## Mục tiêu đánh giá

Đánh giá hiệu quả tổng thể của FedKDL dưới góc nhìn hệ thống.

## Các phương pháp đối chiếu

* FedAvg.
* Top-K Sparsification.
* FLORA.
* FedKDL.

## Chỉ số đánh giá

* mAP@0.5.
* Payload.
* Latency.
* Energy Consumption.
* Participation Rate.

## Kết quả kỳ vọng

FedAvg đạt độ chính xác tương đối tốt nhưng phải trả giá bằng độ trễ và năng lượng rất cao.

Top-K giảm chi phí truyền thông nhưng vẫn tồn tại chi phí tính toán đáng kể.

FLORA tạo ra sự cân bằng tốt giữa độ chính xác và chi phí truyền thông.

FedKDL được kỳ vọng đạt độ chính xác gần FLORA nhưng có payload, độ trễ và năng lượng thấp hơn nhờ cơ chế lượng tử hóa và kiến trúc phân cấp.

Ngoài ra, khi giảm phạm vi truyền thông, FedAvg sẽ xuất hiện nhiều AUV bị cô lập và không thể tham gia huấn luyện. Ngược lại, FedKDL tận dụng Relay trung gian để duy trì tỷ lệ tham gia cao hơn, từ đó đảm bảo tính liên tục của quá trình học liên kết.

Kịch bản này đóng vai trò xác nhận cuối cùng rằng FedKDL đạt được sự cân bằng tốt nhất giữa độ chính xác nhận diện và chi phí vận hành của hệ thống.
