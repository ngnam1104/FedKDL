Cấu trúc 4 Kịch bản Tinh chỉnh:
Kịch bản 1: Rào cản Không gian và Năng lượng của Kiến trúc Phẳng (Topology & Non-IID)

Mục tiêu: Chứng minh mạng Phẳng (Flat) là bất khả thi dưới biển.
Thiết lập: So sánh kiến trúc HFL của FedKDL với các thuật toán Flat (FedAvg, FedProx, SCAFFOLD).
Luận điểm: Các kiến trúc phẳng bắt AUV truyền thẳng lên mặt nước, đòi hỏi công suất âm thanh khổng lồ $\rightarrow$ Vắt kiệt pin cực nhanh. Hơn nữa, do sóng yếu, tỷ lệ rớt gói (Packet Loss) cực cao khiến Gateway không nhận đủ gradient, cộng thêm dữ liệu bị Non-IID theo độ sâu $\rightarrow$ Mô hình phân kỳ hoàn toàn.
Kịch bản 2: Giải phẫu Cơ chế Nén và Sai lệch Không gian con (Compression & Subspace Misalignment)

Mục tiêu: Chứng minh SVD-LoRA là cơ chế nén tối ưu nhất.
Thiết lập: Tất cả các baseline trong Kịch bản 2 đều được đặt trên kiến trúc HFL 3 tầng (để đảm bảo không rớt gói, giả định lý tưởng). Ta so sánh: HFL + Full Parameter, HFL + Top-K (Sparsification), HFL + Naive LoRA, và HFL + SVD-LoRA.
Luận điểm:
Full Param: Vẫn tốn quá nhiều pin viễn thông để truyền hàng MB dữ liệu qua Relay.
Top-K: Phải tính toán toàn bộ đạo hàm ngược (Backprop) ở AUV rồi mới lọc, gây hao pin CPU. Việc vứt bỏ ngẫu nhiên ma trận cũng làm hỏng đặc trưng không gian 2D.
Naive LoRA: Nén tốt, nhưng lấy trung bình trực tiếp sinh ra sai lệch tích chéo (cross-terms mismatch) khiến mAP phân kỳ.
SVD-LoRA (FedKDL): Triệt tiêu hoàn toàn lỗi toán học, duy trì mAP tiệm cận với Centralized.
Kịch bản 3: Nút thắt Chưng cất Tri thức Thị giác 2D (Knowledge Distillation Mismatch)

Mục tiêu: Khẳng định sự thất bại của các kỹ thuật KD cũ trên bài toán Object Detection.
Thiết lập: Áp dụng HFL + SVD-LoRA, thay đổi bộ KD ở Gateway: Không dùng KD, dùng Logit-KD (FedMD), dùng Feature-KD (FedProto), và LoRA-Projection KD (FedKDL).
Luận điểm:
Logit-KD: Bị "thiếu máu" thông tin vì object detection cần trích xuất không gian cực dày đặc (dense prediction).
Feature-KD: Gây xung đột chiều dữ liệu (Feature Mismatch) vì mạng Teacher (YOLO12l) và Student (YOLO12n) khác nhau về số kênh (channels). Ép khớp feature map gây tràn RAM.
LoRA-Projection KD: Khớp nối mượt mà qua các không gian con hạng thấp, giúp Student thừa hưởng sức mạnh thị giác ưu việt.
Kịch bản 4: Đánh giá Chi phí Tổng thể (Joint Cost Assessment)

Mục tiêu: Tổng kết toàn diện bài toán tối ưu hóa.
Thiết lập: Tính toán hàm chi phí $F = \text{Energy} + \text{Latency}$ từ kết quả của cả 3 kịch bản trên.
Luận điểm: Đồ thị hóa chi phí tổng thể qua từng vòng lặp. FedKDL là hệ thống duy nhất liên tục suy giảm chi phí $F$ và duy trì tuổi thọ cho bầy đàn IoUT.
