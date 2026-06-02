# Kế hoạch Hoàn thiện Bài báo FedKDL (Q1 Paper)

Danh sách 7 Sơ đồ Kiến trúc (Conceptual Figures) mang tính quyết định để ăn điểm Reviewers:

- [ ] **Sơ đồ 1: Kiến trúc Tổng quan IoUT**
  - *Tên:* The Overall Architecture of FedKDL in Internet of Underwater Things (IoUT).
  - *Mô tả:* Bức tranh toàn cảnh 3 tầng nước (Deep Layer: AUV đóng băng mạng + LoRA, Mid-water: Relay với SVD-LoRA, Surface: Gateway với bộ não Teacher/Student và LoRA-Projection KD).
- [x] **Sơ đồ 2: Kênh truyền Vật lý & Nhiễu Wenz**
  - *Tên:* Physical Link and Feasibility Graph under Thorp-Wenz Noise.
  - *Mô tả:* Đồ thị liên kết mạng, đường truyền ngắn xanh, đường dài đỏ vỡ vụn do hàm rào cản nhiễu Wenz.
- [] **Sơ đồ 3: Phân mảnh Dữ liệu Sinh thái Biển sâu (Non-IID)** (Còn lỗi, cần sửa sau)
  - *Tên:* Depth-Habitat Non-IID Data Distribution model via Gaussian-Dirichlet mechanisms.
  - *Mô tả:* Lát cắt đại dương, các đường Gaussian đại diện cho sinh vật theo độ sâu để giải thích sự phân mảnh Non-IID.
- [x] **Sơ đồ 4: Giải phẫu mạng YOLO tiêm LoRA và Lượng tử hóa** (Đã vẽ bằng TikZ)
  - *Tên:* Low-Rank Adaptation and Asymmetric Delta INT8 Quantization on YOLO Backbone.
  - *Mô tả:* Bóc tách phần "ruột" AI tại AUV. Luồng bypass với hình thang A, B và khối lượng tử hóa INT8.
- [x] **Sơ đồ 5: Tổng hợp chéo SVD-LoRA tại Relay** (Đã vẽ bằng TikZ)
  - *Tên:* Subspace Misalignment Correction via SVD-LoRA Aggregation at Relay Nodes.
  - *Mô tả:* Minh họa sự vượt trội của FedKDL (SVD tái phân rã) so với FedAvg truyền thống.
- [x] **Sơ đồ 6: Chưng cất Tri thức qua LoRA-Projection tại Gateway** (Đã vẽ bằng TikZ)
  - *Tên:* LoRA-Projection Knowledge Distillation mechanism at Surface Gateway.
  - *Mô tả:* Teacher và Student chưng cất tri thức trực tiếp qua điểm neo $h = A \cdot x$ để tránh tràn RAM.
- [] **Sơ đồ 7: Kiến trúc Tổng thể YOLOv12 tiêm Adaptive LoRA** (Đã vẽ bằng TikZ)
  - *Tên:* Full YOLOv12 Architecture with Adaptive LoRA Injection and Asymmetric Quantization.
  - *Mô tả:* Sơ đồ luồng dữ liệu toàn bộ mạng YOLOv12 (Backbone, PAN-FPN Neck, Head) kèm các khối LoRA bypass và trích xuất $\Delta W_q$.

## Các công việc khác

- [ ] Chèn 7 khung `\begin{figure}` vào file `FedKDL.tex` với Caption chuẩn hàn lâm.
- [ ] Chạy code Python và plot đồ thị cho 7 Kịch bản Thực nghiệm.
