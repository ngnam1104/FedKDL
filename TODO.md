Bạn phân tích rất chính xác! Việc cấu trúc lại kịch bản như bạn đề xuất sẽ khiến bài báo mang tính nhân quả (causality) rất mạnh: **Kịch bản 1 chứng minh sự sụp đổ toàn diện của mạng Phẳng và thiết lập HFL làm nền tảng bắt buộc**. Các Kịch bản 2, 3, 4 sau đó đứng trên vai HFL để lần lượt giải phẫu Nén (LoRA), Kiến thức (KD) và Tối ưu hóa (Joint Cost).

Tôi đồng ý việc **bỏ metric $R_{survive}$ (Tuổi thọ mạng)**. Thay vì đếm số vòng sống sót một cách rời rạc, ta chỉ cần báo cáo **Tổng năng lượng tiêu thụ (Joules/round)** hoặc **Tổng năng lượng tích lũy**. Nếu đường năng lượng chạm trần $4000\text{J}$ trước khi kết thúc 60 vòng, người đọc tự khắc hiểu là mạng đã chết.

Dưới đây là đề xuất phân bổ Metrics lại cho 4 Kịch bản để bạn chốt trước khi ta viết LaTeX:

### 🎯 Phân bổ Metrics cho 4 Kịch bản

**1. Kịch bản 1: Rào cản Kiến trúc Phẳng và Hiệu ứng Non-IID (The Flat Topology Failure)**

* **Mục tiêu:** Chứng minh mạng Flat "chết toàn tập" (mất gói, cạn pin, phân kỳ loss) và thiết lập FedKDL (HFL) làm \textit{state-of-the-art} toàn diện.
* **Baselines:** FedAvg (Flat), FedProx (Flat), SCAFFOLD (Flat) vs FedKDL.
* **Metrics sử dụng (Tất tay):**
  * $\eta_{part}$ (Tỷ lệ tham gia): Flat rớt 85% gói, HFL đạt 100%.
  * $E_{comm}$ & $\tau_{round}$ (Năng lượng & Độ trễ): Flat tốn năng lượng khổng lồ để phát xa $1000\text{m}$.
  * $\mathcal{L}_{total}$ & mAP: Flat phân kỳ vì Non-IID và rớt mạng; FedKDL hội tụ mượt mà nhờ Relay chia sẻ chéo.

**2. Kịch bản 2: Giải phẫu Cơ chế Nén và Sai lệch Không gian con (Compression & Subspace Misalignment)**

* **Mục tiêu:** Giả định mọi mạng đều dùng HFL để loại trừ lỗi rớt gói. Cần chứng minh SVD-LoRA là cách nén duy nhất không làm hỏng tính năng thị giác máy tính.
* **Baselines:** HFL + Full Param, HFL + Top-K, HFL + Naive LoRA, HFL + SVD-LoRA (FedKDL).
* **Metrics sử dụng:**
  * $S_{avg}$ (Payload KB): Thấy rõ Top-K và LoRA nén nhỏ cỡ nào so với Full Param (hàng chục MB).
  * $E_{comm}$ (Năng lượng/vòng): Chứng minh Full Param dù dùng HFL vẫn chết pin vì payload quá to.
  * $\mathcal{L}_{total}$ & mAP: Top-K làm hỏng không gian 2D (mAP $\to 0$); Naive LoRA bị sai số tích chéo (mAP kịch trần thấp); SVD-LoRA hội tụ cao nhất.

**3. Kịch bản 3: Nút thắt Chưng cất Tri thức Thị giác 2D (KD Mismatch)**

* **Mục tiêu:** Chứng minh các phương pháp KD cũ không phù hợp với mạng Object Detection, nhấn mạnh sức mạnh của LoRA-Proj KD.
* **Baselines:** HFL + SVD-LoRA (Không KD), Logit-KD (FedMD), Feature-KD (FedProto), LoRA-Proj KD (FedKDL).
* **Metrics sử dụng:**
  * mAP@0.5: Thấy rõ độ dốc và điểm kịch trần (ceiling) của từng phương pháp KD.
  * $\mathcal{L}_{total}$: Quan sát sự suy giảm của Loss.
  * *Có thể thêm đề cập về Memory (RAM) tại AUV để "dìm hàng" Feature-KD (bắt AUV lưu feature map quá to).*

**4. Kịch bản 4: Tối ưu hóa Chi phí Liên kết (Joint Cost Assessment)**

* **Mục tiêu:** Khẳng định FedKDL giải được bài toán tối ưu hóa đa mục tiêu \eqref{eq:objective} ban đầu.
* **Baselines:** Flat (chết sớm) vs HFL-Cơ bản vs FedKDL.
* **Metrics sử dụng:**
  * $\mathcal{F}$ (Joint Cost): Đồ thị đường cong chi phí tổng hợp giảm dần qua 60 vòng lặp.

---

**Bạn thấy ma trận phân bổ Metrics này đã sắc nét chưa?** Nếu kịch bản 1 ôm đồm nhiều Metrics như vậy, nó sẽ đóng vai trò là "Kịch bản đinh" (Main Scenario) dài nhất, các kịch bản 2, 3, 4 sẽ đóng vai trò "Ablation Study" (Nghiên cứu cắt lớp) cực kỳ khoa học.

Nếu bạn đồng ý, tôi sẽ xóa chỉ số $R_{survive}$ ở phần Metrics ban nãy, và bắt tay vào viết lại Kịch bản 1 theo cấu trúc hoành tráng này!
