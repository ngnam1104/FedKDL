### Setup Thực Nghiệm

**1. Tập dữ liệu huấn luyện (Dataset)**
Quá trình huấn luyện và đánh giá mô hình được thực hiện trên bộ dữ liệu **URPC 2020** (Underwater Robot Professional Contest), mô phỏng môi trường biển sâu với độ nhiễu quang học phức tạp.

* **Quy mô:** 5,543 hình ảnh huấn luyện.
* **Phân lớp mục tiêu (4 loại sinh vật đáy biển):** Hải sâm (holothurian), cầu gai (echinus), điệp (scallop) và sao biển (starfish).

**2. Các chỉ số đánh giá (Evaluation Metrics)**
Hiệu suất và tính khả thi của hệ thống FedKDL được đo lường toàn diện thông qua 6 nhóm chỉ số cốt lõi:

* **Độ chính xác nhận diện (mAP@0.5):** Chỉ số quan trọng nhất đánh giá đồng thời khả năng phân loại và định vị vùng ảnh (bounding box) tại ngưỡng IoU = 0.5.
* **Tỷ lệ tham gia chủ động ($\eta^{(t)}_{part}$):** Tỷ lệ phần trăm các AUV kết nối và gửi bản cập nhật thành công trong một vòng lặp, phản ánh khả năng duy trì mạng lưới dưới các ràng buộc khắt khe về năng lượng và sóng âm.
* **Chi phí truyền thông ($E^{(t)}_{comm}$ & $S_{avg}$):** Đánh giá gánh nặng băng thông và pin, bao gồm năng lượng viễn thông tiêu hao mỗi vòng ($E^{(t)}_{comm}$) và dung lượng tải trung bình ($S_{avg}$ tính bằng KB/round).
* **Độ trễ vận hành ($\tau^{(t)}_{round}$):** Đo lường thời gian trễ của từng chu kỳ huấn luyện và tổng độ trễ tích lũy ($\tau^{(T)}_{cumul}$), phản ánh trực tiếp chi phí thời gian thực khi triển khai đội tàu AUV.
* **Hàm mất mát huấn luyện ($L_{stu}$):** Đánh giá tốc độ hội tụ của mô hình Student. Sự sụt giảm của $L_{stu}$ cho thấy mức độ hiệu quả của phương pháp chưng cất tri thức (Knowledge Distillation) đóng vai trò như gradient bổ trợ.
* **Chi phí tối ưu hóa tổng hợp ($F$):** Thước đo đánh giá hiệu quả toàn diện của hệ thống bằng cách liên kết trực tiếp ba yếu tố xung đột: độ chính xác, mức tiêu thụ năng lượng và độ trễ vào một hàm mục tiêu duy nhất.

### Mô hình và Huấn luyện

* **Mô hình nền tảng:** Teacher: YOLOv12-Large (yolo12l) và Student: YOLOv12-nano (yolo12n), các mô hình nhẹ tối ưu hóa cho thiết bị biên.
* **Kỹ thuật nén & kết hợp (Compression & Fusion):** Sử dụng kỹ thuật LoRA (Low-Rank Adaptation) kết hợp với phương pháp SVD (Singular Value Decomposition) để giảm tham số mô hình mà vẫn giữ được hiệu năng nhận diện.

### Kịch bản 1: Về sự cần thiết của mạng phân cấp

**Các thuật toán đối chứng (Baselines):**

* **FedAvg:** Thuật toán học liên kết cơ bản thực hiện lấy trung bình trọng số tham số mô hình trực tiếp từ các thiết bị tham gia.

* **FedProx:** Giải pháp toán học bổ sung một số hạng phạt cận kề (proximal term) vào hàm mục tiêu cục bộ để hạn chế sự lệch hướng cập nhật của thiết bị biên.

* **SCAFFOLD:** Thuật toán sử dụng các biến kiểm soát (control variates) nhằm giảm thiểu phương sai cập nhật và khắc phục hiện tượng trôi dạt trọng số do dữ liệu không đồng nhất.

Tất cả đều kết nối trực tiếp từ AUV tới trung tâm dữ liệu mà không qua Relay

**Các Metrics sử dụng**: mAP@0.5, tỉ lệ tham gia chủ động ($\eta_{part}^{(t)}$), năng lượng viễn thông tiêu hao ($E^{(t)}_{comm}$), dung lượng tải trung bình ($S_{avg}$), độ trễ vận hành ($\tau^{(t)}_{round}$), hàm mất mát huấn luyện ($L_{stu}$), chi phí tối ưu hóa tổng hợp ($F$)

**Kết quả kỳ vọng (Expected Results):**

* **Kiến trúc mạng phẳng (Flat FL):** Do cự ly truyền trực tiếp vượt quá 1000m gây suy hao âm học nặng, tỷ lệ tham gia chủ động trung bình chỉ đạt tầm 50%; đồng thời tác động đan xen của mất gói tin và dữ liệu Non-IID khiến hàm Loss phân kỳ và độ chính xác mAP@0.5 bị khóa chết dưới ngưỡng 5%.

* **Kiến trúc phân cấp FedKDL:** Bằng cách chia nhỏ lộ trình qua 5 trạm Relay trung gian để khống chế cự ly mỗi chặng dưới 500m , FedKDL kỳ vọng bảo toàn 100% tỷ lệ kết nối bầy đàn ($\eta_{part}^{(t)} = 100.0\%$) , tiết kiệm điện năng viễn thông và giúp mô hình nhận diện hội tụ ổn định với độ chính xác cao.

### Kịch bản 2: So sánh các phương pháp giảm tải tham số truyền thông

**Các thuật toán đối chứng (Baselines):**
(Lưu ý: Tất cả đều được chạy trên nền tảng phân cấp HFL để loại trừ rủi ro mất gói tin)

* **HFL + Full Parameter (FedAvg truyền thống):** Truyền tải nguyên bản toàn bộ tham số mô hình (11.2 MB) từ thiết bị lên trạm trung chuyển.
* **HFL + Top-K Sparsification:** Giảm tải dung lượng viễn thông bằng cách làm thưa (cắt tỉa) gradient, nhưng bắt buộc thiết bị phải thực hiện lan truyền ngược trên toàn bộ tham số gốc.
* **HFL + Naive LoRA:** Đóng băng bộ khung xương sống (backbone), chỉ huấn luyện các ma trận phụ trợ có kích thước nhỏ ($A$ và $B$) và thực hiện lấy trung bình trực tiếp trên các ma trận phân tán này.

**Các Metrics sử dụng**: Năng lượng tính toán nội bộ ($E_{comp}$), năng lượng viễn thông tiêu hao ($E_{comm}$), độ chính xác nhận diện (mAP@0.5), Độ trễ truyền thông ($\tau^{(t)}_{round}$), độ trễ tính toán nội bộ ($T_{comp}$), Tổng chi phí tối ưu hóa ($F$)

**Kết quả kỳ vọng (Expected Results):**

* **HFL + Full Parameter:** Thiết bị cạn kiệt pin ngay ở những vòng lặp đầu tiên do gánh nặng viễn thông ($E_{comm}$) vượt quá ngưỡng cho phép.
* **HFL + Top-K Sparsification:** Thiết bị sập nguồn ở giữa tiến trình do khâu lan truyền ngược tiêu hao quá nhiều năng lượng tính toán ($E_{comp}$), khiến quá trình hội tụ đứt gãy dù khả năng bảo toàn đặc trưng không gian tốt.
* **HFL + Naive LoRA:** Thiết bị sống sót đến cuối chu kỳ nhờ giảm cả $E_{comp}$ và $E_{comm}$, nhưng độ chính xác mAP@0.5 kịch trần ở mức 42.8% do sai số không gian con (cross-term mismatch) sinh ra khi lấy trung bình trực tiếp.
* **HFL + SVD-LoRA (Cơ chế của FedKDL):** Vừa duy trì 100% khả năng sinh tồn của bầy đàn, vừa khắc phục triệt để lỗi toán học bằng cách tổng hợp phần dư $\Delta W$ trước khi phân rã, giúp khôi phục sức mạnh nhận diện tiệm cận đường cơ sở tập trung.

Lưu ý: Kịch bản này sẽ bỏ KD ra để so sánh các phương pháp giảm tải tham số truyền thông.

### Kịch bản 3: Về sự cần thiết của KD (Knowledge Distillation)

**Các thuật toán đối chứng (Baselines):**

* **HFL + SVD-LoRA nguyên bản (Không KD):** Vận hành SVD-LoRA trên kiến trúc phân cấp nhưng để mô hình sinh viên tự học mà không có bất kỳ sự hỗ trợ chưng cất tri thức nào.
* **HFL + SVD-LoRA + Logit-KD (theo hướng FedMD):** Thực hiện chưng cất tri thức thông qua việc đồng bộ hóa chỉ dựa trên lớp phân loại đầu ra (logits) của mô hình.
* **HFL + SVD-LoRA + Feature-KD (theo hướng FedProto):** Yêu cầu các mô hình sinh viên tại thiết bị biên lưu trữ, xử lý và đồng bộ hóa các bản đồ đặc trưng (feature maps) trung gian đa chiều với mô hình giáo viên.

**Các Metrics sử dụng**: Năng lượng tính toán nội bộ ($E_{comp}$), năng lượng viễn thông tiêu hao ($E_{comm}$), độ chính xác nhận diện (mAP@0.5), Độ trễ truyền thông ($\tau^{(t)}_{round}$), độ trễ tính toán nội bộ ($T_{comp}$), Tổng chi phí tối ưu hóa ($F$), mức độ suy giảm hàm mất mát huấn luyện ($L_{stu}$)

**Kết quả kỳ vọng (Expected Results):**

* **HFL + SVD-LoRA nguyên bản:** Hội tụ ổn định nhưng kịch trần độ chính xác ở mức 61.4% do giới hạn năng lực biểu diễn của mô hình sinh viên (YOLOv12n) cỡ nhỏ.
* **HFL + Logit-KD:** Gần như không cải thiện độ chính xác ($\approx 61.8\%$) do việc chỉ dùng logits làm triệt tiêu hoàn toàn thông tin hình học và không gian của bài toán dự đoán mật độ cao (dense prediction).
* **HFL + Feature-KD:** Cải thiện được mAP và tốc độ giảm Loss nhưng gây tràn bộ nhớ (Out-Of-Memory) tại thiết bị biên do tải lượng xử lý các tensor đa chiều vượt quá giới hạn phần cứng của AUV.
* **HFL + LoRA-Proj KD (Cơ chế của FedKDL):** Bứt phá mốc mAP lên 68.7% với tốc độ giảm Loss nhanh nhất nhờ dời toàn bộ gánh nặng tính toán chưng cất lên Gateway mặt nước, đồng thời duy trì an toàn tuyệt đối cho bộ nhớ (RAM/VRAM) của AUV.

### Kịch bản 4: Đánh Giá Chi Phí Tổng Hợp và Mục Tiêu Tối Ưu Hóa (Joint Cost Assessment and Optimization Objective)

**Các thuật toán đối chứng (Baselines):**

* **SCAFFOLD:** Đại diện xuất sắc nhất của nhóm kiến trúc phẳng (Flat FL), bắt buộc truyền tải toàn bộ tham số mô hình và biến điều khiển trực tiếp qua kênh âm học tầm xa.
* **HFL + Top-K:** Ứng viên nén có độ chính xác cao nhất trên nền tảng phân cấp, giúp tối ưu khoảng cách truyền nhưng yêu cầu thiết bị biên phải tính toán gradient đầy đủ.
* **HFL + SVD-LoRA + Feature-KD:** Phương pháp chưng cất tri thức mạnh nhất trong nhóm đối thủ, đòi hỏi lưu trữ và đồng bộ hóa các bản đồ đặc trưng (feature maps) trung gian.

*(Lưu ý: Kịch bản này chọn lọc các đại diện ưu tú nhất từ các kịch bản trước để đối chiếu toàn diện trên cùng một hàm mục tiêu đa biến).*

**Các Metrics sử dụng**: Hàm chi phí tổng hợp đa biến $F$ (kết hợp tổng tiêu thụ năng lượng $E_{total}$ và độ trễ tích lũy $\tau_{max}$), số vòng lặp sinh tồn.

**Kết quả kỳ vọng (Expected Results):**

* **SCAFFOLD:** Hàm $F$ bùng nổ ngay từ những vòng đầu tiên và mạng lưới sụp đổ hoàn toàn do gánh nặng viễn thông vượt quá giới hạn.
* **HFL + Top-K:** Khởi đầu khả quan nhưng đường cong $F$ đứt đoạn ở khoảng vòng 30–40 do thiết bị cạn kiệt pin từ việc tiêu hao năng lượng tính toán ($E_{comp}$) quá lớn.
* **HFL + SVD-LoRA + Feature-KD:** Sống sót qua giới hạn năng lượng nhưng hàm $F$ và độ trễ $\tau_{round}$ bị đẩy lên cao ở các vòng cuối do áp lực quá tải bộ nhớ (RAM/VRAM) tại thiết bị biên.
* **FedKDL (Kiến trúc đề xuất hoàn chỉnh):** Duy trì quỹ đạo suy giảm hàm chi phí $F$ ổn định và nhất quán suốt 60 vòng lặp, giải quyết trọn vẹn bài toán tối ưu hóa đa mục tiêu (năng lượng, tính toán, độ trễ và độ chính xác) mà không vi phạm bất kỳ ràng buộc vật lý nào.
