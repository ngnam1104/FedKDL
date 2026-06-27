# 📋 Danh sách công việc chuẩn bị nộp Đồ án Tốt nghiệp & Sửa Paper FedKDL

> **Trạng thái:** Tự động lọc bỏ các đầu việc Google Drive đã hoàn thành.

---

# 📁 PHẦN 1: Google Drive Submission TODOs (Còn lại)

## 📁 1. Slides (Slide thuyết trình)
- [ ] Tạo thư mục `Slides` trên Google Drive
- [ ] Thiết kế slide PowerPoint trình bày đồ án tốt nghiệp
- [ ] Upload file slide PowerPoint bản gốc (`.pptx`)
- [ ] Upload bản slide xuất ra định dạng PDF (để tránh lỗi font/hiển thị)
- [ ] Chia sẻ quyền truy cập thư mục cho `haipv.expert@gmail.com`
- [ ] Cập nhật link thư mục vào file Excel

## 📁 2. Demo (Video Demo các Case Study)
*Quay video demo các chức năng được lập trình trong thư mục `demo/`:*
- [ ] Tạo thư mục `Demo` trên Google Drive
- [ ] Quay và upload video demo **Detection Demo** (Upload ảnh và chạy YOLOv12 inference nhận diện vật thể dưới nước)
- [ ] Quay và upload video demo **Centralized Raw-Data Demo** (Mô phỏng AUV gửi ảnh thô về Gateway và đánh giá payload)
- [ ] Quay và upload video demo **FedAvg Round Replay** (Replay 3 vòng học liên kết của FedAvg từ kết quả thực nghiệm)
- [ ] Quay và upload video demo **FedKDL Round Replay** (Replay 3 vòng học liên kết của FedKDL từ kết quả thực nghiệm với LoRA/INT8, relay aggregation & Gateway KD)
- [ ] Chia sẻ quyền truy cập thư mục cho `haipv.expert@gmail.com`
- [ ] Cập nhật link thư mục vào file Excel

## 🛠️ Trạng thái chung / Gửi Link Excel
- [ ] Xác nhận tất cả các link thư mục Google Drive đã được cập nhật đầy đủ và chính xác vào file Excel theo dõi.

---

# 📝 PHẦN 2: Sửa Paper FedKDL theo góp ý GVHD & Rà soát nội bộ

## Mục tiêu chính
1. Làm rõ **vì sao sử dụng Federated Learning (FL)**.
2. Phát biểu rõ **các bài toán cơ bản và vấn đề nghiên cứu**, kèm citation phù hợp.
3. Nêu rõ **đóng góp kỹ thuật** của FedKDL.
4. Bổ dung **Results Discussion khoảng 2/3 trang**, bao gồm ý nghĩa đóng góp mới, hạn chế và hướng phát triển.

---

## 🅰️ A. Mức ưu tiên P0 — Bắt buộc sửa trước khi nộp

### A1. Làm rõ quy trình validation/test
- [ ] Kiểm tra lại chính xác các đường cong mAP qua 40 vòng được tính trên:
  - validation set;
  - test set;
  - hay một evaluation split dùng chung.
- [ ] Không viết “peak mAP trên test qua 40 vòng” nếu test set được dùng để chọn checkpoint.
- [ ] Quy trình ưu tiên:
  1. chọn checkpoint bằng validation;
  2. đánh giá test một lần tại checkpoint được chọn.
- [ ] Nếu chưa thể chạy lại, đổi cách gọi thành `evaluation split` và nêu rõ đây không phải test độc lập.
- [ ] Đồng bộ cách gọi trong:
  - Experimental Setup;
  - captions;
  - bảng kết quả;
  - phần mô tả từng RQ;
  - Abstract và Conclusion nếu có.
* **Tiêu chí hoàn thành:** test set không được dùng như validation để chọn peak checkpoint mà không giải thích.

### A2. Tách “hàm mục tiêu P1” và “chỉ số tổng hợp thực nghiệm”
- [ ] Kiểm tra Bảng objective summary: giá trị hiện đang được tính từ
  \[
  \overline{\mathcal L}_{\mathrm{val}}
  +
  \lambda_E \overline E_{\mathrm{total}}
  +
  \lambda_\tau \overline \tau_{\mathrm{total}}.
  \]
- [ ] Không gọi trực tiếp đại lượng này là objective của P1 nếu P1 sử dụng:
  \[
  \mathcal L_{\mathrm{global}}(\theta^{(T)})
  +
  \sum_{t=1}^{T}
  \left(
  \lambda_\tau \tau_{\mathrm{total}}^{(t)}
  +
  \lambda_E E_{\mathrm{total}}^{(t)}
  \right).
  \]
- [ ] Định nghĩa riêng một đại lượng báo cáo, ví dụ:
  ```latex
  \overline{\mathcal F}_{\mathrm{report}}
  =
  \overline{\mathcal L}_{\mathrm{val}}
  +
  \lambda_E\overline E_{\mathrm{total}}
  +
  \lambda_\tau\overline\tau_{\mathrm{total}}.
  ```
- [ ] Đổi tên cột/bảng/hình từ `Total Objective` sang một tên chính xác hơn:
  - `Reported Scalarized Score`;
  - `Composite Evaluation Score`;
  - hoặc `Experimental Aggregate Score`.
- [ ] Giải thích rằng chỉ số này dùng để so sánh thực nghiệm dưới cùng quy ước scaling, không phải nghiệm trực tiếp của P1.
* **Tiêu chí hoàn thành:** người đọc không nhầm chỉ số trong bảng với hàm tối ưu P1.

### A3. Bổ sung mô hình chuyển động AUV
- [ ] Thêm mô hình cập nhật vị trí AUV tại System Model hoặc Experimental Setup.
- [ ] Nêu rõ:
  - mô hình chuyển động sử dụng;
  - cách sinh hướng;
  - mức dịch chuyển \(5\), \(50\), \(100\) m/round;
  - cách xử lý biên không gian;
  - thời điểm cập nhật topology;
  - thời điểm tái liên kết relay.
- [ ] Nếu dùng Gauss--Markov, viết công thức chuẩn và các tham số thực tế.
- [ ] Nếu chỉ dùng bước dịch chuyển cố định/ngẫu nhiên, mô tả đúng implementation, không gọi Gauss--Markov nếu code không dùng.
* **Tiêu chí hoàn thành:** RQ về mobility có mô hình vật lý đủ rõ để tái lập.

### A4. Thay hoặc sửa hình System Model
- [ ] Thay hình System Model cũ bằng bản đã đồng bộ.
- [ ] Loại bỏ ký hiệu cũ:
  - \(DI\);
  - \(E_{\mathrm{tx}}^{s2f}\);
  - \(E_{\mathrm{comp}}=c_{\mathrm{op}}\Phi_i\);
  - battery recursion cũ.
- [ ] Dùng đúng:
  - \(IL\);
  - fixed-rate tại \(\gamma_{\mathrm{tgt}}\);
  - \(E_{\mathrm{comp},u}=\zeta_{\mathrm{op}}C_u(f_{\mathrm{CPU},u})^2\);
  - AUV--relay, relay--relay, relay--gateway.
- [ ] System Model chỉ mô tả kiến trúc vật lý, liên kết, độ trễ và năng lượng; không đưa FedKDL/LoRA/KD vào hình.
- [ ] Đổi nhãn `Uplink Data` thành `Model Update` hoặc thuật ngữ tương đương.
* **Tiêu chí hoàn thành:** hình và phương trình trong System Model không còn thuộc hai phiên bản khác nhau.

### A5. Sửa lỗi tham chiếu phương trình
- [ ] Sửa câu trong Method đang dẫn \(C_i^{(t)}\) tới phương trình năng lượng.
- [ ] Tham chiếu đúng phương trình định nghĩa:
  \[
  C_i^{(t)}=n_iE_{\mathrm{local}}c_{\mathrm{sample}}.
  \]
- [ ] Rà toàn bộ `\eqref{}` trong Method để phát hiện tham chiếu sai nghĩa dù label vẫn tồn tại.
* **Tiêu chí hoàn thành:** mỗi tham chiếu phương trình trỏ đúng đại lượng được mô tả.

### A6. Thống nhất thống kê loss
- [ ] Chọn duy nhất một cách báo cáo:
  - `Mean Validation Loss`; hoặc
  - `Final Validation Loss`.
- [ ] Đồng bộ trong:
  - figure axes;
  - captions;
  - table headers;
  - prose;
  - reported scalarized score.
- [ ] Không dùng đồng thời `Peak Loss`, `Reported Loss`, `Final Loss`, `Average Loss` cho cùng một cột số liệu.
- [ ] Xác minh từng số trong bảng khớp với log cuối.
* **Tiêu chí hoàn thành:** người đọc biết chính xác loss được lấy ở vòng nào hoặc được trung bình như thế nào.

### A7. Giảm causal claim chưa được ablation hỗ trợ
- [ ] Trong mobility experiment, không khẳng định riêng cooperation là nguyên nhân duy nhất tạo độ ổn định nếu không có cấu hình `mobility + no cooperation`.
- [ ] Dùng cách viết thận trọng:
  > The combined relay reassociation and inter-relay cooperation pipeline maintains a similar learning trajectory under the evaluated mobility settings.
- [ ] Giữ causal claim mạnh cho cooperation trong RQ3, nơi có ablation:
  - no cooperation;
  - selective cooperation;
  - nearest cooperation.
* **Tiêu chí hoàn thành:** mọi kết luận nhân quả đều có ablation hoặc so sánh trực tiếp hỗ trợ.

---

## 🅱️ B. Mức ưu tiên P1 — Xử lý trực tiếp góp ý GVHD

### B1. Làm rõ “Vì sao dùng FL?”
- [ ] Bổ sung một câu trực diện sau đoạn mô tả centralized learning vào Introduction:
  > FL is adopted to avoid repeatedly transferring raw underwater imagery to the surface gateway while retaining local data at the AUVs.
- [ ] Ngay sau đó làm rõ vì sao **flat FL chưa đủ**:
  - direct AUV--gateway links có thể không khả thi;
  - detector updates có payload lớn;
  - dữ liệu phân tán Non-IID.
- [ ] Giải thích vì sao chọn HFL:
  > HFL introduces relay-assisted participation and hierarchical aggregation for AUVs that cannot directly reach the gateway.
- [ ] Gắn citation ngay sau từng nhóm claim:
  - FL/FedAvg;
  - HFL;
  - underwater acoustic constraints;
  - underwater FL/HFL.
* **Tiêu chí hoàn thành:** trả lời rõ vì sao không centralized learning và vì sao không flat FL ngay trong Introduction.

### B2. Phát biểu rõ các bài toán cơ bản và vấn đề nghiên cứu
- [ ] Thêm một đoạn ngắn ở cuối phần motivation hoặc trước Contributions, phát biểu 4 vấn đề nghiên cứu:
  - **Problem 1 (Connectivity and participation):** AUV không có direct feasible link tới gateway có thể bị loại khỏi quá trình học.
  - **Problem 2 (Communication-efficient detector synchronization):** Full-model detector synchronization tạo payload, latency và energy lớn.
  - **Problem 3 (Hierarchical aggregation under Non-IID data):** Factor-wise LoRA averaging không bảo toàn weighted average của effective updates.
  - **Problem 4 (Quality recovery after compressed aggregation):** Quantization, truncation và Non-IID aggregation có thể làm giảm chất lượng student.
- [ ] Mỗi vấn đề phải nêu rõ: hiện tượng, hạn chế của phương pháp hiện có, và nhu cầu kỹ thuật dẫn tới FedKDL.

### B3. Rà citation cho các claim nền tảng
- [ ] Kiểm tra từng claim trong Introduction và Related Work (acoustic channel, FL communication, HFL participation, Non-IID, LoRA, KD...).
- [ ] Dùng primary papers cho thuật toán, phương trình, cơ chế, kết quả định lượng (không lạm dụng survey).
- [ ] Claim chưa có nguồn trực tiếp phải viết thận trọng hoặc đánh dấu `[CITATION NEEDED]`.

### B4. Đóng góp kỹ thuật (Contributions)
- [ ] Viết lại Contributions theo dạng `problem → mechanism → technical effect`:
  - **Contribution 1 (Relay-assisted participation):** Three-tier AUV-relay-gateway + feasible association + relay cooperation giúp duy trì participation.
  - **Contribution 2 (Parameter-efficient synchronization):** Layer-wise LoRA + Delta-INT8 + effective-weight aggregation + SVD giúp đồng bộ nhẹ và đúng logic.
  - **Contribution 3 (Gateway-side refinement):** Refinement dựa trên proxy supervision + confidence-weighted classification KD + box geometry KD để cải thiện student.
- [ ] Không coi "joint evaluation" là đóng góp kỹ thuật chính.

---

## 🅲 C. Mức ưu tiên P1 — Thêm Results Discussion khoảng 2/3 trang

### C1. Tạo subsection `Discussion`
- [ ] Chèn subsection Discussion vào sau RQ4 và trước Conclusion (độ dài khoảng 500-700 từ).

### C2. Nội dung Discussion
- [ ] **Đoạn 1 (Ý nghĩa tổng thể):** Định vị FedKDL là một operating point tối ưu giữa các chiều (participation, payload, latency, energy, quality) chứ không chỉ riêng mAP.
- [ ] **Đoạn 2 (Bằng chứng thực nghiệm):** Liên kết các RQ1-RQ4 để hỗ trợ trực tiếp cho các đóng góp kỹ thuật tương ứng.
- [ ] **Đoạn 3 (Hạn chế):** Chỉ ra các mặt giới hạn (chỉ dùng 1 dataset URPC2020, mô phỏng kênh vật lý chưa có contention/retransmission, dữ liệu proxy có nhãn...).
- [ ] **Đoạn 4 (Hướng phát triển):** Đề xuất các hướng mở rộng (domain shift, adaptive LoRA, trajectory control...).

---

## 🅳 D. Mức ưu tiên P2 — Tăng khả năng tái lập

### D1. Bổ sung thông tin setup còn thiếu
- [ ] Thêm các thông số: Tâm độ sâu lớp \(c_h\), \(\sigma_z\), random seed, số lần chạy, input resolution, optimizer cục bộ, phần cứng & phiên bản CUDA/Ultralytics...

### D2. Báo cáo uncertainty nếu có thể
- [ ] Chạy tối thiểu 3 seeds cho các RQ chính để báo cáo mean ± std (nếu còn thời gian).
- [ ] Nếu không chạy lại được, viết thận trọng (không lạm dụng từ `robust`, `stable`).

---

## 🅴 E. Mức ưu tiên P2 — Abstract và Conclusion

### E1. Abstract
- [ ] Giữ kết quả nén payload (10.421 MB -> 0.491 MB), đồng thời đưa thêm kết quả accuracy trade-off.

### E2. Conclusion
- [ ] Rút gọn phần lặp lại phương pháp, tập trung vào kết quả định lượng chính, trade-off và hướng tương lai.

---

## 🅵 F. Mức ưu tiên P2 — Citation và Bibliography
- [ ] Bổ sung citation gốc cho Backpropagation, Thorp/Wenz, AP/mAP, object detection, CIoU/DFL.
- [ ] Chuẩn hóa capitalization, venue, arXiv ID, DOI của BibTeX.

---

## 🄶 G. Mức ưu tiên P2 — Hình, Bảng và Thuật ngữ
- [ ] Thay System Model figure, kiểm tra các sơ đồ kiến trúc và sơ đồ Gateway KD.
- [ ] Đảm bảo bảng kết quả hiển thị thống nhất đơn vị, in đậm/làm nổi bật FedKDL một cách hợp lý.
- [ ] Thống nhất các thuật ngữ: `Non-IID`, `gateway-side refinement`, `effective-weight aggregation`, `truncated-SVD extraction`, `Delta-INT8`, `inter-relay cooperation`.

---

## 🅷 H. Thứ tự thực hiện đề xuất
- [ ] **Giai đoạn 1 (Logic & Đánh giá):** Hoàn thành A1 -> A7.
- [ ] **Giai đoạn 2 (Góp ý GVHD):** Hoàn thành B1 -> B4 và C1 -> C2.
- [ ] **Giai đoạn 3 (Hoàn thiện):** Reproducibility, Abstract/Conclusion, BibTeX, Figures/Tables.

---

## 🅸 I. Checklist kiểm tra cuối trước khi nộp
- [ ] Không còn `TODO`, `[CITATION NEEDED]`, `??`.
- [ ] Không có undefined references/citations.
- [ ] Mọi equation variable được định nghĩa trước khi dùng.
- [ ] Test set không được dùng để chọn peak checkpoint mà không giải thích.
- [ ] P1 và reported experimental score được phân biệt rõ ràng.
- [ ] Mobility model được mô tả.
- [ ] System Model figure khớp với các phương trình.
- [ ] Mọi claim mạnh đều có ablation/citation hỗ trợ.
- [ ] Paper compile sạch (không lỗi) trước khi nộp.
