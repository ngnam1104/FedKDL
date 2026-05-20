# Implementation Plan: Refactoring Research Proposal

## Mục tiêu (Goal)
Tái cấu trúc và tinh chỉnh nội dung file `Research_Proposal.md` theo 6 yêu cầu kỹ thuật: di chuyển bảng tham số, tổng quát hóa các thông số cứng, loại bỏ RL, cập nhật mô hình từ YOLO-Nano sang YOLO26n với hàm loss mới, loại bỏ logic Phân cụm phân cấp (Hierarchical Clustering) để thay bằng kết nối trực tiếp Sensor-Fog, và xóa bỏ hằng số 11 KB.

## User Review Required
> [!IMPORTANT]
> Vui lòng xem xét các nội dung tôi đề xuất thay đổi. Nếu bạn đồng ý, hãy phản hồi để tôi thực thi toàn bộ kịch bản này!

## Proposed Changes

### 1. Di chuyển phần "Tổng hợp các Biến và Tham số hệ thống"
*   **[MODIFY]** `Research_Proposal.md`: Cắt toàn bộ Mục **"8. Tổng hợp các Biến và Tham số hệ thống"** (từ `8.1` đến `8.4`) ở cuối Chương III (System Model).
*   **[MODIFY]** `Research_Proposal.md`: Dán toàn bộ nội dung này vào **Chương V (Experiments)**, đổi tên thành **"V.1. Tổng hợp các Biến và Tham số hệ thống"** (hoặc tích hợp ngay trước bảng Simulation Parameters). 

### 2. Loại bỏ các con số cứng nhắc (Hardcoded Values) từ Chương I đến IV
*   **[MODIFY]** `Research_Proposal.md`: Rà soát toàn bộ văn bản và thay thế các hằng số bằng các biến tổng quát:
    *   `500-1000m`, `100-400m` $\rightarrow$ "độ sâu tương ứng của tầng Sensor và Fog".
    *   `1500 m/s` $\rightarrow$ $c_s$ (vận tốc truyền âm).
    *   `~4 kHz, ~15 kbps` $\rightarrow$ "băng thông $B$ và tốc độ $R$ giới hạn".
    *   `500 J` $\rightarrow$ $E_{init}$ (ngân sách năng lượng khởi tạo).
    *   `100` AUV, `10` trạm Fog $\rightarrow$ $N$ AUV cảm biến, $M$ trạm Fog.
    *   `11 KB`, `11,072 tham số`, `3.072 params LoRA + 8.000 params Head` $\rightarrow$ Kích thước payload $S$ (sẽ loại bỏ hoàn toàn con số 11 KB như yêu cầu thứ 6).
    *   $140 \text{ dB}$ $\rightarrow$ $SL_{max}$.

### 3. Loại bỏ hoàn toàn sự xuất hiện của Reinforcement Learning (RL / Học tăng cường)
*   **[MODIFY]** `Research_Proposal.md`: Tìm và xóa hoặc thay thế các cụm từ "Học tăng cường sâu (DRL)", "Học tăng cường", "thuật toán bầy đàn", "thuật toán điều phối động". Khẳng định lại hệ thống của chúng ta sử dụng **các quy tắc tĩnh xác định (Deterministic Rules)**, không dùng RL.

### 4. Loại bỏ thuật toán "Phân cụm Phân cấp" (Hierarchical Clustering)
*   **[MODIFY]** `Research_Proposal.md`: Tại **Bước 1 (Mục IV.2)**, loại bỏ đoạn nói về "Hierarchical Clustering (Ward / Lance-Williams)".
*   Thay thế bằng cơ chế **Liên kết Trực tiếp (Direct Association)**: Nút cảm biến (Sensor) dưới cùng sẽ tính toán $D_{joint}$ với tất cả các Fog AUV khả thi và kết nối trực tiếp với Fog node có $D_{joint}$ nhỏ nhất. Không có việc phân cụm nội bộ giữa các Sensor với nhau. Sửa đổi lưu đồ Mermaid để phản ánh điều này.

### 5. Chuyển đổi mô hình YOLO-Nano sang YOLO26n và cập nhật Hàm Loss
*   **[MODIFY]** `Research_Proposal.md`: Tìm và thay thế toàn bộ keyword `YOLO-Nano` (hoặc `YOLO-Nano [32]`) thành `YOLO26n [32]`.
*   **[MODIFY]** `Research_Proposal.md`: Thêm đoạn mô tả **Hàm mục tiêu cục bộ (Learning Objective) của YOLO26n**:
    > "Khác với các thế hệ trước sử dụng Distribution Focal Loss (DFL), kiến trúc YOLO26 được thiết kế đặc thù cho thiết bị biên (Edge AI) thông qua cơ chế End-to-End NMS-Free và loại bỏ hoàn toàn DFL nhằm giảm độ trễ tính toán. Hàm mất mát nguyên bản của tác vụ phát hiện đối tượng được cấu thành như sau:
    > $$\mathcal{L}_{YOLO_{26}}(\theta) = \lambda_{box}\mathcal{L}_{box} + \lambda_{cls}\mathcal{L}_{cls} + \lambda_{STAL}\mathcal{L}_{STAL} + \lambda_{Prog}\mathcal{L}_{Prog}$$
    > Trong đó: $\mathcal{L}_{box}, \mathcal{L}_{cls}$: Hàm mất mát định vị hộp bao và phân lớp truyền thống; $\mathcal{L}_{STAL}$ (Scale-Tolerant Anchor Loss): Xử lý sự biến đổi kích thước đột ngột của mục tiêu...; $\mathcal{L}_{Prog}$ (Progressive Loss): Hỗ trợ mô hình tập trung nhận diện mục tiêu sinh học siêu nhỏ..."

### 6. Bỏ giới hạn "11 KB"
*   **[MODIFY]** `Research_Proposal.md`: Do payload size đã được tính lại, tôi sẽ loại bỏ các câu văn khẳng định "khóa cứng kích thước gói tin ở mức $\approx 11$ KB". Thay vào đó, văn bản sẽ dùng ngôn ngữ tổng quát "kích thước gói tin cực nhỏ $S$ đủ khả năng truyền qua kênh âm thanh", con số chi tiết sẽ được trình bày ở phần Experiments.

### 7. Đọc và Bổ sung Trích dẫn từ Thư mục .papers
*   **[RESEARCH & MODIFY]**: Tiến hành đọc nội dung chi tiết các bài báo PDF trong thư mục `.papers` (bao gồm các bài Surveys, phương pháp tối ưu, YOLO, và các Baseline).
*   **[MODIFY]** `Research_Proposal.md`: Dựa trên nội dung thực tế đọc được, chèn thêm các trích dẫn `[x]` vào những vị trí phù hợp trong bản đề xuất (đặc biệt là trong phần Introduction, Related Work và Methodology) để củng cố lập luận học thuật, đảm bảo toàn bộ 37 bài báo đều được tận dụng tối đa.

## Verification Plan
1. Chạy python regex để kiểm tra đảm bảo không còn sót chữ "11 KB", "YOLO-Nano", "RL", "Hierarchical Clustering" trong toàn bộ bài.
2. Kiểm tra phần V (Experiments) đã chứa Bảng biến/tham số từ Chương III chuyển xuống.
3. Review kết quả trực quan trên markdown để đảm bảo cấu trúc mới hợp lý.
