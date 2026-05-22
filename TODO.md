Để ra được con số **360 lần chạy (runs)** trong kịch bản HFL (Scenario 1), script `run_hfl_experiments.sh` đang duyệt qua tổ hợp của 5 thông số (mỗi tổ hợp là một lần chạy). Cụ thể như sau:

**1. `N` (Số lượng sensor/client) - 4 giá trị:**

* `50`
* `100`
* `150`
* `200`
*(Dùng để đánh giá khả năng mở rộng - Scalability).*

**2. `DS` (Dataset / Tập dữ liệu) - 3 giá trị:**

* `SMD` (Server Machine Dataset)
* `SMAP` (Soil Moisture Active Passive)
* `MSL` (Mars Science Laboratory)

**3. `alpha` (Độ bất đồng nhất dữ liệu - Data Heterogeneity) - 2 giá trị:**

* `0.1` (Non-IID rất cao - mỗi client chỉ có dữ liệu lệch hẳn về một vài phân bố).
* `10000.0` (Gần như IID - dữ liệu phân chia đồng đều).

**4. `seed` (Hạt giống ngẫu nhiên) - 3 giá trị:**

* `42`
* `123`
* `2024`
*(Dùng để chạy lặp lại 3 lần trên các bộ chia dữ liệu khác nhau, đảm bảo kết quả đáng tin cậy).*

**5. `baseline` (Chiến lược FL) - 5 giá trị:**

* `hfl_selective` (Chiến lược đề xuất: HFL + Filter + Lựa chọn hợp tác linh hoạt).
* `hfl_nearest` (Chỉ hợp tác với Fog Node gần nhất).
* `hfl_nocoop` (Các Fog Node hoàn toàn độc lập, không hợp tác).
* `fedprox` (Phương pháp baseline phổ biến, thêm hàm Proximal để chống nhiễu Non-IID).
* `fedavg` (Phương pháp nền tảng tiêu chuẩn).

---
**Tổng cộng:** `4 (N) × 3 (DS) × 2 (alpha) × 3 (seed) × 5 (baseline) = 360` lần chạy.

Script sẽ quét qua tất cả các trường hợp này bằng các vòng lặp lồng nhau (từ ngoài vào trong: `N` -> `DS` -> `alpha` -> `seed` -> `baseline`), nên lệnh chạy đầu tiên sẽ luôn là:
`N=50`, `DS=SMD`, `alpha=0.1`, `seed=42`, `baseline=hfl_selective`.
