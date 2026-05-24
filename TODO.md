Viewed run_kdl_experiments.sh:98-135
Searched for "epochs"
Viewed simulator.py:50-85
Viewed simulator.py:1-49
Searched for "train_and_get_payload"
Searched for "train_and_get_payload"
Viewed simulator.py:300-320
Viewed settings.py:1-102
Edited settings.py
Edited run_kdl_experiments.sh
Ran command: `git add run_kdl_experiments.sh config/settings.py ; git commit -m "Reduce rounds to 50 and local epochs to 2 to optimize training time" ; git push`
Used tool: manage_task

Quyết định rất hợp lý! Việc giảm `ROUNDS=50` và giảm `LOCAL_EPOCHS=2` là đòn bẩy kép giúp cắt giảm thời gian huấn luyện một cách đáng kinh ngạc. Mình đã sửa 2 thông số này (trong `run_kdl_experiments.sh` và `config/settings.py`) và push lên GitHub (Commit `ea76373`).

---

### 📉 Bảng Ước Tính Thời Gian MỚI (Giảm 42% tổng thời gian)

Nhờ việc mỗi sensor chỉ chạy 2 epoch thay vì 3 (giảm $\frac{1}{3}$ thời gian local train), kết hợp với số vòng giảm xuống 50:

* **Tốc độ trung bình mới:**
  * `N=20`: **~8.1 phút / vòng** $\rightarrow$ **~6.7 tiếng / kịch bản** (50 vòng). *(Trước đây là 11.5 tiếng)*
  * `N=30`: **~11.5 phút / vòng** $\rightarrow$ **~9.6 tiếng / kịch bản**. *(Trước đây là 16.5 tiếng)*
  * `N=40`: **~14.8 phút / vòng** $\rightarrow$ **~12.3 tiếng / kịch bản**. *(Trước đây là 21.5 tiếng)*
  * `N=50`: **~18.1 phút / vòng** $\rightarrow$ **~15.1 tiếng / kịch bản**. *(Trước đây là 26.5 tiếng)*

* **Tổng thời gian phân bổ (cho 53 kịch bản):**
  * **Group A (N=20):** 18 × 6.7h = **~120.6 tiếng**.
  * **Group B (N=20,30,40,50):** 7 × (6.7h + 9.6h + 12.3h + 15.1h) = **~305.9 tiếng**.
  * **Group C (N=30):** 7 × 9.6h = **~67.2 tiếng**.

🔥 **TỔNG CỘNG:** Từ mức khủng khiếp ~854.5 tiếng (gần 36 ngày), bây giờ toàn bộ thực nghiệm chỉ còn mất khoảng **493 tiếng (~20.5 ngày)** trên 1 GPU. Bạn đã tiết kiệm được hơn **15 ngày** chạy máy!

Bạn có thể tiến hành `git pull` để bắt đầu thực nghiệm ngay nhé. Budget đã hợp lý hơn rất nhiều rồi!
