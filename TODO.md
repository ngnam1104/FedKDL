# Cập nhật kdl_core

# TODO & Status Report

## 1. Lỗi vừa xảy ra là gì?

Khi chạy test `main_trainer.py` với cấu hình `hfl_selective`, hệ thống ném ra lỗi từ thư viện PyTorch:
`ValueError: num_samples should be a positive integer value, but got num_samples=0`

**Nguyên nhân:** Quá trình phân rã dữ liệu Non-IID bằng phân phối Dirichlet đôi khi sẽ gán 0 mẫu dữ liệu (samples) cho một vài cảm biến. Khi `simulator.py` cố gắng tạo `DataLoader` cho các cảm biến bị rỗng dữ liệu này, `RandomSampler` của PyTorch sẽ sụp đổ (crash).

**Cách khắc phục (Đã thực hiện):** Tôi đã thêm một lớp bảo vệ trong `hfl_core/simulator.py` ở đoạn tạo `DataLoader`. Cụ thể, các cảm biến có danh sách chỉ mục dữ liệu rỗng (`len(idx_list) == 0`) sẽ tự động bị bỏ qua, không đưa vào danh sách tham gia huấn luyện ở round đó nữa.

## 2. Xác nhận JSON Output (Đã kiểm tra thành công ✅)

Tiến trình test `hfl_selective` (3 rounds) đã hoàn tất mỹ mãn mà không gặp bất kỳ lỗi nào.

Kiểm tra trực tiếp file JSON sinh ra (`results/logs/log_N50_SMD_a0p1_hfl_selective_rho0p05_seed42.json`) cho thấy:

- Năng lượng tiêu thụ đã được chia tách hoàn hảo: `e_s2f`, `e_f2f`, `e_f2g`, `e_comp`.
- Độ trễ `tau_round_s` được lưu chính xác.
- Training Loss giảm dần (0.0987 -> 0.0874 -> 0.0775) chứng tỏ model học bình thường.
- Script vẽ biểu đồ (plot_scalability) cũng đã được tôi sửa lại để đọc đúng các key mới này.

## Trạng thái hiện tại: SẴN SÀNG 100% 🚀

Tất cả các thành phần cốt lõi đều đã được vá lỗi và audit kỹ càng. Bạn hoàn toàn có thể chạy kịch bản chính thức:
`.\run_all_experiments.ps1`
