# Tương lai / Ý tưởng Cải tiến (TODO)

Tài liệu này lưu trữ các ý tưởng tối ưu hóa có thể được nghiên cứu và áp dụng trong tương lai nhưng chưa đưa vào hệ thống hiện tại để bảo toàn tính chuẩn mực của các baselines.

## 1. Cơ chế EMA / Server Momentum tại Gateway và Relay

**Trạng thái:** Đã triển khai tại Gateway cho riêng họ FedKDL bằng
`BaselineConfig.server_mix` và `fed_cfg.SERVER_MIX_BETA`. FedAvg, FedProx và
FLORA vẫn giữ cập nhật nguyên bản. Với LoRA, phép trộn được thực hiện trên
ma trận hiệu dụng `B @ A` rồi SVD lại, không nội suy riêng hai nhân tử.

**Mô tả:** 
Hiện tại, logic Aggregation của hệ thống (trong `aggregator.py`) đang sử dụng FedAvg thuần túy (Weighted Average). Khi truyền payload bằng INT8, việc lấy trung bình của 30 mô hình đã giải mã từ INT8 (vốn chứa sai số lượng tử) sẽ khuếch đại nhiễu và làm rửa trôi các đặc trưng cục bộ (features) tốt mà các AUV đã học được.

**Ý tưởng:** 
Thay vì ghi đè 100% Global Model bằng bản Aggregated Model mới, ta sử dụng Exponential Moving Average (EMA) - hay còn gọi là **Server Momentum (FedAvgM)**:
```python
Global_State_New = (1 - \beta) * Global_State_Old + \beta * Aggregated_State
```
*(Với $\beta$ thường nằm trong khoảng $0.8$ đến $1.0$)*.

**Phạm vi và tính công bằng:**
- Việc thêm Server Momentum vào tầng Aggregation sẽ tác động đến toàn bộ các thuật toán đang sử dụng hàm `weighted_state_dict_average` (bao gồm `fedavg`, `fedprox`, v.v.). Điều này làm biến đổi bản chất của các baseline chuẩn thành các biến thể "with Server Momentum" (FedAvgM, FedProxM), khiến việc so sánh với nguyên bản không còn liêm chính (unfair comparison).
- **Hướng đánh giá:** So sánh `SERVER_MIX_BETA=1.0` và `0.9` trong một
  ablation riêng. Chưa bật temporal EMA tại Relay vì association có thể đổi
  theo vòng, khiến state riêng của Relay bị stale.
