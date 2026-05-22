Ran terminal command: Get-Content config/settings.py | Select-String "FRACTION"

Ran terminal command: Get-Content config/settings.py | Select-String -i "frac|client"
Completed with input: Get-Content config/settings.py | Select-String -i "frac|client"

Ran terminal command: Get-Content config/settings.py -TotalCount 50 | Select-String "FL" -Context 5,5

Ran terminal command: Get-Content federated_core\base_simulator.py -Tail 50 | Select-String "def"

Ran terminal command: cat run_kdl_experiments.sh | grep "python main_trainer_od.py"
Completed with input: cat run_kdl_experiments.sh | grep "python main_trainer_od.py"

Ran terminal command: Select-String "python main_trainer" run_kdl_experiments.sh

Ran terminal command: Get-Content run_kdl_experiments.sh | Select-String "python"

Ran terminal command: Get-Content run_kdl_experiments.sh | Select-String "main_trainer_od.py" -Context 0,7

Ran terminal command: Get-Content federated_core\base_simulator.py | Select-String "for s_id in self.sensors:" -Context 2,2

Ran terminal command: Get-Content federated_core\base_simulator.py | Select-String "for s_id in" -Context 2,2

Ran terminal command: Get-Content federated_core\base_simulator.py | Select-String "alive_sensors =" -Context 2,2

Dạ, với tốc độ trung bình 1 phút / 1 client, nếu anh chạy *tất cả* các client một cách **tuần tự** thì mình cần nhìn vào thiết lập của hệ thống để nhẩm tính thời gian nhé:

- **1 Vòng (Round):** Log hiện tại đang cấu hình hệ thống train cho $N=50$ clients (tức 50 Sensos/AUVs chạy trong 1 cụm). Điều đó có nghĩa là 1 vòng FL (Global Round) sẽ kéo dài: 
  `50 clients × 1 phút = 50 phút`.
- **1 Kịch bản đầy đủ (1 Run):** Trong log, anh đã cấu hình chạy `rounds=50` (50 vòng FL cho tác vụ 2D). Vậy để xong 1 baseline này, thời gian huấn luyện là:
  `50 phút × 50 vòng = 2.500 phút` (tức là khoảng chừng **41.6 tiếng / gần 2 ngày cắm máy liên tục** cho 1 thuật toán).

**Tuy nhiên, có một chi tiết CỰC KỲ QUAN TRỌNG:**
Trên dòng đầu của file log báo: `[1/240] OD | N=50...`. Anh đang chạy tổng cộng **240 kịch bản/thuật toán** trong file bash `run_kdl_experiments.sh`. 
Nếu cứ để nguyên với setup $N=50$, cắm tuần tự cho toàn bộ 50 clients vào mỗi vòng, tổng thời gian để chạy xong script bash đó sẽ nhẩm lên tới... **hơn 400 ngày**.

**Nguyên nhân:**
Em check file `base_simulator.py` thì thấy anh đang để toàn bộ `alive_sensors` (cả 50 nodes) tham gia vào mỗi vòng FL theo dạng vòng lặp `for` tuần tự (Sequential) chứ không lấy mẫu $10-20\%$ subset (client fraction) như FedAvg thông thường. Đối với Object Detection YOLOv8/v11 thì GPU phải chạy khá lâu cho 1 node.

**Đề xuất hướng xử lý:**
1. **Dùng Subset Sampling:** Thay vì cho 50 nodes train đồng loạt mỗi vòng, anh nên chỉnh code Simulator để chỉ lấy ngẫu nhiên 10% (5 clients) - 20% (10 clients) tham gia vào mỗi Round. (Thời gian sẽ giảm gấp 5 - 10 lần).
2. **Setup cụm test nhỏ:** Nếu đang chạy test để khảo sát vòng đời, hãy giảm số rounds xuống `rounds=5` hoặc `rounds=10` và số thiết bị $N=10$, $N=20$ thôi ạ.

Mục tiêu: Giảm tính toán xuống còn 3 ngày