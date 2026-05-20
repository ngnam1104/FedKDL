# Hướng dẫn Tải và Tích hợp Dữ liệu Thực tế (SMD, SMAP, MSL)

Trong khuôn khổ bài báo (Omeke et al. 2026), 3 bộ dữ liệu được sử dụng để đánh giá kịch bản phát hiện bất thường (Anomaly Detection) là:
1. **SMD (Server Machine Dataset)**
2. **SMAP (Soil Moisture Active Passive)**
3. **MSL (Mars Science Laboratory)**

Vì các bộ dữ liệu này có kích thước khá lớn và yêu cầu một số bước tiền xử lý, tài liệu này hướng dẫn bạn cách tải và đưa dữ liệu vào dự án FedKDL trên máy chủ.

## 1. Nguồn Dữ liệu (Nguồn gốc)

Cả 3 bộ dữ liệu này thường được tổng hợp và xử lý sẵn trong kho lưu trữ chính thức của bài báo **OmniAnomaly** hoặc các benchmark tương đương (như của nhóm nghiên cứu thung lũng Silicon - NetMan).

- **Link GitHub OmniAnomaly (Server Machine Dataset):** [https://github.com/NetManAIOps/OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly)
- **Link Telemanom (SMAP & MSL):** [https://github.com/khundman/telemanom](https://github.com/khundman/telemanom)

## 2. Các Bước Thực Hiện Trên Máy Chủ

### Bước 2.1: Tạo thư mục lưu trữ

Truy cập máy chủ của bạn và di chuyển vào thư mục dự án `FedKDL`. Tạo một thư mục con tên là `datasets` ở thư mục gốc:

```bash
cd /path/to/FedKDL
mkdir datasets
cd datasets
```

### Bước 2.2: Tải Dữ liệu

Tùy vào định dạng mà cộng đồng chia sẻ, bạn có thể clone trực tiếp repo hoặc tải file zip. Dưới đây là cách tải tập dữ liệu đã qua tiền xử lý chuẩn (thường lưu dưới dạng pickle `.pkl` hoặc `.npy`):

*(Ví dụ sử dụng thư viện dataset chuẩn như `TSB-UAD` hoặc tải từ link Google Drive của tác giả bài báo nếu có).*

Do hiện tại link tải trực tiếp file `.pkl` tiền xử lý thường bị đổi, **khuyến nghị** clone repo OmniAnomaly và copy thư mục `ServerMachineDataset` vào `FedKDL/datasets/`:

```bash
git clone https://github.com/NetManAIOps/OmniAnomaly.git
cp -r OmniAnomaly/ServerMachineDataset ./SMD
```

Làm tương tự cho SMAP và MSL.

### Bước 2.3: Tích hợp vào Mã Nguồn FedKDL

Khi bạn đã có các file dữ liệu (ví dụ: `train/*.txt`, `test/*.txt`, `test_label/*.txt`), bạn cần sửa file `fl_core/data/dataloader_1d.py`. 

Trong file `dataloader_1d.py`, tìm đến hàm `load_dataset(name, seed)`. Thay vì gọi `generate_synthetic_timeseries`, hãy trỏ nó tới file đọc dữ liệu trên ổ cứng:

```python
# Ví dụ thay đổi trong dataloader_1d.py

def load_real_smd(data_dir="datasets/SMD"):
    # Đọc file numpy/txt từ thư mục
    import numpy as np
    import os
    
    train_data = np.loadtxt(os.path.join(data_dir, "train", "machine-1-1.txt"), delimiter=",")
    test_data = np.loadtxt(os.path.join(data_dir, "test", "machine-1-1.txt"), delimiter=",")
    test_labels = np.loadtxt(os.path.join(data_dir, "test_label", "machine-1-1.txt"), delimiter=",")
    
    # Kết hợp hoặc chỉ lấy train để train
    # ... tiền xử lý MinMaxScaler ...
    return data, labels

def load_dataset(name: str, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    if name == 'SMD':
        return load_real_smd("datasets/SMD")
    # ... tương tự cho SMAP, MSL ...
```

## 3. Lời Khuyên

1. **Scale Data**: Dữ liệu thực tế thường có độ biến thiên rất lớn giữa các features. Bắt buộc phải dùng `MinMaxScaler` hoặc `StandardScaler` (fit trên tập train, transform trên cả train và test) trước khi đưa vào hàm `SlidingWindowDataset`.
2. **Kích thước Window**: Bài báo gốc không nói rõ window size là bao nhiêu, nhưng thông thường cho SMD, `window_size` thường từ 100 đến 120. (Trong mã nguồn đang default là 10, hãy điều chỉnh `SlidingWindowDataset(window_size=100)` nếu cần).
3. **Chạy Thử Trước**: Hãy dùng synthetic data để test toàn bộ luồng chạy của 3 thí nghiệm (Thí nghiệm 1, 2, 3) trên máy chủ để đảm bảo không bị lỗi OOM (Out Of Memory) hoặc lỗi thuật toán. Sau đó mới cắm đường dẫn dữ liệu thật vào.

## 4. Dữ liệu Object Detection (Kịch bản 2 & 3: URPC 2020)

Đối với các kịch bản sử dụng YOLO (Object Detection dưới nước), chúng ta sử dụng tập dữ liệu **URPC 2020** (Underwater Robot Professional Contest).

### Yêu cầu
Tập dữ liệu URPC 2020 bao gồm ảnh màu RGB dưới nước và các bounding box annotations cho 4 lớp: `holothurian`, `echinus`, `scallop`, `starfish`.

### Hướng dẫn tải và chuẩn bị cho YOLO
1.  **Tải từ Kaggle (Khuyên dùng):**
    ```bash
    pip install kaggle
    kaggle datasets download -d slmhvn/urpc-2020 -p datasets/
    unzip datasets/urpc-2020.zip -d datasets/URPC2020
    ```
2.  **Cấu trúc thư mục YOLO:**
    Tập dữ liệu cần được chuyển đổi sang định dạng YOLO (các file `.txt` tương ứng với mỗi ảnh). Thông thường Kaggle dataset đã có sẵn hoặc bạn có thể dùng công cụ chuyển đổi từ VOC XML sang YOLO txt.
3.  **Tạo file `URPC2020.yaml`:**
    Tạo file `datasets/URPC2020.yaml` với nội dung sau:
    ```yaml
    path: URPC2020 # relative to dataset root
    train: images/train
    val: images/val

    # Classes
    nc: 4
    names: ['holothurian', 'echinus', 'scallop', 'starfish']
    ```
    
Sau khi có file yaml, bạn có thể chạy `scripts/run_scenario3_fedkdl.py` trên server. Hệ thống sẽ tự động dùng dataloader 2D để chia non-IID cho các AUV.
