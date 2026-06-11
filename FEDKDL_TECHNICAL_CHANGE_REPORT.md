# Báo cáo thay đổi kỹ thuật FedKDL

## 1. Mục đích và phạm vi

Báo cáo này tổng hợp các thay đổi gần nhất đối với pipeline huấn luyện FedKDL,
giải thích cơ sở toán học và ghi nhận trạng thái kiểm thử. Các thay đổi tập trung
vào bốn mục tiêu:

1. Tăng đóng góp hữu ích của Knowledge Distillation (KD) mà không thay đổi
   checkpoint warmup.
2. Giảm sai số tích lũy do lượng tử hóa INT8.
3. Bảo toàn hình học LoRA trong tổng hợp phân cấp AUV - Relay - Gateway.
4. Sửa các sai lệch trong mô hình năng lượng, độ trễ và quy tắc hợp tác Relay.

Warmup không bị tăng số epoch và không thay đổi nội dung checkpoint đã thống
nhất. Các baseline chuẩn như FedAvg, FedProx và FLORA không bị áp dụng Server
Mix, nhằm giữ tính công bằng của thực nghiệm.

## 2. Cấu hình chính sau thay đổi

Các tham số đều được tập trung trong `config/settings.py`.

| Nhóm | Tham số chính | Giá trị |
|---|---|---:|
| Local FL | `LOCAL_LR` | `5e-4` |
| Local Head | `LOCAL_HEAD_LR_MULT` | `4.0` |
| Local LoRA | `LOCAL_LORA_LR_MULT` | `1.0` |
| Gateway KD | `KD_LR` | `1e-3` |
| KD Head | `KD_HEAD_LR_MULT` | `4.0` |
| KD LoRA | `KD_LORA_LR_MULT` | `1.0` |
| KD temperature | `KD_TEMPERATURE` | `4.0` |
| KD branch weights | cls / box / projection | `0.45 / 0.35 / 0.20` |
| KD ratio | start / floor | `1.00 / 0.20` |
| Teacher confidence | threshold / gamma | `0.10 / 2.0` |
| Server Mix | `SERVER_MIX_BETA` | `0.90` |
| Relay neighbor weight | nearest / selective | `0.30 / 0.20` |

Local training dùng augmentation nhẹ gồm HSV, translate, scale và horizontal
flip. Mosaic và MixUp vẫn tắt vì mỗi AUV có tập dữ liệu nhỏ và Non-IID; hai phép
augment mạnh này có thể làm thay đổi phân phối cục bộ quá mức.

## 3. Knowledge Distillation mới

### 3.1 Hàm mục tiêu tổng quát

Gateway tối ưu:

```text
L_total = lambda_sup * L_sup + L_KD
```

Trong đó `lambda_sup = 0.5` và:

```text
L_KD = rho_t * (w_cls * L_cls + w_box * L_box + w_proj * L_proj)
```

Với:

```text
w_cls = 0.45, w_box = 0.35, w_proj = 0.20
```

Các trọng số trên biểu diễn tỷ lệ đóng góp mong muốn, không phải hệ số nhân thô
trên ba loss có đơn vị và độ lớn khác nhau.

### 3.2 Chuẩn hóa đóng góp từng nhánh

Gọi supervised reference của một batch là:

```text
R_sup = lambda_sup * |L_sup|
```

Với nhánh KD `j`, code tính hệ số tách khỏi đồ thị gradient:

```text
s_j = clamp(R_sup / (|L_j| + eps), s_min, s_max)
```

Sau đó:

```text
L_j_balanced = w_j * s_j * L_j
```

Tổng các nhánh tiếp tục được chuẩn hóa để đạt tỷ lệ KD mục tiêu:

```text
s_KD = clamp(rho_t * R_sup / (|sum_j L_j_balanced| + eps),
             s_min, s_max)

L_KD = s_KD * sum_j L_j_balanced
```

Do các hệ số chuẩn hóa được tính từ tensor đã `detach`, chúng điều chỉnh độ lớn
nhưng không tạo thêm đường gradient. Khi không chạm giới hạn clamp:

```text
|L_KD| / R_sup ~= rho_t
```

**Tại sao phải dùng 2 tầng chuẩn hóa (Dual Scale)?**
Nếu mạng ổn định và không nhánh nào chạm ngưỡng clamp, hai tầng `s_j` và `s_KD` về mặt toán học sẽ bù trừ nhau, đưa tổng độ lớn (magnitude) của KD về đúng `rho_t * R_sup` và chỉ dùng `L_j` để lấy hướng gradient.
Tuy nhiên, tầng 1 (`s_j`) hoạt động như một cơ chế an toàn (insurance policy): ngăn chặn một nhánh cụ thể (ví dụ `L_proj` thường rất nhỏ ở giai đoạn đầu) bị zeroed out hoặc một nhánh tăng vọt làm hỏng toàn bộ gradient trước khi tầng 2 kịp điều chỉnh toàn cục. Nhờ đó, các trọng số `w_cls`, `w_box`, `w_proj` luôn duy trì đúng tỷ lệ đóng góp thực tế.

Trần `KD_BALANCE_SCALE_MAX` được tăng lên `20.0` để các nhánh KL hoặc projection
vốn nhỏ vẫn có thể đạt tỷ lệ đóng góp cấu hình. Test số học xác nhận với
`L_sup=10`, `lambda_sup=0.5`, các đóng góp lần lượt là `2.25`, `1.75`, `1.00`,
tổng KD bằng `5.0`.

### 3.3 Classification KD có confidence weighting

Teacher probability tại anchor `a`, class `c`:

```text
p_t(a,c) = sigmoid(z_t(a,c) / T)
```

Độ tin cậy anchor:

```text
q_a = max_c sigmoid(z_t(a,c))
```

Trọng số foreground:

```text
m_a = I(q_a > tau) * q_a^gamma
```

Với `tau=0.10`, `gamma=2.0`. Classification KD là binary cross entropy có
temperature:

```text
L_cls =
    T^2 * sum[a,c] m_a * BCEWithLogits(z_s(a,c)/T, p_t(a,c))
    / (C * sum[a] m_a)
```

`T^2` giữ quy mô gradient theo công thức distillation chuẩn. Confidence mask
giảm ảnh hưởng của hàng nghìn background anchors có xác suất gần 0, vốn dễ lấn
át foreground và làm giảm Recall.

### 3.4 Box KD: DFL-KL và CIoU

Box KD cũ dùng MSE trực tiếp giữa các tensor box, không phù hợp khi YOLO biểu
diễn khoảng cách hộp dưới dạng Distribution Focal Loss (DFL). Logic mới gồm:

```text
L_box = lambda_DFL * L_DFL-KL + lambda_CIoU * L_CIoU
```

Với:

```text
lambda_DFL = 1.0
lambda_CIoU = 0.5
```

(Trọng số `lambda_DFL = 1.0` hiện tại không làm thay đổi giá trị nhưng được giữ lại làm placeholder để có thể tune linh hoạt sự tương quan giữa DFL và CIoU trong các thực nghiệm sau này).

DFL-KL truyền phân phối khoảng cách của Teacher:

```text
L_DFL-KL = sum m_a * KL(P_t(d|a) || P_s(d|a)) / (4 * sum m_a)
```

Sau khi giải mã DFL thành tọa độ hộp:

```text
L_CIoU = sum m_a * (1 - CIoU(b_s(a), b_t(a))) / sum m_a
```

Như vậy Student học cả hình dạng phân phối lẫn hình học hộp thực tế.

### 3.5 LoRA Projection KD

Projection của một lớp LoRA là:

```text
h = A * x
```

Student và Teacher được ghép theo stage và theo tỷ lệ độ sâu. Feature map được
adaptive pooling về kích thước nhỏ trước khi tính MSE:

```text
L_proj = mean_k ||normalize(h_s,k) - normalize(h_t,k)||_2^2
```

Cách này giữ tri thức biểu diễn hạng thấp nhưng tránh lưu toàn bộ feature map
Teacher, giảm mạnh VRAM.

### 3.6 Lịch KD và adaptive dropout

Với 60 vòng:

- Giai đoạn đầu: KD mỗi 2 vòng.
- Giai đoạn giữa: KD mỗi 4 vòng.
- Sau `2/3` tổng số vòng: dừng KD và để FL tự tối ưu.

Tỷ lệ `rho_t` giảm tuyến tính từ `1.0` xuống `0.2`. Nếu chất lượng sau các vòng
KD liên tục giảm 5 lần, adaptive dropout tắt KD cho phần còn lại. Chỉ số theo dõi:

```text
quality = mAP50-95 + 0.25 * mAP50 + 0.25 * Recall
```

Teacher detection loss từng được tính nhưng không tham gia công thức mới. Phép
tính dư này đã bị bỏ; Teacher chỉ forward một lần để lấy logits, DFL và
projection.

## 4. Delta-INT8

### 4.1 Vấn đề của lượng tử hóa trọng số tuyệt đối

Lượng tử hóa affine một tensor `x` sử dụng bước:

```text
Delta_x = (max(x) - min(x)) / 255
```

Nếu trọng số có miền giá trị lớn nhưng cập nhật FL rất nhỏ, sai số làm tròn tối
đa khoảng `Delta_x / 2` có thể lớn hơn chính cập nhật. Lượng tử hóa lại trọng số
tuyệt đối qua nhiều vòng còn gây drift tích lũy.

### 4.2 Công thức Delta-INT8

AUV nhận global state `theta_t`, huấn luyện thành `theta_i`, sau đó gửi:

```text
delta_i = theta_i - theta_t
q_i = Q_INT8(delta_i)
```

Relay khôi phục:

```text
theta_i_hat = theta_t + Q_INT8^-1(q_i)
```

Vì miền giá trị của `delta_i` nhỏ hơn miền của `theta_i`, bước lượng tử hóa cũng
nhỏ hơn:

```text
Delta_delta << Delta_theta
```

Kích thước payload không đổi vì số phần tử và số bit mỗi phần tử không đổi.
Delta-INT8 được dùng cho AUV-to-Relay, Relay-to-Relay và Relay-to-Gateway.

BatchNorm state tiếp tục truyền FP32 để bảo vệ `running_mean`, `running_var` và
`num_batches_tracked`.

### 4.3 Kết quả test Delta-INT8

Lệnh:

```powershell
python test_all_scenarios.py --delta-int8-only
```

Kết quả:

```text
PASS Delta-INT8: roundtrip, error reduction, BN/FedAvg, and 60-round drift
```

Test kiểm tra:

- Zero update và constant update.
- Sai số Delta-INT8 nhỏ hơn raw INT8 ít nhất 20 lần trong case kiểm thử.
- FedAvg nhiều client với BN.
- Drift sau 60 vòng.

## 5. Tổng hợp SVD-LoRA

### 5.1 Vì sao không được FedAvg riêng A và B

Với LoRA:

```text
Delta_W_i = B_i * A_i
```

Nếu lấy trung bình riêng:

```text
B_bar = sum p_i B_i
A_bar = sum p_i A_i
```

thì:

```text
B_bar * A_bar = sum_i sum_j p_i p_j B_i A_j
```

Biểu thức xuất hiện các cross-term `B_i A_j` với `i != j`, không tồn tại trong
mục tiêu FedAvg mong muốn:

```text
Delta_W_bar = sum_i p_i B_i A_i
```

### 5.2 SVD-LoRA đúng hình học

Relay và Gateway thực hiện:

```text
Delta_W_bar = sum_i p_i (B_i A_i)
Delta_W_bar = U Sigma V^T
B_new = U_r Sigma_r^(1/2)
A_new = Sigma_r^(1/2) V_r^T
```

Do đó:

```text
B_new A_new = U_r Sigma_r V_r^T
```

Đây là xấp xỉ rank-`r` tối ưu theo chuẩn Frobenius theo định lý
Eckart-Young-Mirsky.

### 5.3 Chuẩn hóa dấu SVD

SVD không xác định duy nhất về dấu:

```text
u_k sigma_k v_k^T = (-u_k) sigma_k (-v_k)^T
```

Hai lần SVD có thể sinh A/B đổi dấu dù `BA` không đổi. Code mới chọn phần tử có
trị tuyệt đối lớn nhất trong mỗi cột `U` làm pivot và ép pivot dương. Phép biến
đổi áp dụng đồng thời lên `U` và `V^T`, nên không thay đổi ma trận hiệu dụng.

Điều này làm biểu diễn A/B ổn định hơn giữa các vòng, có lợi cho Delta-INT8 và
tránh thay đổi tọa độ optimizer không cần thiết. Momentum cũ của riêng tham số
LoRA vẫn được loại sau SVD; momentum Head/BN được giữ vì hệ tọa độ của chúng
không đổi.

## 6. Server Mix tại Gateway

Ý tưởng trong `TODO.md` đã được triển khai có giới hạn cho họ FedKDL:

```text
theta_(t+1) = (1 - beta) * theta_t + beta * theta_agg
```

Với `beta=0.90`:

```text
theta_(t+1) = 0.10 * theta_t + 0.90 * theta_agg
```

Đây là phép interpolation/EMA, chưa phải FedAvgM velocity đầy đủ. Mục tiêu là
giảm dao động global model và giữ lại một phần tri thức vòng trước.

Với LoRA, code không nội suy A và B riêng. Thay vào đó:

```text
Delta_W_mix =
    (1 - beta) * (B_t A_t) + beta * (B_agg A_agg)
```

rồi SVD lại để thu `B_(t+1), A_(t+1)`. Vì vậy Server Mix không tái tạo
cross-term.

Server Mix chỉ bật cho FedKDL và các ablation trực tiếp của FedKDL. FedAvg,
FedProx, FLORA, Naive-LoRA, SCAFFOLD và Top-K giữ cập nhật gốc. Chưa dùng
temporal EMA tại Relay vì association có thể thay đổi theo vòng; state riêng
của một Relay khi đó có nguy cơ stale hoặc đại diện cho tập AUV khác.

## 7. Hợp tác Relay

### 7.1 HFL-Nearest

Relay chọn láng giềng khả thi gần nhất:

```text
m* = argmin_k d(m,k)
```

Không yêu cầu cụm láng giềng lớn hơn. Trọng số:

```text
theta_m_coop = 0.70 * theta_m + 0.30 * theta_m*
```

### 7.2 HFL-Selective

Chỉ Relay có cụm nhỏ mới hợp tác:

```text
c_m <= max(2, 0.75 * mean(c))
```

Ứng viên phải có cụm lớn hơn và nằm trong ngưỡng khoảng cách Q1. Trọng số:

```text
theta_m_coop = 0.80 * theta_m + 0.20 * theta_m*
```

Việc blend LoRA cũng diễn ra trên `BA` rồi SVD, không blend riêng A/B.

Một lỗi cũ đã được sửa: HFL-Nearest trước đây vô tình dùng điều kiện “cụm lớn
hơn” của HFL-Selective, khiến Relay lớn nhất hoặc bằng kích thước không tìm được
partner.

## 8. Kiểm tra kiến trúc ba tầng

### 8.1 AUV-to-Relay

Mỗi local state được tổng hợp theo số mẫu:

```text
theta_m = sum_(i in C_m) n_i / N_m * theta_i
```

Delta-INT8 luôn được giải mã dựa trên đúng global state đã broadcast đầu vòng.

### 8.2 Relay-to-Gateway

Gateway dùng tổng số mẫu của từng cụm:

```text
theta_agg = sum_m N_m / N * theta_m
```

Nếu có cooperation, state sau cooperation là state thực sự được truyền và tổng
hợp. Đường R2R/R2G của FedKDL đi qua codec INT8 giống kích thước payload đã dùng
trong mô hình vật lý.

### 8.3 SVD computation accounting

Relay luôn có một Temp SVD sau intra-cluster aggregation. Final SVD chỉ phát sinh
khi cooperation thực sự xảy ra. Năng lượng và độ trễ SVD nay dựa trên số lần gọi
thực tế thay vì mặc định hai lần cho mọi Relay.

## 9. Sửa metric vật lý

### 9.1 Năng lượng R2R

Trong cooperation, partner là Relay phát model và Relay hiện tại là bên nhận.
Năng lượng truyền:

```text
E_tx = (P_ac / eta_EA + P_c_tx) * S / R
```

trước đây bị trừ vào pin Relay nhận. Code mới trừ đúng vào pin Relay phát.

### 9.2 Độ trễ theo từng nhánh Relay

Độ trễ vòng phải lấy bottleneck của các đường hoàn chỉnh:

```text
tau_round =
    max_m (tau_A2R,m + tau_R2R,m + tau_R2G,m)
    + tau_comp + tau_SVD
```

Logic cũ cộng `tau_A2R` của Relay `m` với `max tau_R2G` toàn mạng, có thể ghép hai
chặng thuộc hai Relay khác nhau và phóng đại độ trễ. Logic mới giữ từng đường
Relay gắn với nhau trước khi lấy max. Test số học cố định xác nhận bottleneck
đúng bằng `11.0 s` thay vì tổng giả `20.0 s`.

## 10. Tối ưu tốc độ huấn luyện

Các thay đổi không tăng batch size hoặc warmup:

- Giữ persistent dataloader và RAM dataset cache theo từng AUV qua các vòng.
- `workers=0` để tránh chi phí spawn process lớn đối với dataset rất nhỏ của
  từng AUV.
- Bỏ local validation sau mỗi AUV vì không ảnh hưởng global model.
- Bỏ teacher criterion không được sử dụng trong Gateway KD.
- Không gọi `torch.cuda.empty_cache()` sau từng AUV trừ khi cấu hình bật.
- AMP tiếp tục được dùng cho local training và KD.

## 11. Trạng thái kiểm thử

### 11.1 Delta-INT8

`PASS` cho round-trip, giảm sai số, BN/FedAvg và drift 60 vòng.

### 11.2 Toàn bộ 18 baseline

Lệnh:

```powershell
python test_all_scenarios.py
```

Kết quả đính kèm xác nhận tất cả 18 baseline `PASS`:

```text
fedkdl, fedavg, fedprox, fedavg_hfl, topk_grad, flora,
naive_lora, scaffold, fedkdl_nocoop, fedkdl_selective,
fedkdl_nokd, fedkdl_proxy_ft, logit_kd, centralized,
fedprox_kdl, fedkdl_nolora, fedkd, fedprox_hfl
```

Test kiểm tra ba lớp:

1. AUV: local update và đúng loại payload.
2. Relay: weighted aggregation, cooperation hoặc bypass đối với flat FL.
3. Gateway: aggregate-only, projection KD, logit KD, proxy fine-tune hoặc
   centralized routing.

Ngoài ra test còn khóa các invariant:

- SVD-LoRA khác Naive-LoRA và không sinh cross-term.
- Server Mix bảo toàn effective LoRA matrix.
- Dấu SVD xác định.
- KD đạt tỷ lệ đóng góp số học mong muốn.
- HFL-Nearest và HFL-Selective chọn đúng partner.
- Độ trễ giữ đúng từng đường Relay.

## 12. Ý nghĩa và giới hạn xác minh

Các test hiện tại chứng minh tính đúng đắn về:

- Dòng dữ liệu qua ba tầng.
- Công thức tổng hợp và trọng số mẫu.
- Round-trip của codec.
- Routing của 18 baseline.
- Các invariant đại số của SVD, Server Mix và KD balancing.

Chúng chưa chứng minh FedKDL chắc chắn tăng mAP trên dữ liệu thật. Điều đó cần
full GPU experiment vì hội tụ còn phụ thuộc teacher quality, proxy distribution,
Non-IID partition và stochastic optimization.

Các metric cần theo dõi ở run tiếp theo:

```text
mAP50-95, mAP50, Precision, Recall
kd_ratio, kd_scale
kd_cls_contrib, kd_box_contrib, kd_proj_contrib
server_mix_beta
tau_a2r, tau_r2r, tau_r2g, tau_svd
e_a2r, e_r2r, e_r2g, e_svd
```

Kỳ vọng hợp lý, không phải cam kết kết quả:

- Delta-INT8 giảm drift lượng tử hóa.
- Server Mix giảm dao động giữa vòng KD và vòng FL.
- Confidence-weighted KD bảo vệ Recall khỏi background domination.
- DFL-KL + CIoU truyền box knowledge phù hợp YOLO hơn Box MSE.
- SVD-LoRA và canonical sign làm quỹ đạo LoRA ổn định hơn.

## 13. File code liên quan

| File | Vai trò |
|---|---|
| `config/settings.py` | Nguồn cấu hình tập trung |
| `federated_core/aggregator.py` | FedAvg, SVD-LoRA, canonical sign, Server Mix |
| `federated_core/workers.py` | Relay cooperation và Gateway aggregation |
| `federated_core/hfl_rules.py` | Quy tắc Nearest/Selective |
| `federated_core/base_simulator.py` | Pipeline ba tầng và accounting SVD/energy |
| `federated_core/metrics.py` | Tính latency theo từng đường Relay |
| `tasks/detection_2d/simulator.py` | Delta-INT8, R2R/R2G codec, lịch Gateway KD |
| `tasks/detection_2d/trainer.py` | Local cache, AMP và augmentation nhẹ |
| `tasks/detection_2d/knowledge_compression/int8_quantization.py` | Codec Delta-INT8 |
| `tasks/detection_2d/knowledge_compression/knowledge_distillation.py` | KD loss mới |
| `tasks/detection_2d/baselines.py` | Contract của 18 baseline |
| `test_all_scenarios.py` | Kiểm thử deterministic toàn pipeline |
