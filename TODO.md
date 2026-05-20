## UPDATE SYSTEM MODEL VÀ PROBLEM FORMULATION

Hệ thống IoUT được định hình bởi không gian 3D $V = [0, L_x] \times [0, L_y] \times [0, H]$ và cấu trúc mạng lưới phân cấp tĩnh. Để tối ưu hóa thuật toán học máy trong môi trường này, hệ thống được mô hình hóa chặt chẽ qua 6 khối mô hình thành phần và một bài toán tối ưu tổng hợp.

### 1. Kiến trúc hệ thống tổng quát (Overall System Architecture)

Kiến trúc liên kết của mạng IoUT được chia thành 3 tầng phân cấp rõ rệt:

* **Tầng Cảm biến (Deep Layer):** Gồm $N$ cảm biến ($s_1, s_2, \dots, s_N$) cố định dưới đáy biển (độ sâu 500–1000m), làm nhiệm vụ thu thập dữ liệu và huấn luyện mô hình cục bộ.
* **Tầng Fog (Mid-water Layer):** Gồm $M$ trạm sương mù (Fog Aggregators / AUVs) lơ lửng ở tầng nước giữa (độ sâu 100–400m). Các AUV này di chuyển theo mô hình Gauss-Markov, có nhiệm vụ thu nhận, tổng hợp cập nhật từ cảm biến và hợp tác trao đổi với các AUV lân cận.
* **Tầng Surface (Surface Gateway):** Một cổng trung tâm duy nhất trên mặt nước ($z \approx 0$) đóng vai trò điều phối quá trình Học liên kết (FL) toàn cục và kết nối với Cloud.

### 2. Mô hình Kênh truyền Âm thanh Dưới nước (Underwater Acoustic Channel Model)

Kênh truyền âm thanh chịu sự chi phối của suy hao vật lý và nhiễu môi trường, quyết định tính khả thi của mọi liên kết mạng.

* **Suy hao truyền dẫn (Transmission Loss - TL):** Tính toán độ suy giảm tín hiệu theo khoảng cách $d$ và tần số $f$:

$$TL(d, f) = 10k \log_{10}(d) + \alpha(f)\frac{d}{1000} \tag{1}$$


* **Hệ số hấp thụ Thorp ($\alpha(f)$):** Tính bằng dB/km tại tần số $f$ (kHz):

$$\alpha(f) = \frac{0.11f^2}{1+f^2} + \frac{44f^2}{4100+f^2} + 2.75 \times 10^{-4}f^2 + 0.003 \tag{2}$$


* **Độ trễ lan truyền (Propagation delay):** Thời gian tín hiệu di chuyển giữa hai nút $u$ và $v$ với tốc độ âm thanh $c_s \approx 1500$ m/s:

$$\tau_{uv} = \frac{d_{uv}}{c_s} \tag{3}$$


* **Mức nhiễu tại dải thông (Noise Level - NL):** Tổng hợp nhiễu nền Wenz ($N_0(f)$) lọt vào bộ thu có dải thông $B$ (Hz):

$$NL(f, B) = N_0(f) + 10 \log_{10}(B) \tag{4}$$


* **Tỷ số Tín hiệu trên Nhiễu (Receiver SNR):** Đánh giá chất lượng tín hiệu qua phương trình sonar thụ động (với $IL$ là suy hao phần cứng):

$$SNR_{uv} = SL_u - TL(d_{uv}, f) - NL(f, B) - IL \tag{5}$$


* **Điều kiện Khả thi Kênh truyền:** Mức nguồn phát tối thiểu ($SL_u^{min}$) để đạt ngưỡng SNR mục tiêu ($\gamma_{tgt}$) không được vượt quá giới hạn phần cứng ($SL_{max}$):

$$SL_u^{min}(u, v) = \gamma_{tgt} + TL(d_{uv}, f) + NL(f, B) + IL \le SL_{max} \tag{6}$$



### 3. Mô hình Năng lượng (Energy Model)

Hệ thống hạch toán năng lượng tiêu thụ theo từng vòng, phân tách rõ điện toán và truyền thông.

* **Công suất phát âm thanh và điện năng ($P_{ac}$, $P_{tx}$):**

$$P_{ac} = \frac{4\pi p_{ref}^2}{\rho_w c_s} 10^{SL^{min}/10}, \quad P_{tx} = \frac{P_{ac}}{\eta_{ea}} \tag{7}$$


* **Tốc độ truyền dẫn (Shannon-type Rate):** Tốc độ tối đa (bps) khi bám sát $\gamma_{tgt}$:

$$R_{uv} = B \log_2\left(1 + 10^{\gamma_{tgt}/10}\right) \tag{8}$$


* **Năng lượng Viễn thông và Điện toán:**
* Phát gói tin $L$ bit: $E_{tx}(L; u, v) = (P_{tx} + P_{c,tx}) \frac{L}{R_{uv}}$.
* Tính toán cục bộ: $E_{comp, i} = \epsilon_{op} \Phi_i$ (với $\Phi_i$ là số FLOPs).


* **Tổng năng lượng vòng học ($E_{round}^t$):** Phân rã theo 3 chặng:

$$E_{round}^t = \underbrace{\sum_{i\in\mathcal{S}} E_{tx}(L_u; i, a_i^t)}_{E_{s2f}^t} + \underbrace{\sum_{m\in\mathcal{F}}\sum_{j\in\mathcal{N}_m^t} E_{tx}(L_f; m, j)}_{E_{f2f}^t} + \underbrace{\sum_{m\in\mathcal{F}} E_{tx}(L_g; m, g)}_{E_{f2g}^t} \tag{9}$$


* **Động lực học Pin:** $E_i^{t+1} = E_i^t - E_{tx}(L_u; i, a_i^t) - E_{comp, i}^t$.

### 4. Mô hình Độ trễ (Latency Model)

Tính tổng độ trễ dựa trên "nút thắt cổ chai" của hệ thống.

* **Độ trễ liên kết đơn lẻ:** Bao gồm trễ lan truyền và trễ truyền tải:

$$\tau_{u \to v}^t = \frac{d_{uv}^t}{c_s} + \frac{L_{u \to v}^t}{R_{uv}^t} \tag{10}$$


* **Độ trễ toàn vòng lặp ($\tau_{round}^t$):** Chặng tốn thời gian nhất quyết định toàn mạng:

$$\tau_{round}^t = \max\left\{ \max_i \tau_{i \to a_i^t}^t, \max_{m,j} \tau_{m \to j}^t, \max_m \tau_{m \to g}^t \right\} + \tau_{comp}^t \tag{11}$$



### 5. Mô hình Tác vụ Nhận diện Đối tượng (Object Detection)

Mỗi nút $i$ lưu trữ tập dữ liệu hình ảnh dưới nước $\mathcal{D}_i = \{I_{i,n}, Y_{i,n}\}_{n=1}^{n_i}$, với $Y_{i,n}$ chứa nhãn phân lớp và tọa độ hộp bao (Bounding Boxes).

* **Hàm mất mát cấu trúc (Learning Objective):** Kết hợp 3 thành phần:

$$\mathcal{L}_{YOLO}(\theta) = \lambda_{box}\mathcal{L}_{box} + \lambda_{cls}\mathcal{L}_{cls} + \lambda_{dfl}\mathcal{L}_{dfl} \tag{12}$$



(Trong đó: $\mathcal{L}_{box}$ định vị hộp bao, $\mathcal{L}_{cls}$ phân lớp, và $\mathcal{L}_{dfl}$ tối ưu ranh giới dưới nước).
* **Hàm mục tiêu cục bộ:**

$$F_i(\theta) = \frac{1}{n_i} \sum_{n=1}^{n_i} \mathcal{L}_{YOLO}(I_{i,n}, Y_{i,n}; \theta) \tag{13}$$



### 6. Mô hình Học liên kết Phân cấp (Hierarchical FL Model)

Quá trình FL diễn ra qua 3 pha đồng bộ trong 1 vòng học $t$:

* **Tổng hợp nội cụm (Intra-cluster aggregation):** Trạm Fog $m$ gom cập nhật từ tập thành viên $\mathcal{C}_m^t$:

$$\theta_{m}^{t+1/2} = \theta^{t} + \sum_{i\in\mathcal{C}_{m}^{t}} \frac{n_{i}}{\sum_{k\in\mathcal{C}_{m}^{t}}n_{k}} \Delta\theta_{i}^{t} \tag{14}$$


* **Pha trộn hợp tác (Cooperative fog mixing):** Trạm Fog $m$ kết hợp mô hình với tập $\mathcal{N}_m^t$ trạm lân cận:

$$\tilde{\theta}_m^{t+1} = \sum_{j\in\{m\}\cup\mathcal{N}_m^t} \alpha_{m,j}^{t} \theta_{j}^{t+1/2} \tag{15}$$


* **Tổng hợp Toàn cục (Global Aggregation):** Gateway trên mặt nước tính trung bình có trọng số:

$$\Theta^{t+1} = \sum_{m\in\mathcal{F}} \frac{\sum_{i\in\mathcal{C}_{m}^{t}}n_{i}}{\sum_{k\in\mathcal{S}}n_{k}} \tilde{\theta}_{m}^{t+1} \tag{16}$$



### 7. Bài toán Tối ưu hóa Tổng hợp (Problem Formulation)

Hệ thống phải tự động ra các quyết định điều khiển: liên kết $a_i^t$, tập hợp tác $\mathcal{N}_m^t$, và hệ số pha trộn $\alpha_{m,j}^t$. Bài toán tối ưu hóa tổ hợp (Non-convex and Combinatorial Optimization) được đặt ra nhằm tối thiểu hóa hàm mục tiêu:

$$\min_{\{\theta, a_i^t, \mathcal{N}_m^t, \alpha_m^t\}} \left\{ F(\theta^T) + \lambda_E \sum_{t=0}^{T-1} E_{round}^t + \lambda_\tau \sum_{t=0}^{T-1} \tau_{round}^t \right\} \tag{17}$$

Chịu sự chi phối của các ràng buộc vật lý:

* **(C1) Association:** $a_i^t \in \mathcal{F}, \quad \forall i, t$.
* **(C2) Cooperation:** $\alpha_{m,j}^t \ge 0, \sum_{j} \alpha_{m,j}^t = 1$, và $|\mathcal{N}_m^t| \le K$.
* **(C3) Energy Reserve:** $E_i^{t+1} \ge E_{min}, \quad \forall i, t$.
* **(C4) Latency Deadline:** $\tau_{round}^t \le \tau_{max}, \quad \forall t$.
* **(C5) Acoustic Feasibility:** $SNR_l^t \ge \gamma_{tgt}, \quad \forall l, t$.