# FedKDL — Thực nghiệm TODO

> Cập nhật: 2026-06-07. Theo cấu trúc 4 kịch bản trong `FedKDL-vi.tex`.

---

## ✅ Code & Config (Đã chốt)

- [x] LR Config chốt:
  - **Warmup** (5 ep, AdamW, FP32): `head×3.0 → 6e-3`, `lora×0.5 → 1e-3`
  - **Centralized LoRA** (150 ep, AdamW, FP32): `head×1.0 → 2e-3`, `lora×0.25 → 5e-4`
  - **FL Local SGD** (2 ep/round, SGD, FP32): `head×2.0 → 1e-3`, `lora×1.0 → 5e-4`
  - **Teacher YOLO12l** (300 ep, AdamW, AMP): `head×1.0 → 2e-4`, `lora×0.25 → 5e-5`
- [x] FedBN: BN params được train (`requires_grad=True`) nhưng không gửi lên server
- [x] `_local_bn_state` được persist qua các FL round (fix fed-forget-bn bug)
- [x] `final_eval()` bypass `fuse()` để tránh phá LoRAConv2d layers

---

## 🔬 Kịch bản 1 — Flat Topology Failure

> Mục tiêu: Chứng minh Flat FL "chết toàn tập" → HFL là bắt buộc.
> Baselines: FedAvg, FedProx, SCAFFOLD (Flat) vs. FedKDL (HFL)

- [ ] **Chạy FedAvg Flat** 60 vòng trên N=30, ghi lại `η_part`, `E_comm`, `τ_round`, `mAP`
- [ ] **Chạy FedProx Flat** 60 vòng (mu=0.01)
- [ ] **Chạy SCAFFOLD Flat** 60 vòng
- [ ] **Chạy FedKDL (HFL)** 60 vòng — kết quả kỳ vọng: `η_part=100%`, `mAP≈68.7%`
- [ ] Điền kết quả vào `Table 1` (`tab:topology_stats`) trong LaTeX
- [ ] Vẽ plot: `E_comm (J) vs Rounds` — Flat cạn pin sớm, HFL an toàn

---

## 🔬 Kịch bản 2 — Compression & Subspace Misalignment

> Mục tiêu: SVD-LoRA là cách nén duy nhất không phá không gian đặc trưng thị giác.
> Baselines: HFL + Full Param, HFL + Top-K, HFL + Naive LoRA, HFL + SVD-LoRA

- [ ] **Chạy HFL + Full Param** — kỳ vọng: cạn pin vòng 5-10 (`E_comm > E_limit`)
- [ ] **Chạy HFL + Top-K** (sparsity=0.01) — kỳ vọng: cạn pin vòng 30-40 (`E_comp cao`)
- [ ] **Chạy HFL + Naive LoRA** — kỳ vọng: hoàn tất 60 vòng nhưng mAP kịch trần ≈ 42.8%
- [ ] **Chạy HFL + SVD-LoRA** — kỳ vọng: hoàn tất 60 vòng, mAP tiệm cận Top-K trước khi nó chết
- [ ] Vẽ `Figure (scenario2_energy_map.pdf)`: 2 subplots — Cumulative Energy vs Rounds + mAP vs Rounds

---

## 🔬 Kịch bản 3 — Knowledge Distillation Mismatch

> Mục tiêu: LoRA-Proj KD phá trần mAP mà không gây OOM tại AUV.
> Baselines: HFL+SVD-LoRA (No KD), Logit-KD, Feature-KD, LoRA-Proj KD

- [ ] **Chạy HFL + SVD-LoRA (No KD)** — kỳ vọng: mAP ≈ 61.4% (trần của student nhỏ)
- [ ] **Chạy Logit-KD** — kỳ vọng: mAP ≈ 61.8% (gần như không cải thiện)
- [ ] **Chạy Feature-KD** — kỳ vọng: Loss giảm nhanh hơn nhưng OOM hoặc RAM áp lực cao
- [ ] **Chạy LoRA-Proj KD (FedKDL)** — kỳ vọng: mAP bứt phá ≈ 68.7%, Loss hội tụ dốc nhất
- [ ] Vẽ `Figure (scenario3_kd_mismatch.pdf)`: 2 subplots — Training Loss + mAP vs Rounds

---

## 🔬 Kịch bản 4 — Joint Cost Assessment

> Mục tiêu: Chứng minh FedKDL tối ưu hóa `F = E_total + λ*τ` toàn vòng đời.
> Baselines: SCAFFOLD (Flat), HFL+Top-K, HFL+SVD-LoRA+Feature-KD, FedKDL

- [ ] Lấy kết quả từ 3 kịch bản trên, tính `F` cho từng phương pháp
- [ ] Vẽ `Figure (scenario4_joint_cost.pdf)`: `F vs Rounds` — FedKDL là đường duy nhất không đứt

---

## 📄 LaTeX / Paper

- [ ] Điền số thực vào `Table 1` (Kịch bản 1) — thay các placeholder `> 500`, `> 1800`
- [ ] Thay placeholder `mAP 68.7%` bằng số thực từ thực nghiệm
- [ ] Thay placeholder `42.8%` (Naive LoRA) và `61.4%` (No KD) bằng số thực
- [ ] Tạo file `figs/scenario2_energy_map.pdf`
- [ ] Tạo file `figs/scenario3_kd_mismatch.pdf`
- [ ] Tạo file `figs/scenario4_joint_cost.pdf`
- [ ] Thay `[demo]` trong `\usepackage[demo]{graphicx}` bằng gói thật khi có hình

---

## 🛠 Kỹ thuật còn lại

- [ ] Xác nhận `mAP50` đạt ≥ 68% sau khi centralized train xong với LR config mới
- [ ] Chạy FL full pipeline `main_trainer_od.py` và verify `mAP` không bị tụt so với centralized
- [ ] Kiểm tra `E_comm` tính đúng trong `simulator.py` (dùng Thorp-Wenz model)
- [ ] Double-check `S_avg` payload KB vs. lý thuyết (`rank=8`, INT8 → ≈127KB)
