#!/usr/bin/env bash
# test_pipeline.sh
# Script chạy thử nghiệm (smoke test) siêu nhanh để đảm bảo code không bị lỗi (crash)
# Chỉ chạy: N=50, 1 Vòng (Round=1) trên bộ dữ liệu nhỏ nhất để test logic.

set -e

if [[ -f ".venv/bin/python" ]]; then
  PYTHON=".venv/bin/python"
elif [[ -f ".venv/Scripts/python.exe" ]]; then
  PYTHON=".venv/Scripts/python.exe"
else
  PYTHON="${PYTHON:-python}"
fi

echo "============================================================"
echo "BẮT ĐẦU SMOKE TEST (KIỂM TRA LỖI LOGIC CODE)"
echo "============================================================"

# ---- TEST 2D (OD) ----
N_2D=50
DS_2D="URPC"
ALPHA_2D="0.5" # URPC thường dùng 0.5 để kiểm tra tính năng Heterogeneity
SEED_2D="42"
ALPHA_STR_2D="0p5"

TOPO_2D="environments/topo/N_${N_2D}/topo_N${N_2D}_seed${SEED_2D}.pkl"
DATA_2D="environments/data/${DS_2D}/N_${N_2D}/data_N${N_2D}_${DS_2D}_a${ALPHA_STR_2D}_seed${SEED_2D}.pkl"

BASELINES_2D=(
    "fedkdl"
    "fedkdl_r4"
    "full_param_kd"
    "full_param_nokd"
    "lora_head_kd_noint8"
    "head_kd_int8_nolora"
    "lora_head_int8_nokd"
    "fedavg"
    "fedprox"
    "centralized"
)

if [[ -f "$TOPO_2D" && -f "$DATA_2D" ]]; then
    for baseline in "${BASELINES_2D[@]}"; do
        echo ""
        echo ">>> [TEST 2D] Đang kiểm tra baseline: $baseline"
        $PYTHON main_trainer_od.py \
            --topo "$TOPO_2D" \
            --data "$DATA_2D" \
            --baseline "$baseline" \
            --rounds 2 \
            --out-dir "results/test_logs" \
            --log-dir "results/test_logs"
    done
else
    echo "⚠️ Bỏ qua test 2D vì thiếu file môi trường (Chạy utils/generate_all_envs.py trước)."
fi

echo "============================================================"
echo "🎉 XONG! TẤT CẢ CÁC KỊCH BẢN ĐỀU CHẠY THÀNH CÔNG KHÔNG CRASH!"
echo "============================================================"
