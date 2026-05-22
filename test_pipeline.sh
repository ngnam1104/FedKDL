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

# ---- TEST 1D (HFL) ----
N_1D=50
DS_1D="SMD"
ALPHA_1D="10000.0"
SEED_1D="42"
ALPHA_STR_1D="10000p0"

TOPO_1D="environments/topo/N_${N_1D}/topo_N${N_1D}_seed${SEED_1D}.pkl"
DATA_1D="environments/data/${DS_1D}/N_${N_1D}/data_N${N_1D}_${DS_1D}_a${ALPHA_STR_1D}_seed${SEED_1D}.pkl"

BASELINES_1D=(
    "hfl_selective"
    "hfl_nearest"
    "hfl_nocoop"
    "fedprox"
    "fedavg"
    "centralized"
)

if [[ -f "$TOPO_1D" && -f "$DATA_1D" ]]; then
    for baseline in "${BASELINES_1D[@]}"; do
        echo ""
        echo ">>> [TEST 1D] Đang kiểm tra baseline: $baseline"
        $PYTHON main_trainer.py \
            --topo "$TOPO_1D" \
            --data "$DATA_1D" \
            --baseline "$baseline" \
            --rounds 1 \
            --out-dir "results/test_logs" \
            --log-dir "results/test_logs"
    done
else
    echo "⚠️ Bỏ qua test 1D vì thiếu file môi trường."
fi

# ---- TEST 2D (OD) ----
N_2D=50
DS_2D="URPC"
ALPHA_2D="10000.0"
SEED_2D="42"
ALPHA_STR_2D="10000p0"

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
            --rounds 1 \
            --out-dir "results/test_logs" \
            --log-dir "results/test_logs"
    done
else
    echo "⚠️ Bỏ qua test 2D vì thiếu file môi trường (Chạy utils/generate_all_envs.py trước)."
fi

echo "============================================================"
echo "🎉 XONG! TẤT CẢ CÁC KỊCH BẢN ĐỀU CHẠY THÀNH CÔNG KHÔNG CRASH!"
echo "============================================================"
