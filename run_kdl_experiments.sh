#!/usr/bin/env bash
# Scenario 2/3 (2D OD): grid train + plot.
# Decoupled version to hit the 62-hour budget (~78 hours max).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "============================================================"
echo "[START RUN] Timestamp: $(date +"%Y-%m-%d %H:%M:%S")"
echo "============================================================"

if [[ -f ".venv/bin/python" ]]; then
  PYTHON=".venv/bin/python"
elif [[ -f ".venv/Scripts/python.exe" ]]; then
  PYTHON=".venv/Scripts/python.exe"
else
  PYTHON="${PYTHON:-python}"
fi

OUT_DIR="results/logs_kdl"
ENVS_DIR="environments"
STDOUT_DIR="results/train_logs/kdl"
mkdir -p "$OUT_DIR" "$STDOUT_DIR"
export PYTHONIOENCODING=utf-8

# =========================================================
# Cấu hình thử nghiệm (Chỉnh sửa các tham số tại đây)
# =========================================================
ROUNDS=50
SEED=42
DS="URPC"
M_FOGS_2D=4       # Thay đổi số lượng Fog tại đây (vd: 4, 5, 10...)
# =========================================================

GEN_ENV_ARGS=()
# Kiểm tra nếu M_FOGS_2D được gán giá trị thì tự động thêm cờ
if [[ -n "$M_FOGS_2D" ]]; then
  echo "[KDL] Overriding fog count for 2D topologies: M_FOGS_2D=$M_FOGS_2D"
  GEN_ENV_ARGS=(--m-fogs "$M_FOGS_2D")
fi

# Sinh dữ liệu môi trường riêng cho mạng lớn (N=20, 30, 40, 50)
echo "[KDL] Generating topologies and data partitions..."
"$PYTHON" utils/generate_all_envs.py --n 20 --dataset "$DS" "${GEN_ENV_ARGS[@]}"
"$PYTHON" utils/generate_all_envs.py --n 30 --dataset "$DS" "${GEN_ENV_ARGS[@]}"
"$PYTHON" utils/generate_all_envs.py --n 40 --dataset "$DS" "${GEN_ENV_ARGS[@]}"
"$PYTHON" utils/generate_all_envs.py --n 50 --dataset "$DS" "${GEN_ENV_ARGS[@]}"

# Pre-train Teacher & Khởi động ấm Student
echo "[KDL] Đang tiến hành chuẩn bị các mô hình Teacher và Student..."
"$PYTHON" scripts/fedkdl/pretrain.py

# =========================================================
# Hàm chạy chung để tránh lặp code (Giữ nguyên đoạn này trở xuống)
# =========================================================
total_tasks=53
current_task=0

run_baseline() {
  local n=$1
  local alpha=$2
  local baseline=$3
  local lora_rank=${4:-""}  # Optional: lora rank override

  current_task=$((current_task + 1))
  local topo="${ENVS_DIR}/2d/topo/N_${n}/topo_N${n}_seed${SEED}.pkl"
  local alpha_str="${alpha//./p}"
  local data="${ENVS_DIR}/2d/data/${DS}/N_${n}/data_N${n}_${DS}_a${alpha_str}_seed${SEED}.pkl"

  if [[ ! -f "$topo" || ! -f "$data" ]]; then
    echo "[Warning] Missing env: N=$n DS=$DS alpha=$alpha seed=$SEED"
    return
  fi

  local log_json="${OUT_DIR}/log_N${n}_${DS}_a${alpha_str}_${baseline}_seed${SEED}.json"
  # Skip nếu JSON tồn tại và dủ lớn (> 1KB) — tránh skip file rỗng do crash
  if [[ -s "$log_json" ]] && [[ $(stat -c%s "$log_json" 2>/dev/null || echo 0) -gt 1024 ]]; then
    echo "[$current_task/$total_tasks] SKIP (complete log exists): $log_json"
    return 0
  elif [[ -f "$log_json" ]]; then
    echo "[$current_task/$total_tasks] OVERWRITE (incomplete/crash log): $log_json"
    rm -f "$log_json"
  fi

  echo "[$current_task/$total_tasks] OD | N=$n | alpha=$alpha | baseline=$baseline${lora_rank:+ | lora_rank=$lora_rank}"
  set +eo pipefail

  local TS=$(date +"%Y%m%d_%H%M%S")
  local log_file="$STDOUT_DIR/raw_bash_output_${n}_${alpha_str}_${TS}.log"

  local extra_args=""
  if [[ -n "$lora_rank" ]]; then
    extra_args="--lora-rank $lora_rank"
  fi

  "$PYTHON" main_trainer_od.py \
    --topo "$topo" --data "$data" \
    --baseline "$baseline" --rounds "$ROUNDS" \
    --out-dir "$OUT_DIR" --log-dir "$STDOUT_DIR" \
    $extra_args 2>&1 | tee -a "$log_file"
  local rc=${PIPESTATUS[0]}
  set -eo pipefail
  if [[ $rc -ne 0 ]]; then
    echo "[Error] Run failed (exit $rc). Check $log_file"
    # Xóa JSON bị viết dở do crash để lần sau sẽ chạy lại
    rm -f "$log_json"
  fi
}

echo ""
echo ""
echo "=== GROUP A1: KDL-Accelerated Baselines ==="
# N=20, Alpha=2.0
# Đã áp dụng toàn bộ KDL (LoRA+INT8+KD) vào các chiến lược truyền thống.
# fedkdl (hfl_selective_kdl) chạy ĐẦU TIÊN
KDL_BASELINES=(
  "fedkdl" "fedavg_kdl" "fedprox_kdl" "hfl_nocoop_kdl" "hfl_nearest_kdl"
)
for b in "${KDL_BASELINES[@]}"; do
  run_baseline 20 2.0 "$b"
done

echo ""
echo "=== GROUP A2: FedKDL Ablation Studies ==="
# N=20, Alpha=2.0
# GIẢI THÍCH CƠ CHẾ PARSE TÊN BASELINE TRONG CODE:
# 1. Mặc định các chiến lược ablation KHÔNG CÓ chữ 'noint8' đều được áp dụng nén INT8 (giảm 4 lần payload).
#    VD: 'full_param_kd' gửi toàn bộ 2.6M tham số, nhưng được ép xuống INT8 (1 byte/param) nên payload chỉ còn ~2.5 MB.
# 2. Nếu tên có chữ 'noint8' (như 'lora_head_kd_noint8'), code sẽ giữ nguyên định dạng Float32 (4 bytes/param).
# 3. Tương tự: 'nolora' sẽ gửi full param; 'nokd' sẽ bỏ qua bước Knowledge Distillation tại Gateway.
ABLATION_BASELINES=(
  "fedkdl_r4" "full_param_kd" "full_param_nokd" 
  "lora_head_kd_noint8" "head_kd_int8_nolora" "lora_head_int8_nokd"
)
for b in "${ABLATION_BASELINES[@]}"; do
  if [[ "$b" == "fedkdl_r4" ]]; then
    run_baseline 20 2.0 "$b" 4  # r=4 ablation
  else
    run_baseline 20 2.0 "$b"
  fi
done

echo ""
echo "=== GROUP A3: Classic Full-Param Baselines ==="
# N=20, Alpha=2.0
# GIẢI THÍCH:
# Với các thuật toán truyền thống (nằm trong mảng classic_baselines của python code),
# hệ thống sẽ tự động ép cờ: use_lora=False và use_int8=False.
# Do đó các thuật toán dưới đây sẽ truyền 100% tham số ở chuẩn FP32 nguyên bản (Payload ~10.5 MB).
CLASSIC_BASELINES=(
  "centralized" "fedavg" "fedprox" "fedkd" "hfl_nocoop" "hfl_nearest" "hfl_selective"
)
for b in "${CLASSIC_BASELINES[@]}"; do
  run_baseline 20 2.0 "$b"
done

echo ""
echo "=== GROUP B: Scalability ==="
# N=20, 30, 40, 50 — với 5 Fog, cần ít nhất N=20 để ý nghĩa thống kê (4 sensor/fog)
# Áp dụng công nghệ nén KDL lên tất cả, chỉ so sánh sự khác biệt của thuật toán gom nhóm.
MAIN_BASELINES=("fedkdl" "fedavg_kdl" "fedprox_kdl" "hfl_nocoop_kdl" "hfl_nearest_kdl" "centralized" "fedkd")
for n in 20 30 40 50; do
  for b in "${MAIN_BASELINES[@]}"; do
    run_baseline "$n" 2.0 "$b"
  done
done

echo ""
echo "=== GROUP C: Heterogeneity ==="
# N=20, Alpha=10000.0 (Non-IID rất thấp -> IID)
for b in "${MAIN_BASELINES[@]}"; do
  run_baseline 20 10000.0 "$b"
done

echo ""
echo "[KDL] Training done. Generating plots..."
"$PYTHON" scripts/fedkdl/plot_od_comparison.py
"$PYTHON" scripts/fedkdl/plot_od_scalability.py
"$PYTHON" scripts/fedkdl/plot_heterogeneity.py
"$PYTHON" scripts/fedkdl/eval_baselines.py --results-dir "$OUT_DIR"
"$PYTHON" scripts/od/plot_ablation.py

echo "[KDL] All done."
echo "  JSON (plot):  $OUT_DIR/*.json"
echo "  Stdout logs:  $STDOUT_DIR/*.stdout.log"
echo "  Figures:      results/scenario3/"
