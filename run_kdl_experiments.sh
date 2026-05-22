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

# Sinh dữ liệu môi trường riêng cho mạng nhỏ (N=10, 15, 20)
echo "[KDL] Generating topologies and data partitions for N=10, 15, 20..."
"$PYTHON" utils/generate_all_envs.py --n 10 --dataset URPC
"$PYTHON" utils/generate_all_envs.py --n 15 --dataset URPC
"$PYTHON" utils/generate_all_envs.py --n 20 --dataset URPC

# Cấu hình chung
ROUNDS=20
SEED=42
DS="URPC"

# Hàm chạy chung để tránh lặp code
# Usage: run_baseline N ALPHA BASELINE
total_tasks=22
current_task=0

run_baseline() {
  local n=$1
  local alpha=$2
  local baseline=$3

  current_task=$((current_task + 1))
  local topo="${ENVS_DIR}/2d/topo/N_${n}/topo_N${n}_seed${SEED}.pkl"
  local alpha_str="${alpha//./p}"
  local data="${ENVS_DIR}/2d/data/${DS}/N_${n}/data_N${n}_${DS}_a${alpha_str}_seed${SEED}.pkl"

  if [[ ! -f "$topo" || ! -f "$data" ]]; then
    echo "[Warning] Missing env: N=$n DS=$DS alpha=$alpha seed=$SEED"
    return
  fi

  local log_json="${OUT_DIR}/log_N${n}_${DS}_a${alpha_str}_${baseline}_seed${SEED}.json"
  if [[ -f "$log_json" ]]; then
    echo "[$current_task/$total_tasks] Overwriting existing JSON: $log_json"
  fi

  echo "[$current_task/$total_tasks] OD | N=$n | alpha=$alpha | baseline=$baseline"
  set +e
  "$PYTHON" main_trainer_od.py \
    --topo "$topo" --data "$data" \
    --baseline "$baseline" --rounds "$ROUNDS" \
    --out-dir "$OUT_DIR" --log-dir "$STDOUT_DIR" 2>&1 | tee -a "$STDOUT_DIR/raw_bash_output_${n}_${alpha_str}.log"
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "[Error] Run failed (exit $rc). Check ${STDOUT_DIR}/raw_bash_output_${n}_${alpha_str}.log"
  fi
}

echo ""
echo "=== GROUP A: Ablation & Comparison ==="
# N=10, Alpha=0.5, 10 baselines
ALL_BASELINES=(
  "fedkdl" "fedkdl_r4" "full_param_kd" "full_param_nokd" 
  "lora_head_kd_noint8" "head_kd_int8_nolora" "lora_head_int8_nokd"
  "fedavg" "fedprox" "centralized"
)
for b in "${ALL_BASELINES[@]}"; do
  run_baseline 10 0.5 "$b"
done

echo ""
echo "=== GROUP B: Scalability ==="
# N=15, 20 (N=10 đã có ở Group A), Alpha=0.5, 3 baselines
MAIN_BASELINES=("fedkdl" "fedavg" "centralized")
for n in 15 20; do
  for b in "${MAIN_BASELINES[@]}"; do
    run_baseline "$n" 0.5 "$b"
  done
done

echo ""
echo "=== GROUP C: Heterogeneity ==="
# N=10, Alpha=10000.0, 3 baselines
for b in "${MAIN_BASELINES[@]}"; do
  run_baseline 10 10000.0 "$b"
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
