#!/usr/bin/env bash
# Scenario 2/3 (2D OD): grid train + plot.
# Mỗi run: main_trainer_od.py tự lưu JSON (results/logs_kdl) + stdout log (results/train_logs/kdl).
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

N_LIST=(50 100 150 200)
DATASETS=(URPC)
ALPHAS=(0.5 10000.0)
SEEDS=(42 123 2024)

# Danh sách baselines
# Các cấu hình Ablation (Ablation Study):
# fedkdl: Bản gốc (LoRA + Head, INT8, có KD)
# full_param_kd: Truyền toàn bộ mô hình (INT8, có KD)
# full_param_nokd: Truyền toàn bộ mô hình (INT8, không KD)
# lora_head_kd_noint8: LoRA + Head (Float32, có KD)
# head_kd_int8_nolora: Chỉ Head (INT8, có KD)
# lora_head_int8_nokd: LoRA + Head (INT8, không KD)
# Các FL Baselines: fedavg, fedprox, centralized
BASELINES=(
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

ROUNDS=20
OUT_DIR="results/logs_kdl"
ENVS_DIR="environments"
STDOUT_DIR="results/train_logs/kdl"

mkdir -p "$OUT_DIR" "$STDOUT_DIR"
export PYTHONIOENCODING=utf-8

total=$(( ${#N_LIST[@]} * ${#DATASETS[@]} * ${#ALPHAS[@]} * ${#SEEDS[@]} * ${#BASELINES[@]} ))
count=0

for n in "${N_LIST[@]}"; do
  for ds in "${DATASETS[@]}"; do
    for alpha in "${ALPHAS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        topo="${ENVS_DIR}/topo/N_${n}/topo_N${n}_seed${seed}.pkl"
        alpha_str="${alpha//./p}"
        data="${ENVS_DIR}/data/${ds}/N_${n}/data_N${n}_${ds}_a${alpha_str}_seed${seed}.pkl"

        if [[ ! -f "$topo" || ! -f "$data" ]]; then
          echo "[Warning] Missing env: N=$n DS=$ds alpha=$alpha seed=$seed — run utils/generate_all_envs.py"
          count=$(( count + ${#BASELINES[@]} ))
          continue
        fi

        for baseline in "${BASELINES[@]}"; do
          count=$((count + 1))
          log_json="${OUT_DIR}/log_N${n}_${ds}_a${alpha_str}_${baseline}_seed${seed}.json"
          if [[ -f "$log_json" ]]; then
            echo "[$count/$total] Overwriting existing JSON: $log_json"
          fi

          echo "[$count/$total] OD | N=$n | DS=$ds | alpha=$alpha | seed=$seed | baseline=$baseline"
          set +e
          "$PYTHON" main_trainer_od.py \
            --topo "$topo" --data "$data" \
            --baseline "$baseline" --rounds "$ROUNDS" \
            --out-dir "$OUT_DIR" --log-dir "$STDOUT_DIR"
          rc=$?
          set -e
          if [[ $rc -ne 0 ]]; then
            echo "[Error] Run failed (exit $rc). Check ${STDOUT_DIR}/log_N${n}_*.stdout.log"
          fi
        done
      done
    done
  done
done

echo ""
echo "[KDL] Training done. Generating plots..."
"$PYTHON" scripts/fedkdl/plot_od_comparison.py
"$PYTHON" scripts/fedkdl/plot_od_scalability.py
"$PYTHON" scripts/fedkdl/plot_heterogeneity.py
"$PYTHON" scripts/fedkdl/eval_baselines.py --results-dir "$OUT_DIR"
echo "[KDL] All done."
echo "  JSON (plot):  $OUT_DIR/*.json"
echo "  Stdout logs:  $STDOUT_DIR/*.stdout.log"
echo "  Figures:      results/scenario3/"
