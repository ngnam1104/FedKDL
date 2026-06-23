#!/usr/bin/env bash
# S1: Remaining Baselines (fedkdl_selective, fedkdl_32bit)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
DS="${DS:-URPC}"
N="${N:-30}"
M="${M:-8}"
ALPHA="${ALPHA:-1.0}"
SEED="${SEED:-1109}"
ROUNDS="${ROUNDS:-40}"
ENVS_DIR="${ENVS_DIR:-environments}"
OUT_DIR="${OUT_DIR:-results/logs/N_${N}/M_${M}}"
LOG_DIR="${LOG_DIR:-results/train_logs/N_${N}/M_${M}}"

# 2 runs: fedkdl_selective, fedkdl_32bit
DEFAULT_BASELINES=("fedkdl_selective" "fedkdl_32bit")
if [[ -n "${BASELINES_OVERRIDE:-}" ]]; then
  read -r -a BASELINES <<< "$BASELINES_OVERRIDE"
else
  BASELINES=("${DEFAULT_BASELINES[@]}")
fi

mkdir -p "$OUT_DIR" "$LOG_DIR"
alpha_str="${ALPHA//./p}"
topo="${ENVS_DIR}/2d/topo/N_${N}/topo_N${N}_seed${SEED}.pkl"
data="${ENVS_DIR}/2d/data/${DS}/N_${N}/data_N${N}_${DS}_a${alpha_str}_seed${SEED}.pkl"

"$PYTHON" utils/generate_all_envs.py --n "$N" --dataset "$DS" --m-relays "$M" --alphas "$ALPHA" --seeds "$SEED"

echo "[S1] baselines=${BASELINES[*]} rounds=${ROUNDS}"
for baseline in "${BASELINES[@]}"; do
  ts="$(date +"%Y%m%d_%H%M%S")"
  log_file="${LOG_DIR}/raw_S1_MIXED_${baseline}_N${N}_M${M}_a${alpha_str}_seed${SEED}_${ts}.log"
  echo "[S1] baseline=${baseline} rounds=${ROUNDS} log=${log_file}"
  WANDB_MODE=disabled PYTHONUNBUFFERED=1 "$PYTHON" -u main_trainer_od.py \
    --topo "$topo" \
    --data "$data" \
    --baseline "$baseline" \
    --rounds "$ROUNDS" \
    --out-dir "$OUT_DIR" \
    --log-dir "$LOG_DIR" \
    2>&1 | tee -a "$log_file"
done
