#!/usr/bin/env bash
# S2: Mobility sweep
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

mkdir -p "$OUT_DIR" "$LOG_DIR"
alpha_str="${ALPHA//./p}"
topo="${ENVS_DIR}/2d/topo/N_${N}/topo_N${N}_seed${SEED}.pkl"
data="${ENVS_DIR}/2d/data/${DS}/N_${N}/data_N${N}_${DS}_a${alpha_str}_seed${SEED}.pkl"

"$PYTHON" utils/generate_all_envs.py --n "$N" --dataset "$DS" --m-relays "$M" --alphas "$ALPHA" --seeds "$SEED"

# RUN_MOBILITY_SWEEP is forced to 1
RUN_MOBILITY_SWEEP=1
if [[ "${RUN_MOBILITY_SWEEP:-1}" == "1" ]]; then
  MOBILITY_ROUNDS="${MOBILITY_ROUNDS:-$ROUNDS}"
  MOBILITY_BASELINE="${MOBILITY_BASELINE:-fedkdl}"
  MOBILITY_OUT_DIR="${MOBILITY_OUT_DIR:-results/mobility_velocity/logs/N_${N}/M_${M}}"
  MOBILITY_LOG_DIR="${MOBILITY_LOG_DIR:-results/mobility_velocity/train_logs/N_${N}/M_${M}}"
  SPEED_LABELS=("normal" "fast")
  SPEEDS=("0.8333" "1.6667")
  MAX_SPEEDS=("2.0" "4.0")
  if [[ "${INCLUDE_STRESS:-0}" == "1" ]]; then
    SPEED_LABELS+=("stress")
    SPEEDS+=("3.0")
    MAX_SPEEDS+=("6.0")
  fi

  mkdir -p "$MOBILITY_OUT_DIR" "$MOBILITY_LOG_DIR"
  echo "[S2-mobility] baseline=${MOBILITY_BASELINE} rounds=${MOBILITY_ROUNDS} move_energy=off"
  for i in "${!SPEEDS[@]}"; do
    speed="${SPEEDS[$i]}"
    max_speed="${MAX_SPEEDS[$i]}"
    label="${SPEED_LABELS[$i]}"
    ts="$(date +"%Y%m%d_%H%M%S")"
    log_file="${MOBILITY_LOG_DIR}/raw_S2_MOBILITY_${label}_v${speed}_${MOBILITY_BASELINE}_N${N}_M${M}_a${alpha_str}_seed${SEED}_${ts}.log"
    echo "[S2-mobility] label=${label} speed=${speed} max_speed=${max_speed} baseline=${MOBILITY_BASELINE} log=${log_file}"
    FEDKDL_MOBILITY_ENABLED=1 \
    FEDKDL_MOVE_ENERGY_ENABLED=0 \
    FEDKDL_MOBILITY_DT="${FEDKDL_MOBILITY_DT:-30.0}" \
    FEDKDL_GM_MEAN_SPEED="$speed" \
    FEDKDL_GM_MAX_SPEED="$max_speed" \
    WANDB_MODE=disabled \
    PYTHONUNBUFFERED=1 \
    "$PYTHON" -u main_trainer_od.py \
      --topo "$topo" \
      --data "$data" \
      --baseline "$MOBILITY_BASELINE" \
      --rounds "$MOBILITY_ROUNDS" \
      --out-dir "$MOBILITY_OUT_DIR/${label}/${MOBILITY_BASELINE}" \
      --log-dir "$MOBILITY_LOG_DIR" \
      2>&1 | tee -a "$log_file"
  done
fi
