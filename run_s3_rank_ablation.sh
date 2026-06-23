#!/usr/bin/env bash
# S3: Rank Ablation
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

# RUN_RANK_ABLATION is forced to 1
RUN_RANK_ABLATION=1
if [[ "${RUN_RANK_ABLATION:-1}" == "1" ]]; then
  RANK_ROUNDS="${RANK_ROUNDS:-$ROUNDS}"
  RANK_BASELINE="${RANK_BASELINE:-fedkdl}"
  RANK_OUT_DIR="${RANK_OUT_DIR:-results/lora_rank_ablation/logs/N_${N}/M_${M}}"
  RANK_LOG_DIR="${RANK_LOG_DIR:-results/lora_rank_ablation/train_logs/N_${N}/M_${M}}"
  VARIANT_LABELS=("r4_4" "r2_4")
  BACKBONE_RANKS=("4" "2")
  NECK_RANKS=("4" "4")

  mkdir -p "$RANK_OUT_DIR" "$RANK_LOG_DIR"
  echo "[S3-rank] baseline=${RANK_BASELINE} rounds=${RANK_ROUNDS}"
  for i in "${!VARIANT_LABELS[@]}"; do
    label="${VARIANT_LABELS[$i]}"
    b_rank="${BACKBONE_RANKS[$i]}"
    n_rank="${NECK_RANKS[$i]}"
    ts="$(date +"%Y%m%d_%H%M%S")"
    log_file="${RANK_LOG_DIR}/raw_S3_RANK_${label}_${RANK_BASELINE}_N${N}_M${M}_a${alpha_str}_seed${SEED}_${ts}.log"
    echo "[S3-rank] variant=${label} backbone=${b_rank} neck=${n_rank} baseline=${RANK_BASELINE} log=${log_file}"
    FEDKDL_LORA_RANK="$n_rank" \
    FEDKDL_LORA_BACKBONE_RANK="$b_rank" \
    FEDKDL_LORA_NECK_RANK="$n_rank" \
    FEDKDL_WARMUP_SUFFIX="$label" \
    WANDB_MODE=disabled \
    PYTHONUNBUFFERED=1 \
    "$PYTHON" -u main_trainer_od.py \
      --topo "$topo" \
      --data "$data" \
      --baseline "$RANK_BASELINE" \
      --rounds "$RANK_ROUNDS" \
      --out-dir "$RANK_OUT_DIR/${label}/${RANK_BASELINE}" \
      --log-dir "$RANK_LOG_DIR" \
      2>&1 | tee -a "$log_file"
  done
fi
