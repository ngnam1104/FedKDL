#!/usr/bin/env bash
# Job 3/3: KD ablation with classification-logit and LoRA-projection matching.
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
GPU="${GPU:-2}"
DS="${DS:-URPC}"
N="${N:-30}"
M="${M:-8}"
ALPHA="${ALPHA:-1.0}"
SEED="${SEED:-1109}"
ROUNDS="${ROUNDS:-40}"
ENVS_DIR="${ENVS_DIR:-environments}"
OUT_DIR="${OUT_DIR:-results/logs/N_${N}/M_${M}}"
LOG_DIR="${LOG_DIR:-results/train_logs/N_${N}/M_${M}}"
BASELINE="logit_proj_kd"

alpha_tag="${ALPHA//./p}"
topo="${ENVS_DIR}/2d/topo/N_${N}/topo_N${N}_seed${SEED}.pkl"
data="${ENVS_DIR}/2d/data/${DS}/N_${N}/data_N${N}_${DS}_a${alpha_tag}_seed${SEED}.pkl"

for required in "$topo" "$data"; do
  if [[ ! -f "$required" ]]; then
    echo "[ERROR] Missing input: $required" >&2
    echo "Prepare it first with:" >&2
    echo "  $PYTHON utils/generate_all_envs.py --dataset $DS --n $N --m-relays $M --alphas $ALPHA --seeds $SEED" >&2
    exit 2
  fi
done

mkdir -p "$OUT_DIR" "$LOG_DIR"
echo "[START] baseline=$BASELINE gpu=$GPU rounds=$ROUNDS N=$N M=$M alpha=$ALPHA seed=$SEED"

CUDA_VISIBLE_DEVICES="$GPU" \
WANDB_MODE="${WANDB_MODE:-disabled}" \
PYTHONUNBUFFERED=1 \
"$PYTHON" -u main_trainer_od.py \
  --topo "$topo" \
  --data "$data" \
  --baseline "$BASELINE" \
  --rounds "$ROUNDS" \
  --out-dir "$OUT_DIR" \
  --log-dir "$LOG_DIR"

