#!/usr/bin/env bash
# Scenario 2/3 (2D OD): grid train + plot.
# Phase 5 defaults: N=30, M=5, Non-IID alpha=1.0, seed=1104
# Baseline groups follow experiment_design.md (4 RQs).
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
# Cấu hình thử nghiệm cố định (Phase 5)
# =========================================================
ROUNDS="${ROUNDS:-60}"
DS="URPC"
M_RELAYS_2D=5
N_AUVS=30
read -r -a ALPHA_VALUES <<< "${ALPHAS:-${ALPHA:-1.0}}"
read -r -a SEED_VALUES <<< "${SEEDS:-${SEED:-1104}}"
# =========================================================

echo "[KDL] Generating topologies and data partitions..."
"$PYTHON" utils/generate_all_envs.py \
  --n "$N_AUVS" --dataset "$DS" --m-relays "$M_RELAYS_2D" \
  --alphas "${ALPHA_VALUES[@]}" --seeds "${SEED_VALUES[@]}"

# =========================================================
# BƯỚC 1: Pre-train Teacher LoRA
# =========================================================
echo "[KDL] Bat dau huan luyen Teacher LoRA (YOLO12l)..."
if [[ -f "yolo12l_lora_pretrained.pt" ]]; then
  echo "[KDL] yolo12l_lora_pretrained.pt da ton tai, BO QUA buoc nay."
else
  set +e
  "$PYTHON" scripts/fedkdl/train_teacher_lora.py
  TEACHER_RC=$?
  set -e
  if [[ $TEACHER_RC -ne 0 ]]; then
    echo "[Warning] train_teacher_lora.py exit=$TEACHER_RC."
    if [[ ! -f "yolo12l_lora_pretrained.pt" ]]; then
      exit 1
    fi
  fi
fi

# =========================================================
# BƯỚC 2: Warmup Student LoRA
# =========================================================
echo "[KDL] Dang Warm-up Student voi LoRA..."
if [[ -f "yolo12n_warmup.pt" ]]; then
  echo "[KDL] yolo12n_warmup.pt da ton tai, BO QUA buoc Warm-up."
else
  set +e
  "$PYTHON" scripts/fedkdl/train_student_warmup.py --mode warmup --epochs-warmup 5
  STUDENT_RC=$?
  set -e
  if [[ $STUDENT_RC -ne 0 ]]; then
    echo "[Warning] train_student_warmup.py exit=$STUDENT_RC."
    if [[ ! -f "yolo12n_warmup.pt" ]]; then
      exit 1
    fi
  fi
fi

# =========================================================
# BƯỚC 3: Bake checkpoint warmup cho kiến trúc YOLO gốc
# Không train thêm; mọi baseline dùng cùng một trạng thái warmup.
# =========================================================
echo "[KDL] Dang tao ban warmup baked cho cac baseline khong LoRA..."
if [[ -f "yolo12n_head_warmup.pt" ]]; then
  echo "[KDL] yolo12n_head_warmup.pt da ton tai, BO QUA."
else
  set +e
  "$PYTHON" scripts/fedkdl/train_student_warmup.py --mode warmup_fullparam --epochs-warmup 5
  FP_RC=$?
  set -e
  if [[ $FP_RC -ne 0 ]]; then
    echo "[Warning] bake warmup exit=$FP_RC."
    if [[ ! -f "yolo12n_head_warmup.pt" ]]; then
      exit 1
    fi
  fi
fi

# =========================================================
# Dinh nghia cac nhom baseline theo tung RQ
# experiment_design.md:
#   RQ1 - Ket noi va on dinh:  fedavg (flat), fedprox (flat) vs. fedkdl (HFL)
#   RQ2 - Nen truyen thong:    fedavg_hfl, topk_grad, flora, fedkdl (all HFL)
#   RQ3 - Non-IID va Relay:    fedavg_hfl, scaffold, flora,
#                              fedkdl_nocoop (HFL), fedkdl (HFL)
#   RQ4 - Gateway KD ablation: fedkdl_nokd (HFL), logit_kd (HFL), fedkdl (HFL),
#                              centralized (flat)
#
# fedkdl xuat hien o moi RQ -> chay truoc, dung chung log (idempotent skip).
#
# Topology:
#   Flat (hfl=False): fedavg, fedprox, fedkd, centralized
#   HFL  (hfl=True):  fedavg_hfl, fedprox_hfl, flora, scaffold, topk_grad,
#                     fedkdl, fedkdl_nocoop, logit_kd, fedkdl_nokd, ...
#
# NGUYEN TAC: RQ1 = Flat topology; RQ2/3/4 = HFL topology.
# Do do: RQ1 dung "fedavg" (flat), RQ2/3 dung "fedavg_hfl" (HFL).
# =========================================================

# RQ1: So sanh ket noi/on dinh - TAT CA FLAT (flat vs. fedkdl HFL)
RQ1_BASELINES=("fedavg" "fedprox")

# RQ2: Nen truyen thong - TAT CA HFL
# fedavg_hfl = HFL FedAvg (reference HFL, khong nen)
# topk_grad  = HFL + Top-K sparse gradient
# flora      = HFL + LoRA Float32 (no INT8)
# fedkdl     = HFL + LoRA INT8 + KD (da chay)
RQ2_BASELINES=("fedavg_hfl" "topk_grad" "flora")

# RQ3: Non-IID va Relay Cooperation - TAT CA HFL
# fedavg_hfl  = HFL FedAvg (reference, khong xu ly non-IID)
# scaffold    = HFL + Control Variates (xu ly drift)
# flora       = HFL + LoRA (da chay RQ2, skip neu co log)
# fedkdl_nocoop = HFL + LoRA INT8 + KD, khong relay coop
# fedkdl      = HFL + LoRA INT8 + KD + relay coop (da chay)
RQ3_BASELINES=("fedavg_hfl" "scaffold" "flora" "fedkdl_nocoop" "fedkdl_selective")

# Ablation extras (neu con thoi gian)
ABLATION_BASELINES=("fedprox_kdl" "fedkdl_nolora" "fedkdl_proxy_ft" "fedkd" "fedprox_hfl" "naive_lora")

# RQ4: Gateway KD Ablation - TAT CA HFL
# fedkdl_nokd = HFL + LoRA INT8, khong KD
# fedkdl_proxy_ft = HFL + LoRA INT8, thay KD bang Gateway Finetune
# logit_kd    = HFL + LoRA INT8 + Logit KD (KL-divergence)
# fedkdl      = HFL + LoRA INT8 + Projection KD (da chay)
RQ4_BASELINES=("fedkdl_proxy_ft" "fedkdl_nokd" "logit_kd")

# Reference and component ablations.
REFERENCE_BASELINES=("centralized" "fedkd" "fedprox_kdl" "fedkdl_nolora" "fedprox_hfl")

# Unique default suite: 16 baselines. fedkdl_proxy_ft is optional because it is
# an auxiliary supervised proxy experiment, not a required experiment_gaps RQ.
RUN_BASELINES=(
  "fedkdl"
  "fedavg"
  "fedprox"
  "fedavg_hfl"
  "topk_grad"
  "flora"
  "naive_lora"
  "scaffold"
  "fedkdl_nocoop"
  "fedkdl_selective"
  "fedkdl_proxy_ft"
  "fedkdl_nokd"
  "logit_kd"
  "centralized"
  "fedprox_kdl"
  "fedkdl_nolora"
  "fedkd"
  "fedprox_hfl"
)

total_tasks=$((${#RUN_BASELINES[@]} * ${#ALPHA_VALUES[@]} * ${#SEED_VALUES[@]}))
current_task=0

run_baseline() {
  local baseline=$1
  local alpha=$2
  local seed=$3
  current_task=$((current_task + 1))

  local topo="${ENVS_DIR}/2d/topo/N_${N_AUVS}/topo_N${N_AUVS}_seed${seed}.pkl"
  local alpha_str="${alpha//./p}"
  local data="${ENVS_DIR}/2d/data/${DS}/N_${N_AUVS}/data_N${N_AUVS}_${DS}_a${alpha_str}_seed${seed}.pkl"

  if [[ ! -f "$topo" || ! -f "$data" ]]; then
    echo "[Warning] Missing env: N=$N_AUVS DS=$DS alpha=$alpha seed=$seed"
    return
  fi

  local log_json="${OUT_DIR}/log_N${N_AUVS}_${DS}_a${alpha_str}_${baseline}_seed${seed}.json"
  if [[ -s "$log_json" ]] && [[ $(stat -c%s "$log_json" 2>/dev/null || echo 0) -gt 1024 ]]; then
    echo "[$current_task/$total_tasks] SKIP (complete log exists): $log_json"
    return 0
  elif [[ -f "$log_json" ]]; then
    echo "[$current_task/$total_tasks] OVERWRITE (incomplete log): $log_json"
    rm -f "$log_json"
  fi

  echo "[$current_task/$total_tasks] OD | N=$N_AUVS | alpha=$alpha | seed=$seed | baseline=$baseline"
  set +eo pipefail

  local TS=$(date +"%Y%m%d_%H%M%S")
  local log_file="$STDOUT_DIR/raw_bash_output_${baseline}_${N_AUVS}_${alpha_str}_seed${seed}_${TS}.log"

  "$PYTHON" main_trainer_od.py \
    --topo "$topo" --data "$data" \
    --baseline "$baseline" --rounds "$ROUNDS" \
    --out-dir "$OUT_DIR" --log-dir "$STDOUT_DIR" \
    2>&1 | tee -a "$log_file"

  local rc=${PIPESTATUS[0]}
  set -eo pipefail
  if [[ $rc -ne 0 ]]; then
    echo "[Error] Run failed (exit $rc). Check $log_file"
    rm -f "$log_json"
  fi
}

echo ""
echo "=== Experiment design ==="
echo "RQ1 flat:         ${RQ1_BASELINES[*]}"
echo "RQ2 hierarchical: ${RQ2_BASELINES[*]} fedkdl"
echo "RQ3 hierarchical: ${RQ3_BASELINES[*]} fedkdl"
echo "RQ4 hierarchical: ${RQ4_BASELINES[*]} fedkdl"
echo "References:       ${REFERENCE_BASELINES[*]}"
echo ""
echo "=== Running ${total_tasks} experiment jobs ==="
for b in "${RUN_BASELINES[@]}"; do
  for alpha in "${ALPHA_VALUES[@]}"; do
    for seed in "${SEED_VALUES[@]}"; do
      run_baseline "$b" "$alpha" "$seed"
    done
  done
done

echo ""
echo "[KDL] All experiments completed."
