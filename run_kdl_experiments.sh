#!/usr/bin/env bash
# Scenario 2/3 (2D OD): grid train + plot.
# Tái cấu trúc Phase 5: Cố định N=50, M=10, Non-IID.
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
ROUNDS=50
SEED=1104
DS="URPC"
M_RELAYS_2D=5
N_AUVS=30
ALPHA="1.0"
# =========================================================

echo "[KDL] Generating topologies and data partitions..."
"$PYTHON" utils/generate_all_envs.py --n "$N_AUVS" --dataset "$DS" --m-relays "$M_RELAYS_2D" --alphas "$ALPHA"

# =========================================================
# BƯỚC 1: Pre-train Teacher LoRA
# =========================================================
echo "[KDL] Bắt đầu huấn luyện Teacher LoRA (YOLO12l)..."
if [[ -f "yolo12l_lora_pretrained.pt" ]]; then
  echo "[KDL] yolo12l_lora_pretrained.pt đã tồn tại, BỎ QUA bước này."
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
echo "[KDL] Đang Warm-up Student với LoRA..."
if [[ -f "yolo12n_warmup.pt" ]]; then
  echo "[KDL] yolo12n_warmup.pt đã tồn tại, BỎ QUA bước Warm-up."
else
  set +e
  "$PYTHON" scripts/fedkdl/train_student_warmup.py --mode warmup --epochs-warmup 3
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
# Định nghĩa các mảng Task (Sắp xếp theo RQ, fedkdl chạy đầu tiên)
# =========================================================
MAIN_BASELINES=("fedkdl")

# RQ1: Kết nối và ổn định (Topology Flat)
RQ1_BASELINES=("fedavg" "fedprox")

# RQ2 & RQ3: Nén truyền thông, Non-IID và Phân cấp (Topology HFL)
RQ2_RQ3_BASELINES=("fedavg_hfl" "fedprox_hfl" "flora" "scaffold" "fedkdl_nocoop" "topk_grad")

# RQ4: Gateway KD (Topology HFL/Flat)
RQ4_BASELINES=("logit_kd" "fedkdl_nokd" "centralized")

total_tasks=$(( ${#MAIN_BASELINES[@]} + ${#RQ1_BASELINES[@]} + ${#RQ2_RQ3_BASELINES[@]} + ${#RQ4_BASELINES[@]} ))
current_task=0

run_baseline() {
  local baseline=$1
  current_task=$((current_task + 1))
  
  local topo="${ENVS_DIR}/2d/topo/N_${N_AUVS}/topo_N${N_AUVS}_seed${SEED}.pkl"
  local alpha_str="${ALPHA//./p}"
  local data="${ENVS_DIR}/2d/data/${DS}/N_${N_AUVS}/data_N${N_AUVS}_${DS}_a${alpha_str}_seed${SEED}.pkl"

  if [[ ! -f "$topo" || ! -f "$data" ]]; then
    echo "[Warning] Missing env: N=$N_AUVS DS=$DS alpha=$ALPHA seed=$SEED"
    return
  fi

  local log_json="${OUT_DIR}/log_N${N_AUVS}_${DS}_a${alpha_str}_${baseline}_seed${SEED}.json"
  if [[ -s "$log_json" ]] && [[ $(stat -c%s "$log_json" 2>/dev/null || echo 0) -gt 1024 ]]; then
    echo "[$current_task/$total_tasks] SKIP (complete log exists): $log_json"
    return 0
  elif [[ -f "$log_json" ]]; then
    echo "[$current_task/$total_tasks] OVERWRITE (incomplete log): $log_json"
    rm -f "$log_json"
  fi

  echo "[$current_task/$total_tasks] OD | N=$N_AUVS | alpha=$ALPHA | baseline=$baseline"
  set +eo pipefail

  local TS=$(date +"%Y%m%d_%H%M%S")
  local log_file="$STDOUT_DIR/raw_bash_output_${baseline}_${N_AUVS}_${alpha_str}_${TS}.log"

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
echo "=== GROUP 1: Reference Baseline ==="
for b in "${MAIN_BASELINES[@]}"; do
  run_baseline "$b"
done

echo ""
echo "=== GROUP 2: RQ1 (Flat Topology) ==="
for b in "${RQ1_BASELINES[@]}"; do
  run_baseline "$b"
done

echo ""
echo "=== GROUP 3: RQ2 & RQ3 (HFL Topology) ==="
for b in "${RQ2_RQ3_BASELINES[@]}"; do
  run_baseline "$b"
done

echo ""
echo "=== GROUP 4: RQ4 (Gateway KD & Centralized) ==="
for b in "${RQ4_BASELINES[@]}"; do
  run_baseline "$b"
done

echo ""
echo "[KDL] Training done. All experiments completed for Phase 5 setup."
