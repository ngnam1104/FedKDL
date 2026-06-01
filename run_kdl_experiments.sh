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
# Định nghĩa các mảng Task (Mô-đun hóa dễ mở rộng)
# =========================================================
MAIN_BASELINES=("fedkdl" "fedkdl_selective" "fedprox_kdl" "fedkd")
SOTA_LORA_BASELINES=("fedkdl_nolora") # Thêm các phương pháp LoRA mới vào đây sau này
SOTA_KD_BASELINES=("fedkdl_nokd" "fedkdl_proxy_ft")     # Thêm các phương pháp KD mới vào đây sau này

total_tasks=$(( ${#MAIN_BASELINES[@]} + ${#SOTA_LORA_BASELINES[@]} + ${#SOTA_KD_BASELINES[@]} ))
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
  local log_file="$STDOUT_DIR/raw_bash_output_${N_AUVS}_${alpha_str}_${TS}.log"

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
echo "=== GROUP 1: Main Baselines ==="
for b in "${MAIN_BASELINES[@]}"; do
  run_baseline "$b"
done

echo ""
echo "=== GROUP 2: SOTA LoRA Propagation & Ablation ==="
for b in "${SOTA_LORA_BASELINES[@]}"; do
  run_baseline "$b"
done

echo ""
echo "=== GROUP 3: SOTA Knowledge Distillation & Ablation ==="
for b in "${SOTA_KD_BASELINES[@]}"; do
  run_baseline "$b"
done

echo ""
echo "[KDL] Training done. All experiments completed for Phase 5 setup."
# Vẽ biểu đồ sẽ được thiết lập ở 1 module độc lập (do cấu trúc JSON có thể thay đổi sau đợt tái cấu trúc)
