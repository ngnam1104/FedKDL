#!/usr/bin/env bash
# quick_start.sh — Setup server + datasets + environments + chạy grid HFL & KDL.
#
# Ví dụ:
#   chmod +x quick_start.sh run_hfl_experiments.sh run_kdl_experiments.sh
#   ./quick_start.sh                    # full pipeline
#   ./quick_start.sh --train-only       # đã setup, chỉ train
#   ./quick_start.sh --hfl-only         # chỉ 1D
#   ./quick_start.sh --setup-only       # venv + pip, không train
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# --- Defaults ---
DO_SETUP=1
DO_DOWNLOAD=1
DO_ENV_GEN=1
DO_HFL=1
DO_KDL=1
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR=".venv"
MASTER_LOG="results/quick_start/master.log"

# --- Parse args ---
usage() {
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
  echo ""
  echo "Options:"
  echo "  --setup-only       Chỉ tạo venv + cài requirements (+ kagglehub)"
  echo "  --data-only        Chỉ tải datasets"
  echo "  --env-only         Chỉ sinh environments/*.pkl"
  echo "  --train-only       Bỏ setup/download/env-gen; chỉ chạy experiment scripts"
  echo "  --hfl-only         Chỉ run_hfl_experiments.sh"
  echo "  --kdl-only         Chỉ run_kdl_experiments.sh"
  echo "  --skip-setup       Bỏ bước tạo venv + cài requirements"
  echo "  --skip-download    Bỏ bước tải dataset"
  echo "  --skip-env-gen     Bỏ generate_all_envs.py"
  echo "  --skip-train       Dừng sau setup/data/env (không train)"
  echo "  --python PATH      Python dùng tạo venv (mặc định: python3)"
  echo "  -h, --help         Hiện trợ giúp"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --setup-only)   DO_DOWNLOAD=0; DO_ENV_GEN=0; DO_HFL=0; DO_KDL=0 ;;
    --data-only)    DO_SETUP=0; DO_ENV_GEN=0; DO_HFL=0; DO_KDL=0 ;;
    --env-only)     DO_SETUP=0; DO_DOWNLOAD=0; DO_HFL=0; DO_KDL=0 ;;
    --train-only)   DO_SETUP=0; DO_DOWNLOAD=0; DO_ENV_GEN=0 ;;
    --hfl-only)     DO_KDL=0 ;;
    --kdl-only)     DO_HFL=0 ;;
    --skip-setup)   DO_SETUP=0 ;;
    --skip-download) DO_DOWNLOAD=0 ;;
    --skip-env-gen)  DO_ENV_GEN=0 ;;
    --skip-train)    DO_HFL=0; DO_KDL=0 ;;
    --python)        PYTHON_BIN="$2"; shift ;;
    -h|--help)       usage; exit 0 ;;
    *) echo "[Error] Unknown option: $1"; usage; exit 1 ;;
  esac
  shift
done

mkdir -p results/quick_start
MASTER_LOG="results/quick_start/master.log"

log() {
  local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  echo "$msg" | tee -a "$MASTER_LOG"
}

run_step() {
  local name="$1"
  shift
  log "========== $name =========="
  if "$@"; then
    log "OK: $name"
    return 0
  else
    local rc=$?
    log "FAILED ($rc): $name"
    return "$rc"
  fi
}

resolve_python() {
  if [[ -x "$VENV_DIR/bin/python" ]]; then
    PYTHON="$VENV_DIR/bin/python"
  elif [[ -f "$VENV_DIR/Scripts/python.exe" ]]; then
    PYTHON="$VENV_DIR/Scripts/python.exe"
  else
    PYTHON="${PYTHON:-$PYTHON_BIN}"
  fi
  log "Python: $($PYTHON -c 'import sys; print(sys.executable)')"
}

# --- Step 0: Preflight ---
preflight() {
  log "Repo root: $ROOT"
  command -v git >/dev/null 2>&1 || { log "[Warn] git không có — tải SMD có thể lỗi"; }
  if [[ ! -f requirements.txt ]]; then
    log "[Error] requirements.txt không tồn tại. Chạy script từ thư mục FedKDL."
    exit 1
  fi
  chmod +x quick_start.sh run_hfl_experiments.sh run_kdl_experiments.sh 2>/dev/null || true
}

# --- Step 1: Venv + pip ---
setup_venv() {
  if [[ ! -d "$VENV_DIR" ]]; then
    log "Tạo venv: $PYTHON_BIN -m venv $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR" || return 1
  else
    log "Venv đã tồn tại: $VENV_DIR"
  fi
  resolve_python
  "$PYTHON" -m pip install --upgrade pip wheel || return 1
  "$PYTHON" -m pip install -r requirements.txt || return 1
  # Hỗ trợ download URPC qua Kaggle (tùy chọn)
  "$PYTHON" -m pip install kagglehub 2>/dev/null || log "[Warn] kagglehub cài thất bại — URPC có thể cần tải tay"
  "$PYTHON" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
}

# --- Step 2: Datasets ---
load_kaggle_env() {
  if [[ -f "$ROOT/.env" ]]; then
    log "Nạp .env (KAGGLE_API_TOKEN, …)"
    set -a
    # shellcheck disable=SC1091
    source "$ROOT/.env"
    set +a
  fi
  if [[ -n "${KAGGLE_API_TOKEN:-}" ]]; then
    log "KAGGLE_API_TOKEN: đã set (URPC/SMAP qua Kaggle)"
  else
    log "[Warn] Chưa có KAGGLE_API_TOKEN — URPC có thể fail. Xem README.md & .env.example"
  fi
}

# --- Step 2: Datasets ---
download_data() {
  resolve_python
  load_kaggle_env
  log "Tải datasets (SMD, SMAP/MSL, URPC) — có thể lâu..."
  set +e
  "$PYTHON" utils/download_datasets.py --all
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    log "[Warn] download_datasets.py exit $rc — kiểm tra từng dataset bên dưới"
  fi
  check_datasets
}

check_datasets() {
  local ok=1
  for d in datasets/SMD datasets/SMAP_MSL datasets/URPC2020; do
    if [[ -d "$d" ]] && [[ -n "$(ls -A "$d" 2>/dev/null || true)" ]]; then
      log "  [OK] $d"
    else
      log "  [MISSING] $d"
      ok=0
    fi
  done
  if [[ ! -f datasets/URPC2020.yaml ]]; then
    log "  [MISSING] datasets/URPC2020.yaml"
    ok=0
  else
    log "  [OK] datasets/URPC2020.yaml"
  fi
  if [[ $ok -eq 0 ]]; then
    log "[Warn] Một số dataset thiếu — HFL/KDL có thể skip run tương ứng"
  fi
}

# --- Step 3: Environments ---
generate_envs() {
  resolve_python
  log "Sinh environments/ (topo + data partition)..."
  "$PYTHON" utils/generate_all_envs.py || return 1
  count_env_files
}

count_env_files() {
  local topo n_data
  topo=$(find environments/topo -name '*.pkl' 2>/dev/null | wc -l | tr -d ' ')
  n_data=$(find environments/data -name '*.pkl' 2>/dev/null | wc -l | tr -d ' ')
  log "  Topo pkls: $topo | Data pkls: $n_data (kỳ vọng topo≈12, data≈48)"
}

# --- Step 4: Train grids ---
run_experiments() {
  if [[ $DO_HFL -eq 1 ]]; then
    log "Khởi chạy HFL grid (CPU-friendly)..."
    ./run_hfl_experiments.sh 2>&1 | tee results/quick_start/run_hfl_$(date +"%Y%m%d_%H%M%S").log
  fi
  if [[ $DO_KDL -eq 1 ]]; then
    log "Khởi chạy KDL grid (ưu tiên GPU)..."
    ./run_kdl_experiments.sh 2>&1 | tee results/quick_start/run_kdl_$(date +"%Y%m%d_%H%M%S").log
  fi
}

# --- Main ---
main() {
  preflight
  log "FedKDL quick_start — setup=$DO_SETUP download=$DO_DOWNLOAD env=$DO_ENV_GEN hfl=$DO_HFL kdl=$DO_KDL"

  if [[ $DO_SETUP -eq 1 ]]; then
    run_step "1/4 Setup venv + requirements" setup_venv || exit 1
  fi
  resolve_python

  if [[ $DO_DOWNLOAD -eq 1 ]]; then
    run_step "2/4 Download datasets" download_data || log "[Warn] Download không hoàn hảo — xem log"
  fi

  if [[ $DO_ENV_GEN -eq 1 ]]; then
    run_step "3/4 Generate environments" generate_envs || exit 1
  fi

  if [[ $DO_HFL -eq 1 || $DO_KDL -eq 1 ]]; then
    log "Gợi ý: chạy trong tmux nếu SSH có thể đứt: tmux new -s fedkdl './quick_start.sh --train-only'"
    run_step "4/4 Run experiment grids" run_experiments || log "[Warn] Một phần train/plot có thể lỗi — xem results/quick_start/run_*.log"
  else
    log "Bỏ bước train (--skip-train hoặc --setup-only)."
  fi

  log "========== quick_start hoàn tất =========="
  log "Master log:     $ROOT/$MASTER_LOG"
  log "JSON HFL:       $ROOT/results/logs/"
  log "JSON KDL:       $ROOT/results/logs_kdl/"
  log "Stdout HFL:     $ROOT/results/train_logs/hfl/"
  log "Stdout KDL:     $ROOT/results/train_logs/kdl/"
  log "Plots HFL:      results/convergence|scalability|heterogeneity|real_benchmark/"
  log "Plots KDL:      results/scenario3/"
}

main "$@"
