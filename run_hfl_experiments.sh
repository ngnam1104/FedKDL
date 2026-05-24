#!/usr/bin/env bash
# Scenario 1 (HFL / 1D): grid train + plot.
# Mỗi run: main_trainer.py tự lưu JSON (results/logs) + stdout log (results/train_logs/hfl).
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
DATASETS=(SMD SMAP MSL)
ALPHAS=(1.0 10000.0)
SEEDS=(42 123 2024)
BASELINES=(hfl_selective hfl_nearest hfl_nocoop fedprox fedavg centralized)

ROUNDS=30
RHO_S=0.05
OUT_DIR="results/logs"
ENVS_DIR="environments"
STDOUT_DIR="results/train_logs/hfl"

mkdir -p "$OUT_DIR" "$STDOUT_DIR"
export PYTHONIOENCODING=utf-8

total=$(( ${#N_LIST[@]} * ${#DATASETS[@]} * ${#ALPHAS[@]} * ${#SEEDS[@]} * ${#BASELINES[@]} ))
count=0

M_FOGS_1D=10
GEN_ENV_ARGS=()
if [[ -n "$M_FOGS_1D" ]]; then
  echo "[HFL] Overriding fog count for 1D topologies: M_FOGS_1D=$M_FOGS_1D"
  GEN_ENV_ARGS=(--m-fogs "$M_FOGS_1D")
fi

echo "[HFL] Generating topologies and data partitions for 1D..."
"$PYTHON" utils/generate_all_envs.py "${GEN_ENV_ARGS[@]}"

for n in "${N_LIST[@]}"; do
  for alpha in "${ALPHAS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      topo="${ENVS_DIR}/1d/topo/N_${n}/topo_N${n}_seed${seed}.pkl"
      
      for ds in "${DATASETS[@]}"; do
        alpha_str="${alpha//./p}"
        data="${ENVS_DIR}/1d/data/${ds}/N_${n}/data_N${n}_${ds}_a${alpha_str}_seed${seed}.pkl"

        if [[ ! -f "$topo" || ! -f "$data" ]]; then
          echo "[Warning] Missing env: N=$n DS=$ds alpha=$alpha seed=$seed — run utils/generate_all_envs.py"
          count=$(( count + ${#BASELINES[@]} ))
          continue
        fi

        for baseline in "${BASELINES[@]}"; do
          count=$((count + 1))
          rho_str="${RHO_S//./p}"
          log_json="${OUT_DIR}/log_N${n}_${ds}_a${alpha_str}_${baseline}_rho${rho_str}_seed${seed}.json"
          if [[ -s "$log_json" ]]; then
            echo "[$count/$total] SKIPPING: $log_json already exists and is not empty."
            continue
          fi

          echo "[$count/$total] N=$n | alpha=$alpha | seed=$seed | DS=$ds | baseline=$baseline"
          set +e
          "$PYTHON" main_trainer.py \
            --topo "$topo" --data "$data" \
            --baseline "$baseline" --rho-s "$RHO_S" \
            --rounds "$ROUNDS" --out-dir "$OUT_DIR" --log-dir "$STDOUT_DIR"
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
echo "[HFL] Training done. Generating plots..."
"$PYTHON" scripts/hfl/plot_convergence.py
"$PYTHON" scripts/hfl/plot_scalability.py
"$PYTHON" scripts/hfl/plot_heterogeneity.py
"$PYTHON" scripts/hfl/plot_real_benchmark.py
"$PYTHON" scripts/hfl/plot_tradeoff.py
echo "[HFL] All done."
echo "  JSON (plot):  $OUT_DIR/*.json"
echo "  Stdout logs:  $STDOUT_DIR/*.stdout.log"
echo "  Figures:      results/convergence|scalability|heterogeneity|real_benchmark/"
