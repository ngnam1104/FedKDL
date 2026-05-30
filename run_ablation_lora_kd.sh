#!/usr/bin/env bash
# Script chạy ablation study: Có KD vs Không KD với LoRA HFL (EMD=0)

OUT_DIR="results/logs_kdl"
STDOUT_DIR="results/train_logs/kdl"
mkdir -p "$OUT_DIR" "$STDOUT_DIR"

echo "=========================================================="
echo "=== RUNNING EXPERIMENT 1: KHONG KD (fedkdl_nokd) ==="
echo "=========================================================="
python main_trainer_od.py \
  --topo environments/2d/topo/N_20/topo_N20_seed1104.pkl \
  --data environments/2d/data/URPC/N_20/data_N20_URPC_a1p0_seed1104.pkl \
  --baseline fedkdl_nokd --rounds 60 2>&1 | tee "$STDOUT_DIR/ablation_nokd.log"

echo ""
echo "=========================================================="
echo "=== RUNNING EXPERIMENT 2: CO KD (fedkdl) ==="
echo "=========================================================="
python main_trainer_od.py \
  --topo environments/2d/topo/N_20/topo_N20_seed1104.pkl \
  --data environments/2d/data/URPC/N_20/data_N20_URPC_a1p0_seed1104.pkl \
  --baseline fedkdl --rounds 60 2>&1 | tee "$STDOUT_DIR/ablation_kd.log"

echo "Done!"
