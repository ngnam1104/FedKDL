"""
main_trainer_sota.py
Entry point cho SOTA Baseline (Jiang et al., 2025).
Chạy riêng biệt, KHÔNG ảnh hưởng main_trainer_od.py gốc.

Cách dùng:
    python main_trainer_sota.py \
        --topo environments/topo/N_20/topo_N20_seed42.pkl \
        --data environments/data/URPC/N_20/data_N20_URPC_a2.0_seed42.pkl \
        --rounds 50 \
        --out-dir results/logs_sota \
        --log-dir results/train_logs/sota
"""
import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

import torch
import ultralytics
ultralytics.settings.update({'datasets_dir': str(Path('datasets').absolute())})

from config.settings import network_cfg, fed_cfg
from tasks.detection_2d_lwkd_dcp.simulator import SimulatorSOTA
from utils.log_export import build_experiment_bundle
from utils.train_io import build_experiment_paths, run_trainer_with_artifacts


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


def parse_args():
    parser = argparse.ArgumentParser("SOTA Baseline (Jiang et al., 2025)")
    parser.add_argument("--topo",    type=str, required=True)
    parser.add_argument("--data",    type=str, required=True)
    parser.add_argument("--rounds",  type=int, default=None)
    parser.add_argument("--out-dir", type=str, default="results/logs_sota")
    parser.add_argument("--log-dir", type=str, default="results/train_logs/sota")
    return parser.parse_args()


def main():
    args = parse_args()
    topo_path = Path(args.topo)
    data_path = Path(args.data)

    if not topo_path.exists() or not data_path.exists():
        print("[Error] Environment files not found.")
        sys.exit(1)

    if args.rounds is not None:
        fed_cfg.GLOBAL_ROUNDS = {"1D": args.rounds, "2D": args.rounds}

    T_rounds = fed_cfg.GLOBAL_ROUNDS["2D"]
    device   = "cuda" if torch.cuda.is_available() else "cpu"

    # Parse metadata từ tên file (giống main_trainer_od.py)
    stem   = data_path.stem
    parts  = stem.split("_")
    N      = int(parts[1][1:])
    dataset    = parts[2]
    alpha_str  = parts[3][1:]
    seed       = int(parts[4][4:])

    network_cfg.N_SENSORS = N

    paths = build_experiment_paths(
        task="2D",
        out_dir=args.out_dir,
        log_dir=args.log_dir,
        N=N,
        dataset=dataset,
        alpha_str=alpha_str,
        baseline="sota_jiang2025",
        seed=seed,
    )

    def _train():
        sim = SimulatorSOTA(
            topo_path=str(topo_path),
            data_path=str(data_path),
            test_yaml=(
                "datasets/URPC2020.yaml"
                if "urpc" in dataset.lower()
                else "coco8.yaml"
            ),
            student_ckpt=(
                "yolo11n_pretrained.pt"
                if Path("yolo11n_pretrained.pt").exists()
                else "yolo11n.pt"
            ),
            teacher_ckpt=(
                "yolo12l_pretrained.pt"
                if Path("yolo12l_pretrained.pt").exists()
                else "yolo12l.pt"
            ),
            device=device,
        )

        print(f"\n[SOTA Trainer] baseline=sota_jiang2025 | rounds={T_rounds} | device={device}")
        print(f"[SOTA Trainer] topo={topo_path}")
        print(f"[SOTA Trainer] data={data_path}")

        history = sim.run(T_rounds=T_rounds)

        bundle = {
            "metadata": {
                "task":     "2D",
                "baseline": "sota_jiang2025",
                "paper":    "Jiang et al. (2025) FL for IoUT w/ Lightweight Distillation & DCP",
                "rounds":   T_rounds,
                "N":        N,
                "dataset":  dataset,
                "alpha":    alpha_str,
                "seed":     seed,
                "topo_path": str(topo_path),
                "data_path": str(data_path),
                "payload_note": "Full YOLO11n Float32 (~5.4 MB per sensor per round, no LoRA, no INT8)",
                "kd_note": "Local KD at Sensor (Teacher YOLO12l), DCP preprocessing, no Gateway KD",
            },
            "metrics": history,
        }

        import gc
        del sim
        gc.collect()
        torch.cuda.empty_cache()

        return bundle

    run_trainer_with_artifacts(paths, _train, encoder_cls=NumpyEncoder)


if __name__ == "__main__":
    main()
