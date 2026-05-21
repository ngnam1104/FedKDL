import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch

from config.settings import NetworkConfig, AcousticChannelConfig, EnergyConfig, FedKDLConfig
from tasks.detection_2d.simulator import Simulator2D
from utils.log_export import build_experiment_bundle
from utils.train_io import build_experiment_paths, run_trainer_with_artifacts

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def parse_args():
    parser = argparse.ArgumentParser("FedKDL OD Trainer")
    parser.add_argument("--topo", type=str, required=True, help="Đường dẫn file topo (.pkl)")
    parser.add_argument("--data", type=str, required=True, help="Đường dẫn file data partition (.pkl)")
    parser.add_argument("--baseline", type=str, required=True, help="fedkdl hoặc baseline_od")
    parser.add_argument("--rounds", type=int, default=None, help="Ghi đè số vòng (GLOBAL_ROUNDS)")
    parser.add_argument("--out-dir", type=str, default="results/logs_kdl",
                        help="Thư mục JSON metrics (scripts/fedkdl đọc từ đây)")
    parser.add_argument("--log-dir", type=str, default="results/train_logs/kdl",
                        help="Thư mục stdout .log từng run (debug / tư liệu)")
    parser.add_argument("--lora-rank", type=int, default=None, help="Ghi đè LORA_RANK (4 hoặc 8)")
    return parser.parse_args()

def main():
    args = parse_args()
    topo_path = Path(args.topo)
    data_path = Path(args.data)
    
    if not topo_path.exists() or not data_path.exists():
        print(f"[Error] Environment files not found.")
        sys.exit(1)
        
    net_cfg = NetworkConfig()
    ac_cfg  = AcousticChannelConfig()
    en_cfg  = EnergyConfig()
    fed_cfg = FedKDLConfig()
    
    if args.rounds is not None:
        fed_cfg.GLOBAL_ROUNDS = {"1D": args.rounds, "2D": args.rounds}
    if args.lora_rank is not None:
        fed_cfg.LORA_RANK = args.lora_rank

    T_rounds = fed_cfg.GLOBAL_ROUNDS["2D"]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    stem = data_path.stem
    parts = stem.split("_")
    N = int(parts[1][1:])
    dataset = parts[2]
    alpha_str = parts[3][1:]
    seed = int(parts[4][4:])

    paths = build_experiment_paths(
        task="2D",
        out_dir=args.out_dir,
        log_dir=args.log_dir,
        N=N,
        dataset=dataset,
        alpha_str=alpha_str,
        baseline=args.baseline,
        seed=seed,
    )

    def _train():
        sim = Simulator2D(
            topo_path=str(topo_path),
            data_path=str(data_path),
            baseline=args.baseline,
            test_yaml="datasets/URPC2020.yaml",
            device=device,
        )
        print(f"\n[Trainer 2D] baseline={args.baseline} rounds={T_rounds} lora_rank={fed_cfg.LORA_RANK} device={device}")
        print(f"[Trainer 2D] topo={topo_path}")
        print(f"[Trainer 2D] data={data_path}")
        history = sim.run(T_rounds=T_rounds, baseline=args.baseline)
        return build_experiment_bundle(
            sim,
            history,
            metadata={
                "task": "2D",
                "baseline": args.baseline,
                "rounds": T_rounds,
                "lora_rank": fed_cfg.LORA_RANK,
                "N": N,
                "dataset": dataset,
                "alpha": alpha_str,
                "seed": seed,
                "topo_path": str(topo_path),
                "data_path": str(data_path),
            },
        )

    run_trainer_with_artifacts(paths, _train, encoder_cls=NumpyEncoder)

if __name__ == "__main__":
    main()
