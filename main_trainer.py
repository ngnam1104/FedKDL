import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
from config.settings import NetworkConfig, AcousticChannelConfig, EnergyConfig, FedKDLConfig
from tasks.anomaly_1d.simulator import Simulator1D
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
    parser = argparse.ArgumentParser("FedKDL Unified Trainer")
    parser.add_argument("--topo", type=str, required=True, help="Đường dẫn file topo (.pkl)")
    parser.add_argument("--data", type=str, required=True, help="Đường dẫn file data partition (.pkl)")
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--rho-s", type=float, default=0.05)
    parser.add_argument("--rounds", type=int, default=None, help="Ghi đè số vòng (GLOBAL_ROUNDS)")
    parser.add_argument("--out-dir", type=str, default="results/logs",
                        help="Thư mục JSON metrics (scripts/hfl đọc từ đây)")
    parser.add_argument("--log-dir", type=str, default="results/train_logs/hfl",
                        help="Thư mục stdout .log từng run (debug / tư liệu)")
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
    
    fed_cfg.RHO_S = args.rho_s
    if args.rounds is not None:
        fed_cfg.GLOBAL_ROUNDS = {"1D": args.rounds, "2D": args.rounds}
    
    T_rounds = fed_cfg.GLOBAL_ROUNDS["1D"]
    
    stem = data_path.stem
    parts = stem.split("_")
    N = int(parts[1][1:])
    dataset = parts[2]
    alpha_str = parts[3][1:]
    seed = int(parts[4][4:])

    paths = build_experiment_paths(
        task="1D",
        out_dir=args.out_dir,
        log_dir=args.log_dir,
        N=N,
        dataset=dataset,
        alpha_str=alpha_str,
        baseline=args.baseline,
        seed=seed,
        rho_s=args.rho_s,
    )

    def _train():
        sim = Simulator1D(
            topo_path=str(topo_path),
            data_path=str(data_path),
            baseline=args.baseline,
            device="cpu",
        )
        print(f"\n[Trainer 1D] baseline={args.baseline} rounds={T_rounds} rho_s={args.rho_s}")
        print(f"[Trainer 1D] topo={topo_path}")
        print(f"[Trainer 1D] data={data_path}")
        history = sim.run(T_rounds=T_rounds, baseline=args.baseline)
        return build_experiment_bundle(
            sim,
            history,
            metadata={
                "task": "1D",
                "baseline": args.baseline,
                "rho_s": args.rho_s,
                "rounds": T_rounds,
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
