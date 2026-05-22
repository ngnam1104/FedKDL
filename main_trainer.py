import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
from config.settings import network_cfg, acoustic_cfg, energy_cfg, fed_cfg
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
        if args.baseline == 'centralized':
            print(f"\n[Trainer 1D] RUNNING CENTRALIZED TRAINING ON {dataset}")
            from tasks.anomaly_1d.dataloader import load_dataset, SlidingWindowDataset, make_val_loader
            from tasks.anomaly_1d.autoencoder import SmallAutoencoder
            from tasks.anomaly_1d.trainer import local_sgd
            from torch.utils.data import DataLoader
            from federated_core.metrics import anomaly_threshold, point_adjusted_f1
            import torch
            
            # 1. Load data
            train_data, train_labels, test_data, test_labels = load_dataset(dataset, seed=seed)
            split_idx = int(len(train_data) * 0.7)
            train_ds = SlidingWindowDataset(train_data[:split_idx], train_labels[:split_idx], window_size=10)
            val_ds   = SlidingWindowDataset(train_data[split_idx:], train_labels[split_idx:], window_size=10)
            test_ds  = SlidingWindowDataset(test_data,  test_labels,  window_size=10)
            
            train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
            val_loader = make_val_loader(val_ds, batch_size=256)
            test_loader = make_val_loader(test_ds, batch_size=256)
            
            # 2. Init model
            sample_batch, _ = next(iter(train_loader))
            model = SmallAutoencoder(input_dim=sample_batch.shape[1]).to("cpu")
            
            # 3. Train
            # Centralized runs for T_rounds epochs for equivalence
            _, avg_loss = local_sgd(
                model=model,
                dataloader=train_loader,
                epochs=T_rounds,
                lr=fed_cfg.LOCAL_LR,
                device="cpu",
            )
            
            # 4. Evaluate
            model.eval()
            val_errors = []
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    errs = model.reconstruction_error(x_val).cpu().numpy()
                    normal_errs = errs[y_val.numpy() == 0]
                    val_errors.extend(normal_errs)
            
            tau_A = anomaly_threshold(np.array(val_errors), percentile=99.0)
            
            test_errors = []
            test_labels_list = []
            with torch.no_grad():
                for x_test, y_test in test_loader:
                    errs = model.reconstruction_error(x_test)
                    test_errors.extend(errs.cpu().numpy())
                    test_labels_list.extend(y_test.numpy())
            
            pa_f1, prec, rec = point_adjusted_f1(np.array(test_labels_list), np.array(test_errors), tau_A)
            
            print(f"[Centralized] PA-F1: {pa_f1:.4f}")
            
            history = {
                'round': list(range(1, T_rounds + 1)),
                'PA-F1': [pa_f1] * T_rounds,
                'tau_round_s': [0] * T_rounds,
                'avg_payload_kb': [0] * T_rounds,
                'e_total': [0] * T_rounds,
                'e_cumul': [0] * T_rounds,
                'loss': [avg_loss] * T_rounds,
            }
            
            return {
                "metadata": {
                    "task": "1D",
                    "baseline": args.baseline,
                    "rho_s": args.rho_s,
                    "rounds": T_rounds,
                    "N": N,
                    "dataset": dataset,
                    "alpha": alpha_str,
                    "seed": seed,
                },
                "history": history
            }

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
