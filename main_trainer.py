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
        import torch
        device = "cpu" # Ép buộc chạy CPU cho mạng 1D để tránh nghẽn cổ chai GPU
        # Initialize Simulator first to get dataloaders and network info
        sim = Simulator1D(
            topo_path=str(topo_path),
            data_path=str(data_path),
            baseline=args.baseline,
            device=device,
        )

        if args.baseline == 'centralized':
            print(f"\n[Trainer 1D] RUNNING CENTRALIZED TRAINING ON {data_path}")
            
            # Tính toán các chỉ số vật lý cho Centralized
            total_samples = sum(len(loader.dataset) for loader in sim.train_loaders.values())
            if total_samples == 0:
                total_samples = 1000 # Fallback
                
            from physics_models.latency import comp_delay_dynamic
            from physics_models.energy import e_comp_dynamic
            from config.settings import fed_cfg, energy_cfg as en_cfg
            
            tau_comp_gw = comp_delay_dynamic(
                n_samples=total_samples,
                n_local_epochs=1,
                flops_per_sample=fed_cfg.MODEL_FLOPS_PER_SAMPLE["1D"],
                flop_multiplier=fed_cfg.FLOP_MULTIPLIER["1D"],
                f_cpu=en_cfg.F_CPU * 5
            )
            
            e_comp_gw = e_comp_dynamic(
                n_samples=total_samples,
                n_local_epochs=1,
                flops_per_sample=fed_cfg.MODEL_FLOPS_PER_SAMPLE["1D"],
                epsilon_op=en_cfg.EPSILON_OP["1D"],
                flop_multiplier=fed_cfg.FLOP_MULTIPLIER["1D"]
            )
            
            # Raw data transmission in round 1
            # For 1D, each sample is 10 floats (40 bytes)
            raw_payload_kb = (total_samples * 40) / 1024.0
            
            from physics_models.energy import e_tx
            e_tx_raw_total = 0.0
            for sid, s in sim.sensors.items():
                if ('sensor', sid, 'gateway', 0) in sim.G:
                    link = sim.G[('sensor', sid, 'gateway', 0)]
                else:
                    link = next(iter(sim.G.values())) # fallback
                e_tx_raw_total += e_tx(s.n_samples * 40 * 8, link.R_bps, link.SL_min, en_cfg.ETA_EA, en_cfg.P_C_TX)

            from tasks.anomaly_1d.dataloader import load_dataset, SlidingWindowDataset
            from torch.utils.data import DataLoader
            from utils.env_manager import EnvironmentManager
            from federated_core.metrics import anomaly_threshold, point_adjusted_f1
            import torch
            
            data_part = EnvironmentManager.load_data_partition(str(data_path))
            train_data_split, train_labels_split, val_parts, val_labels_parts, test_parts, test_labels_parts = load_dataset(dataset, seed=seed, per_channel_eval=True)
            
            train_ds = SlidingWindowDataset(train_data_split, train_labels_split, window_size=10)
            train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
            
            val_loaders_per_channel = []
            test_loaders_per_channel = []
            
            for v_d, v_l in zip(val_parts, val_labels_parts):
                if len(v_d) >= 10:
                    ds = SlidingWindowDataset(v_d, v_l, window_size=10)
                    val_loaders_per_channel.append(DataLoader(ds, batch_size=256, shuffle=False))
                else:
                    val_loaders_per_channel.append(None)
                    
            for t_d, t_l in zip(test_parts, test_labels_parts):
                if len(t_d) >= 10:
                    ds = SlidingWindowDataset(t_d, t_l, window_size=10)
                    test_loaders_per_channel.append(DataLoader(ds, batch_size=256, shuffle=False))
                else:
                    test_loaders_per_channel.append(None)
            
            sample_batch, _ = next(iter(train_loader))
            input_dim = sample_batch.shape[1]
            
            from tasks.anomaly_1d.autoencoder import SmallAutoencoder
            model = SmallAutoencoder(input_dim=input_dim).to(device)
            
            from tasks.anomaly_1d.trainer import local_sgd
            pa_f1_history = []
            f1_std_history = []
            prec_history = []
            rec_history = []
            prec_std_history = []
            rec_std_history = []
            auc_roc_history = []
            pr_auc_history = []
            loss_history = []
            
            # Train T_rounds
            for t in range(T_rounds):
                model.train()
                _, avg_loss = local_sgd(
                    model=model,
                    dataloader=train_loader,
                    epochs=fed_cfg.LOCAL_EPOCHS,
                    lr=fed_cfg.LOCAL_LR,
                    mu=0.0,
                    device=device,
                )
                
                # Evaluate sau mỗi vòng
                model.eval()
                
                from federated_core.metrics import anomaly_threshold, point_adjusted_f1_components, best_f1_components
                
                total_tp_pa = 0
                total_fp_pa = 0
                total_fn_pa = 0
                total_tp_std = 0
                total_fp_std = 0
                total_fn_std = 0
                
                global_test_labels = []
                global_test_errors = []
                
                for v_loader, t_loader in zip(val_loaders_per_channel, test_loaders_per_channel):
                    if v_loader is None or t_loader is None:
                        continue
                        
                    val_errors = []
                    with torch.no_grad():
                        for x_val, y_val in v_loader:
                            x_val = x_val.to(device)
                            errs = model.reconstruction_error(x_val).cpu().numpy()
                            normal_errs = errs[y_val.numpy() == 0]
                            val_errors.extend(normal_errs)
                            
                    if len(val_errors) == 0:
                        continue
                        
                    test_errors = []
                    test_labels_list = []
                    with torch.no_grad():
                        for x_test, y_test in t_loader:
                            x_test = x_test.to(device)
                            errs = model.reconstruction_error(x_test).cpu().numpy()
                            test_errors.extend(errs)
                            test_labels_list.extend(y_test.numpy())
                            
                    if len(test_errors) == 0:
                        continue
                        
                    global_test_labels.extend(test_labels_list)
                    global_test_errors.extend(test_errors)
                        
                    if fed_cfg.ANOMALY_EVAL_MODE == "best_f1":
                        comps = best_f1_components(np.array(test_labels_list), np.array(test_errors))
                        tp_pa, fp_pa, fn_pa, tp_std, fp_std, fn_std = comps
                    else:
                        tau_A = anomaly_threshold(np.array(val_errors), percentile=fed_cfg.ANOMALY_PERCENTILE)
                        comps = point_adjusted_f1_components(np.array(test_labels_list), np.array(test_errors), tau_A)
                        tp_pa, fp_pa, fn_pa, tp_std, fp_std, fn_std = comps
                        
                    total_tp_pa += tp_pa
                    total_fp_pa += fp_pa
                    total_fn_pa += fn_pa
                    total_tp_std += tp_std
                    total_fp_std += fp_std
                    total_fn_std += fn_std
                
                prec = total_tp_pa / (total_tp_pa + total_fp_pa) if (total_tp_pa + total_fp_pa) > 0 else 0.0
                rec = total_tp_pa / (total_tp_pa + total_fn_pa) if (total_tp_pa + total_fn_pa) > 0 else 0.0
                pa_f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                
                prec_std = total_tp_std / (total_tp_std + total_fp_std) if (total_tp_std + total_fp_std) > 0 else 0.0
                rec_std = total_tp_std / (total_tp_std + total_fn_std) if (total_tp_std + total_fn_std) > 0 else 0.0
                f1_std = 2 * prec_std * rec_std / (prec_std + rec_std) if (prec_std + rec_std) > 0 else 0.0
                
                auc_roc = 0.0
                pr_auc = 0.0
                if len(global_test_labels) > 0 and len(np.unique(global_test_labels)) > 1:
                    try:
                        from sklearn.metrics import roc_auc_score, average_precision_score
                        auc_roc = roc_auc_score(global_test_labels, global_test_errors)
                        pr_auc = average_precision_score(global_test_labels, global_test_errors)
                    except ImportError:
                        pass
                
                pa_f1_history.append(pa_f1)
                f1_std_history.append(f1_std)
                auc_roc_history.append(auc_roc)
                pr_auc_history.append(pr_auc)
                prec_history.append(prec)
                rec_history.append(rec)
                prec_std_history.append(prec_std)
                rec_std_history.append(rec_std)
                loss_history.append(avg_loss)
                
                print(f"   -> [Centralized Training] Round {t+1}/{T_rounds} | Loss: {avg_loss:.4f} | PA-F1: {pa_f1:.4f} | AUC: {auc_roc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}        ", end="\r", flush=True)
                
            print() # Xuống dòng khi kết thúc vòng lặp
            
            tau_cumul_s = [tau_comp_gw * t for t in range(1, T_rounds + 1)]
            e_cumul = [e_tx_raw_total + e_comp_gw * t for t in range(1, T_rounds + 1)]
            avg_payload_kb_arr = [raw_payload_kb] + [0] * (T_rounds - 1) if T_rounds > 0 else []
            e_total_arr = [e_tx_raw_total + e_comp_gw] + [e_comp_gw] * (T_rounds - 1) if T_rounds > 0 else []
            
            history = {
                'round': list(range(1, T_rounds + 1)),
                'PA-F1': pa_f1_history,
                'F1-Score': f1_std_history,
                'AUC-ROC': auc_roc_history,
                'PR-AUC': pr_auc_history,
                'Prec': prec_history,
                'Rec': rec_history,
                'Prec-Std': prec_std_history,
                'Rec-Std': rec_std_history,
                'loss': loss_history,
                'alive': [N] * T_rounds,
                'tau_round_s': [tau_comp_gw] * T_rounds,
                'tau_cumul_s': tau_cumul_s,
                'avg_payload_kb': avg_payload_kb_arr,
                'payload_cumul_kb': [raw_payload_kb] * T_rounds,
                'e_total': e_total_arr,
                'e_cumul': e_cumul,
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
                "metrics": history
            }

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
