import os
import sys



if not hasattr(sys.stdout, 'encoding'):
    sys.stdout.encoding = 'utf-8'
import json
import argparse
from pathlib import Path
import numpy as np
import torch

from config.settings import network_cfg, acoustic_cfg, energy_cfg, fed_cfg
from detection_2d.baselines import BASELINE_CONFIGS, parse_baseline_config
from detection_2d.simulator import Simulator2D
from utils.log_export import build_experiment_bundle
from utils.image_payload import image_bytes_by_owner, list_unique_image_files
from utils.train_io import build_experiment_paths, run_trainer_with_artifacts

import ultralytics
ultralytics.settings.update({'datasets_dir': str(Path('datasets').absolute())})

# Register sys.modules shims so that checkpoints saved under the old
# 'tasks.detection_2d.*' package path can still be unpickled.
import detection_2d.compat  # noqa: F401  (side-effect import)

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
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        choices=tuple(BASELINE_CONFIGS),
        help="2D experiment baseline",
    )
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
        
    
    if args.rounds is not None:
        fed_cfg.GLOBAL_ROUNDS = {"1D": args.rounds, "2D": args.rounds}
    if args.lora_rank is not None:
        fed_cfg.LORA_RANK = args.lora_rank
        
    # [TỐI ƯU HÓA] Bỏ hardcode ép LOCAL_EPOCHS=3 để tôn trọng LOCAL_EPOCHS=2 từ config/settings.py
    # fed_cfg.LOCAL_EPOCHS = 3

    T_rounds = fed_cfg.GLOBAL_ROUNDS["2D"]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    stem = data_path.stem
    parts = stem.split("_")
    N = int(parts[1][1:])
    dataset = parts[2]
    alpha_str = parts[3][1:]
    seed = int(parts[4][4:])
    
    network_cfg.N_AUVS = N

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
        baseline_cfg = parse_baseline_config(args.baseline)
        # Train warmup once, then expose two architecture views of the same
        # function: LoRAConv2d for LoRA methods and baked YOLO modules for all
        # full-model, Top-K, and SCAFFOLD methods.
        from scripts.fedkdl.train_student_warmup import ensure_warmup_checkpoints
        warmup_pt, head_warmup_pt = ensure_warmup_checkpoints(
            epochs=getattr(fed_cfg, "STUDENT_WARMUP_EPOCHS", 5)
        )
        if not warmup_pt.exists() or not head_warmup_pt.exists():
            raise RuntimeError(
                "Warmup preparation failed: both yolo12n_warmup.pt and "
                "yolo12n_head_warmup.pt are required before training."
            )
        if baseline_cfg.use_lora:
            chosen_student_ckpt = str(warmup_pt)
            print(f"[Auto-Warmup] {args.baseline}: dùng checkpoint LoRA {warmup_pt.name}.")
        else:
            chosen_student_ckpt = str(head_warmup_pt)
            print(f"[Auto-Warmup] {args.baseline}: dùng checkpoint đã bake {head_warmup_pt.name}.")

        teacher_ckpt = Path(getattr(fed_cfg, "TEACHER_CKPT", "teacher_lora_best.pt"))
        if not teacher_ckpt.exists():
            fallback_teachers = [
                Path("yolo12l_lora_pretrained.pt"),
                Path("yolo12l_pretrained.pt"),
                Path("yolo12l.pt"),
            ]
            teacher_ckpt = next((p for p in fallback_teachers if p.exists()), teacher_ckpt)

        # Initialize Simulator first to get total_samples and network info
        sim = Simulator2D(
            topo_path=str(topo_path),
            data_path=str(data_path),
            baseline=args.baseline,
            test_yaml="datasets/URPC2020.yaml" if "urpc" in dataset.lower() else "coco8.yaml",
            student_ckpt=chosen_student_ckpt,
            teacher_ckpt=str(teacher_ckpt),
            device=device,
        )


        if args.baseline == 'centralized':
            print(f"\n[Trainer 2D] RUNNING CENTRALIZED TRAINING ON {data_path}")
            
            total_samples = sum(getattr(s, 'n_samples', 0) for s in sim.auvs.values())
            if total_samples == 0:
                total_samples = 4000 # Fallback
            
            from physics_models.latency import comp_delay_dynamic
            from physics_models.energy import e_comp
            from config.settings import energy_cfg as en_cfg
            
            # In centralized, Gateway trains 1 epoch per round.
            tau_comp_gw = comp_delay_dynamic(
                n_samples=total_samples,
                n_local_epochs=1,
                flops_per_sample=fed_cfg.MODEL_FLOPS_PER_SAMPLE["2D"],
                flop_multiplier=fed_cfg.FLOP_MULTIPLIER["2D"],
                f_cpu=en_cfg.F_CPU * 5 # Assuming GW is 5x faster
            )
            
            e_comp_gw = e_comp(
                n_samples=total_samples,
                local_epochs=1,
                flops_per_sample=fed_cfg.MODEL_FLOPS_PER_SAMPLE["2D"],
                epsilon_op=en_cfg.EPSILON_OP["2D"],
                flop_multiplier=fed_cfg.FLOP_MULTIPLIER["2D"],
                f_cpu=en_cfg.F_CPU * 5
            )
            
            # Raw data transmission in round 1, measured from the encoded image
            # files owned by each AUV rather than a fixed per-image estimate.
            image_dir = Path("datasets/URPC2020/URPC2020/train/images")
            image_paths = list_unique_image_files(image_dir)
            owner_image_bytes = image_bytes_by_owner(
                image_paths,
                sim.data_part.auv_data_indices,
            )
            raw_payload_kb = sum(owner_image_bytes.values()) / 1024.0
            
            # E_tx for raw data (Assume directly sent to Gateway)
            from physics_models.energy import e_tx
            from physics_models.latency import comm_delay
            e_tx_raw_total = 0.0
            tau_tx_raw_max = 0.0
            for sid, s in sim.auvs.items():
                if ('auv', sid, 'gateway', 0) in sim.G:
                    link = sim.G[('auv', sid, 'gateway', 0)]
                else:
                    link = next(iter(sim.G.values())) # fallback
                bits = owner_image_bytes.get(sid, 0) * 8
                e_tx_raw_total += e_tx(bits, link.R_bps, link.SL_min, en_cfg.ETA_EA, en_cfg.P_C_TX)
                tau_tx_s = comm_delay(bits, link.R_bps, getattr(link, 'distance', 1000.0))
                if tau_tx_s > tau_tx_raw_max:
                    tau_tx_raw_max = tau_tx_s

            from scripts.fedkdl.train_student_warmup import run_centralized_lora
            
            # Centralized upper bound uses the same LoRA+Head parameterization
            # and differential LR policy configured for the warmup/student path.
            test_yaml = "datasets/URPC2020.yaml" if "urpc" in dataset.lower() else "coco8.yaml"
            centralized_result = run_centralized_lora(
                epochs=T_rounds,
                data_yaml=test_yaml,
                project=args.out_dir,
                name=f"centralized_{stem}",
                device=device,
            )
            
            import pandas as pd
            results_csv = centralized_result["results_csv"]
            if not results_csv.exists():
                raise FileNotFoundError(f"Centralized results were not written: {results_csv}")
            results_df = pd.read_csv(results_csv)
            results_df.columns = [str(column).strip() for column in results_df.columns]

            def _metric_series(*names: str) -> list[float]:
                for name in names:
                    if name in results_df:
                        return results_df[name].fillna(0.0).astype(float).tolist()
                return [0.0] * len(results_df)

            map50_95_arr = _metric_series("metrics/mAP50-95(B)", "metrics/mAP50-95")
            map50_arr = _metric_series("metrics/mAP50(B)", "metrics/mAP50")
            precision_arr = _metric_series("metrics/precision(B)", "metrics/precision")
            recall_arr = _metric_series("metrics/recall(B)", "metrics/recall")
            train_box_arr = _metric_series("train/box_loss")
            train_cls_arr = _metric_series("train/cls_loss")
            train_dfl_arr = _metric_series("train/dfl_loss")
            val_box_arr = _metric_series("val/box_loss")
            val_cls_arr = _metric_series("val/cls_loss")
            val_dfl_arr = _metric_series("val/dfl_loss")
            train_loss_arr = [
                box + cls + dfl
                for box, cls, dfl in zip(train_box_arr, train_cls_arr, train_dfl_arr)
            ]
            val_loss_arr = [
                box + cls + dfl
                for box, cls, dfl in zip(val_box_arr, val_cls_arr, val_dfl_arr)
            ]

            epochs_done = len(results_df)
            if epochs_done == 0:
                raise RuntimeError("Centralized training produced an empty results.csv")
            print(
                f"[Centralized] mAP50-95: {map50_95_arr[-1]:.4f} | "
                f"mAP50: {map50_arr[-1]:.4f}"
            )

            tau_round_s_arr = [tau_tx_raw_max + tau_comp_gw] + [tau_comp_gw] * (epochs_done - 1)
            tau_cumul_s = [tau_tx_raw_max + tau_comp_gw * t for t in range(1, epochs_done + 1)]
            e_cumul = [e_tx_raw_total + e_comp_gw * t for t in range(1, epochs_done + 1)]
            avg_payload_kb_arr = [raw_payload_kb] + [0] * (epochs_done - 1)
            e_total_arr = [e_tx_raw_total + e_comp_gw] + [e_comp_gw] * (epochs_done - 1)
            
            history = {
                'round': list(range(1, epochs_done + 1)),
                'mAP50-95': map50_95_arr,
                'mAP50': map50_arr,
                'Prec': precision_arr,
                'Rec': recall_arr,
                'val_box_loss': val_box_arr,
                'val_cls_loss': val_cls_arr,
                'val_dfl_loss': val_dfl_arr,
                'loss': train_loss_arr,
                'val_loss': val_loss_arr,
                'alive': [N] * epochs_done,
                'tau_round_s': tau_round_s_arr,
                'tau_a2r': [tau_tx_raw_max] + [0.0] * (epochs_done - 1),
                'tau_r2r': [0.0] * epochs_done,
                'tau_r2g': [0.0] * epochs_done,
                'tau_comp': [tau_comp_gw] * epochs_done,
                'tau_svd': [0.0] * epochs_done,
                'tau_cumul_s': tau_cumul_s,
                'avg_payload_kb': avg_payload_kb_arr,
                'payload_cumul_kb': [raw_payload_kb] * epochs_done,
                'e_total': e_total_arr,
                'e_a2r': [e_tx_raw_total] + [0.0] * (epochs_done - 1),
                'e_r2r': [0.0] * epochs_done,
                'e_r2g': [0.0] * epochs_done,
                'e_comp': [e_comp_gw] * epochs_done,
                'e_svd': [0.0] * epochs_done,
                'e_cumul': e_cumul,
            }
            
            del sim
            import gc
            gc.collect()
            import torch
            torch.cuda.empty_cache()
            
            return {
                "metadata": {
                    "task": "2D",
                    "baseline": args.baseline,
                    "rounds": T_rounds,
                    "N": N,
                    "dataset": dataset,
                    "alpha": alpha_str,
                    "seed": seed,
                },
                "metrics": history
            }
            
        print(f"\n[Trainer 2D] baseline={args.baseline} rounds={T_rounds} lora_rank={fed_cfg.LORA_RANK} device={device}")
        print(f"[Trainer 2D] topo={topo_path}")
        print(f"[Trainer 2D] data={data_path}")
        history = sim.run(T_rounds=T_rounds, baseline=args.baseline)
        
        bundle = build_experiment_bundle(
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
        
        del sim
        import gc
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        
        return bundle

    run_trainer_with_artifacts(paths, _train, encoder_cls=NumpyEncoder)

if __name__ == "__main__":
    main()
