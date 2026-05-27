import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch

from config.settings import network_cfg, acoustic_cfg, energy_cfg, fed_cfg
from tasks.detection_2d.simulator import Simulator2D
from utils.log_export import build_experiment_bundle
from utils.train_io import build_experiment_paths, run_trainer_with_artifacts

import ultralytics
ultralytics.settings.update({'datasets_dir': str(Path('datasets').absolute())})

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
    
    network_cfg.N_SENSORS = N

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
        # Initialize Simulator first to get total_samples and network info
        sim = Simulator2D(
            topo_path=str(topo_path),
            data_path=str(data_path),
            baseline=args.baseline,
            test_yaml="datasets/URPC2020.yaml" if "urpc" in dataset.lower() else "coco8.yaml",
            student_ckpt="yolo11n_pretrained.pt" if Path("yolo11n_pretrained.pt").exists() else "yolo11n.pt",
            teacher_ckpt="yolo12l_pretrained.pt" if Path("yolo12l_pretrained.pt").exists() else "yolo12l.pt",
            device=device,
        )

        if args.baseline == 'centralized':
            print(f"\n[Trainer 2D] RUNNING CENTRALIZED TRAINING ON {data_path}")
            
            total_samples = sum(getattr(s, 'n_samples', 0) for s in sim.sensors.values())
            if total_samples == 0:
                total_samples = 4000 # Fallback
            
            from physics_models.latency import comp_delay_dynamic
            from physics_models.energy import e_comp_dynamic
            from config.settings import energy_cfg as en_cfg
            
            # In centralized, Gateway trains 1 epoch per round.
            tau_comp_gw = comp_delay_dynamic(
                n_samples=total_samples,
                n_local_epochs=1,
                flops_per_sample=fed_cfg.MODEL_FLOPS_PER_SAMPLE["2D"],
                flop_multiplier=fed_cfg.FLOP_MULTIPLIER["2D"],
                f_cpu=en_cfg.F_CPU * 5 # Assuming GW is 5x faster
            )
            
            e_comp_gw = e_comp_dynamic(
                n_samples=total_samples,
                n_local_epochs=1,
                flops_per_sample=fed_cfg.MODEL_FLOPS_PER_SAMPLE["2D"],
                epsilon_op=en_cfg.EPSILON_OP["2D"],
                flop_multiplier=fed_cfg.FLOP_MULTIPLIER["2D"]
            )
            
            # Raw data transmission in round 1
            raw_payload_kb = total_samples * 500 # Assume 500KB per image
            
            # E_tx for raw data (Assume directly sent to Gateway)
            from physics_models.energy import e_tx
            from physics_models.latency import comm_delay
            e_tx_raw_total = 0.0
            tau_tx_raw_max = 0.0
            for sid, s in sim.sensors.items():
                if ('sensor', sid, 'gateway', 0) in sim.G:
                    link = sim.G[('sensor', sid, 'gateway', 0)]
                else:
                    link = next(iter(sim.G.values())) # fallback
                n_samples_s = getattr(s, 'n_samples', 100)
                bits = n_samples_s * 500 * 1024 * 8
                e_tx_raw_total += e_tx(bits, link.R_bps, link.SL_min, en_cfg.ETA_EA, en_cfg.P_C_TX)
                tau_tx_s = comm_delay(bits, link.R_bps, getattr(link, 'distance', 1000.0))
                if tau_tx_s > tau_tx_raw_max:
                    tau_tx_raw_max = tau_tx_s

            from ultralytics import YOLO
            
            # Khởi tạo mô hình dựa trên config full_param, LORA, v.v.
            # Centralized thường train full params hoặc có thể test LoRA. Ở đây giả định train full params.
            model = YOLO("yolo11n.pt")
            
            # Train trực tiếp trên dataset yaml tương ứng
            test_yaml = "datasets/URPC2020.yaml" if "urpc" in dataset.lower() else "coco8.yaml"
            results = model.train(
                data=test_yaml,
                epochs=T_rounds,
                imgsz=640,
                batch=getattr(fed_cfg, 'LOCAL_BATCH_SIZE', 16),
                workers=getattr(fed_cfg, 'DATALOADER_WORKERS', 4),
                device=device,
                project=args.out_dir,
                name=f"centralized_{stem}",
                verbose=True
            )
            
            map50_95 = float(results.box.map)
            map50 = float(results.box.map50)
            mp = float(np.mean(results.box.mp)) if hasattr(results.box, 'mp') else 0.0
            mr = float(np.mean(results.box.mr)) if hasattr(results.box, 'mr') else 0.0
            
            # Ultralytics results.results_dict contains detailed losses
            val_box_loss = 0.0
            val_cls_loss = 0.0
            val_dfl_loss = 0.0
            train_box_loss = 0.0
            train_cls_loss = 0.0
            train_dfl_loss = 0.0
            
            if hasattr(results, 'results_dict'):
                val_box_loss = results.results_dict.get('val/box_loss', 0.0)
                val_cls_loss = results.results_dict.get('val/cls_loss', 0.0)
                val_dfl_loss = results.results_dict.get('val/dfl_loss', 0.0)
                
                train_box_loss = results.results_dict.get('train/box_loss', 0.0)
                train_cls_loss = results.results_dict.get('train/cls_loss', 0.0)
                train_dfl_loss = results.results_dict.get('train/dfl_loss', 0.0)
                
            total_val_loss = val_box_loss + val_cls_loss + val_dfl_loss
            total_train_loss = train_box_loss + train_cls_loss + train_dfl_loss
                
            print(f"[Centralized] mAP50-95: {map50_95:.4f} | mAP50: {map50:.4f}")
            
            tau_round_s_arr = [tau_tx_raw_max + tau_comp_gw] + [tau_comp_gw] * (T_rounds - 1) if T_rounds > 0 else []
            tau_cumul_s = [tau_tx_raw_max + tau_comp_gw * t for t in range(1, T_rounds + 1)]
            e_cumul = [e_tx_raw_total + e_comp_gw * t for t in range(1, T_rounds + 1)]
            avg_payload_kb_arr = [raw_payload_kb] + [0] * (T_rounds - 1) if T_rounds > 0 else []
            e_total_arr = [e_tx_raw_total + e_comp_gw] + [e_comp_gw] * (T_rounds - 1) if T_rounds > 0 else []
            
            history = {
                'round': list(range(1, T_rounds + 1)),
                'mAP50-95': [map50_95] * T_rounds,
                'mAP50': [map50] * T_rounds,
                'Prec': [mp] * T_rounds,
                'Rec': [mr] * T_rounds,
                'val_box_loss': [val_box_loss] * T_rounds,
                'val_cls_loss': [val_cls_loss] * T_rounds,
                'val_dfl_loss': [val_dfl_loss] * T_rounds,
                'loss': [total_train_loss] * T_rounds,
                'val_loss': [total_val_loss] * T_rounds,
                'alive': [N] * T_rounds,
                'tau_round_s': tau_round_s_arr,
                'tau_cumul_s': tau_cumul_s,
                'avg_payload_kb': avg_payload_kb_arr,
                'payload_cumul_kb': [raw_payload_kb] * T_rounds,
                'e_total': e_total_arr,
                'e_cumul': e_cumul,
            }
            
            del model
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
