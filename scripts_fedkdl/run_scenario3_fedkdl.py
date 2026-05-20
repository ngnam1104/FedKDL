"""
run_scenario3_fedkdl.py
Kịch bản 3: FedKDL (KD-LoRA-INT8 + HFL-Selective) vs baselines.
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config.settings import fed_cfg, network_cfg
from kdl_core.data.dataloader_2d import create_client_datasets_yolo
from kdl_core.simulator_od import ODSimulator


def run():
    yaml_dir = "datasets/urpc_splits_s3"
    client_yamls = create_client_datasets_yolo(
        "datasets/URPC2020.yaml", yaml_dir, network_cfg.N_SENSORS
    )
    test_yaml = client_yamls[0]

    results = {}
    for baseline in ["fedavg", "hfl_nocoop", "hfl_selective"]:
        print(f"\n>>> Running: {baseline}")
        sim = ODSimulator(client_yamls, test_yaml, device="cpu")
        results[baseline] = sim.run(baseline=baseline, use_kd_lora_int8=True)

    os.makedirs("results/scenario3", exist_ok=True)

    colors = {"fedavg": "orange", "hfl_nocoop": "steelblue", "hfl_selective": "green"}
    labels = {"fedavg": "FedAvg + KD-LoRA-INT8", "hfl_nocoop": "HFL-NoCoop + KD-LoRA-INT8",
              "hfl_selective": "FedKDL (Ours)"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for key, hist in results.items():
        axes[0].plot(hist['round'], hist['map'], label=labels[key], color=colors[key])
        axes[1].plot(hist['round'], hist['alive'], label=labels[key], color=colors[key])
        axes[2].plot(hist['round'], hist['energy_cumul_J'], label=labels[key], color=colors[key])

    axes[0].set_title("mAP@0.5:0.95 vs Round"); axes[0].set_xlabel("Round"); axes[0].set_ylabel("mAP")
    axes[1].set_title("Alive AUVs vs Round"); axes[1].set_xlabel("Round"); axes[1].set_ylabel("Alive AUVs")
    axes[2].set_title("Cumulative Energy (J)"); axes[2].set_xlabel("Round"); axes[2].set_ylabel("E (J)")

    for ax in axes:
        ax.grid(True); ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("results/scenario3/fedkdl_comparison.png", dpi=150)
    print("Lưu biểu đồ tại results/scenario3/fedkdl_comparison.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        fed_cfg.GLOBAL_ROUNDS = 2
        fed_cfg.LOCAL_EPOCHS = 1
    run()
