"""
run_scenario2_bottleneck.py
Kịch bản 2: Chứng minh sự sụp đổ của mạng AUV khi truyền YOLO26n nguyên bản.
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config.settings import fed_cfg, network_cfg
from hfl_core.data.dataloader_2d import create_client_datasets_yolo
from hfl_core.simulator_od import ODSimulator


def run():
    yaml_dir = "datasets/urpc_splits_s2"
    client_yamls = create_client_datasets_yolo(
        "datasets/URPC2020.yaml", yaml_dir, network_cfg.N_SENSORS
    )
    test_yaml = client_yamls[0]

    sim = ODSimulator(client_yamls, test_yaml, device="cpu")
    history = sim.run(baseline="fedavg", use_kd_lora_int8=False)

    os.makedirs("results/scenario2", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history['round'], history['alive'], color='red', marker='x', label="Full YOLO26n")
    axes[0].set_title("AUV Survival over Rounds")
    axes[0].set_xlabel("Round"); axes[0].set_ylabel("Alive AUVs")
    axes[0].grid(True); axes[0].legend()

    axes[1].plot(history['round'], history['avg_payload_kb'], color='orange', marker='s', label="Full YOLO26n")
    axes[1].set_yscale('log')
    axes[1].set_title("Avg Payload Size (KB) — log scale")
    axes[1].set_xlabel("Round"); axes[1].set_ylabel("Payload (KB)")
    axes[1].grid(True, which='both'); axes[1].legend()

    plt.tight_layout()
    plt.savefig("results/scenario2/bottleneck.png", dpi=150)
    print("Lưu biểu đồ tại results/scenario2/bottleneck.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        fed_cfg.GLOBAL_ROUNDS = 2
        fed_cfg.LOCAL_EPOCHS = 1
    run()
