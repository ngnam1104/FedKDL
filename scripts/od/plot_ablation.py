"""
plot_ablation.py
Đọc logs JSON (bài toán 2D - Object Detection) để vẽ Ablation Study.
Bao gồm: Bar Charts (Accuracy, Energy, Payload), Convergence Line Chart, và Radar Chart.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from utils.plot_styles import setup_global_plot_style, get_style

def create_radar_chart(ax, angles, values, label, color):
    """Hàm phụ trợ vẽ 1 polygon trên radar chart."""
    values = np.concatenate((values, [values[0]]))
    ax.plot(angles, values, 'o-', linewidth=2, label=label, color=color)
    ax.fill(angles, values, alpha=0.25, color=color)

def plot_ablation():
    setup_global_plot_style()
    os.makedirs("results/ablation", exist_ok=True)

    metrics_data = defaultdict(lambda: defaultdict(list))
    
    log_files = glob.glob("results/logs_kdl/*.json")
    if not log_files:
        log_files = glob.glob("results/test_logs/*.json")

    for f in log_files:
        with open(f, "r", encoding="utf-8") as file:
            data = json.load(file)

        meta = data.get("metadata", {})
        baseline = meta.get("baseline")
        task = meta.get("task")
        
        # Chỉ xét 2D
        if task != "2D": continue
        
        metrics = data.get("metrics", {})
        energy = data.get("energy_consumption", {})
        latency = data.get("latency_history", {})
        
        # Centralized and standard FL might have different structures
        if "mAP50-95" in metrics and metrics["mAP50-95"]:
            metrics_data[baseline]["mAP"].append(metrics["mAP50-95"][-1])
            metrics_data[baseline]["mAP_history"].append(metrics["mAP50-95"])
        elif "mAP50-95" in data.get("history", {}):
            metrics_data[baseline]["mAP"].append(data["history"]["mAP50-95"][-1])
            metrics_data[baseline]["mAP_history"].append(data["history"]["mAP50-95"])

        if "e_s2f" in energy:
            total_e = (sum(energy.get("e_s2f", [])) + sum(energy.get("e_f2f", [])) + 
                       sum(energy.get("e_f2g", [])) + sum(energy.get("e_comp", [])))
            metrics_data[baseline]["energy"].append(total_e)
        elif "e_total" in data.get("history", {}):
            metrics_data[baseline]["energy"].append(np.sum(data["history"]["e_total"]))

        if "avg_payload_kb" in metrics and metrics["avg_payload_kb"]:
            metrics_data[baseline]["payload"].append(np.mean(metrics["avg_payload_kb"]))
        elif "avg_payload_kb" in data.get("history", {}):
            metrics_data[baseline]["payload"].append(np.mean(data["history"]["avg_payload_kb"]))
            
        if "tau_round_s" in latency and latency["tau_round_s"]:
            metrics_data[baseline]["latency"].append(np.sum(latency["tau_round_s"]))
        elif "tau_round_s" in data.get("history", {}):
            metrics_data[baseline]["latency"].append(np.sum(data["history"]["tau_round_s"]))

    if not metrics_data:
        print("[Warning] Không tìm thấy dữ liệu ablation 2D.")
        return

    # Các kịch bản muốn so sánh
    baselines = [
        "fedkdl", 
        "full_param_nokd", 
        "lora_head_kd_noint8", 
        "head_kd_int8_nolora", 
        "lora_head_int8_nokd",
        "fedavg",
        "centralized"
    ]
    
    # Lọc những baseline có thật trong logs
    baselines = [b for b in baselines if b in metrics_data]
    
    labels_map = {
        "fedkdl": "FedKDL (Full)",
        "full_param_nokd": "Full Param + No KD",
        "lora_head_kd_noint8": "No INT8",
        "head_kd_int8_nolora": "No LoRA",
        "lora_head_int8_nokd": "No KD",
        "fedavg": "FedAvg",
        "centralized": "Centralized"
    }

    # Tính mean cho từng metric
    b_map = []
    b_energy = []
    b_payload = []
    b_latency = []
    
    for b in baselines:
        b_map.append(np.mean(metrics_data[b]["mAP"]) if metrics_data[b].get("mAP") else 0.0)
        b_energy.append(np.mean(metrics_data[b]["energy"]) if metrics_data[b].get("energy") else 0.0)
        b_payload.append(np.mean(metrics_data[b]["payload"]) if metrics_data[b].get("payload") else 0.0)
        b_latency.append(np.mean(metrics_data[b]["latency"]) if metrics_data[b].get("latency") else 0.0)

    # ==========================================
    # 1. BARCHART: Accuracy, Energy, Payload
    # ==========================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(baselines))
    colors = plt.cm.tab10(np.linspace(0, 1, len(baselines)))

    # Acc
    axes[0].bar(x, b_map, color=colors, edgecolor='black')
    axes[0].set_title("(a) Accuracy (mAP@50-95)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([labels_map[b] for b in baselines], rotation=45, ha='right')
    axes[0].set_ylabel("mAP")

    # Energy
    axes[1].bar(x, b_energy, color=colors, edgecolor='black')
    axes[1].set_title("(b) Total Energy (J)")
    axes[1].set_yscale('log')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([labels_map[b] for b in baselines], rotation=45, ha='right')
    axes[1].set_ylabel("Energy (Log Scale)")

    # Payload
    axes[2].bar(x, b_payload, color=colors, edgecolor='black')
    axes[2].set_title("(c) Comm. Cost (Payload KB)")
    axes[2].set_yscale('log')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([labels_map[b] for b in baselines], rotation=45, ha='right')
    axes[2].set_ylabel("Payload (KB)")

    plt.tight_layout()
    plt.savefig("results/ablation/fig_ablation_bars.png", dpi=150)
    plt.close()

    # ==========================================
    # 2. CONVERGENCE LINE CHART
    # ==========================================
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, b in enumerate(baselines):
        histories = metrics_data[b].get("mAP_history", [])
        if not histories: continue
        min_len = min(len(h) for h in histories)
        mat = np.array([h[:min_len] for h in histories])
        mean_map = np.mean(mat, axis=0)
        rounds = np.arange(1, min_len + 1)
        ax.plot(rounds, mean_map, label=labels_map[b], color=colors[i], marker='o', linewidth=2)
        
    ax.set_title("Ablation: Convergence Behaviour")
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("mAP@50-95")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/ablation/fig_ablation_convergence.png", dpi=150)
    plt.close()

    # ==========================================
    # 3. RADAR CHART
    # ==========================================
    # Normalize metrics: 0 = worst, 1 = best
    def normalize_max(arr, higher_is_better=True):
        arr = np.array(arr)
        if len(arr) == 0 or np.max(arr) == np.min(arr): return np.zeros_like(arr)
        if higher_is_better:
            return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        else:
            return (np.max(arr) - arr) / (np.max(arr) - np.min(arr))

    norm_map = normalize_max(b_map, True)
    norm_energy = normalize_max(b_energy, False)
    norm_payload = normalize_max(b_payload, False)
    norm_latency = normalize_max(b_latency, False)

    categories = ['Accuracy', 'Energy Efficiency', 'Payload Efficiency', 'Latency Efficiency']
    N_cat = len(categories)
    angles = [n / float(N_cat) * 2 * np.pi for n in range(N_cat)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)

    for i, b in enumerate(baselines):
        # Chọn các kịch bản tiêu biểu để vẽ radar (vẽ nhiều quá sẽ rối)
        if b not in ["fedkdl", "lora_head_kd_noint8", "head_kd_int8_nolora", "lora_head_int8_nokd", "fedavg"]:
            continue
            
        idx = baselines.index(b)
        vals = [norm_map[idx], norm_energy[idx], norm_payload[idx], norm_latency[idx]]
        create_radar_chart(ax, angles, vals, labels_map[b], colors[i])

    ax.set_title("FedKDL Ablation Radar Chart", size=15, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig("results/ablation/fig_ablation_radar.png", dpi=150)
    plt.close()

    print("Đã lưu các biểu đồ Ablation (Bars, Convergence, Radar) vào thư mục results/ablation/")

if __name__ == "__main__":
    plot_ablation()
