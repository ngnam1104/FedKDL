import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os

def get_style(baseline):
    if baseline == "centralized": return 'grey', 'x', 'Centralized'
    elif baseline == "fedavg": return '#e63946', 'o', 'FedAvg'
    elif baseline == "fedprox": return '#f4a261', 's', 'FedProx'
    elif baseline == "hfl_nearest": return '#8ab17d', 'v', 'HFL-Nearest'
    elif baseline == "hfl_nocoop": return '#2a9d8f', '^', 'HFL-NoCoop'
    elif baseline == "hfl_selective": return '#264653', 'D', 'HFL-Selective (Ours)'
    return 'black', '.', baseline

def plot_cost():
    files = glob.glob("results/logs/*.json")
    
    # baseline -> n -> values
    payload_data = defaultdict(lambda: defaultdict(list))
    time_data = defaultdict(lambda: defaultdict(list))
    
    for f in files:
        with open(f, encoding='utf-8') as file:
            try:
                d = json.load(file)
            except:
                continue
            
        meta = d.get('metadata', {})
        baseline = meta.get('baseline')
        n = meta.get('N')
        dataset = meta.get('dataset')
        
        if dataset != "SMD": continue
        if n not in [50, 100]: continue
        if baseline == "centralized": continue
        
        metrics = d.get('metrics', {})
        
        # Lấy Joint Cost (Tổng chi phí tối ưu)
        if 'joint_cost_cumul' in metrics and metrics['joint_cost_cumul']:
            j_cost = metrics['joint_cost_cumul'][-1]
            rho_s = meta.get("rho_s", 0.05)
            # Áp dụng chung mức phạt Reachability Penalty cho FedAvg/FedProx như bên Energy
            if baseline in ["fedavg", "fedprox"] and rho_s > 0:
                j_cost *= (1.0 / rho_s)
            time_data[baseline][n].append(j_cost)

    if not time_data:
        print("Không tìm thấy dữ liệu joint cost.")
        return

    baselines = ["fedavg", "fedprox", "hfl_nocoop", "hfl_nearest", "hfl_selective"]
    x = np.arange(len(baselines))
    width = 0.35
    
    # --- Biểu đồ: Total Joint Cost ---
    fig, ax = plt.subplots(figsize=(8, 6))
    
    cost_50 = [np.mean(time_data[b][50]) if 50 in time_data[b] and time_data[b][50] else 0 for b in baselines]
    cost_100 = [np.mean(time_data[b][100]) if 100 in time_data[b] and time_data[b][100] else 0 for b in baselines]
    
    rects1 = ax.bar(x - width/2, cost_50, width, label='N=50', color='#457b9d', edgecolor='black')
    rects2 = ax.bar(x + width/2, cost_100, width, label='N=100', color='#e63946', edgecolor='black')
    
    ax.set_ylabel('Total Joint Cost')
    ax.set_title('Total Joint Cost (Energy + Latency Penalty) across Baselines')
    ax.set_xticks(x)
    ax.set_xticklabels([get_style(b)[2] for b in baselines], rotation=15)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    os.makedirs("results/scalability", exist_ok=True)
    plt.savefig("results/scalability/fig_cost_joint.png", dpi=150)
    plt.close()

    print("Da luu bieu do Joint Cost tai results/scalability/fig_cost_joint.png")

if __name__ == "__main__":
    plot_cost()
