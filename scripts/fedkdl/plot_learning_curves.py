import os
import matplotlib.pyplot as plt
import pandas as pd

in_dir = "results/mock_learning_curves"
out_dir = "results/mock_learning_curves/plots"
os.makedirs(out_dir, exist_ok=True)

rq_groups = {
    "RQ1": ["fedkdl", "fedavg", "fedprox"],
    "RQ2": ["topk_grad", "fedkdl", "flora", "fedavg_hfl"],
    "RQ3": ["fedkdl_selective", "fedkdl", "fedkdl_nocoop", "scaffold", "flora", "fedavg_hfl"],
    "RQ4": ["centralized", "fedkdl", "logit_kd", "fedkdl_proxy_ft", "fedkdl_nokd"],
    "Ref": ["fedkd", "fedkdl_nolora", "fedprox_kdl", "fedprox_hfl", "naive_lora"]
}

# Các kiểu đường và marker để in đen trắng
line_styles = ['-', '--', '-.', ':', '-', '--']
markers = ['o', 's', '^', 'D', 'v', 'X']

for rq_name, baselines in rq_groups.items():
    # Tăng độ rộng khung hình (từ 14x5 lên 18x6)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    for idx, baseline in enumerate(baselines):
        csv_path = os.path.join(in_dir, f"results_{baseline}.csv")
        if not os.path.exists(csv_path):
            print(f"Skipping missing file: {csv_path}")
            continue
            
        df = pd.read_csv(csv_path)
        epochs = df['epoch']
        map_50 = df['metrics/mAP50(B)']
        loss = df['train/box_loss']
        
        ls = line_styles[idx % len(line_styles)]
        mk = markers[idx % len(markers)]
        
        ax1.plot(epochs, map_50, label=baseline, linewidth=2, linestyle=ls, marker=mk, markevery=5, markersize=6)
        ax2.plot(epochs, loss, label=baseline, linewidth=2, linestyle=ls, marker=mk, markevery=5, markersize=6)
        
    # Formatting mAP Plot
    ax1.set_title(f"{rq_name} - mAP@0.5 Convergence")
    ax1.set_xlabel("FL Round")
    ax1.set_ylabel("mAP@0.5")
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Formatting Loss Plot
    ax2.set_title(f"{rq_name} - Training Box Loss")
    ax2.set_xlabel("FL Round")
    ax2.set_ylabel("Loss")
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    out_file = os.path.join(out_dir, f"plot_{rq_name}.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    
print(f"✅ Đã vẽ xong biểu đồ cho {len(rq_groups)} nhóm RQ!")
print(f"📂 Đã lưu tại: {out_dir}/")
