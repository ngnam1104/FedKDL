import matplotlib.pyplot as plt
import numpy as np

from plot_common import LABELS, load_summary, plot_mean_std, save, setup_style

setup_style()
data = load_summary()
methods = ["fedavg_hfl", "naive_lora", "flora", "topk_grad", "fedkdl"]

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

# Subplot 1: Joint Cost vs N_AUV
plot_mean_std(axes[0], data, methods, "joint_cost")
axes[0].set(xlabel="Number of AUVs", ylabel="Joint Objective Cost")
axes[0].set_yscale("log") # <--- Added log scale to fix crushed lines
axes[0].legend()

# Subplot 2: Energy/Latency Breakdown
rows = data[(data["N_AUV"] == 60) & data["baseline"].isin(methods)].set_index("baseline")
from config.settings import fed_cfg
latency = np.array([rows.loc[m, "tau_round_s_mean"] for m in methods]) * fed_cfg.LAMBDA_TAU
energy = np.array([rows.loc[m, "e_total_j_mean"] for m in methods]) * fed_cfg.LAMBDA_E
x = np.arange(len(methods))

axes[1].bar(x - 0.2, latency, 0.4, label=r"$\lambda_\tau \tau$ (Latency)")
axes[1].bar(x + 0.2, energy, 0.4, label=r"$\lambda_E E$ (Energy)")
axes[1].set_yscale("log") # <--- Added log scale here as well since fedavg_hfl dominates
axes[1].set_xticks(x, [LABELS[m] for m in methods], rotation=15)
axes[1].set_xlabel("Baseline Method")
axes[1].set_ylabel("Cost Contribution at N=60")
axes[1].legend()

save(fig, "K2_joint_cost")
