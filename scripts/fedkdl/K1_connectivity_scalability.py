import matplotlib.pyplot as plt
import numpy as np

from plot_common import (
    LABELS,
    T,
    load_learning,
    load_summary,
    plot_mean_std,
    save,
    setup_style,
)

setup_style()
data = load_summary()
methods = ["fedavg", "fedkdl"]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
for index, method in enumerate(methods):
    rows = data[data["baseline"] == method].sort_values("N_AUV")
    x_values = rows["N_AUV"].to_numpy()
    rate = 100.0 * rows["participation_rate_mean"].to_numpy()
    rate_std = 100.0 * rows["participation_rate_std"].fillna(0).to_numpy()
    participants = rows["participants_mean"].to_numpy()
    participants_std = rows["participants_std"].fillna(0).to_numpy()
    label = LABELS[method]
    color = ("#C44E52", "#2A9D8F")[index]
    marker = ("o", "s")[index]
    axes[0].plot(x_values, rate, label=label, color=color, marker=marker, linewidth=2)
    axes[0].fill_between(x_values, rate - rate_std, rate + rate_std, color=color, alpha=0.14)
    axes[1].plot(
        x_values, participants, label=label, color=color, marker=marker, linewidth=2
    )
    axes[1].fill_between(
        x_values,
        participants - participants_std,
        participants + participants_std,
        color=color,
        alpha=0.14,
    )
axes[0].set(xlabel=T("Number of AUVs"), ylabel=T("Participation Rate (%)"), ylim=(0, 108))
axes[1].set(xlabel=T("Number of AUVs"), ylabel=T("Participating AUVs"))
axes[0].legend()
axes[1].legend()


save(fig, "K1_connectivity_scalability")
