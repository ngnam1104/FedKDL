import matplotlib.pyplot as plt
import numpy as np

from plot_common import LABELS, load_learning, load_summary, save, setup_style

setup_style()
data = load_summary()
methods = ["fedavg_hfl", "naive_lora", "flora", "topk_grad", "fedkdl"]
rows = data[(data["N_AUV"] == 30) & data["baseline"].isin(methods)].set_index("baseline")

# Convert KB to MB
payload_mb = np.array([rows.loc[m, "payload_per_auv_kb_mean"] for m in methods]) / 1024.0

# Load best mAP@0.5 from learning curves
map_scores = np.array([load_learning(m)["map50"].max() for m in methods])

x = np.arange(len(methods))
width = 0.36

fig, ax1 = plt.subplots(figsize=(8.5, 4.5))
ax2 = ax1.twinx()

bar1 = ax1.bar(
    x - width / 2,
    payload_mb,
    width,
    color="#7A9CC6",
    edgecolor="black",
    label="Payload per AUV (MB)",
)

bar2 = ax2.bar(
    x + width / 2,
    map_scores,
    width,
    color="#E8A15B",
    edgecolor="black",
    label="Best mAP@0.5",
)

ax1.set_xticks(x, [LABELS.get(m, m) for m in methods])
ax1.set_xlabel("Compression / PEFT Method")
ax1.set_ylabel("Transmission Payload per AUV (MB)")
ax2.set_ylabel("Mean Average Precision (mAP@0.5)")

# Set limits
ax1.set_ylim(0, 12)
ax2.set_ylim(0.4, 0.75)

# Add value labels on top of payload bars
for i, v in enumerate(payload_mb):
    ax1.text(x[i] - width / 2, v + 0.2, f"{v:.2f}", ha='center', va='bottom', fontsize=9, rotation=90)

# Add legends
ax1.legend(
    [bar1, bar2],
    ["Payload per AUV (MB)", "Best mAP@0.5"],
    loc="upper center",
    bbox_to_anchor=(0.5, 1.15),
    ncol=2,
)

save(fig, "K2_payload_comparison")
