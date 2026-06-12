import matplotlib.pyplot as plt
import numpy as np

from plot_common import LABELS, load_learning, load_summary, save, setup_style

setup_style()
methods = ["fedkdl_nocoop", "fedkdl_selective", "fedkdl"]
physics = load_summary()
physics = physics[physics["N_AUV"] == 30].set_index("baseline")

joint_cost = np.array([physics.loc[method, "joint_cost_mean"] for method in methods])
accuracy = np.array([load_learning(method)["map50"].max() for method in methods])

x = np.arange(len(methods))
width = 0.36

fig, ax = plt.subplots(figsize=(7.5, 4.5))
accuracy_axis = ax.twinx()

cost_bars = ax.bar(
    x - width / 2,
    joint_cost,
    width,
    color="#7A9CC6",
    edgecolor="black",
    label="Joint Objective Cost",
)
accuracy_bars = accuracy_axis.bar(
    x + width / 2,
    accuracy,
    width,
    color="#E8A15B",
    edgecolor="black",
    label="Best mAP@0.5",
)

ax.set_xticks(x, [LABELS[method] for method in methods])
ax.set_xlabel("Relay Cooperation Strategy")
ax.set_ylabel("Joint Objective Cost")
accuracy_axis.set_ylabel("Mean Average Precision")

ax.set_ylim(0, 45)
accuracy_axis.set_ylim(0, 0.75)

# Add legend
ax.legend(
    [cost_bars, accuracy_bars],
    ["Joint Objective Cost", "Best mAP@0.5"],
    loc="upper center",
    ncol=2,
)

save(fig, "K3_rq3_ablation")
