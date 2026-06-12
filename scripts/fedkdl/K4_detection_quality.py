import matplotlib.pyplot as plt
import numpy as np

from plot_common import LABELS, load_learning, save, setup_style

setup_style()
methods = ["fedkdl_nokd", "fedkdl_proxy_ft", "logit_kd", "fedkdl", "centralized"]
values = []
for method in methods:
    frame = load_learning(method)
    best = frame.loc[frame["map50"].idxmax()]
    values.append((best["map50"], best["map5095"]))
x = np.arange(len(methods))
width = 0.36
fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.bar(x - width / 2, [v[0] for v in values], width, label="mAP@0.5")
ax.bar(x + width / 2, [v[1] for v in values], width, label="mAP@0.5:0.95")
ax.set_xticks(x, [LABELS[m] for m in methods], rotation=15)
ax.set_xlabel("Baseline Method")
ax.set_ylabel("Detection Quality (mAP)")
ax.legend()
save(fig, "K4_detection_quality")
