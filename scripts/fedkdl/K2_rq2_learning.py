from plot_common import plot_learning_panels

plot_learning_panels(
    ["fedavg_hfl", "naive_lora", "flora", "topk_grad", "fedkdl"],
    "K2_rq2_learning",
    "Communication-efficient methods",
)
