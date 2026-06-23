from plot_common import plot_learning_panels

plot_learning_panels(
    ["fedkdl_nokd", "logit_kd", "fedkdl_proxy_ft", "fedkdl", "centralized"],
    "K4_rq4_learning",
    "Gateway knowledge distillation",
)
