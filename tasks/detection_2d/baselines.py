from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineConfig:
    full_param: bool
    use_lora: bool
    use_int8: bool
    use_gateway_kd: bool
    use_gateway_proxy_ft: bool = False
    local_kd: bool = False
    topk_grad: bool = False
    fedprox: bool = False
    hfl: bool = True
    coop_rule: str = 'nocoop'
    scaffold: bool = False
    # KD ablation controls
    logit_kd_only: bool = False
    logit_box_kd_only: bool = False
    logit_proj_kd_only: bool = False
    lora_aggregation: str = 'svd'
    server_mix: bool = False

    def __post_init__(self) -> None:
        if self.lora_aggregation not in {'svd', 'naive'}:
            raise ValueError(f"Unsupported LoRA aggregation: {self.lora_aggregation}")
        if self.full_param and self.use_lora:
            raise ValueError("A baseline cannot enable full-parameter and LoRA training together")

    @property
    def coop(self) -> bool:
        return self.coop_rule != 'nocoop'


# fmt: off
BASELINE_CONFIGS = {
    # ── RQ1: Flat baselines ────────────────────────────────────────────────
    'fedavg':           BaselineConfig(full_param=True,  use_lora=False, use_int8=False, use_gateway_kd=False, hfl=False),
    'fedprox':          BaselineConfig(full_param=True,  use_lora=False, use_int8=False, use_gateway_kd=False, hfl=False, fedprox=True),

    # ── RQ2/3: HFL baselines (no KD) ──────────────────────────────────────
    'fedavg_hfl':       BaselineConfig(full_param=True,  use_lora=False, use_int8=False, use_gateway_kd=False),
    'fedprox_hfl':      BaselineConfig(full_param=True,  use_lora=False, use_int8=False, use_gateway_kd=False, fedprox=True),
    'flora':            BaselineConfig(full_param=False,  use_lora=True,  use_int8=False, use_gateway_kd=False, lora_aggregation='svd'),
    'naive_lora':       BaselineConfig(full_param=False,  use_lora=True,  use_int8=False, use_gateway_kd=False, lora_aggregation='naive'),
    'scaffold':         BaselineConfig(full_param=True,  use_lora=False, use_int8=False, use_gateway_kd=False, scaffold=True),
    'topk_grad':        BaselineConfig(full_param=True,  use_lora=False, use_int8=False, use_gateway_kd=False, topk_grad=True),

    # ── Primary: FedKDL family (HFL + LoRA INT8 + Gateway KD) ─────────────
    'fedkdl':           BaselineConfig(full_param=False, use_lora=True,  use_int8=True,  use_gateway_kd=True,  coop_rule='nearest', server_mix=True),
    'fedkdl_selective': BaselineConfig(full_param=False, use_lora=True,  use_int8=True,  use_gateway_kd=True,  coop_rule='selective', server_mix=True),
    'fedkdl_nocoop':    BaselineConfig(full_param=False, use_lora=True,  use_int8=True,  use_gateway_kd=True,  coop_rule='nocoop', server_mix=True),

    # ── RQ4: KD ablation ──────────────────────────────────────────────────
    'fedkdl_nokd':      BaselineConfig(full_param=False, use_lora=True,  use_int8=True,  use_gateway_kd=False, coop_rule='nearest', server_mix=True),
    'logit_kd':         BaselineConfig(full_param=False, use_lora=True,  use_int8=True,  use_gateway_kd=True,  coop_rule='nearest', logit_kd_only=True, server_mix=True),
    'logit_box_kd':     BaselineConfig(full_param=False, use_lora=True,  use_int8=True,  use_gateway_kd=True,  coop_rule='nearest', logit_box_kd_only=True, server_mix=True),
    'logit_proj_kd':    BaselineConfig(full_param=False, use_lora=True,  use_int8=True,  use_gateway_kd=True,  coop_rule='nearest', logit_proj_kd_only=True, server_mix=True),

    # ── Ablation: component removal ────────────────────────────────────────
    'fedkdl_32bit':    BaselineConfig(full_param=True,  use_lora=False, use_int8=False, use_gateway_kd=True,  coop_rule='nearest', server_mix=True),
    'fedkdl_proxy_ft':  BaselineConfig(full_param=False, use_lora=True,  use_int8=True,  use_gateway_kd=False, use_gateway_proxy_ft=True, coop_rule='nearest', server_mix=True),
    'fedprox_kdl':      BaselineConfig(full_param=False, use_lora=True,  use_int8=True,  use_gateway_kd=True,  coop_rule='nearest', fedprox=True, server_mix=True),

    # ── Reference: Flat + KD ──────────────────────────────────────────────
    'fedkd':            BaselineConfig(full_param=True,  use_lora=False, use_int8=False, use_gateway_kd=True,  hfl=False),
    'centralized':      BaselineConfig(full_param=False, use_lora=True,  use_int8=False, use_gateway_kd=False, hfl=False),
}
# fmt: on

STANDARD_BASELINES = (
    'fedkdl',
    'fedavg',
    'fedprox',
    'fedavg_hfl',
    'topk_grad',
    'flora',
    'naive_lora',
    'scaffold',
    'fedkdl_nocoop',
    'fedkdl_selective',
    'fedkdl_nokd',
    'fedkdl_proxy_ft',
    'logit_kd',
    'logit_box_kd',
    'logit_proj_kd',
    'centralized',
    'fedprox_kdl',
    'fedkdl_32bit',
    'fedkd',
    'fedprox_hfl',
)

OPTIONAL_BASELINES = ()


def parse_baseline_config(baseline: str) -> BaselineConfig:
    """Return the 2D experiment configuration for each baseline."""
    try:
        return BASELINE_CONFIGS[baseline]
    except KeyError as exc:
        known = ', '.join(BASELINE_CONFIGS)
        raise ValueError(f"Unknown 2D baseline '{baseline}'. Expected one of: {known}") from exc
