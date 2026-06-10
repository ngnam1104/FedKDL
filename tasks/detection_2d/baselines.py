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
    logit_kd_only: bool = False

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
    'flora':            BaselineConfig(full_param=False,  use_lora=True,  use_int8=False, use_gateway_kd=False),
    'scaffold':         BaselineConfig(full_param=True,  use_lora=False, use_int8=False, use_gateway_kd=False, scaffold=True),
    'topk_grad':        BaselineConfig(full_param=True,  use_lora=False, use_int8=False, use_gateway_kd=False, topk_grad=True),

    # ── Primary: FedKDL family (HFL + LoRA INT8 + Gateway KD) ─────────────
    'fedkdl':           BaselineConfig(full_param=False, use_lora=True,  use_int8=True,  use_gateway_kd=True,  coop_rule='nearest'),
    'fedkdl_selective': BaselineConfig(full_param=False, use_lora=True,  use_int8=True,  use_gateway_kd=True,  coop_rule='selective'),
    'fedkdl_nocoop':    BaselineConfig(full_param=False, use_lora=True,  use_int8=True,  use_gateway_kd=True,  coop_rule='nocoop'),

    # ── RQ4: KD ablation ──────────────────────────────────────────────────
    'fedkdl_nokd':      BaselineConfig(full_param=False, use_lora=True,  use_int8=True,  use_gateway_kd=False, coop_rule='nearest'),
    'logit_kd':         BaselineConfig(full_param=False, use_lora=True,  use_int8=True,  use_gateway_kd=True,  coop_rule='nearest', logit_kd_only=True),

    # ── Ablation: component removal ────────────────────────────────────────
    'fedkdl_nolora':    BaselineConfig(full_param=True,  use_lora=False, use_int8=False, use_gateway_kd=True,  coop_rule='nearest'),
    'fedkdl_proxy_ft':  BaselineConfig(full_param=False, use_lora=True,  use_int8=True,  use_gateway_kd=False, use_gateway_proxy_ft=True, coop_rule='nearest'),
    'fedprox_kdl':      BaselineConfig(full_param=False, use_lora=True,  use_int8=True,  use_gateway_kd=True,  coop_rule='nearest', fedprox=True),

    # ── Reference: Flat + KD ──────────────────────────────────────────────
    'fedkd':            BaselineConfig(full_param=True,  use_lora=False, use_int8=False, use_gateway_kd=True,  hfl=False, local_kd=True),
    'centralized':      BaselineConfig(full_param=False, use_lora=True,  use_int8=False, use_gateway_kd=False, hfl=False),
}
# fmt: on


def parse_baseline_config(baseline: str) -> BaselineConfig:
    """Return the 2D experiment configuration for each baseline."""
    return BASELINE_CONFIGS.get(baseline, BASELINE_CONFIGS['fedkdl'])
