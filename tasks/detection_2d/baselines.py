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


BASELINE_CONFIGS = {
    'fedavg': BaselineConfig(True, False, False, False, hfl=False),
    'fedprox': BaselineConfig(True, False, False, False, fedprox=True, hfl=False),
    'fedavg_hfl': BaselineConfig(True, False, False, False),
    'fedprox_hfl': BaselineConfig(True, False, False, False, fedprox=True),
    'flora': BaselineConfig(False, True, False, False),
    'scaffold': BaselineConfig(True, False, False, False, scaffold=True),
    'fedkdl': BaselineConfig(False, True, True, True, coop_rule='nearest'),
    'fedkdl_selective': BaselineConfig(False, True, True, True, coop_rule='selective'),
    'fedkdl_nocoop': BaselineConfig(False, True, True, True),
    'logit_kd': BaselineConfig(False, True, True, True, coop_rule='nearest', logit_kd_only=True),
    'fedprox_kdl': BaselineConfig(False, True, True, True, fedprox=True, coop_rule='nearest'),
    'fedkd': BaselineConfig(True, False, False, True, local_kd=True, hfl=False),
    'topk_grad': BaselineConfig(True, False, False, False, topk_grad=True),
    'centralized': BaselineConfig(False, True, False, False, hfl=False),
    'fedkdl_nokd': BaselineConfig(False, True, True, False, coop_rule='nearest'),
    'fedkdl_nolora': BaselineConfig(True, False, False, True, coop_rule='nearest'),
    'fedkdl_proxy_ft': BaselineConfig(False, True, True, False, use_gateway_proxy_ft=True, coop_rule='nearest'),
}


def parse_baseline_config(baseline: str) -> BaselineConfig:
    """Return the 2D experiment configuration for each baseline."""
    return BASELINE_CONFIGS.get(baseline, BASELINE_CONFIGS['fedkdl'])
