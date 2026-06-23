# Finalized Baseline Metrics

This folder contains baselines that are considered finished for the current
N=30, M=8, alpha=1.0, seed=1109 experiment set. New run scripts should not
schedule these baselines again unless explicitly re-running final tables.

Finalized baselines:

- `fedavg`
- `fedprox`
- `fedavg_hfl`
- `fedprox_hfl`
- `centralized`
- `topk_grad`
- `flora`
- `fedkdl_nokd`
- `scaffold`

Pending groups are scheduled by:

- `run_s1_proxy_proj_prox.sh`
- `run_s2_logit_box_nocoop.sh`
- `run_s3_fedkdl_family.sh`
