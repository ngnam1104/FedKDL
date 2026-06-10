# Baseline Logic Audit

Audit date: 2026-06-10. The experiment contract is `experiment_design.md`;
the method details are cross-checked against `.docs/FedKDL-vi.tex`.

## Baseline Matrix

| ID | Runtime logic | Audit |
|---|---|---|
| `fedavg` | Flat, full parameter, Float32, FedAvg, no KD | Matches RQ1 |
| `fedprox` | Flat FedAvg plus proximal gradient term | Matches RQ1 |
| `fedavg_hfl` | HFL, full parameter, no relay cooperation, no KD | Matches the HFL control in RQ2/RQ3 |
| `fedprox_hfl` | HFL full parameter plus FedProx, no cooperation | Valid extra reference |
| `topk_grad` | HFL full-parameter local training; Top-K model-delta upload with persistent error feedback | Valid Top-K update compression; it is not per-minibatch gradient sparsification |
| `flora` | HFL LoRA+Head, Float32, SVD-LoRA aggregation, no cooperation/KD | Now correctly represents SVD-LoRA without INT8 compression. |
| `naive_lora` | HFL LoRA+Head, Float32, independent weighted averaging of LoRA A/B | Matches the paper's naive-LoRA control. |
| `scaffold` | HFL full parameter; client/server control variates; no cooperation/KD | Matches SCAFFOLD structure; control deltas are averaged by participating client count |
| `fedkdl` | HFL LoRA+Head, INT8, SVD-LoRA, nearest feasible cooperation, gateway projection KD | Matches the proposed pipeline |
| `fedkdl_nocoop` | `fedkdl` without relay cooperation | Matches RQ3 ablation |
| `fedkdl_selective` | `fedkdl` with selective Q1-distance cooperation | Valid cooperation ablation |
| `fedkdl_nokd` | `fedkdl` without gateway KD | Matches RQ4 No-KD |
| `logit_kd` | `fedkdl` topology/compression with logit-only soft-target KD | Matches Logit-KD intent; implementation uses foreground-masked sigmoid BCE, not softmax KL |
| `centralized` | Centralized LoRA+Head training on the full dataset using the custom differential-LR trainer | Matches the configured centralized upper bound |
| `fedprox_kdl` | `fedkdl` plus local FedProx term | Valid component ablation |
| `fedkdl_nolora` | HFL full-parameter Float32 plus gateway KD | Valid no-LoRA ablation; LoRA projection loss is naturally zero, leaving logit/box KD |
| `fedkd` | Flat full-parameter FL plus the repository's gateway KD | Internally consistent, but needs the intended FedKD paper to verify that this is the canonical algorithm |

`fedkdl_proxy_ft` is an optional seventeenth experiment, not part of the
standard 16-baseline suite.

## Important Interpretation

Physical metrics are deterministic for a fixed baseline, topology, seed and
configuration, because there is no learned resource scheduler. They are not
necessarily constant over rounds: AUV mobility changes feasible links and
clusters, batteries can deplete, participation can change, and KD is periodic.

Therefore:

- Use curves for YOLO metrics: training loss, validation loss, mAP50,
  mAP50-95, precision and recall.
- Use tables or grouped bars for payload, energy and latency summaries.
- Use round curves for physical metrics only when demonstrating mobility,
  battery depletion, participation loss or constraint violations.
- Aggregate multiple seeds with mean and standard deviation. A single seed
  supports a case study, not a robustness claim.

## Plot Plan

### RQ1: Connectivity And Stability

- Two-panel curve: participation rate and mAP50 versus round.
- Secondary loss curve for FedAvg, FedProx and FedKDL.
- Summary table: mean participation, isolated-AUV rounds, final mAP50,
  cumulative energy, mean/max round latency and number of `TAU_MAX` violations.

### RQ2: Communication Compression

- Grouped bars: average AUV payload, relay payload, cumulative energy and
  mean/max latency.
- mAP50 and mAP50-95 convergence curves.
- Pareto scatter: final mAP50 versus average payload, with energy encoded by
  marker size. This communicates the compression/accuracy trade-off directly.

### RQ3: Non-IID And Relay Cooperation

- Run an alpha grid such as `0.1 0.5 1.0 10000.0` with at least three seeds.
- Plot final mAP50 versus Dirichlet alpha with mean and standard-deviation band.
- Plot per-round mAP50/loss for one representative strong Non-IID setting.
- Add a grouped bar for final mAP50 comparing no cooperation, selective
  cooperation and nearest cooperation.

### RQ4: Gateway KD

- Two-panel convergence plot: loss and mAP50 versus round.
- Grouped bars: final/best mAP50, mAP50-95 and rounds-to-threshold.
- Optional KD diagnostic plot: logit, box and LoRA-projection loss components.
- Show centralized training as an upper-bound reference, not as a federated
  communication competitor.

## Context Still Needed

Please provide the intended references for:

1. FLORA, if it means a specific published algorithm rather than the paper's
   naive-LoRA baseline.
2. FedKD, because the current ID means flat full-parameter FL plus gateway KD.
3. Top-K, if the experiment must reproduce per-minibatch gradient
   sparsification rather than Top-K model-update compression.
4. Relay cooperation policy. The paper prose says nearest feasible neighbor,
   while its parameter table mentions a cluster-size threshold. The current
   `nearest` mode chooses the nearest feasible relay with a larger active
   cluster; `selective` additionally applies the under-populated-cluster and
   Q1-distance filters.

The paper and `experiment_design.md` also disagree on the KD competitors:
`.docs/FedKDL-vi.tex` includes Feature-KD, while `experiment_design.md` uses
Centralized training. The current 16-baseline suite follows
`experiment_design.md`; Feature-KD is not implemented.

The paper's flat-topology scenario also lists flat SCAFFOLD, whereas
`experiment_design.md` places SCAFFOLD in hierarchical RQ3. The current suite
again follows `experiment_design.md`.

One fairness choice still needs confirmation: LoRA methods start from
`yolo12n_warmup.pt`, while full-parameter/no-LoRA methods start from
`yolo12n.pt` to avoid loading LoRA modules into a no-LoRA model. If warmup is
intended as common pretraining rather than a FedKDL component, save a baked
warmup checkpoint and initialize every baseline from that same effective
weight state.
