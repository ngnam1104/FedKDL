"""
Deterministic dummy tests for the 2D FedKDL algorithm family.

The real Simulator2D path needs YOLO checkpoints, URPC YAML files, and a GPU.
This file instead validates the algorithmic data path with tiny tensors:

  AUV local update -> optional compression -> relay aggregation/cooperation
  -> gateway aggregation -> optional KD/proxy/scaffold metadata handling.

Each baseline gets explicit numeric expected outputs. The test is intentionally
small enough to run in CI or on a login node before launching expensive runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import torch

from federated_core.aggregator import fedavg_global, svd_lora_aggregate
from federated_core.hfl_rules import blend_state_dicts
from tasks.detection_2d.baselines import BASELINE_CONFIGS, BaselineConfig
from tasks.detection_2d.knowledge_compression.int8_quantization import pack_payload, unpack_payload
from tasks.detection_2d.knowledge_compression.topk_sparsification import (
    SparseFloatPayload,
    TopKCompressor,
    flatten_state_dict,
    unflatten_state_dict,
)


@dataclass
class DummyResult:
    head: torch.Tensor
    lora_product: torch.Tensor | None
    payload_kb: float
    map50: float
    scaffold_c: torch.Tensor | None = None


def assert_close(actual: torch.Tensor, expected: torch.Tensor, name: str, atol: float = 1e-4) -> None:
    if not torch.allclose(actual.float(), expected.float(), atol=atol, rtol=0):
        raise AssertionError(f"{name}: expected {expected.tolist()}, got {actual.tolist()}")


def assert_scalar(actual: float, expected: float, name: str, atol: float = 1e-6) -> None:
    if not math.isclose(float(actual), float(expected), abs_tol=atol, rel_tol=0):
        raise AssertionError(f"{name}: expected {expected}, got {actual}")


def make_global_state(use_lora: bool, full_param: bool = False) -> Dict[str, torch.Tensor]:
    if use_lora:
        return {
            "model.10.conv.lora_B": torch.tensor([[1.0], [2.0]]),
            "model.10.conv.lora_A": torch.tensor([[3.0, 4.0]]),
            "model.21.head.weight": torch.tensor([10.0, 20.0]),
        }
    return {
        "model.0.weight": torch.tensor([1.0, 2.0]) if full_param else torch.tensor([10.0, 20.0]),
        "model.21.head.weight": torch.tensor([10.0, 20.0]),
    }


def has_kd(cfg: BaselineConfig) -> bool:
    return cfg.use_gateway_kd or cfg.local_kd


def local_update(global_state: Dict[str, torch.Tensor], client_id: int, cfg: BaselineConfig) -> Dict[str, torch.Tensor]:
    """Return deterministic local states with simple arithmetic expectations."""
    out: Dict[str, torch.Tensor] = {}
    for key, value in global_state.items():
        if "lora_B" in key:
            out[key] = value * (1.0 + 0.1 * client_id)
        elif "lora_A" in key:
            out[key] = value.clone()
        elif "head" in key:
            delta = float(client_id)
            if cfg.fedprox:
                delta *= 0.5
            out[key] = value + delta
        else:
            delta = 0.2 * float(client_id)
            if cfg.fedprox:
                delta *= 0.5
            out[key] = value + delta

    if cfg.scaffold:
        out["__scaffold_delta_c__"] = {
            key: torch.ones_like(value) * (0.01 * client_id)
            for key, value in global_state.items()
        }
    return out


def maybe_pack_roundtrip(state: Dict[str, torch.Tensor], template: Dict[str, torch.Tensor], use_int8: bool) -> tuple[Dict[str, torch.Tensor], float]:
    if not use_int8:
        params = sum(v.numel() for v in state.values() if torch.is_tensor(v))
        return state, params * 4 / 1024.0
    payload, kb = pack_payload(state)
    return unpack_payload(payload, template), kb


def topk_roundtrip(global_state: Dict[str, torch.Tensor], state: Dict[str, torch.Tensor]) -> tuple[Dict[str, torch.Tensor], float]:
    delta = {k: state[k] - global_state[k] for k in global_state}
    flat, shapes = flatten_state_dict(delta)
    compressor = TopKCompressor(total_params=flat.numel(), rho_s=0.5)
    idx, values = compressor.compress(flat)
    payload = SparseFloatPayload(idx, values, total_params=flat.numel(), shapes=shapes)
    recovered_delta = unflatten_state_dict(payload.decompress(), shapes)
    recovered = {k: global_state[k] + recovered_delta[k] for k in global_state}
    return recovered, payload.payload_bytes / 1024.0


def aggregate_cluster(states: List[Dict[str, torch.Tensor]], samples: List[int]) -> Dict[str, torch.Tensor]:
    return svd_lora_aggregate(states, samples)


def run_dummy_pipeline(baseline: str) -> DummyResult:
    cfg = BASELINE_CONFIGS[baseline]
    torch.manual_seed(0)
    global_state = make_global_state(cfg.use_lora, full_param=cfg.full_param)

    client_samples = [1, 3, 2, 4]
    client_states = []
    payload_kb = 0.0
    for client_id in range(1, 5):
        state = local_update(global_state, client_id, cfg)
        if cfg.topk_grad:
            state, kb = topk_roundtrip(global_state, state)
        else:
            state, kb = maybe_pack_roundtrip(state, global_state, cfg.use_int8)
        payload_kb += kb / 4.0
        client_states.append(state)

    if not cfg.hfl:
        gateway_state = aggregate_cluster(client_states, client_samples)
    else:
        relay0 = aggregate_cluster(client_states[:2], client_samples[:2])
        relay1 = aggregate_cluster(client_states[2:], client_samples[2:])
        if cfg.coop:
            alpha = 0.8 if cfg.coop_rule == 'selective' else 0.7
            relay0 = blend_state_dicts(relay0, relay1, alpha=alpha)
        gateway_state = fedavg_global([relay0, relay1], [sum(client_samples[:2]), sum(client_samples[2:])])

    map50 = 0.50
    if has_kd(cfg):
        map50 += 0.10
    if cfg.use_gateway_proxy_ft:
        map50 += 0.05
    if cfg.fedprox:
        map50 -= 0.02
    if cfg.topk_grad:
        map50 -= 0.03

    lora_product = None
    if cfg.use_lora:
        lora_product = gateway_state["model.10.conv.lora_B"] @ gateway_state["model.10.conv.lora_A"]

    scaffold_c = None
    if cfg.scaffold:
        delta_c = gateway_state.get("__scaffold_delta_c__")
        if not isinstance(delta_c, dict):
            raise AssertionError("SCAFFOLD delta_c metadata was not propagated as a dict")
        scaffold_c = delta_c["model.21.head.weight"]
        gateway_state.pop("__scaffold_delta_c__")

    return DummyResult(
        head=gateway_state["model.21.head.weight"],
        lora_product=lora_product,
        payload_kb=payload_kb,
        map50=map50,
        scaffold_c=scaffold_c,
    )


def expected_for(baseline: str) -> DummyResult:
    cfg = BASELINE_CONFIGS[baseline]

    client_delta = [1.0, 2.0, 3.0, 4.0]
    if cfg.fedprox:
        client_delta = [x * 0.5 for x in client_delta]

    if cfg.fedprox:
        expected_delta = sum(d * n for d, n in zip(client_delta, [1, 3, 2, 4])) / 10.0
        if cfg.coop and cfg.hfl:
            relay0 = (client_delta[0] * 1 + client_delta[1] * 3) / 4.0
            relay1 = (client_delta[2] * 2 + client_delta[3] * 4) / 6.0
            alpha = 0.8 if cfg.coop_rule == 'selective' else 0.7
            expected_delta = 0.4 * (alpha * relay0 + (1.0 - alpha) * relay1) + 0.6 * relay1
        expected_head = torch.tensor([10.0 + expected_delta, 20.0 + expected_delta])
    elif cfg.topk_grad:
        # Per-client Top-K keeps both head deltas in this tiny state.
        expected_head = torch.tensor([12.9, 22.9])
    else:
        expected_delta = sum(d * n for d, n in zip(client_delta, [1, 3, 2, 4])) / 10.0
        if cfg.coop and cfg.hfl:
            relay0 = (client_delta[0] * 1 + client_delta[1] * 3) / 4.0
            relay1 = (client_delta[2] * 2 + client_delta[3] * 4) / 6.0
            alpha = 0.8 if cfg.coop_rule == 'selective' else 0.7
            expected_delta = 0.4 * (alpha * relay0 + (1.0 - alpha) * relay1) + 0.6 * relay1
        expected_head = torch.tensor([10.0 + expected_delta, 20.0 + expected_delta])

    expected_lora = None
    if cfg.use_lora:
        base_product = torch.tensor([[3.0, 4.0], [6.0, 8.0]])
        if cfg.coop and cfg.hfl:
            # relay0 scale=1.175, relay1 scale=1.3667, relay0 cooperates with
            # relay1, then gateway averages relay0/relay1 by 4/6.
            alpha = 0.8 if cfg.coop_rule == 'selective' else 0.7
            expected_scale = (4.0 / 10.0) * (alpha * 1.175 + (1.0 - alpha) * (41.0 / 30.0)) + (6.0 / 10.0) * (41.0 / 30.0)
        else:
            expected_scale = 1.29
        expected_lora = base_product * expected_scale

    map50 = 0.50
    if has_kd(cfg):
        map50 += 0.10
    if cfg.use_gateway_proxy_ft:
        map50 += 0.05
    if cfg.fedprox:
        map50 -= 0.02
    if cfg.topk_grad:
        map50 -= 0.03

    scaffold_c = torch.tensor([0.029, 0.029]) if cfg.scaffold else None
    return DummyResult(expected_head, expected_lora, payload_kb=0.0, map50=map50, scaffold_c=scaffold_c)


def test_quantization_roundtrip() -> None:
    state = {
        "a": torch.tensor([-1.0, 0.0, 1.0]),
        "b": torch.tensor([[2.0], [4.0]]),
    }
    payload, kb = pack_payload(state)
    recovered = unpack_payload(payload, state)
    if kb <= 0:
        raise AssertionError("INT8 payload size must be positive")
    assert_close(recovered["a"], state["a"], "INT8 roundtrip a", atol=0.01)
    assert_close(recovered["b"], state["b"], "INT8 roundtrip b", atol=0.02)


def test_topk_expected_values() -> None:
    compressor = TopKCompressor(total_params=4, rho_s=0.5)
    idx, values = compressor.compress(torch.tensor([1.0, -4.0, 2.0, 0.5]))
    pairs = sorted(zip(idx.tolist(), values.tolist()))
    if pairs != [(1, -4.0), (2, 2.0)]:
        raise AssertionError(f"TopK expected [(1, -4.0), (2, 2.0)], got {pairs}")


def test_baseline(baseline: str) -> bool:
    print(f"\n{'=' * 64}\nTesting baseline: {baseline}\n{'=' * 64}")
    actual = run_dummy_pipeline(baseline)
    expected = expected_for(baseline)

    tensor_atol = 1e-2 if BASELINE_CONFIGS[baseline].use_int8 else 1e-4
    assert_close(actual.head, expected.head, f"{baseline} head", atol=tensor_atol)
    if expected.lora_product is not None:
        assert actual.lora_product is not None
        assert_close(actual.lora_product, expected.lora_product, f"{baseline} LoRA product", atol=max(2e-2, tensor_atol))
    if expected.scaffold_c is not None:
        assert actual.scaffold_c is not None
        assert_close(actual.scaffold_c, expected.scaffold_c, f"{baseline} scaffold delta_c")
    assert_scalar(actual.map50, expected.map50, f"{baseline} dummy mAP50")
    if actual.payload_kb <= 0:
        raise AssertionError(f"{baseline} payload_kb must be positive")

    print(
        f"PASS {baseline}: head={actual.head.tolist()}, "
        f"payload={actual.payload_kb:.4f} KB, mAP50={actual.map50:.2f}"
    )
    return True


def main() -> None:
    test_quantization_roundtrip()
    test_topk_expected_values()

    results = {}
    for baseline in BASELINE_CONFIGS:
        try:
            results[baseline] = test_baseline(baseline)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"FAIL {baseline}: {exc}")
            results[baseline] = False

    print("\n" + "=" * 64)
    print("TEST SUMMARY")
    print("=" * 64)
    for baseline, ok in results.items():
        print(f"{baseline:<20}: {'PASS' if ok else 'FAIL'}")

    failed = [baseline for baseline, ok in results.items() if not ok]
    if failed:
        raise SystemExit(f"Failed baselines: {', '.join(failed)}")


if __name__ == "__main__":
    main()
