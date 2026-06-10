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
from typing import Any, Dict, List

import torch

from federated_core.aggregator import (
    svd_lora_aggregate,
    weighted_state_dict_average,
)
from federated_core.hfl_rules import blend_state_dicts
from federated_core.workers import BaseGateway
from tasks.detection_2d.baselines import (
    BASELINE_CONFIGS,
    OPTIONAL_BASELINES,
    STANDARD_BASELINES,
    BaselineConfig,
)
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
    auv_payload_kind: str = ""
    auv_head: torch.Tensor | None = None
    relay0_head: torch.Tensor | None = None
    relay1_head: torch.Tensor | None = None
    relay0_after_coop_head: torch.Tensor | None = None
    relay_bypassed: bool = False
    gateway_mode: str = ""


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


def encode_auv_payload(
    global_state: Dict[str, torch.Tensor],
    state: Dict[str, torch.Tensor],
    cfg: BaselineConfig,
) -> tuple[Any, float, str]:
    """Encode one AUV update exactly as the selected transport family expects."""
    if cfg.topk_grad:
        delta = {key: state[key] - global_state[key] for key in global_state}
        flat, shapes = flatten_state_dict(delta)
        compressor = TopKCompressor(total_params=flat.numel(), rho_s=0.5)
        idx, values = compressor.compress(flat)
        payload = SparseFloatPayload(idx, values, total_params=flat.numel(), shapes=shapes)
        return payload, payload.payload_bytes / 1024.0, "topk_sparse"
    if cfg.use_int8:
        payload, kb = pack_payload(state)
        return payload, kb, "int8"

    params = sum(value.numel() for value in state.values() if torch.is_tensor(value))
    return state, params * 4 / 1024.0, "float32"


def decode_auv_payload(
    global_state: Dict[str, torch.Tensor],
    payload: Any,
    payload_kind: str,
) -> Dict[str, torch.Tensor]:
    """Decode the AUV transport at the relay/gateway boundary."""
    if payload_kind == "topk_sparse":
        if not isinstance(payload, SparseFloatPayload):
            raise AssertionError("Top-K AUV payload must be SparseFloatPayload")
        recovered_delta = unflatten_state_dict(payload.decompress(), payload.shapes)
        return {key: global_state[key] + recovered_delta[key] for key in global_state}
    if payload_kind == "int8":
        if not isinstance(payload, (bytes, bytearray)):
            raise AssertionError("INT8 AUV payload must be bytes")
        return unpack_payload(payload, global_state)
    if not isinstance(payload, dict):
        raise AssertionError("Float32 AUV payload must be a state dict")
    return payload


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


def aggregate_cluster(
    states: List[Dict[str, torch.Tensor]],
    samples: List[int],
    lora_aggregation: str,
) -> Dict[str, torch.Tensor]:
    delta_c_updates = [
        state["__scaffold_delta_c__"]
        for state in states
        if "__scaffold_delta_c__" in state
    ]
    model_states = [
        {key: value for key, value in state.items() if key != "__scaffold_delta_c__"}
        for state in states
    ]
    if lora_aggregation == "svd":
        result = svd_lora_aggregate(model_states, samples)
    else:
        result = weighted_state_dict_average(model_states, samples)
    if len(delta_c_updates) == len(states):
        result["__scaffold_delta_c__"] = {
            key: sum(delta[key] for delta in delta_c_updates) / len(delta_c_updates)
            for key in delta_c_updates[0]
        }
        result["__scaffold_client_count__"] = len(delta_c_updates)
    return result


def run_dummy_pipeline(baseline: str) -> DummyResult:
    cfg = BASELINE_CONFIGS[baseline]
    torch.manual_seed(0)
    global_state = make_global_state(cfg.use_lora, full_param=cfg.full_param)

    client_samples = [1, 3, 2, 4]
    client_states = []
    payload_kb = 0.0
    payload_kinds = set()
    for client_id in range(1, 5):
        state = local_update(global_state, client_id, cfg)
        payload, kb, payload_kind = encode_auv_payload(global_state, state, cfg)
        state = decode_auv_payload(global_state, payload, payload_kind)
        payload_kb += kb / 4.0
        payload_kinds.add(payload_kind)
        client_states.append(state)

    relay0_head = None
    relay1_head = None
    relay0_after_coop_head = None
    if not cfg.hfl:
        gateway = BaseGateway(global_state)
        gateway.aggregate_global(
            dict(enumerate(client_states)),
            dict(enumerate(client_samples)),
            lora_aggregation=cfg.lora_aggregation,
        )
        gateway_state = gateway.global_state_dict
    else:
        relay0 = aggregate_cluster(client_states[:2], client_samples[:2], cfg.lora_aggregation)
        relay1 = aggregate_cluster(client_states[2:], client_samples[2:], cfg.lora_aggregation)
        relay0_head = relay0["model.21.head.weight"].clone()
        relay1_head = relay1["model.21.head.weight"].clone()
        if cfg.coop:
            alpha = 0.8 if cfg.coop_rule == 'selective' else 0.7
            relay0 = blend_state_dicts(relay0, relay1, alpha=alpha)
        relay0_after_coop_head = relay0["model.21.head.weight"].clone()
        gateway = BaseGateway(global_state)
        gateway.aggregate_global(
            {0: relay0, 1: relay1},
            {0: sum(client_samples[:2]), 1: sum(client_samples[2:])},
            lora_aggregation=cfg.lora_aggregation,
        )
        gateway_state = gateway.global_state_dict

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

    if len(payload_kinds) != 1:
        raise AssertionError(f"{baseline} produced mixed AUV payload kinds: {payload_kinds}")
    if baseline == "centralized":
        gateway_mode = "centralized_train"
    elif cfg.use_gateway_proxy_ft:
        gateway_mode = "proxy_finetune"
    elif cfg.logit_kd_only:
        gateway_mode = "logit_kd"
    elif cfg.use_gateway_kd:
        gateway_mode = "projection_kd" if cfg.use_lora else "full_model_kd"
    else:
        gateway_mode = "aggregate_only"

    return DummyResult(
        head=gateway_state["model.21.head.weight"],
        lora_product=lora_product,
        payload_kb=payload_kb,
        map50=map50,
        scaffold_c=scaffold_c,
        auv_payload_kind=next(iter(payload_kinds)),
        auv_head=client_states[0]["model.21.head.weight"],
        relay0_head=relay0_head,
        relay1_head=relay1_head,
        relay0_after_coop_head=relay0_after_coop_head,
        relay_bypassed=not cfg.hfl,
        gateway_mode=gateway_mode,
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

    scaffold_c = torch.tensor([0.025, 0.025]) if cfg.scaffold else None
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


def test_lora_aggregation_strategies() -> None:
    states = [
        {
            "layer.lora_B": torch.tensor([[1.0], [0.0]]),
            "layer.lora_A": torch.tensor([[1.0, 0.0]]),
        },
        {
            "layer.lora_B": torch.tensor([[0.0], [1.0]]),
            "layer.lora_A": torch.tensor([[0.0, 1.0]]),
        },
    ]
    naive = weighted_state_dict_average(states, [3, 1])
    svd = svd_lora_aggregate(states, [3, 1])
    naive_product = naive["layer.lora_B"] @ naive["layer.lora_A"]
    svd_product = svd["layer.lora_B"] @ svd["layer.lora_A"]

    assert_close(
        naive_product,
        torch.tensor([[0.5625, 0.1875], [0.1875, 0.0625]]),
        "naive LoRA A/B average",
    )
    assert_close(
        svd_product,
        torch.tensor([[0.75, 0.0], [0.0, 0.0]]),
        "SVD LoRA effective-weight average",
    )
    if torch.allclose(naive_product, svd_product):
        raise AssertionError("Naive FLORA and SVD-LoRA aggregation must remain distinct")


def test_baseline_contracts() -> None:
    expected = {
        'fedavg': dict(hfl=False, full_param=True, fedprox=False),
        'fedprox': dict(hfl=False, full_param=True, fedprox=True),
        'fedavg_hfl': dict(hfl=True, full_param=True, coop_rule='nocoop'),
        'fedprox_hfl': dict(hfl=True, full_param=True, fedprox=True, coop_rule='nocoop'),
        'flora': dict(
            hfl=True,
            use_lora=True,
            use_int8=False,
            use_gateway_kd=False,
            coop_rule='nocoop',
            lora_aggregation='svd',
        ),
        'naive_lora': dict(
            hfl=True,
            use_lora=True,
            use_int8=False,
            use_gateway_kd=False,
            coop_rule='nocoop',
            lora_aggregation='naive',
        ),
        'scaffold': dict(hfl=True, full_param=True, scaffold=True, coop_rule='nocoop'),
        'topk_grad': dict(hfl=True, full_param=True, topk_grad=True, coop_rule='nocoop'),
        'fedkdl': dict(hfl=True, use_lora=True, use_int8=True, use_gateway_kd=True, coop_rule='nearest'),
        'fedkdl_selective': dict(hfl=True, use_lora=True, use_int8=True, use_gateway_kd=True, coop_rule='selective'),
        'fedkdl_nocoop': dict(hfl=True, use_lora=True, use_int8=True, use_gateway_kd=True, coop_rule='nocoop'),
        'fedkdl_nokd': dict(hfl=True, use_lora=True, use_int8=True, use_gateway_kd=False, coop_rule='nearest'),
        'fedkdl_proxy_ft': dict(
            hfl=True,
            use_lora=True,
            use_int8=True,
            use_gateway_kd=False,
            use_gateway_proxy_ft=True,
            coop_rule='nearest',
        ),
        'logit_kd': dict(hfl=True, use_gateway_kd=True, logit_kd_only=True, coop_rule='nearest'),
        'fedprox_kdl': dict(hfl=True, use_lora=True, use_int8=True, use_gateway_kd=True, fedprox=True),
        'fedkdl_nolora': dict(hfl=True, full_param=True, use_lora=False, use_gateway_kd=True),
        'fedkd': dict(hfl=False, full_param=True, use_gateway_kd=True, local_kd=False),
        'centralized': dict(hfl=False, use_lora=True, full_param=False, use_gateway_kd=False),
    }
    if len(STANDARD_BASELINES) != 18:
        raise AssertionError(f"Expected 18 standard baselines, got {len(STANDARD_BASELINES)}")
    if set(expected) != set(STANDARD_BASELINES):
        raise AssertionError("STANDARD_BASELINES does not match the experiment contract table")
    if set(STANDARD_BASELINES) & set(OPTIONAL_BASELINES):
        raise AssertionError("Standard and optional baseline lists must be disjoint")

    for baseline, fields in expected.items():
        cfg = BASELINE_CONFIGS[baseline]
        for field, value in fields.items():
            actual = getattr(cfg, field)
            if actual != value:
                raise AssertionError(f"{baseline}.{field}: expected {value!r}, got {actual!r}")


def test_baseline(baseline: str) -> bool:
    print(f"\n{'=' * 64}\nTesting baseline: {baseline}\n{'=' * 64}")
    actual = run_dummy_pipeline(baseline)
    expected = expected_for(baseline)

    # Affine INT8 has a worst-case rounding error of scale / 2. The dummy
    # head spans 10 units, so Δ/2 = (20 - 10) / 255 / 2 ≈ 0.01961.
    tensor_atol = 2.1e-2 if BASELINE_CONFIGS[baseline].use_int8 else 1e-4
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

    cfg = BASELINE_CONFIGS[baseline]

    # Layer 1: AUV local update and transport.
    expected_payload_kind = "topk_sparse" if cfg.topk_grad else ("int8" if cfg.use_int8 else "float32")
    if actual.auv_payload_kind != expected_payload_kind:
        raise AssertionError(
            f"{baseline} AUV payload: expected {expected_payload_kind}, got {actual.auv_payload_kind}"
        )
    expected_auv_delta = 0.5 if cfg.fedprox else 1.0
    assert actual.auv_head is not None
    assert_close(
        actual.auv_head,
        torch.tensor([10.0 + expected_auv_delta, 20.0 + expected_auv_delta]),
        f"{baseline} AUV layer",
        atol=tensor_atol,
    )

    # Layer 2: Relay aggregation/cooperation, or an explicit bypass for flat methods.
    if cfg.hfl:
        if actual.relay_bypassed:
            raise AssertionError(f"{baseline} unexpectedly bypassed the relay layer")
        relay_scale = 0.5 if cfg.fedprox else 1.0
        expected_relay0 = torch.tensor([10.0 + 1.75 * relay_scale, 20.0 + 1.75 * relay_scale])
        expected_relay1 = torch.tensor([
            10.0 + (11.0 / 3.0) * relay_scale,
            20.0 + (11.0 / 3.0) * relay_scale,
        ])
        assert actual.relay0_head is not None
        assert actual.relay1_head is not None
        assert_close(actual.relay0_head, expected_relay0, f"{baseline} Relay 0", atol=tensor_atol)
        assert_close(actual.relay1_head, expected_relay1, f"{baseline} Relay 1", atol=tensor_atol)
        expected_after_coop = expected_relay0
        if cfg.coop:
            alpha = 0.8 if cfg.coop_rule == "selective" else 0.7
            expected_after_coop = alpha * expected_relay0 + (1.0 - alpha) * expected_relay1
        assert actual.relay0_after_coop_head is not None
        assert_close(
            actual.relay0_after_coop_head,
            expected_after_coop,
            f"{baseline} Relay cooperation",
            atol=tensor_atol,
        )
    else:
        if not actual.relay_bypassed:
            raise AssertionError(f"{baseline} must bypass the relay layer")
        if actual.relay0_head is not None or actual.relay1_head is not None:
            raise AssertionError(f"{baseline} flat pipeline unexpectedly produced relay states")

    # Layer 3: Gateway algorithm routing and final global state.
    if baseline == "centralized":
        expected_gateway_mode = "centralized_train"
    elif cfg.use_gateway_proxy_ft:
        expected_gateway_mode = "proxy_finetune"
    elif cfg.logit_kd_only:
        expected_gateway_mode = "logit_kd"
    elif cfg.use_gateway_kd:
        expected_gateway_mode = "projection_kd" if cfg.use_lora else "full_model_kd"
    else:
        expected_gateway_mode = "aggregate_only"
    if actual.gateway_mode != expected_gateway_mode:
        raise AssertionError(
            f"{baseline} Gateway mode: expected {expected_gateway_mode}, got {actual.gateway_mode}"
        )

    print(
        f"PASS {baseline}: AUV={actual.auv_payload_kind}, "
        f"Relay={'HFL' if cfg.hfl else 'bypass'}, Gateway={actual.gateway_mode}, "
        f"head={actual.head.tolist()}, payload={actual.payload_kb:.4f} KB, "
        f"mAP50={actual.map50:.2f}"
    )
    return True


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser("Deterministic FedKDL baseline tests")
    parser.add_argument("--include-optional", action="store_true")
    args = parser.parse_args()

    test_baseline_contracts()
    test_quantization_roundtrip()
    test_topk_expected_values()
    test_lora_aggregation_strategies()

    results = {}
    baselines = list(STANDARD_BASELINES)
    if args.include_optional:
        baselines.extend(OPTIONAL_BASELINES)
    for baseline in baselines:
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
