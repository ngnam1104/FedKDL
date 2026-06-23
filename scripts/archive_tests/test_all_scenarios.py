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
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from federated_core.aggregator import (
    mix_server_state,
    svd_lora_aggregate,
    weighted_state_dict_average,
)
from federated_core.base_simulator import BaseSimulator
from federated_core.hfl_rules import blend_state_dicts
from federated_core.hfl_rules import find_coop_partner
from federated_core.metrics import EnergyTracker, LatencyTracker, physical_joint_cost
from federated_core.workers import BaseGateway
from tasks.detection_2d.baselines import (
    BASELINE_CONFIGS,
    OPTIONAL_BASELINES,
    STANDARD_BASELINES,
    BaselineConfig,
)
from tasks.detection_2d.knowledge_compression.int8_quantization import (
    SparseINT8Payload,
    pack_delta_payload,
    pack_payload,
    unpack_delta_payload,
    unpack_payload,
)
from tasks.detection_2d.knowledge_compression.knowledge_distillation import (
    _compose_balanced_kd,
)
from tasks.detection_2d.knowledge_compression.topk_sparsification import (
    SparseFloatPayload,
    TopKCompressor,
    flatten_state_dict,
    unflatten_state_dict,
)
from config.settings import fed_cfg
from physics_models.latency import max_participant_samples
from physics_models.topology import gateway_disconnected_relays


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
        "model.0.weight": torch.arange(38, dtype=torch.float32) if full_param else torch.tensor([10.0, 20.0]),
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
    payload, kb, payload_kind, _ = encode_auv_payload_with_stateful_topk(
        global_state,
        state,
        cfg,
    )
    return payload, kb, payload_kind


def encode_auv_payload_with_stateful_topk(
    global_state: Dict[str, torch.Tensor],
    state: Dict[str, torch.Tensor],
    cfg: BaselineConfig,
    topk_compressor: TopKCompressor | None = None,
) -> tuple[Any, float, str, TopKCompressor | None]:
    """Encode one AUV update, optionally preserving Top-K error feedback."""
    if cfg.topk_grad:
        delta = {key: state[key] - global_state[key] for key in global_state}
        flat, shapes = flatten_state_dict(delta)
        if topk_compressor is None or topk_compressor.total_params != flat.numel():
            topk_compressor = TopKCompressor(total_params=flat.numel(), rho_s=fed_cfg.RHO_S)
        idx, values = topk_compressor.compress(flat)
        payload = SparseINT8Payload(idx, values, total_params=flat.numel(), shapes=shapes)
        return payload, payload.payload_bytes / 1024.0, "topk_int8", topk_compressor
    if cfg.use_int8:
        delta = {key: state[key] - global_state[key] for key in global_state}
        payload, kb = pack_payload(delta)
        return payload, kb, "int8", None

    params = sum(value.numel() for value in state.values() if torch.is_tensor(value))
    return state, params * 4 / 1024.0, "float32", None


def decode_auv_payload(
    global_state: Dict[str, torch.Tensor],
    payload: Any,
    payload_kind: str,
) -> Dict[str, torch.Tensor]:
    """Decode the AUV transport at the relay/gateway boundary."""
    if payload_kind == "topk_int8":
        if not isinstance(payload, SparseINT8Payload):
            raise AssertionError("Top-K AUV payload must be SparseINT8Payload")
        recovered_delta = unflatten_state_dict(payload.decompress(), payload.shapes)
        return {key: global_state[key] + recovered_delta[key] for key in global_state}
    if payload_kind == "int8":
        if not isinstance(payload, (bytes, bytearray)):
            raise AssertionError("INT8 AUV payload must be bytes")
        recovered_delta = unpack_payload(payload, global_state)
        return {
            key: global_state[key] + recovered_delta[key]
            for key in global_state
        }
    if not isinstance(payload, dict):
        raise AssertionError("Float32 AUV payload must be a state dict")
    return payload


def maybe_pack_roundtrip(state: Dict[str, torch.Tensor], template: Dict[str, torch.Tensor], use_int8: bool) -> tuple[Dict[str, torch.Tensor], float]:
    if not use_int8:
        params = sum(v.numel() for v in state.values() if torch.is_tensor(v))
        return state, params * 4 / 1024.0
    payload, kb = pack_payload(state)
    return unpack_payload(payload, template), kb


def relay_delta_roundtrip(
    state: Dict[str, torch.Tensor],
    global_state: Dict[str, torch.Tensor],
    use_int8: bool,
) -> Dict[str, torch.Tensor]:
    if not use_int8:
        return state
    payload, _ = pack_delta_payload(state, global_state)
    return unpack_delta_payload(payload, global_state)


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


def run_dummy_pipeline_with_state(
    baseline: str,
    initial_global_state: Dict[str, torch.Tensor] | None = None,
    topk_compressors: Dict[int, TopKCompressor] | None = None,
) -> tuple[DummyResult, Dict[str, torch.Tensor]]:
    cfg = BASELINE_CONFIGS[baseline]
    torch.manual_seed(0)
    if initial_global_state is None:
        global_state = make_global_state(cfg.use_lora, full_param=cfg.full_param)
    else:
        global_state = {
            key: value.clone()
            for key, value in initial_global_state.items()
            if torch.is_tensor(value)
        }

    client_samples = [1, 3, 2, 4]
    client_states = []
    payload_kb = 0.0
    payload_kinds = set()
    for client_id in range(1, 5):
        state = local_update(global_state, client_id, cfg)
        topk_compressor = None
        if topk_compressors is not None:
            topk_compressor = topk_compressors.get(client_id)
        payload, kb, payload_kind, topk_compressor = encode_auv_payload_with_stateful_topk(
            global_state,
            state,
            cfg,
            topk_compressor=topk_compressor,
        )
        if cfg.topk_grad and topk_compressors is not None and topk_compressor is not None:
            topk_compressors[client_id] = topk_compressor
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
            server_mix_beta=fed_cfg.SERVER_MIX_BETA if cfg.server_mix else 1.0,
        )
        gateway_state = gateway.global_state_dict
    else:
        relay0 = aggregate_cluster(client_states[:2], client_samples[:2], cfg.lora_aggregation)
        relay1 = aggregate_cluster(client_states[2:], client_samples[2:], cfg.lora_aggregation)
        relay0_head = relay0["model.21.head.weight"].clone()
        relay1_head = relay1["model.21.head.weight"].clone()
        if cfg.coop:
            alpha = 1.0 - (
                fed_cfg.COOP_NEIGHBOR_WEIGHT_SELECTIVE
                if cfg.coop_rule == 'selective'
                else fed_cfg.COOP_NEIGHBOR_WEIGHT_NEAREST
            )
            relay1_on_r2r = relay_delta_roundtrip(relay1, global_state, cfg.use_int8)
            relay0 = blend_state_dicts(relay0, relay1_on_r2r, alpha=alpha)
        relay0_after_coop_head = relay0["model.21.head.weight"].clone()
        relay0 = relay_delta_roundtrip(relay0, global_state, cfg.use_int8)
        relay1 = relay_delta_roundtrip(relay1, global_state, cfg.use_int8)
        gateway = BaseGateway(global_state)
        gateway.aggregate_global(
            {0: relay0, 1: relay1},
            {0: sum(client_samples[:2]), 1: sum(client_samples[2:])},
            lora_aggregation=cfg.lora_aggregation,
            server_mix_beta=fed_cfg.SERVER_MIX_BETA if cfg.server_mix else 1.0,
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
    # Component ablation baselines use partial KD — slightly lower than full KD.
    if getattr(cfg, 'logit_kd_only', False):
        map50 -= 0.05
    elif getattr(cfg, 'logit_box_kd_only', False):
        map50 -= 0.03
    elif getattr(cfg, 'logit_proj_kd_only', False):
        map50 -= 0.04

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
    elif getattr(cfg, 'logit_kd_only', False):
        gateway_mode = "logit_kd"
    elif getattr(cfg, 'logit_box_kd_only', False):
        gateway_mode = "logit_box_kd"
    elif getattr(cfg, 'logit_proj_kd_only', False):
        gateway_mode = "logit_proj_kd"
    elif cfg.use_gateway_kd:
        gateway_mode = "projection_kd" if cfg.use_lora else "full_model_kd"
    else:
        gateway_mode = "aggregate_only"

    result = DummyResult(
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
    next_global_state = {
        key: value.clone()
        for key, value in gateway_state.items()
        if torch.is_tensor(value)
    }
    return result, next_global_state


def run_dummy_pipeline(baseline: str) -> DummyResult:
    result, _ = run_dummy_pipeline_with_state(baseline)
    return result


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
            alpha = 1.0 - (
                fed_cfg.COOP_NEIGHBOR_WEIGHT_SELECTIVE
                if cfg.coop_rule == 'selective'
                else fed_cfg.COOP_NEIGHBOR_WEIGHT_NEAREST
            )
            expected_delta = 0.4 * (alpha * relay0 + (1.0 - alpha) * relay1) + 0.6 * relay1
        server_beta = fed_cfg.SERVER_MIX_BETA if cfg.server_mix else 1.0
        expected_head = torch.tensor([
            10.0 + server_beta * expected_delta,
            20.0 + server_beta * expected_delta,
        ])
    elif cfg.topk_grad:
        # Per-client Top-K keeps both head deltas in this tiny state.
        expected_head = torch.tensor([12.9, 22.9])
    else:
        expected_delta = sum(d * n for d, n in zip(client_delta, [1, 3, 2, 4])) / 10.0
        if cfg.coop and cfg.hfl:
            relay0 = (client_delta[0] * 1 + client_delta[1] * 3) / 4.0
            relay1 = (client_delta[2] * 2 + client_delta[3] * 4) / 6.0
            alpha = 1.0 - (
                fed_cfg.COOP_NEIGHBOR_WEIGHT_SELECTIVE
                if cfg.coop_rule == 'selective'
                else fed_cfg.COOP_NEIGHBOR_WEIGHT_NEAREST
            )
            expected_delta = 0.4 * (alpha * relay0 + (1.0 - alpha) * relay1) + 0.6 * relay1
        server_beta = fed_cfg.SERVER_MIX_BETA if cfg.server_mix else 1.0
        expected_head = torch.tensor([
            10.0 + server_beta * expected_delta,
            20.0 + server_beta * expected_delta,
        ])

    expected_lora = None
    if cfg.use_lora:
        base_product = torch.tensor([[3.0, 4.0], [6.0, 8.0]])
        if cfg.coop and cfg.hfl:
            # relay0 scale=1.175, relay1 scale=1.3667, relay0 cooperates with
            # relay1, then gateway averages relay0/relay1 by 4/6.
            alpha = 1.0 - (
                fed_cfg.COOP_NEIGHBOR_WEIGHT_SELECTIVE
                if cfg.coop_rule == 'selective'
                else fed_cfg.COOP_NEIGHBOR_WEIGHT_NEAREST
            )
            expected_scale = (4.0 / 10.0) * (alpha * 1.175 + (1.0 - alpha) * (41.0 / 30.0)) + (6.0 / 10.0) * (41.0 / 30.0)
        else:
            expected_scale = 1.29
        server_beta = fed_cfg.SERVER_MIX_BETA if cfg.server_mix else 1.0
        expected_scale = 1.0 + server_beta * (expected_scale - 1.0)
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
    # Component ablation baselines use partial KD — slightly lower than full KD.
    if getattr(cfg, 'logit_kd_only', False):
        map50 -= 0.05
    elif getattr(cfg, 'logit_box_kd_only', False):
        map50 -= 0.03
    elif getattr(cfg, 'logit_proj_kd_only', False):
        map50 -= 0.04

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


def _max_state_error(
    actual: Dict[str, torch.Tensor],
    expected: Dict[str, torch.Tensor],
) -> float:
    return max(
        (actual[key].float() - expected[key].float()).abs().max().item()
        for key in expected
    )


def _delta_int8_roundtrip(
    global_state: Dict[str, torch.Tensor],
    local_state: Dict[str, torch.Tensor],
) -> tuple[Dict[str, torch.Tensor], float]:
    payload, payload_kb = pack_delta_payload(local_state, global_state)
    return unpack_delta_payload(payload, global_state), payload_kb


def test_delta_int8_zero_and_constant_updates() -> None:
    global_state = {
        "layer.weight": torch.tensor([1000.0, -500.0, 25.0, 0.0]),
        "model.0.bn.running_mean": torch.tensor([0.25, -0.50]),
    }
    local_state = {
        "layer.weight": global_state["layer.weight"] + 0.125,
        "model.0.bn.running_mean": global_state["model.0.bn.running_mean"].clone(),
    }
    recovered, payload_kb = _delta_int8_roundtrip(global_state, local_state)

    if payload_kb <= 0:
        raise AssertionError("Delta-INT8 payload size must be positive")
    assert_close(recovered["layer.weight"], local_state["layer.weight"], "constant delta", atol=1e-7)
    assert_close(
        recovered["model.0.bn.running_mean"],
        local_state["model.0.bn.running_mean"],
        "zero BN delta",
        atol=1e-7,
    )


def test_delta_int8_reduces_quantization_error() -> None:
    global_state = {
        "layer.weight": torch.tensor([-1000.0, -250.0, 500.0, 1000.0]),
        "model.21.head.weight": torch.tensor([-400.0, -30.0, 70.0, 600.0]),
    }
    delta = {
        "layer.weight": torch.tensor([0.030, -0.020, 0.010, -0.040]),
        "model.21.head.weight": torch.tensor([0.004, -0.003, 0.002, -0.001]),
    }
    local_state = {
        key: global_state[key] + delta[key]
        for key in global_state
    }

    raw_payload, raw_kb = pack_payload(local_state)
    raw_recovered = unpack_payload(raw_payload, global_state)
    delta_recovered, delta_kb = _delta_int8_roundtrip(global_state, local_state)

    raw_error = _max_state_error(raw_recovered, local_state)
    delta_error = _max_state_error(delta_recovered, local_state)
    if delta_kb != raw_kb:
        raise AssertionError(
            f"Raw and delta INT8 must have equal shape-based payload size: {raw_kb} vs {delta_kb}"
        )
    if not delta_error < raw_error * 0.05:
        raise AssertionError(
            f"Delta-INT8 should reduce error by at least 20x; raw={raw_error:.6g}, "
            f"delta={delta_error:.6g}"
        )


def test_delta_int8_multiclient_fedavg_with_bn() -> None:
    global_state = {
        "layer.weight": torch.tensor([100.0, -80.0, 40.0, -20.0]),
        "model.21.head.weight": torch.tensor([10.0, 20.0, 30.0, 40.0]),
        "model.0.bn.running_mean": torch.tensor([0.2, -0.3]),
        "model.0.bn.running_var": torch.tensor([1.0, 1.5]),
        "model.0.bn.num_batches_tracked": torch.tensor(100, dtype=torch.int64),
    }
    client_deltas = [
        {
            "layer.weight": torch.tensor([0.02, -0.01, 0.03, -0.04]),
            "model.21.head.weight": torch.tensor([0.04, -0.02, 0.01, -0.03]),
            "model.0.bn.running_mean": torch.tensor([0.01, -0.02]),
            "model.0.bn.running_var": torch.tensor([0.03, -0.01]),
            "model.0.bn.num_batches_tracked": torch.tensor(2, dtype=torch.int64),
        },
        {
            "layer.weight": torch.tensor([-0.03, 0.04, -0.01, 0.02]),
            "model.21.head.weight": torch.tensor([-0.01, 0.03, -0.04, 0.02]),
            "model.0.bn.running_mean": torch.tensor([-0.03, 0.01]),
            "model.0.bn.running_var": torch.tensor([-0.02, 0.04]),
            "model.0.bn.num_batches_tracked": torch.tensor(4, dtype=torch.int64),
        },
        {
            "layer.weight": torch.tensor([0.01, 0.02, -0.02, -0.01]),
            "model.21.head.weight": torch.tensor([0.02, 0.01, -0.01, -0.02]),
            "model.0.bn.running_mean": torch.tensor([0.02, 0.03]),
            "model.0.bn.running_var": torch.tensor([0.01, 0.02]),
            "model.0.bn.num_batches_tracked": torch.tensor(6, dtype=torch.int64),
        },
    ]
    samples = [1, 3, 6]
    ideal_clients = []
    recovered_clients = []
    for delta in client_deltas:
        local_state = {
            key: global_state[key] + delta[key]
            for key in global_state
        }
        ideal_clients.append(local_state)
        recovered, _ = _delta_int8_roundtrip(global_state, local_state)
        recovered_clients.append(recovered)

        # BN tensors bypass INT8 and must survive the transport exactly.
        for key in global_state:
            if "bn" in key:
                assert_close(recovered[key], local_state[key], f"Delta-INT8 exact {key}", atol=0.0)

    expected = weighted_state_dict_average(ideal_clients, samples)
    actual = weighted_state_dict_average(recovered_clients, samples)
    assert_close(actual["layer.weight"], expected["layer.weight"], "Delta-INT8 FedAvg weight", atol=2e-4)
    assert_close(actual["model.21.head.weight"], expected["model.21.head.weight"], "Delta-INT8 FedAvg head", atol=2e-4)
    assert_close(
        actual["model.0.bn.running_mean"],
        expected["model.0.bn.running_mean"],
        "Delta-INT8 FedAvg BN mean",
        atol=0.0,
    )
    assert_close(
        actual["model.0.bn.running_var"],
        expected["model.0.bn.running_var"],
        "Delta-INT8 FedAvg BN variance",
        atol=0.0,
    )


def test_delta_int8_multiround_drift() -> None:
    ideal = {
        "layer.weight": torch.tensor([-1000.0, -250.0, 500.0, 1000.0]),
        "model.21.head.weight": torch.tensor([-400.0, -30.0, 70.0, 600.0]),
    }
    raw_global = {key: value.clone() for key, value in ideal.items()}
    delta_global = {key: value.clone() for key, value in ideal.items()}

    for round_idx in range(1, 61):
        scale = 1.0 + (round_idx % 7) * 0.1
        update = {
            "layer.weight": torch.tensor([0.030, -0.020, 0.010, -0.040]) * scale,
            "model.21.head.weight": torch.tensor([0.004, -0.003, 0.002, -0.001]) * scale,
        }
        ideal = {key: ideal[key] + update[key] for key in ideal}

        raw_local = {key: raw_global[key] + update[key] for key in raw_global}
        raw_payload, _ = pack_payload(raw_local)
        raw_global = unpack_payload(raw_payload, raw_global)

        delta_local = {key: delta_global[key] + update[key] for key in delta_global}
        delta_global, _ = _delta_int8_roundtrip(delta_global, delta_local)

    raw_error = _max_state_error(raw_global, ideal)
    delta_error = _max_state_error(delta_global, ideal)
    if delta_error > 0.01:
        raise AssertionError(f"Delta-INT8 accumulated excessive 60-round drift: {delta_error:.6g}")
    if not delta_error < raw_error * 0.05:
        raise AssertionError(
            f"Delta-INT8 60-round drift should be at least 20x lower; "
            f"raw={raw_error:.6g}, delta={delta_error:.6g}"
        )


def run_delta_int8_tests() -> None:
    test_quantization_roundtrip()
    test_delta_int8_zero_and_constant_updates()
    test_delta_int8_reduces_quantization_error()
    test_delta_int8_multiclient_fedavg_with_bn()
    test_delta_int8_multiround_drift()


def test_topk_expected_values() -> None:
    compressor = TopKCompressor(total_params=4, rho_s=0.5)
    idx, values = compressor.compress(torch.tensor([1.0, -4.0, 2.0, 0.5]))
    pairs = sorted(zip(idx.tolist(), values.tolist()))
    if pairs != [(1, -4.0), (2, 2.0)]:
        raise AssertionError(f"TopK expected [(1, -4.0), (2, 2.0)], got {pairs}")


def test_topk_baseline_uses_five_percent_sparsity() -> None:
    assert_scalar(fed_cfg.RHO_S, 0.05, "Top-K rho_s")
    cfg = BASELINE_CONFIGS["topk_grad"]
    global_state = make_global_state(cfg.use_lora, full_param=cfg.full_param)
    state = local_update(global_state, client_id=1, cfg=cfg)
    payload, _, payload_kind = encode_auv_payload(global_state, state, cfg)
    if payload_kind != "topk_int8":
        raise AssertionError(f"Top-K payload kind must be topk_int8, got {payload_kind}")
    if payload.total_params != 40:
        raise AssertionError(f"Top-K dummy state must have 40 params, got {payload.total_params}")
    if len(payload.indices) != 2:
        raise AssertionError(f"Top-K K must be 5% of 40 = 2, got {len(payload.indices)}")


def test_topk_optional_sweep_ratios() -> None:
    expected = {
        "topk_grad_10": (0.10, 4),
        "topk_grad_20": (0.20, 8),
    }
    for baseline, (ratio, expected_k) in expected.items():
        cfg = BASELINE_CONFIGS[baseline]
        if cfg.topk_ratio != ratio:
            raise AssertionError(f"{baseline}.topk_ratio: expected {ratio}, got {cfg.topk_ratio}")
        global_state = make_global_state(cfg.use_lora, full_param=cfg.full_param)
        state = local_update(global_state, client_id=1, cfg=cfg)
        payload, _, payload_kind = encode_auv_payload(global_state, state, cfg)
        if payload_kind != "topk_int8":
            raise AssertionError(f"{baseline} payload kind must be topk_int8, got {payload_kind}")
        if len(payload.indices) != expected_k:
            raise AssertionError(f"{baseline} K: expected {expected_k}, got {len(payload.indices)}")


def test_topk_flatten_indices_are_stable_across_key_order_and_rounds() -> None:
    round1_delta = {
        "z.weight": torch.tensor([100.0, 1.0]),
        "a.bias": torch.tensor([-5.0, 0.0, 6.0]),
    }
    flat1, shapes1 = flatten_state_dict(round1_delta)
    shape_keys1 = [key for key, _, _ in shapes1]
    if shape_keys1 != ["a.bias", "z.weight"]:
        raise AssertionError(f"Top-K flatten order must be sorted, got {shape_keys1}")

    compressor = TopKCompressor(total_params=flat1.numel(), rho_s=0.4)
    idx1, values1 = compressor.compress(flat1)
    payload1 = SparseFloatPayload(idx1, values1, total_params=flat1.numel(), shapes=shapes1)
    recovered1 = unflatten_state_dict(payload1.decompress(), payload1.shapes)
    assert_close(recovered1["a.bias"], torch.tensor([0.0, 0.0, 6.0]), "Top-K round 1 a.bias")
    assert_close(recovered1["z.weight"], torch.tensor([100.0, 0.0]), "Top-K round 1 z.weight")

    # Same logical tensors, intentionally inserted in the opposite order.
    # The compressor error buffer is positional, so this catches key-order
    # drift that would update the wrong tensor slot in later rounds.
    round2_delta = {
        "a.bias": torch.tensor([1.0, 0.0, 2.0]),
        "z.weight": torch.tensor([3.0, 4.0]),
    }
    flat2, shapes2 = flatten_state_dict(round2_delta)
    shape_keys2 = [key for key, _, _ in shapes2]
    if shape_keys2 != shape_keys1:
        raise AssertionError(f"Top-K flatten order changed across rounds: {shape_keys1} vs {shape_keys2}")

    idx2, values2 = compressor.compress(flat2)
    payload2 = SparseFloatPayload(idx2, values2, total_params=flat2.numel(), shapes=shapes2)
    recovered2 = unflatten_state_dict(payload2.decompress(), payload2.shapes)
    assert_close(recovered2["a.bias"], torch.tensor([-4.0, 0.0, 0.0]), "Top-K round 2 a.bias")
    assert_close(recovered2["z.weight"], torch.tensor([0.0, 5.0]), "Top-K round 2 z.weight")


def test_topk_int8_payload_bits_are_counted_without_len() -> None:
    class PayloadSizer:
        _payload_to_bits = BaseSimulator._payload_to_bits

    idx = torch.tensor([0, 3], dtype=torch.long)
    values = torch.tensor([1.0, -2.0])
    payload = SparseINT8Payload(
        idx,
        values,
        total_params=8,
        shapes=[("w", torch.Size([2, 4]), 8)],
    )
    sizer = PayloadSizer()
    assert_scalar(
        sizer._payload_to_bits(payload),
        payload.payload_bits,
        "Top-K INT8 sparse payload bits",
    )


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


def test_server_mix_preserves_effective_lora_geometry() -> None:
    old_state = {
        "layer.lora_B": torch.tensor([[1.0], [2.0]]),
        "layer.lora_A": torch.tensor([[3.0, 4.0]]),
        "head": torch.tensor([10.0, 20.0]),
    }
    aggregated_state = {
        "layer.lora_B": torch.tensor([[2.0], [4.0]]),
        "layer.lora_A": torch.tensor([[3.0, 4.0]]),
        "head": torch.tensor([14.0, 24.0]),
    }
    mixed = mix_server_state(old_state, aggregated_state, beta=0.75, lora_aggregation="svd")
    mixed_product = mixed["layer.lora_B"] @ mixed["layer.lora_A"]
    expected_product = (
        0.25 * (old_state["layer.lora_B"] @ old_state["layer.lora_A"])
        + 0.75 * (aggregated_state["layer.lora_B"] @ aggregated_state["layer.lora_A"])
    )
    assert_close(mixed_product, expected_product, "Server mix effective LoRA product")
    assert_close(mixed["head"], torch.tensor([13.0, 23.0]), "Server mix head")


def test_svd_factor_signs_are_canonical() -> None:
    state = {
        "layer.lora_B": torch.tensor([[-2.0, 0.0], [0.0, -1.0]]),
        "layer.lora_A": torch.eye(2),
    }
    aggregated = svd_lora_aggregate([state], [1.0])
    B = aggregated["layer.lora_B"]
    for column in range(B.shape[1]):
        pivot = B[:, column].abs().argmax()
        if B[pivot, column] < 0:
            raise AssertionError("SVD LoRA factors must use deterministic positive pivots")
    assert_close(
        B @ aggregated["layer.lora_A"],
        state["layer.lora_B"] @ state["layer.lora_A"],
        "canonical SVD preserves effective LoRA matrix",
    )


def test_kd_component_contributions() -> None:
    supervised = torch.tensor(10.0)
    components = {
        'cls': torch.tensor(2.0),
        'box': torch.tensor(1.0),
        'proj': torch.tensor(0.5),
    }
    (
        supervised_weighted,
        weighted_components,
        weighted_kd,
        _,
        kd_ratio,
    ) = _compose_balanced_kd(
        loss_stu=supervised,
        stu_weight=0.5,
        component_values=components,
        component_weights={'cls': 0.45, 'box': 0.35, 'proj': 0.20},
        kd_lambda=1.0,
        balance_by_supervised=True,
        scale_min=0.001,
        scale_max=20.0,
    )
    assert_scalar(supervised_weighted.item(), 5.0, "KD supervised reference")
    # Current KD composition intentionally uses a single-level weighted sum:
    # component_i * normalized_weight_i * kd_lambda. It no longer rescales the
    # KD branch to match the supervised loss magnitude.
    assert_scalar(weighted_components['cls'].item(), 0.90, "KD cls contribution", atol=1e-6)
    assert_scalar(weighted_components['box'].item(), 0.35, "KD box contribution", atol=1e-6)
    assert_scalar(weighted_components['proj'].item(), 0.10, "KD projection contribution", atol=1e-6)
    assert_scalar(weighted_kd.item(), 1.35, "KD total contribution", atol=1e-6)
    assert_scalar(kd_ratio.item(), 0.27, "KD/supervised ratio", atol=1e-6)


def test_nearest_and_selective_partner_rules() -> None:
    class Link:
        def __init__(self, distance: float):
            self.distance = distance

    graph = {
        ('relay', 0, 'relay', 1): Link(10.0),
        ('relay', 0, 'relay', 2): Link(20.0),
    }
    cluster_sizes = {0: 9, 1: 9, 2: 12}

    nearest = find_coop_partner(
        0,
        cluster_sizes,
        graph,
        require_larger_cluster=False,
    )
    if nearest != 1:
        raise AssertionError(f"HFL-Nearest must choose closest feasible relay 1, got {nearest}")

    selective = find_coop_partner(
        0,
        cluster_sizes,
        graph,
        require_larger_cluster=True,
    )
    if selective != 2:
        raise AssertionError(f"HFL-Selective must borrow from larger relay 2, got {selective}")


def test_latency_keeps_relay_paths_coupled() -> None:
    class Link:
        def __init__(self, rate: float):
            self.R_bps = rate
            self.distance = 0.0

    graph = {
        ('auv', 0, 'relay', 0): Link(10.0),
        ('auv', 1, 'relay', 1): Link(100.0),
        ('relay', 0, 'gateway', 0): Link(100.0),
        ('relay', 1, 'gateway', 0): Link(10.0),
    }
    metrics = LatencyTracker().compute_round_latency(
        G=graph,
        association={0: 0, 1: 1},
        cooperation_partners={},
        tau_comp=0.0,
        tau_svd=0.0,
        auv_payload_bits=100.0,
        relay_model_bits=100.0,
    )
    assert_scalar(metrics['tau_round'], 11.0, "per-relay latency bottleneck")
    assert_scalar(metrics['tau_a2r'], 10.0, "max AUV-to-relay latency")
    assert_scalar(metrics['tau_r2g'], 10.0, "max relay-to-gateway latency")


def test_physics_accounting_contracts() -> None:
    expected_lora_payload_kb = fed_cfg.LORA_INT8_PAYLOAD_BYTES_2D / 1024.0
    assert_scalar(
        expected_lora_payload_kb,
        505.84765625,
        "current serialized LoRA+Head+BN INT8 payload",
    )

    tracker = EnergyTracker()
    tracker.add_round(
        round_idx=1,
        e_a2r=12.0,
        e_r2r=7.0,
        e_r2g=5.0,
        e_comp=3.0,
        e_svd=1.0,
        e_a2r_rx=2.0,
        e_r2r_rx=1.5,
        e_r2g_rx=0.5,
    )
    row = tracker.history[-1]
    assert_scalar(row['e_rx'], 4.0, "total RX energy")
    assert_scalar(row['round_total'], 28.0, "TX+RX+computation round energy")
    assert_scalar(tracker.cumulative_energy, 28.0, "cumulative energy")

    max_samples = max_participant_samples([80, 125, 100])
    if max_samples != 125:
        raise AssertionError(f"tau_comp workload must use max=125, got {max_samples}")
    if max_participant_samples([]) != 100:
        raise AssertionError("empty participant workload must use the documented default")

    cost = physical_joint_cost(
        energy=28.0,
        latency=11.0,
        lambda_e=0.01,
        lambda_tau=0.02,
    )
    assert_scalar(cost, 0.50, "physical joint cost")

    class Link:
        def __init__(self, rate: float):
            self.R_bps = rate
            self.distance = 0.0

    graph_with_missing_r2g = {
        ('auv', 0, 'relay', 0): Link(10.0),
        ('auv', 1, 'relay', 1): Link(20.0),
        ('relay', 0, 'gateway', 0): Link(10.0),
    }
    metrics = LatencyTracker().compute_round_latency(
        G=graph_with_missing_r2g,
        association={0: 0, 1: 1},
        cooperation_partners={},
        tau_comp=2.0,
        tau_svd=1.0,
        auv_payload_bits=100.0,
        relay_model_bits=100.0,
    )
    assert_scalar(metrics['tau_round'], 23.0, "missing R2G link is skipped")

    class Topology:
        M = 3

    gateway_graph = {
        ('relay', 0, 'gateway', 0): object(),
        ('relay', 2, 'gateway', 0): object(),
    }
    missing_relays = gateway_disconnected_relays(Topology(), gateway_graph)
    if missing_relays != [1]:
        raise AssertionError(
            f"Expected relay 1 to lack gateway connectivity, got {missing_relays}"
        )


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
        'fedkdl': dict(hfl=True, use_lora=True, use_int8=True, use_gateway_kd=True, coop_rule='nearest', server_mix=True),
        'fedkdl_selective': dict(hfl=True, use_lora=True, use_int8=True, use_gateway_kd=True, coop_rule='selective', server_mix=True),
        'fedkdl_nocoop': dict(hfl=True, use_lora=True, use_int8=True, use_gateway_kd=True, coop_rule='nocoop', server_mix=True),
        'fedkdl_nokd': dict(hfl=True, use_lora=True, use_int8=True, use_gateway_kd=False, coop_rule='nearest', server_mix=True),
        'fedkdl_proxy_ft': dict(
            hfl=True,
            use_lora=True,
            use_int8=True,
            use_gateway_kd=False,
            use_gateway_proxy_ft=True,
            coop_rule='nearest',
            server_mix=True,
        ),
        'logit_kd': dict(hfl=True, use_gateway_kd=True, logit_kd_only=True, coop_rule='nearest', server_mix=True),
        'logit_box_kd': dict(hfl=True, use_gateway_kd=True, logit_box_kd_only=True, coop_rule='nearest', server_mix=True),
        'logit_proj_kd': dict(hfl=True, use_gateway_kd=True, logit_proj_kd_only=True, coop_rule='nearest', server_mix=True),
        'fedprox_kdl': dict(hfl=True, use_lora=True, use_int8=True, use_gateway_kd=True, fedprox=True, server_mix=True),
        'fedkdl_32bit': dict(hfl=True, full_param=True, use_lora=False, use_gateway_kd=True, server_mix=True),
        'fedkd': dict(hfl=False, full_param=True, use_gateway_kd=True, local_kd=False),
        'centralized': dict(hfl=False, use_lora=True, full_param=False, use_gateway_kd=False),
    }
    if len(STANDARD_BASELINES) != 20:
        raise AssertionError(f"Expected 20 standard baselines, got {len(STANDARD_BASELINES)}")
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

    optional_expected = {
        'topk_grad_10': dict(hfl=True, full_param=True, topk_grad=True, topk_ratio=0.10, coop_rule='nocoop'),
        'topk_grad_20': dict(hfl=True, full_param=True, topk_grad=True, topk_ratio=0.20, coop_rule='nocoop'),
    }
    if set(optional_expected) != set(OPTIONAL_BASELINES):
        raise AssertionError("OPTIONAL_BASELINES does not match the Top-K sweep contract")
    for baseline, fields in optional_expected.items():
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
    expected_payload_kind = "topk_int8" if cfg.topk_grad else ("int8" if cfg.use_int8 else "float32")
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
            alpha = 1.0 - (
                fed_cfg.COOP_NEIGHBOR_WEIGHT_SELECTIVE
                if cfg.coop_rule == "selective"
                else fed_cfg.COOP_NEIGHBOR_WEIGHT_NEAREST
            )
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
    elif getattr(cfg, 'logit_kd_only', False):
        expected_gateway_mode = "logit_kd"
    elif getattr(cfg, 'logit_box_kd_only', False):
        expected_gateway_mode = "logit_box_kd"
    elif getattr(cfg, 'logit_proj_kd_only', False):
        expected_gateway_mode = "logit_proj_kd"
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


def test_two_round_baseline(baseline: str) -> bool:
    cfg = BASELINE_CONFIGS[baseline]
    initial_global = make_global_state(cfg.use_lora, full_param=cfg.full_param)
    topk_compressors: Dict[int, TopKCompressor] = {}

    round1, state_after_round1 = run_dummy_pipeline_with_state(
        baseline,
        initial_global_state=initial_global,
        topk_compressors=topk_compressors,
    )
    round2, state_after_round2 = run_dummy_pipeline_with_state(
        baseline,
        initial_global_state=state_after_round1,
        topk_compressors=topk_compressors,
    )

    tensor_atol = 2.1e-2 if cfg.use_int8 else 1e-4
    step1 = round1.head - initial_global["model.21.head.weight"]
    step2 = round2.head - round1.head
    assert_close(step2, step1, f"{baseline} two-round head increment", atol=tensor_atol)

    if round2.payload_kb <= 0:
        raise AssertionError(f"{baseline} round 2 payload_kb must be positive")
    if round2.auv_payload_kind != round1.auv_payload_kind:
        raise AssertionError(
            f"{baseline} payload kind changed across rounds: "
            f"{round1.auv_payload_kind} -> {round2.auv_payload_kind}"
        )
    if round2.gateway_mode != round1.gateway_mode:
        raise AssertionError(
            f"{baseline} gateway mode changed across rounds: "
            f"{round1.gateway_mode} -> {round2.gateway_mode}"
        )
    if set(state_after_round2) != set(initial_global):
        raise AssertionError(
            f"{baseline} global tensor keys changed across two rounds: "
            f"{sorted(initial_global)} -> {sorted(state_after_round2)}"
        )

    for key, value in state_after_round2.items():
        if not torch.isfinite(value.float()).all():
            raise AssertionError(f"{baseline} round 2 produced non-finite values in {key}")

    if cfg.topk_grad:
        if set(topk_compressors) != {1, 2, 3, 4}:
            raise AssertionError(f"{baseline} did not keep one Top-K compressor per client")
        residual = sum(
            compressor.error_buffer.abs().sum().item()
            for compressor in topk_compressors.values()
        )
        if residual <= 0.0:
            raise AssertionError(f"{baseline} Top-K error feedback buffer did not persist to round 2")

    print(
        f"PASS {baseline} two-round: "
        f"head_r1={round1.head.tolist()}, head_r2={round2.head.tolist()}"
    )
    return True


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser("Deterministic FedKDL baseline tests")
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument(
        "--delta-int8-only",
        action="store_true",
        help="Run only focused Delta-INT8 transport and drift tests.",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=tuple(STANDARD_BASELINES + OPTIONAL_BASELINES),
        help="Run dummy pipeline checks only for these baselines.",
    )
    args = parser.parse_args()

    run_delta_int8_tests()
    if args.delta_int8_only:
        print("PASS Delta-INT8: roundtrip, error reduction, BN/FedAvg, and 60-round drift")
        return

    test_baseline_contracts()
    test_topk_expected_values()
    test_topk_baseline_uses_five_percent_sparsity()
    test_topk_optional_sweep_ratios()
    test_topk_flatten_indices_are_stable_across_key_order_and_rounds()
    test_topk_int8_payload_bits_are_counted_without_len()
    test_lora_aggregation_strategies()
    test_server_mix_preserves_effective_lora_geometry()
    test_svd_factor_signs_are_canonical()
    test_kd_component_contributions()
    test_nearest_and_selective_partner_rules()
    test_latency_keeps_relay_paths_coupled()
    test_physics_accounting_contracts()

    results = {}
    if args.baselines:
        baselines = list(dict.fromkeys(args.baselines))
    else:
        baselines = list(STANDARD_BASELINES)
    if args.include_optional and not args.baselines:
        baselines.extend(OPTIONAL_BASELINES)
    for baseline in baselines:
        try:
            test_baseline(baseline)
            test_two_round_baseline(baseline)
            results[baseline] = True
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
