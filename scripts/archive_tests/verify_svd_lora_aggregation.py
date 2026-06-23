"""
Verify SVD-LoRA aggregation invariants with synthetic tensors.

Run:
    python scripts/fedkdl/verify_svd_lora_aggregation.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from federated_core.aggregator import svd_lora_aggregate
from federated_core.hfl_rules import blend_state_dicts


def _best_rank_r(matrix: torch.Tensor, rank: int) -> torch.Tensor:
    U, S, Vh = torch.linalg.svd(matrix.double(), full_matrices=False)
    keep = min(rank, S.numel())
    return (U[:, :keep] @ torch.diag(S[:keep]) @ Vh[:keep, :]).float()


def _effective(sd: dict, prefix: str) -> torch.Tensor:
    return sd[f"{prefix}.lora_B"].float() @ sd[f"{prefix}.lora_A"].float()


def _make_state(prefix: str, out_dim: int, in_dim: int, rank: int, seed: int) -> dict:
    generator = torch.Generator().manual_seed(seed)
    return {
        f"{prefix}.lora_A": torch.randn(rank, in_dim, generator=generator) * 0.05,
        f"{prefix}.lora_B": torch.randn(out_dim, rank, generator=generator) * 0.05,
        "head.weight": torch.randn(4, out_dim, generator=generator) * 0.01,
        "head.bias": torch.randn(4, generator=generator) * 0.01,
    }


def test_identical_client_preserves_effective_update() -> None:
    prefix = "model.10.cv1.conv"
    state = _make_state(prefix, out_dim=16, in_dim=27, rank=4, seed=1)
    agg = svd_lora_aggregate([state, state], [0.25, 0.75])
    assert torch.allclose(_effective(agg, prefix), _effective(state, prefix), atol=1e-6)


def test_weighted_clients_matches_best_rank_projection() -> None:
    prefix = "model.10.cv1.conv"
    s1 = _make_state(prefix, out_dim=16, in_dim=27, rank=4, seed=2)
    s2 = _make_state(prefix, out_dim=16, in_dim=27, rank=4, seed=3)
    weights = [0.3, 0.7]
    target = weights[0] * _effective(s1, prefix) + weights[1] * _effective(s2, prefix)
    expected = _best_rank_r(target, rank=4)
    agg = svd_lora_aggregate([s1, s2], weights)
    assert torch.allclose(_effective(agg, prefix), expected, atol=1e-6)
    assert torch.allclose(agg["head.weight"], weights[0] * s1["head.weight"] + weights[1] * s2["head.weight"])
    assert torch.allclose(agg["head.bias"], weights[0] * s1["head.bias"] + weights[1] * s2["head.bias"])


def test_nonfinite_client_fails_fast() -> None:
    prefix = "model.10.cv1.conv"
    s1 = _make_state(prefix, out_dim=16, in_dim=27, rank=4, seed=4)
    s2 = _make_state(prefix, out_dim=16, in_dim=27, rank=4, seed=5)
    s2[f"{prefix}.lora_A"][0, 0] = float("nan")
    try:
        svd_lora_aggregate([s1, s2], [0.5, 0.5])
    except RuntimeError as exc:
        assert "Non-finite LoRA factors" in str(exc)
    else:
        raise AssertionError("Expected non-finite LoRA factors to fail fast")


def test_hfl_blend_uses_same_svd_path() -> None:
    prefix = "model.10.cv1.conv"
    s1 = _make_state(prefix, out_dim=16, in_dim=27, rank=4, seed=6)
    s2 = _make_state(prefix, out_dim=16, in_dim=27, rank=4, seed=7)
    alpha = 0.8
    target = alpha * _effective(s1, prefix) + (1.0 - alpha) * _effective(s2, prefix)
    expected = _best_rank_r(target, rank=4)
    blended = blend_state_dicts(s1, s2, alpha=alpha)
    assert torch.allclose(_effective(blended, prefix), expected, atol=1e-6)


def main() -> None:
    test_identical_client_preserves_effective_update()
    test_weighted_clients_matches_best_rank_projection()
    test_nonfinite_client_fails_fast()
    test_hfl_blend_uses_same_svd_path()
    print("SVD-LoRA aggregation checks passed.")


if __name__ == "__main__":
    main()
