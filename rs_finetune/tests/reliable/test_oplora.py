"""Tests for OPLoRA (#16)."""

import pytest
import torch

from reliable.oplora import build_oplora_projectors


def test_build_oplora_projectors_shapes_and_idempotence(frozen_pretrained_weight):
    w = frozen_pretrained_weight(d_out=16, d_in=8)
    P_L, P_R = build_oplora_projectors(w, preserve_k=3)
    assert P_L.shape == (16, 16)
    assert P_R.shape == (8, 8)
    # Orthogonal projector must be idempotent: P @ P ≈ P
    assert torch.allclose(P_L @ P_L, P_L, atol=1e-5)
    assert torch.allclose(P_R @ P_R, P_R, atol=1e-5)


def test_build_oplora_projectors_rejects_out_of_range_k(frozen_pretrained_weight):
    w = frozen_pretrained_weight(d_out=8, d_in=4)
    with pytest.raises(ValueError, match="preserve_k"):
        build_oplora_projectors(w, preserve_k=100)
    with pytest.raises(ValueError, match="preserve_k"):
        build_oplora_projectors(w, preserve_k=-1)


def test_oplora_layer_zero_init_forward_matches_base(frozen_pretrained_weight):
    from reliable.oplora import OPLoRALayer

    w = frozen_pretrained_weight(d_out=16, d_in=8)
    layer = OPLoRALayer(d_in=8, d_out=16, rank=4, base_weight=w, preserve_k=3)
    x = torch.randn(4, 8)
    base_out = x @ w.T
    assert torch.allclose(layer(x), base_out, atol=1e-5)


def test_oplora_layer_preserves_top_k_after_arbitrary_update(
    frozen_pretrained_weight,
):
    from reliable.oplora import OPLoRALayer

    w = frozen_pretrained_weight(d_out=16, d_in=8)
    layer = OPLoRALayer(d_in=8, d_out=16, rank=4, base_weight=w, preserve_k=3)

    # Force a large non-zero update to A and B.
    with torch.no_grad():
        layer.A.copy_(torch.randn_like(layer.A))
        layer.B.copy_(torch.randn_like(layer.B))

    delta = layer.B @ layer.A
    effective = w + layer.P_L @ delta @ layer.P_R

    _U_pre, S_pre, _Vh_pre = torch.linalg.svd(w, full_matrices=False)
    _U_eff, S_eff, _Vh_eff = torch.linalg.svd(effective, full_matrices=False)
    # OPLoRA guarantee: the top-k singular triples of W remain exact singular
    # triples of (W + P_L @ delta @ P_R).  The LoRA delta lives entirely in the
    # orthogonal complement of span(U_k) ⊗ span(V_k), so each preserved triple
    # (u_i, σ_i, v_i) is untouched.  New, possibly larger, singular values may
    # appear from the perturbation, so S_pre[:k] need not equal S_eff[:k]
    # positionally; we check that each original value appears somewhere in the
    # full spectrum of the effective weight.
    for i, s in enumerate(S_pre[:3]):
        diffs = (S_eff - s).abs()
        assert diffs.min() < 1e-4, (
            f"S_pre[{i}]={s:.6f} not found in S_eff (min diff={diffs.min():.2e})"
        )
