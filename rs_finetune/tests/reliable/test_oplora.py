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
