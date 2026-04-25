"""Tests for LoRA-Null initialization (#20)."""

import pytest
import torch

from reliable.lora_null_init import compute_activation_null_basis


def test_compute_activation_null_basis_shape_and_orthonormal():
    # 100 samples of 8-dim activations; null rank 5.
    activations = torch.randn(100, 8)
    U_null = compute_activation_null_basis(activations, null_rank=5)
    assert U_null.shape == (8, 5)
    # Columns orthonormal.
    gram = U_null.T @ U_null
    assert torch.allclose(gram, torch.eye(5), atol=1e-4)


def test_compute_activation_null_basis_rejects_bad_rank():
    activations = torch.randn(20, 8)
    with pytest.raises(ValueError, match="null_rank"):
        compute_activation_null_basis(activations, null_rank=-1)
    with pytest.raises(ValueError, match="null_rank"):
        compute_activation_null_basis(activations, null_rank=9)


def test_compute_activation_null_basis_rejects_non_2d():
    x3d = torch.randn(2, 3, 4)
    with pytest.raises(ValueError, match="must be 2D"):
        compute_activation_null_basis(x3d, null_rank=2)


def test_init_lora_a_in_null_space_kills_subset_activations(
    frozen_pretrained_weight,
):
    """After init, (B @ A) @ x ≈ 0 for any x ∈ span(subset activations).
    Since B is zero at init, the assertion reduces to checking A @ x ≈ 0."""
    from reliable.lora_layer import LoRALayer
    from reliable.lora_null_init import init_lora_a_in_null_space

    # Structured activations: signal on first 4 dims, null on last 4 dims.
    activations = torch.cat(
        [torch.randn(50, 4), torch.zeros(50, 4)], dim=1
    )
    w = frozen_pretrained_weight(d_out=16, d_in=8)
    lora = LoRALayer(d_in=8, d_out=16, rank=4, base_weight=w)
    init_lora_a_in_null_space(lora, activations, null_rank=4)

    projected = activations @ lora.A.T  # (50, 4)
    assert projected.abs().mean() < 1e-4


def test_init_lora_a_rank_mismatch_raises(frozen_pretrained_weight):
    from reliable.lora_layer import LoRALayer
    from reliable.lora_null_init import init_lora_a_in_null_space

    activations = torch.randn(50, 8)
    w = frozen_pretrained_weight(d_out=16, d_in=8)
    lora = LoRALayer(d_in=8, d_out=16, rank=3, base_weight=w)
    with pytest.raises(ValueError, match="rank"):
        init_lora_a_in_null_space(lora, activations, null_rank=5)
