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
