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
