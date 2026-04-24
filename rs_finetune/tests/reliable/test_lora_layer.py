"""Tests for the base LoRA layer."""

import torch

from reliable.lora_layer import LoRALayer


def test_lora_zero_init_forward_matches_base(frozen_pretrained_weight):
    w = frozen_pretrained_weight(d_out=32, d_in=16)
    lora = LoRALayer(d_in=16, d_out=32, rank=4, base_weight=w)
    # Base weight is a buffer, never a parameter.
    assert "base_weight" in dict(lora.named_buffers())
    assert "base_weight" not in dict(lora.named_parameters())
    # Zero-init B → delta is zero → forward equals base.
    x = torch.randn(2, 16)
    base_out = x @ w.T
    lora_out = lora(x)
    assert torch.allclose(base_out, lora_out, atol=1e-6)


def test_lora_with_base_bias_applies_bias(frozen_pretrained_weight):
    """When a frozen base_bias is passed, zero-init forward equals
    ``x @ W.T + b`` — bias from the wrapped Linear is preserved."""
    w = frozen_pretrained_weight(d_out=8, d_in=4)
    b = torch.randn(8)
    b.requires_grad_(False)
    lora = LoRALayer(d_in=4, d_out=8, rank=2, base_weight=w, base_bias=b)
    assert "base_bias" in dict(lora.named_buffers())
    assert "base_bias" not in dict(lora.named_parameters())
    x = torch.randn(3, 4)
    expected = x @ w.T + b
    assert torch.allclose(lora(x), expected, atol=1e-6)
