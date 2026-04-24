"""Tests for the LastN-LoRA placement helper (#29)."""

import pytest
import torch
import torch.nn as nn

from reliable.last_n_placement import attach_lora_to_last_n


def test_attach_last_n_wraps_only_tail(tiny_mock_multispec_backbone):
    model = tiny_mock_multispec_backbone(n_channels=4, embed_dim=32)
    # The mock wraps a 2-layer transformer; attach LoRA to the last 1.
    attach_lora_to_last_n(model, last_n=1, rank=4)
    # Block 0 is untouched; block 1 has an attached LoRA registry.
    assert not hasattr(model.transformer.layers[0], "_lora_registry")
    assert hasattr(model.transformer.layers[1], "_lora_registry")


def test_attach_last_n_greater_than_depth_attaches_all(
    tiny_mock_multispec_backbone,
):
    model = tiny_mock_multispec_backbone(n_channels=4, embed_dim=32)
    attach_lora_to_last_n(model, last_n=99, rank=4)
    assert all(hasattr(lay, "_lora_registry") for lay in model.transformer.layers)


def test_attach_last_n_zero_is_noop(tiny_mock_multispec_backbone):
    model = tiny_mock_multispec_backbone(n_channels=4, embed_dim=32)
    attach_lora_to_last_n(model, last_n=0, rank=4)
    assert all(
        not hasattr(lay, "_lora_registry") for lay in model.transformer.layers
    )


def test_attach_last_n_rejects_non_transformer_model():
    bare = nn.Sequential(nn.Linear(4, 4))
    with pytest.raises(ValueError, match="transformer.layers"):
        attach_lora_to_last_n(bare, last_n=2, rank=4)


def test_attach_last_n_replaces_linears_on_tail(tiny_mock_multispec_backbone):
    """Tail layer's attention out_proj becomes a LoRALayer wrapping the
    original Linear; earlier layers keep plain nn.Linear."""
    from reliable.lora_layer import LoRALayer

    model = tiny_mock_multispec_backbone(n_channels=4, embed_dim=32)
    attach_lora_to_last_n(model, last_n=1, rank=4)

    tail = model.transformer.layers[-1]
    assert isinstance(tail.self_attn.out_proj, LoRALayer)

    head = model.transformer.layers[0]
    assert isinstance(head.self_attn.out_proj, nn.Linear)
    assert not isinstance(head.self_attn.out_proj, LoRALayer)
