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


def test_attach_last_n_replaces_mlp_linears_on_tail(tiny_mock_multispec_backbone):
    """Tail layer's MLP linears (linear1, linear2) become LoRALayers
    wrapping the original weights; earlier layers keep plain nn.Linear.

    Attention out_proj is intentionally left alone because
    nn.MultiheadAttention reads out_proj.weight directly rather than
    calling the module."""
    from reliable.lora_layer import LoRALayer

    model = tiny_mock_multispec_backbone(n_channels=4, embed_dim=32)
    attach_lora_to_last_n(model, last_n=1, rank=4)

    tail = model.transformer.layers[-1]
    assert isinstance(tail.linear1, LoRALayer)
    assert isinstance(tail.linear2, LoRALayer)

    head = model.transformer.layers[0]
    assert isinstance(head.linear1, nn.Linear)
    assert isinstance(head.linear2, nn.Linear)
    assert not isinstance(head.linear1, LoRALayer)
    assert not isinstance(head.linear2, LoRALayer)


def test_attach_last_n_preserves_forward_at_init(
    tiny_mock_multispec_backbone, synthetic_multispec_batch
):
    """With zero-init B and bias preserved, wrapping tail MLP linears must
    not change the model's forward output — in eval AND train mode (the
    latter relies on the conftest fixture's dropout=0.0)."""
    import copy

    model = tiny_mock_multispec_backbone(n_channels=4, embed_dim=32)
    reference = copy.deepcopy(model)
    attach_lora_to_last_n(model, last_n=1, rank=4)

    x = synthetic_multispec_batch(n_channels=3)
    channel_ids = [0, 1, 2]

    # Eval-mode equivalence (Module.__init__ default).
    with torch.no_grad():
        ref_out_eval = reference(x, channel_ids=channel_ids)
        mod_out_eval = model(x, channel_ids=channel_ids)
    assert torch.allclose(ref_out_eval, mod_out_eval, atol=1e-5)

    # Train-mode equivalence — depends on conftest's dropout=0.0 in the
    # mock backbones, otherwise stochastic dropout would diverge the two
    # deep-copied instances even with zero-init LoRA.
    reference.train()
    model.train()
    with torch.no_grad():
        ref_out_train = reference(x, channel_ids=channel_ids)
        mod_out_train = model(x, channel_ids=channel_ids)
    assert torch.allclose(ref_out_train, mod_out_train, atol=1e-5)


def test_attach_last_n_with_oplora_adapter_class(
    tiny_mock_multispec_backbone,
):
    """attach_lora_to_last_n accepts an adapter_cls kwarg so OPLoRALayer
    can be placed via the same code path as LoRALayer."""
    from reliable.lora_layer import LoRALayer
    from reliable.oplora import OPLoRALayer

    model = tiny_mock_multispec_backbone(n_channels=4, embed_dim=32)
    attach_lora_to_last_n(
        model, last_n=1, rank=4, adapter_cls=OPLoRALayer, preserve_k=2
    )
    tail = model.transformer.layers[-1]
    assert isinstance(tail.linear1, OPLoRALayer)
    assert not isinstance(tail.linear1, LoRALayer)
    head = model.transformer.layers[0]
    assert isinstance(head.linear1, nn.Linear)
