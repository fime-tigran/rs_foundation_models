"""#29 LastN-LoRA placement helper.

Attaches LoRA adapters to the last N layers of a transformer-style backbone
so earlier layers stay bit-identical to the pretrained model.

For each targeted layer, replaces ``self_attn.out_proj`` (an ``nn.Linear``)
with a :class:`reliable.lora_layer.LoRALayer` wrapping the original weight
as a frozen base, then sets a ``_lora_registry`` marker for introspection.
"""

import torch.nn as nn

from reliable.lora_layer import LoRALayer


def _wrap_linear_with_lora(linear: nn.Linear, rank: int) -> LoRALayer:
    return LoRALayer(
        d_in=linear.in_features,
        d_out=linear.out_features,
        rank=rank,
        base_weight=linear.weight.detach().clone(),
    )


def attach_lora_to_last_n(model: nn.Module, last_n: int, rank: int) -> None:
    """Attach LoRA adapters to the last ``last_n`` transformer layers.

    No-op when ``last_n <= 0``. Raises :class:`ValueError` when the model
    does not expose a ``transformer.layers`` attribute.
    """
    if last_n <= 0:
        return
    transformer = getattr(model, "transformer", None)
    if transformer is None or not hasattr(transformer, "layers"):
        raise ValueError("Model has no `transformer.layers` to attach LoRA to")
    layers = transformer.layers
    total = len(layers)
    first = max(0, total - last_n)
    for idx in range(first, total):
        layer = layers[idx]
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "out_proj"):
            layer.self_attn.out_proj = _wrap_linear_with_lora(
                layer.self_attn.out_proj, rank=rank
            )
        layer._lora_registry = {"rank": rank}
