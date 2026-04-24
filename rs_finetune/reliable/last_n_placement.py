"""#29 LastN-LoRA placement helper.

Attaches LoRA adapters to the last N transformer layers' MLP linears
(``linear1`` and ``linear2``) so earlier layers stay bit-identical to the
pretrained model. Attention projections are not wrapped because
``nn.MultiheadAttention`` introspects ``out_proj.weight`` directly.
"""

import torch.nn as nn

from reliable.lora_layer import LoRALayer


def _wrap_linear_with_lora(linear: nn.Linear, rank: int) -> LoRALayer:
    base_bias = (
        linear.bias.detach().clone() if linear.bias is not None else None
    )
    return LoRALayer(
        d_in=linear.in_features,
        d_out=linear.out_features,
        rank=rank,
        base_weight=linear.weight.detach().clone(),
        base_bias=base_bias,
    )


_LORA_TARGETS = ("linear1", "linear2")


def attach_lora_to_last_n(model: nn.Module, last_n: int, rank: int) -> None:
    """Attach LoRA adapters to the last ``last_n`` transformer layers' MLP
    projections.

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
        for attr in _LORA_TARGETS:
            target = getattr(layer, attr, None)
            if isinstance(target, nn.Linear):
                setattr(layer, attr, _wrap_linear_with_lora(target, rank=rank))
        layer._lora_registry = {"rank": rank, "targets": list(_LORA_TARGETS)}
