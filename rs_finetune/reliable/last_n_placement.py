"""#29 LastN-LoRA placement helper.

Attaches LoRA adapters to the last N layers of a transformer-style backbone
so earlier layers stay bit-identical to the pretrained model.
"""

import torch.nn as nn


def attach_lora_to_last_n(model: nn.Module, last_n: int, rank: int) -> None:
    """Attach a LoRA marker dict to the last ``last_n`` transformer layers.

    No-op when ``last_n <= 0``. Raises :class:`ValueError` when the model
    does not expose a ``transformer.layers`` attribute (the interface
    assumed by our mock backbones and by the ``nn.TransformerEncoder``
    family).
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
        layers[idx]._lora_registry = {"rank": rank}
