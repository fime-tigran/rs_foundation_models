"""#2 Hard Channel Mask — non-learnable per-channel gate for LoRA residuals."""

from collections.abc import Iterable

import torch


def build_hard_channel_mask(
    training_channel_ids: Iterable[int], n_channels: int
) -> torch.Tensor:
    ids = list(training_channel_ids)
    for c in ids:
        if not 0 <= c < n_channels:
            raise ValueError(
                f"Channel id {c} out of range [0, {n_channels})"
            )
    mask = torch.zeros(n_channels)
    mask[ids] = 1.0
    mask.requires_grad_(False)
    return mask


def apply_hard_channel_mask(
    residual: torch.Tensor,
    mask: torch.Tensor,
    channel_ids: list[int],
) -> torch.Tensor:
    """Multiply per-channel residual by the hard mask.

    Args:
        residual: ``(B, C, ...)`` where ``C == len(channel_ids)``.
        mask: ``(n_channels,)`` frozen buffer from ``build_hard_channel_mask``.
        channel_ids: which channels each of the ``C`` positions corresponds to.
    """
    if residual.shape[1] != len(channel_ids):
        raise ValueError(
            f"residual has {residual.shape[1]} channel positions but "
            f"channel_ids has {len(channel_ids)}"
        )
    gate = mask[torch.tensor(channel_ids, device=residual.device)]
    shape = [1, len(channel_ids)] + [1] * (residual.ndim - 2)
    return residual * gate.view(*shape)
