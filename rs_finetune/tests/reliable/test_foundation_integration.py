"""Integration tests verifying #29 + #2 compose.

Full LoRA-on-attention-with-channel-gating integration lives in Phase 2
once the head and channel-aware LoRA wrapper ship. This file only asserts
the utilities plumb together without type errors."""

import torch

from reliable.channel_mask import apply_hard_channel_mask, build_hard_channel_mask


def test_mask_gates_a_synthetic_per_channel_residual():
    mask = build_hard_channel_mask(
        training_channel_ids=[0, 1, 2], n_channels=12
    )
    # Pretend we have a per-channel residual for channels [0, 1, 2, 6, 10, 11]
    # where 6, 10, 11 are unseen (NIR + SAR).
    channel_ids = [0, 1, 2, 6, 10, 11]
    residual = torch.ones(2, len(channel_ids), 64)
    gated = apply_hard_channel_mask(residual, mask, channel_ids=channel_ids)
    # Training positions (0, 1, 2) unchanged.
    assert torch.equal(gated[:, :3, :], residual[:, :3, :])
    # Unseen positions zeroed.
    assert torch.equal(gated[:, 3:, :], torch.zeros_like(gated[:, 3:, :]))
