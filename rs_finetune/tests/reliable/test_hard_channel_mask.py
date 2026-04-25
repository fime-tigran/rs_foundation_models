"""Tests for Hard Channel Mask (#2)."""

import pytest
import torch

from reliable.channel_mask import build_hard_channel_mask


def test_hard_channel_mask_is_non_learnable_and_correct():
    mask = build_hard_channel_mask(
        training_channel_ids=[0, 1, 2], n_channels=12
    )
    assert mask.requires_grad is False
    assert mask.shape == (12,)
    expected = torch.zeros(12)
    expected[[0, 1, 2]] = 1.0
    assert torch.equal(mask, expected)


def test_hard_channel_mask_rejects_out_of_range_ids():
    with pytest.raises(ValueError, match="out of range"):
        build_hard_channel_mask(training_channel_ids=[0, 12], n_channels=12)
    with pytest.raises(ValueError, match="out of range"):
        build_hard_channel_mask(training_channel_ids=[-1, 0], n_channels=12)


def test_apply_hard_channel_mask_zeros_unseen_channels():
    from reliable.channel_mask import apply_hard_channel_mask

    residual = torch.ones(2, 12, 64)
    mask = build_hard_channel_mask(
        training_channel_ids=[0, 1, 2], n_channels=12
    )
    gated = apply_hard_channel_mask(residual, mask, channel_ids=list(range(12)))
    # Training channels unchanged; unseen zeroed.
    assert torch.equal(gated[:, :3, :], residual[:, :3, :])
    assert torch.equal(gated[:, 3:, :], torch.zeros_like(gated[:, 3:, :]))


def test_apply_hard_channel_mask_shape_mismatch_raises():
    from reliable.channel_mask import apply_hard_channel_mask

    residual = torch.ones(2, 4, 32)
    mask = build_hard_channel_mask(
        training_channel_ids=[0, 1, 2], n_channels=12
    )
    with pytest.raises(ValueError, match="channel"):
        apply_hard_channel_mask(residual, mask, channel_ids=[0, 1, 2])
