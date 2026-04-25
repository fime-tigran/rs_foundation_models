"""Tests for Channel-Token Hierarchical Randomized Smoothing FT (#15)."""

import torch

from reliable.ch_rs_ft import smooth_channel_tokens


def test_smooth_channel_tokens_train_perturbs_subset():
    tokens = torch.zeros(2, 12, 8)
    torch.manual_seed(0)
    smoothed = smooth_channel_tokens(
        tokens, sigma=1.0, p_smooth=0.5, training=True
    )
    # Some channels were noised (non-zero), some kept clean (zero).
    nonzero_per_batch = (smoothed.abs().sum(dim=-1) > 0).sum(dim=-1)
    assert (nonzero_per_batch > 0).all()
    assert (nonzero_per_batch < 12).all()


def test_smooth_channel_tokens_eval_passthrough():
    tokens = torch.randn(2, 12, 8)
    out = smooth_channel_tokens(
        tokens, sigma=1.0, p_smooth=0.5, training=False
    )
    assert torch.equal(tokens, out)


def test_smooth_channel_tokens_sigma_zero_passthrough():
    tokens = torch.randn(2, 12, 8)
    out = smooth_channel_tokens(
        tokens, sigma=0.0, p_smooth=0.5, training=True
    )
    assert torch.equal(tokens, out)


def test_smooth_channel_tokens_p_zero_passthrough():
    tokens = torch.randn(2, 12, 8)
    out = smooth_channel_tokens(
        tokens, sigma=1.0, p_smooth=0.0, training=True
    )
    assert torch.equal(tokens, out)
