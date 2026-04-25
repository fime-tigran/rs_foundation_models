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


def test_mc_smooth_predict_returns_majority_and_count():
    import torch.nn

    from reliable.ch_rs_ft import mc_smooth_predict

    # Toy classifier: returns deterministic class 3 regardless of input.
    class _Const(torch.nn.Module):
        def forward(self, x):  # x: (B, C, D) → (B, num_classes)
            return torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0]] * x.shape[0])

    model = _Const()
    tokens = torch.zeros(1, 4, 8)
    pred, votes = mc_smooth_predict(
        model, tokens, n_mc=10, sigma=0.1, p_smooth=0.5,
    )
    # Returns class with shape (B,) and a vote count (B,) for the majority.
    assert pred.shape == (1,)
    assert votes.shape == (1,)
    assert pred.item() == 3
    assert votes.item() == 10  # constant model → 10/10 votes


def test_mc_smooth_predict_n_mc_one_runs_once():
    import torch.nn

    from reliable.ch_rs_ft import mc_smooth_predict

    call_count = {"n": 0}

    class _Counter(torch.nn.Module):
        def forward(self, x):
            call_count["n"] += 1
            return torch.zeros(x.shape[0], 5)

    pred, votes = mc_smooth_predict(
        _Counter(), torch.zeros(1, 4, 8),
        n_mc=1, sigma=0.1, p_smooth=0.5,
    )
    assert call_count["n"] == 1


def test_mc_smooth_predict_rejects_n_mc_zero():
    import pytest

    from reliable.ch_rs_ft import mc_smooth_predict

    class _Stub(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 5)

    with pytest.raises(ValueError, match="n_mc"):
        mc_smooth_predict(
            _Stub(), torch.zeros(1, 4, 8),
            n_mc=0, sigma=0.1, p_smooth=0.5,
        )
