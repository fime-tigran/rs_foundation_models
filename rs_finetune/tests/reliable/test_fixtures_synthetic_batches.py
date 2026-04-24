"""Smoke tests for synthetic-batch fixtures."""

import torch


def test_synthetic_multispec_batch_default_shape(synthetic_multispec_batch):
    x = synthetic_multispec_batch(n_channels=3)
    assert x.shape == (4, 3, 32, 32)
    assert x.dtype == torch.float32


def test_synthetic_multispec_batch_variable_channels(synthetic_multispec_batch):
    for c in (1, 3, 4, 10, 12):
        x = synthetic_multispec_batch(n_channels=c)
        assert x.shape == (4, c, 32, 32)


def test_synthetic_per_channel_features_shape(synthetic_per_channel_features):
    f = synthetic_per_channel_features(n_channels=3)
    assert f.shape == (4, 3, 64)
    assert f.dtype == torch.float32


def test_synthetic_per_channel_features_variable_channels(
    synthetic_per_channel_features,
):
    for c in (1, 3, 4, 10, 12):
        f = synthetic_per_channel_features(n_channels=c)
        assert f.shape == (4, c, 64)


def test_synthetic_labels_shape_and_range(synthetic_labels):
    y = synthetic_labels(num_classes=10)
    assert y.shape == (4,)
    assert y.dtype == torch.int64
    assert (y >= 0).all() and (y < 10).all()


def test_synthetic_labels_respects_num_classes(synthetic_labels):
    for k in (2, 5, 10, 100):
        y = synthetic_labels(num_classes=k)
        assert (y >= 0).all() and (y < k).all()


def test_synthetic_multispec_batch_is_deterministic_under_seed(
    synthetic_multispec_batch,
):
    # The autouse _seed_torch fixture in tests/conftest.py seeds to 42.
    torch.manual_seed(42)
    a = synthetic_multispec_batch(n_channels=3)
    torch.manual_seed(42)
    b = synthetic_multispec_batch(n_channels=3)
    assert torch.equal(a, b)
