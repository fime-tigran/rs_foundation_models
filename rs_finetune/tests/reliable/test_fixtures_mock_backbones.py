"""Smoke tests for mock backbone fixtures.

Mocks stand in for the five real encoders in the comparison table:
  multispec-aware: χViT, TerraFM, DOFA
  RGB-only:        DINOv2, DINOv3
"""

import pytest
import torch


def test_tiny_mock_multispec_backbone_per_channel_features(
    tiny_mock_multispec_backbone, synthetic_multispec_batch
):
    """Multispec backbone returns one feature vector per input channel."""
    model = tiny_mock_multispec_backbone(n_channels=12, embed_dim=64)
    x = synthetic_multispec_batch(n_channels=3)
    feats = model(x, channel_ids=[0, 1, 2])
    assert feats.shape == (4, 3, 64)


def test_tiny_mock_multispec_backbone_variable_channel_count(
    tiny_mock_multispec_backbone, synthetic_multispec_batch
):
    """Same backbone handles different channel subsets at forward time."""
    model = tiny_mock_multispec_backbone(n_channels=12, embed_dim=64)
    for ids in ([0, 1, 2], [0, 1, 2, 6], [10, 11], list(range(12))):
        x = synthetic_multispec_batch(n_channels=len(ids))
        feats = model(x, channel_ids=ids)
        assert feats.shape == (4, len(ids), 64)


def test_tiny_mock_multispec_backbone_has_params(tiny_mock_multispec_backbone):
    model = tiny_mock_multispec_backbone(n_channels=12, embed_dim=64)
    assert any(p.requires_grad for p in model.parameters())


def test_tiny_mock_rgb_only_backbone_global_feature(
    tiny_mock_rgb_only_backbone, synthetic_multispec_batch
):
    """RGB-only backbone accepts (B, 3, H, W) and returns global CLS (B, D)."""
    model = tiny_mock_rgb_only_backbone(embed_dim=64)
    x = synthetic_multispec_batch(n_channels=3)
    feats = model(x)
    assert feats.shape == (4, 64)


def test_tiny_mock_rgb_only_backbone_rejects_non_rgb(
    tiny_mock_rgb_only_backbone, synthetic_multispec_batch
):
    """RGB-only backbone signals a clear error on non-3-channel input."""
    model = tiny_mock_rgb_only_backbone(embed_dim=64)
    x_4ch = synthetic_multispec_batch(n_channels=4)
    with pytest.raises((ValueError, RuntimeError)):
        model(x_4ch)


def test_tiny_mock_multispec_backbone_forward_fast(
    tiny_mock_multispec_backbone, synthetic_multispec_batch
):
    """Sanity: tiny mock is small enough to forward quickly (no asserts on
    timing — this test just confirms the forward returns without error on a
    larger channel set)."""
    model = tiny_mock_multispec_backbone(n_channels=12, embed_dim=64)
    x = synthetic_multispec_batch(n_channels=12)
    feats = model(x, channel_ids=list(range(12)))
    assert torch.isfinite(feats).all()
