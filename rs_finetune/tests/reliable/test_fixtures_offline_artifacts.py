"""Smoke tests for frozen-weight and offline-artifact fixtures."""

import torch


def test_frozen_pretrained_weight_shape(frozen_pretrained_weight):
    w = frozen_pretrained_weight(d_out=64, d_in=32)
    assert w.shape == (64, 32)


def test_frozen_pretrained_weight_no_grad(frozen_pretrained_weight):
    w = frozen_pretrained_weight(d_out=64, d_in=32)
    assert w.requires_grad is False


def test_frozen_pretrained_weight_square(frozen_pretrained_weight):
    w = frozen_pretrained_weight(d_out=16, d_in=16)
    assert w.shape == (16, 16)


def test_frozen_pretrained_weight_dtype_is_float32(frozen_pretrained_weight):
    w = frozen_pretrained_weight(d_out=8, d_in=4)
    assert w.dtype == torch.float32


def test_tmp_artifact_dir_exists(tmp_artifact_dir):
    assert tmp_artifact_dir.exists()
    assert tmp_artifact_dir.is_dir()


def test_tmp_artifact_dir_writable(tmp_artifact_dir):
    target = tmp_artifact_dir / "sample.pt"
    torch.save(torch.randn(3, 3), target)
    assert target.exists()
    loaded = torch.load(target, weights_only=True)
    assert loaded.shape == (3, 3)
