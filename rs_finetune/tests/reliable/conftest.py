"""Shared fixtures for reliable-solutions test suite.

See ``.cursor/plans/reliable-solutions-test-plan.md`` §1.2 for the fixture
inventory. Fixtures are added one at a time under TDD.
"""

import pytest
import torch
import torch.nn as nn

_BATCH = 4
_IMG_HW = 32
_FEATURE_DIM = 64


class _TinyMockMultispecBackbone(nn.Module):
    """Channel-aware mock: one shared patch-embed per channel, tiny transformer,
    returns ``(B, n_ch, embed_dim)`` per-channel features. Stands in for
    χViT / TerraFM / DOFA at the post-embedding-generator interface."""

    def __init__(self, n_channels: int, embed_dim: int = 64, depth: int = 2,
                 patch: int = 4):
        super().__init__()
        self.n_channels = n_channels
        self.embed_dim = embed_dim
        self.patch = patch
        # Single-channel patch embed shared across all channels.
        self.patch_embed = nn.Conv2d(1, embed_dim, kernel_size=patch,
                                     stride=patch)
        # Learnable per-channel bias (simulates channel_embed table).
        self.channel_embed = nn.Parameter(torch.randn(n_channels, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=embed_dim * 2,
            batch_first=True, dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x: torch.Tensor, channel_ids: list[int]) -> torch.Tensor:
        # x: (B, C, H, W) with C == len(channel_ids)
        if x.shape[1] != len(channel_ids):
            raise ValueError(
                f"x has {x.shape[1]} channels but channel_ids has "
                f"{len(channel_ids)}"
            )
        B, C, _, _ = x.shape
        per_channel_feats = []
        for ch_idx, cid in enumerate(channel_ids):
            single = x[:, ch_idx : ch_idx + 1]          # (B, 1, H, W)
            tokens = self.patch_embed(single)            # (B, D, H/p, W/p)
            tokens = tokens.flatten(2).transpose(1, 2)   # (B, n_patches, D)
            tokens = tokens + self.channel_embed[cid]
            tokens = self.transformer(tokens)
            per_channel_feats.append(tokens.mean(dim=1))  # (B, D)
        return torch.stack(per_channel_feats, dim=1)      # (B, C, D)


class _TinyMockRGBOnlyBackbone(nn.Module):
    """RGB-only mock: 3-channel patch embed, tiny transformer, returns a single
    global CLS-equivalent ``(B, embed_dim)`` feature. Stands in for DINOv2/v3."""

    def __init__(self, embed_dim: int = 64, depth: int = 2, patch: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch,
                                     stride=patch)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=embed_dim * 2,
            batch_first=True, dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 3:
            raise ValueError(
                f"RGB-only backbone expected 3 channels, got {x.shape[1]}"
            )
        tokens = self.patch_embed(x)                 # (B, D, H/p, W/p)
        tokens = tokens.flatten(2).transpose(1, 2)   # (B, n_patches, D)
        tokens = self.transformer(tokens)
        return tokens.mean(dim=1)                    # (B, D)

# 12-band order used throughout the codebase:
# [B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12, VV, VH]
# Indices:  0    1    2    3    4    5    6    7    8    9   10  11
_B08_INDEX = 6
_VV_INDEX = 10
_VH_INDEX = 11


@pytest.fixture
def training_channel_ids():
    """RGB training-band indices in the 12-band order (B02, B03, B04)."""
    return [0, 1, 2]


@pytest.fixture
def eval_superset_channel_ids():
    """RGB training bands + B08 (NIR) for the Priority-B superset scenario."""
    return [0, 1, 2, _B08_INDEX]


@pytest.fixture
def eval_no_overlap_channel_ids():
    """SAR bands (VV, VH) for the Priority-A no-overlap scenario."""
    return [_VV_INDEX, _VH_INDEX]


@pytest.fixture
def synthetic_multispec_batch():
    """Factory for ``(B=4, n_channels, H=32, W=32)`` Gaussian batches."""

    def _make(n_channels: int) -> torch.Tensor:
        return torch.randn(_BATCH, n_channels, _IMG_HW, _IMG_HW)

    return _make


@pytest.fixture
def synthetic_per_channel_features():
    """Factory for post-embedding-generator features ``(B=4, n_channels, D=64)``."""

    def _make(n_channels: int) -> torch.Tensor:
        return torch.randn(_BATCH, n_channels, _FEATURE_DIM)

    return _make


@pytest.fixture
def synthetic_labels():
    """Factory for integer labels ``(B=4,)`` in ``[0, num_classes)``."""

    def _make(num_classes: int) -> torch.Tensor:
        return torch.randint(low=0, high=num_classes, size=(_BATCH,), dtype=torch.int64)

    return _make


@pytest.fixture
def frozen_pretrained_weight():
    """Factory for a frozen ``(d_out, d_in)`` weight matrix.

    Stand-in for any pretrained backbone weight we want to keep untouched
    during TDD of LoRA/projection techniques.
    """

    def _make(d_out: int, d_in: int) -> torch.Tensor:
        w = torch.randn(d_out, d_in)
        w.requires_grad_(False)
        return w

    return _make


@pytest.fixture
def tmp_artifact_dir(tmp_path):
    """Writable directory for cached offline artifacts (SVD bases, prototypes,
    Gaussians, conformal thresholds). Wraps pytest's ``tmp_path``.
    """
    d = tmp_path / "reliable_artifacts"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def tiny_mock_multispec_backbone():
    """Factory for a channel-aware mock backbone (χViT/TerraFM/DOFA stand-in)."""

    def _make(n_channels: int, embed_dim: int = _FEATURE_DIM) -> nn.Module:
        return _TinyMockMultispecBackbone(n_channels=n_channels, embed_dim=embed_dim)

    return _make


@pytest.fixture
def tiny_mock_rgb_only_backbone():
    """Factory for an RGB-only mock backbone (DINOv2/v3 stand-in)."""

    def _make(embed_dim: int = _FEATURE_DIM) -> nn.Module:
        return _TinyMockRGBOnlyBackbone(embed_dim=embed_dim)

    return _make
