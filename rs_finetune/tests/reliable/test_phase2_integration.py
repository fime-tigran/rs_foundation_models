"""Integration smoke test for Phase 2 training regularizers."""

import torch
import torch.nn as nn

from reliable.cdsd import EMATeacher, cdsd_loss, channel_dropout
from reliable.ch_rs_ft import smooth_channel_tokens
from reliable.lsmm_aux_head import LSMMHead


def test_phase2_regularizers_compose_on_synthetic_forward():
    # Tiny encoder: per-channel feature in, per-channel-pooled out.
    class _Encoder(nn.Module):
        def __init__(self, c, d):
            super().__init__()
            self.proj = nn.Linear(1, d)

        def forward(self, x):                     # x: (B, C, H, W)
            B, C, H, W = x.shape
            tokens = x.mean(dim=(2, 3))           # (B, C)
            return self.proj(tokens.unsqueeze(-1))  # (B, C, D)

    encoder = _Encoder(c=12, d=4)
    teacher = EMATeacher(encoder, momentum=0.99)

    x = torch.randn(2, 12, 8, 8)
    x_dropped = channel_dropout(x, p=0.3, min_keep=1, training=True)

    student_tokens = encoder(x_dropped)            # (B, 12, 4)
    teacher_tokens = teacher(x)                    # (B, 12, 4) under no_grad
    cdsd = cdsd_loss(student_tokens, teacher_tokens, lambda_distill=0.5)
    assert torch.isfinite(cdsd)

    # CH-RS-FT smoothing on the channel-token stream.
    smoothed = smooth_channel_tokens(
        student_tokens, sigma=0.1, p_smooth=0.5, training=True,
    )
    assert smoothed.shape == student_tokens.shape

    # LSMM aux loss on a pooled feature.
    pooled = student_tokens.mean(dim=1)            # (B, 4)
    head = LSMMHead(
        d=4, n_endmembers=8, n_bands=12,
        srf_matrix=torch.randn(3, 12),
        endmembers=torch.randn(12, 8),
    )
    rgb = x[:, :3].mean(dim=(2, 3))                # (B, 3) pooled RGB
    lsmm = head.reconstruction_loss(pooled, rgb, lambda_lsmm=0.3)
    assert torch.isfinite(lsmm)
