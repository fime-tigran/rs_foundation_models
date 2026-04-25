"""Tests for LSMM auxiliary head (#23)."""

import torch

from reliable.lsmm_aux_head import LSMMHead


def test_lsmm_head_endmembers_and_srf_are_buffers():
    K, D, n_bands = 16, 64, 12
    endmembers = torch.randn(n_bands, K)
    srf_matrix = torch.randn(3, n_bands)
    head = LSMMHead(
        d=D, n_endmembers=K, n_bands=n_bands,
        srf_matrix=srf_matrix, endmembers=endmembers,
    )
    bufs = dict(head.named_buffers())
    params = dict(head.named_parameters())
    assert "endmembers" in bufs and "endmembers" not in params
    assert "srf_matrix" in bufs and "srf_matrix" not in params


def test_lsmm_head_abundances_are_non_negative():
    K, D, n_bands = 8, 32, 12
    head = LSMMHead(
        d=D, n_endmembers=K, n_bands=n_bands,
        srf_matrix=torch.randn(3, n_bands),
        endmembers=torch.randn(n_bands, K),
    )
    feats = torch.randn(4, D) * 5.0  # large negative inputs possible
    alpha = head.predict_abundances(feats)
    assert alpha.shape == (4, K)
    assert (alpha >= 0).all()


def test_lsmm_reconstruction_loss_finite_and_lambda_zero_kills():
    K, D, n_bands = 8, 32, 12
    head = LSMMHead(
        d=D, n_endmembers=K, n_bands=n_bands,
        srf_matrix=torch.randn(3, n_bands),
        endmembers=torch.randn(n_bands, K),
    )
    feats = torch.randn(4, D)
    x_rgb = torch.randn(4, 3)
    loss = head.reconstruction_loss(feats, x_rgb, lambda_lsmm=0.5)
    assert torch.isfinite(loss)
    loss_zero = head.reconstruction_loss(feats, x_rgb, lambda_lsmm=0.0)
    assert loss_zero.item() == 0.0
