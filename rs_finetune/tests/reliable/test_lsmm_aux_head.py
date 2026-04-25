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
