"""#23 LSMM — Linear Spectral Mixing Model auxiliary reconstruction head.

The head predicts per-patch abundance vectors ``α ∈ R^K`` from input
features, then reconstructs an RGB observation as ``x_RGB ≈ SRF · E · α``,
where ``E`` is a frozen ``(n_bands, K)`` endmember dictionary (offline VCA
on the pretraining corpus) and ``SRF`` is a frozen ``(3, n_bands)`` matrix
of Sentinel-2 spectral response functions.

Used as a training-time regularizer: the reconstruction loss steers the
encoder toward features consistent with a physics-grounded multispectral
manifold. The head is discarded at eval — the auxiliary signal does not
ride into deployment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSMMHead(nn.Module):
    def __init__(
        self,
        d: int,
        n_endmembers: int,
        n_bands: int,
        srf_matrix: torch.Tensor,
        endmembers: torch.Tensor,
    ):
        super().__init__()
        if endmembers.shape != (n_bands, n_endmembers):
            raise ValueError(
                f"endmembers must have shape ({n_bands}, {n_endmembers}); "
                f"got {tuple(endmembers.shape)}"
            )
        if srf_matrix.shape != (3, n_bands):
            raise ValueError(
                f"srf_matrix must have shape (3, {n_bands}); "
                f"got {tuple(srf_matrix.shape)}"
            )
        self.register_buffer("endmembers", endmembers.detach().clone())
        self.register_buffer("srf_matrix", srf_matrix.detach().clone())
        self.abundance_predictor = nn.Linear(d, n_endmembers)

    def predict_abundances(self, features: torch.Tensor) -> torch.Tensor:
        """Return non-negative abundances of shape ``(..., n_endmembers)``."""
        raw = self.abundance_predictor(features)
        return F.softplus(raw)
