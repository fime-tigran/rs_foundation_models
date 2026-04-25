"""#16 OPLoRA — orthogonal projection on LoRA updates.

Given a frozen base weight ``W ∈ R^{d_out x d_in}`` with SVD
``W = U Σ Vᵀ`` and a "preserve rank" ``k``, the OPLoRA projectors are::

    P_L = I_{d_out} - U[:, :k] @ U[:, :k]ᵀ      # (d_out, d_out)
    P_R = I_{d_in}  - V[:, :k] @ V[:, :k]ᵀ      # (d_in,  d_in)

Any LoRA delta ``ΔW = B @ A`` is double-projected as ``P_L @ ΔW @ P_R``
before being added to ``W``. This preserves the top-``k`` singular triples
of ``W`` exactly under fine-tuning.
"""

import torch
import torch.nn as nn


def build_oplora_projectors(
    weight: torch.Tensor, preserve_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if preserve_k < 0:
        raise ValueError(f"preserve_k must be non-negative, got {preserve_k}")
    d_out, d_in = weight.shape
    if preserve_k > min(d_out, d_in):
        raise ValueError(
            f"preserve_k={preserve_k} exceeds min(d_out={d_out}, d_in={d_in})"
        )
    U, _S, Vh = torch.linalg.svd(weight, full_matrices=False)
    U_k = U[:, :preserve_k]
    V_k = Vh[:preserve_k, :].T
    P_L = torch.eye(d_out, device=weight.device) - U_k @ U_k.T
    P_R = torch.eye(d_in, device=weight.device) - V_k @ V_k.T
    return P_L, P_R


class OPLoRALayer(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        rank: int,
        base_weight: torch.Tensor,
        preserve_k: int,
        base_bias: torch.Tensor | None = None,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank
        self.register_buffer("base_weight", base_weight)
        if base_bias is not None:
            self.register_buffer("base_bias", base_bias)
        else:
            self.base_bias = None
        P_L, P_R = build_oplora_projectors(base_weight, preserve_k=preserve_k)
        self.register_buffer("P_L", P_L)
        self.register_buffer("P_R", P_R)
        self.A = nn.Parameter(torch.randn(rank, d_in) * 0.01)
        self.B = nn.Parameter(torch.zeros(d_out, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = x @ self.base_weight.T
        if self.base_bias is not None:
            base = base + self.base_bias
        delta = self.B @ self.A                       # (d_out, d_in)
        projected = self.P_L @ delta @ self.P_R
        return base + x @ projected.T
