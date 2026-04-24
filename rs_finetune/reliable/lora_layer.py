"""Base LoRA module with zero-init B.

Given a frozen pretrained linear weight ``W ∈ R^{d_out x d_in}`` and a low
rank ``r``, the LoRA forward is::

    y = x @ W.T + x @ (B @ A).T

where ``A ∈ R^{r x d_in}`` is initialised small (Gaussian) and ``B ∈
R^{d_out x r}`` is zero-init so that ``B @ A`` is zero at construction and
the forward equals the base projection exactly.
"""

import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    def __init__(self, d_in: int, d_out: int, rank: int, base_weight: torch.Tensor):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank
        self.register_buffer("base_weight", base_weight)
        self.A = nn.Parameter(torch.randn(rank, d_in) * 0.01)
        self.B = nn.Parameter(torch.zeros(d_out, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = x @ self.base_weight.T
        delta = self.B @ self.A                 # (d_out, d_in)
        return base + x @ delta.T
