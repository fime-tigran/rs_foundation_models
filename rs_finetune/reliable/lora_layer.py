"""Base LoRA module with zero-init B and optional frozen base bias."""

import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        rank: int,
        base_weight: torch.Tensor,
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
        self.A = nn.Parameter(torch.randn(rank, d_in) * 0.01)
        self.B = nn.Parameter(torch.zeros(d_out, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = x @ self.base_weight.T
        if self.base_bias is not None:
            base = base + self.base_bias
        delta = self.B @ self.A
        return base + x @ delta.T
