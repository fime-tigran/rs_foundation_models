"""#9 CDSD — channel-dropout self-distillation.

Three building blocks:

- :class:`EMATeacher` — clones a student module and tracks its parameters
  via an exponential moving average. Used as the iBOT-style soft-target
  source for the distillation loss.
- :func:`channel_dropout` — randomly zeros a subset of input channels in
  ``train`` mode; passthrough in ``eval`` mode or when ``p == 0``.
- :func:`cdsd_loss` — cosine-distance distillation between dropped-student
  patch tokens and full-teacher patch tokens, weighted by lambda_distill.
"""

import copy

import torch
import torch.nn as nn


class EMATeacher(nn.Module):
    """Wraps a student module and exposes a frozen, EMA-tracked copy.

    Args:
        student: the module to track.
        momentum: EMA coefficient. ``teacher = m * teacher + (1 - m) * student``.
    """

    def __init__(self, student: nn.Module, momentum: float = 0.996):
        super().__init__()
        if not (0.0 <= momentum <= 1.0):
            raise ValueError(f"momentum must be in [0, 1], got {momentum}")
        self.module = copy.deepcopy(student)
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.momentum = momentum

    @torch.no_grad()
    def update(self, student: nn.Module) -> None:
        """Take one EMA step toward the student's current parameters."""
        for tp, sp in zip(self.module.parameters(), student.parameters()):
            tp.mul_(self.momentum).add_(sp.detach(), alpha=1.0 - self.momentum)

    @torch.no_grad()
    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.module(*args, **kwargs)
