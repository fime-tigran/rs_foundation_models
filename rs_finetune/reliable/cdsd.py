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
import torch.nn.functional as F


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


def channel_dropout(
    x: torch.Tensor, p: float, min_keep: int, training: bool
) -> torch.Tensor:
    """Randomly zero a subset of channels in ``x`` (shape ``(B, C, *)``).

    No-op when ``training is False`` or ``p == 0``. Always retains at
    least ``min_keep`` channels per batch element.
    """
    if not training or p == 0.0:
        return x
    if not 0.0 < p <= 1.0:
        raise ValueError(f"p must be in (0, 1], got {p}")
    B, C = x.shape[0], x.shape[1]
    n_drop = max(0, min(C - min_keep, int(round(p * C))))
    if n_drop == 0:
        return x
    out = x.clone()
    for b in range(B):
        idx = torch.randperm(C, device=x.device)[:n_drop]
        out[b, idx] = 0.0
    return out


def cdsd_loss(
    student_tokens: torch.Tensor,
    teacher_tokens: torch.Tensor,
    lambda_distill: float,
) -> torch.Tensor:
    """Mean cosine distance between student and teacher patch tokens,
    scaled by ``lambda_distill``.

    Both inputs have shape ``(B, N, D)``. Returns a scalar loss tensor.
    """
    if student_tokens.shape != teacher_tokens.shape:
        raise ValueError(
            f"shape mismatch: student {tuple(student_tokens.shape)} vs "
            f"teacher {tuple(teacher_tokens.shape)}"
        )
    cos_sim = F.cosine_similarity(student_tokens, teacher_tokens, dim=-1)
    distance = 1.0 - cos_sim
    return lambda_distill * distance.mean()
