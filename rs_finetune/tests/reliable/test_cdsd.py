"""Tests for CDSD (#9 channel-dropout self-distillation)."""

import torch
import torch.nn as nn

from reliable.cdsd import EMATeacher


def test_ema_teacher_clones_student_at_init():
    student = nn.Linear(8, 4)
    teacher = EMATeacher(student, momentum=0.99)
    # Same parameter values at init.
    for sp, tp in zip(student.parameters(), teacher.module.parameters()):
        assert torch.equal(sp, tp)
    # Teacher params have requires_grad=False.
    for tp in teacher.module.parameters():
        assert tp.requires_grad is False
