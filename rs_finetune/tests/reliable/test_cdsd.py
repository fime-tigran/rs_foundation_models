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


def test_ema_teacher_momentum_one_freezes_teacher():
    student = nn.Linear(8, 4)
    teacher = EMATeacher(student, momentum=1.0)
    pre = [p.clone() for p in teacher.module.parameters()]
    # Move student weights.
    with torch.no_grad():
        for p in student.parameters():
            p.add_(torch.randn_like(p))
    teacher.update(student)
    for p_pre, p_post in zip(pre, teacher.module.parameters()):
        assert torch.equal(p_pre, p_post)


def test_ema_teacher_momentum_zero_copies_student():
    student = nn.Linear(8, 4)
    teacher = EMATeacher(student, momentum=0.0)
    with torch.no_grad():
        for p in student.parameters():
            p.add_(torch.randn_like(p))
    teacher.update(student)
    for sp, tp in zip(student.parameters(), teacher.module.parameters()):
        assert torch.allclose(sp, tp)


def test_channel_dropout_train_drops_at_least_one():
    from reliable.cdsd import channel_dropout

    x = torch.ones(2, 4, 8, 8)
    torch.manual_seed(0)
    y = channel_dropout(x, p=0.5, min_keep=1, training=True)
    # At least one channel was zeroed in each batch element.
    for b in range(2):
        zeroed = (y[b].abs().sum(dim=(1, 2)) == 0).sum().item()
        assert zeroed >= 1


def test_channel_dropout_eval_passthrough():
    from reliable.cdsd import channel_dropout

    x = torch.randn(2, 4, 8, 8)
    y = channel_dropout(x, p=0.5, min_keep=1, training=False)
    assert torch.equal(x, y)


def test_channel_dropout_p_zero_passthrough():
    from reliable.cdsd import channel_dropout

    x = torch.randn(2, 4, 8, 8)
    y = channel_dropout(x, p=0.0, min_keep=1, training=True)
    assert torch.equal(x, y)
