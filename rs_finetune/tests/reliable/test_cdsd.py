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


def test_cdsd_loss_zero_when_tokens_match():
    from reliable.cdsd import cdsd_loss

    tokens = torch.randn(2, 16, 32)
    loss = cdsd_loss(tokens, tokens.clone(), lambda_distill=1.0)
    assert loss.item() < 1e-5


def test_cdsd_loss_positive_when_tokens_differ():
    from reliable.cdsd import cdsd_loss

    student = torch.randn(2, 16, 32)
    teacher = torch.randn(2, 16, 32)
    loss = cdsd_loss(student, teacher, lambda_distill=1.0)
    assert loss.item() > 0


def test_cdsd_loss_scales_with_lambda():
    from reliable.cdsd import cdsd_loss

    student = torch.randn(2, 16, 32)
    teacher = torch.randn(2, 16, 32)
    loss_a = cdsd_loss(student, teacher, lambda_distill=1.0).item()
    loss_b = cdsd_loss(student, teacher, lambda_distill=0.5).item()
    assert abs(loss_b - 0.5 * loss_a) < 1e-5


def test_ema_teacher_forward_has_no_grad():
    student = nn.Linear(8, 4)
    teacher = EMATeacher(student, momentum=0.99)
    x = torch.randn(2, 8, requires_grad=True)
    y = teacher(x)
    assert y.requires_grad is False
    assert y.grad_fn is None
