# Reliable-Core Phase 2 — Training Regularizers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the three training-time regularizer techniques from
`.cursor/plans/reliable-solutions.md` as composable modules: #9 CDSD
(channel-dropout self-distillation), #23 LSMM (linear spectral-mixing
auxiliary head), #15 CH-RS-FT (channel-token randomized smoothing). Plus a
small Phase-1 follow-up: inject the adapter class into
`attach_lora_to_last_n` so OPLoRA can use the same placement helper as
LoRALayer.

**Architecture:** Three new self-contained modules under `rs_finetune/
reliable/`. CDSD adds an EMA-teacher wrapper and a within-subset dropout-
plus-distillation loss; LSMM adds a frozen-endmember reconstruction head
that is discarded at eval; CH-RS-FT adds Gaussian noise on channel tokens
plus a Monte-Carlo aggregator that returns a majority vote and a vote-
count-based confidence statistic. All three operate at the post-embedding-
generator level (per-channel features), so they remain backbone-agnostic
and portable across χViT / TerraFM / DOFA / DINOv2 / DINOv3.

**Tech stack:** PyTorch 2.x, pytest, uv. Reuses Phase 1 fixtures from
`rs_finetune/tests/reliable/conftest.py`.

**Prerequisites already done:**
- Phase 1 `reliable/` package: `lora_layer.py`, `last_n_placement.py`,
  `oplora.py`, `lora_null_init.py`, `channel_mask.py`. 70 tests green.
- Plan: `.cursor/plans/2026-04-25-reliable-lora-foundation-plan.md`.
- Spec: `.cursor/plans/reliable-solutions.md` §A (universal core), §
  Reliable optional add-ons.

**Out of scope for this plan:**
- VCA endmember extraction from the actual pretraining corpus (real
  endmember dictionaries are computed offline by a separate one-off
  script — Phase 7's responsibility). This plan exercises LSMM with
  synthetic endmembers loaded from `tmp_artifact_dir`.
- Cohen-style or Clopper-Pearson certified radii for CH-RS-FT. This plan
  ships a simpler vote-count confidence statistic; real PAC certificates
  land in a Phase 7 paper-table task.
- Integration with `train_classifier.py` / `train_segmenter.py` /
  `train_change.py` and CLI flag wiring. That is Phase 7.
- Heads (#32 APH, #14 NCI-PIH, #28 MCSE) — Phase 3.

---

## File structure

**New production files** (all under `rs_finetune/reliable/`):

```
reliable/
    cdsd.py                  # #9  EMATeacher + channel_dropout + cdsd_loss
    lsmm_aux_head.py         # #23 LSMMHead with frozen endmembers
    ch_rs_ft.py              # #15 smooth_channel_tokens + mc_smooth_predict
```

**Existing production file modified (Phase 1 follow-up):**

```
reliable/last_n_placement.py # inject adapter_cls parameter
```

**New test files** (all under `rs_finetune/tests/reliable/`):

```
test_cdsd.py                 # 6 tests
test_lsmm_aux_head.py        # 5 tests
test_ch_rs_ft.py             # 5 tests
test_phase2_integration.py   # 1 smoke test
```

**Modified test file:**

```
test_last_n_lora.py          # 1 new test for adapter_cls injection
```

**Modified design doc:**

```
.cursor/plans/reliable-solutions.md  # mark Phase 2 complete
```

---

## Execution rules

- **TDD iron law.** One behaviour per red-green cycle. Write one test, watch
  it RED for a unique reason, write minimum code, verify GREEN, commit.
- **If a test passes on its first run, accept it as a "protective
  regression"** — explicitly labelled in the task description, committed
  with `test:` prefix not `feat:`.
- **All commands run from `rs_finetune/`** (cwd). Test runner is `./run_
  tests.sh`.
- **Run full regression** (`./run_tests.sh`) after every commit.
- **One commit per task.** Do NOT amend prior commits.

---

## Task 1: Inject `adapter_cls` parameter into `attach_lora_to_last_n` (Phase-1 follow-up)

Phase 1 final review flagged this as the first refactor of Phase 2: the
placement helper hard-codes `LoRALayer` in `_wrap_linear_with_lora`, so
OPLoRA can't reuse it. Inject a class parameter so callers pick `LoRALayer`
or `OPLoRALayer` at attach time.

**Files:**
- Modify: `rs_finetune/reliable/last_n_placement.py`
- Modify: `rs_finetune/tests/reliable/test_last_n_lora.py`

- [ ] **Step 1: Write the failing test**

Append to `rs_finetune/tests/reliable/test_last_n_lora.py`:

```python
def test_attach_last_n_with_oplora_adapter_class(
    tiny_mock_multispec_backbone,
):
    """attach_lora_to_last_n accepts an adapter_cls kwarg so OPLoRALayer
    can be placed via the same code path as LoRALayer."""
    from reliable.lora_layer import LoRALayer
    from reliable.oplora import OPLoRALayer

    model = tiny_mock_multispec_backbone(n_channels=4, embed_dim=32)
    attach_lora_to_last_n(
        model, last_n=1, rank=4, adapter_cls=OPLoRALayer, preserve_k=2
    )
    tail = model.transformer.layers[-1]
    assert isinstance(tail.linear1, OPLoRALayer)
    assert not isinstance(tail.linear1, LoRALayer)
    head = model.transformer.layers[0]
    assert isinstance(head.linear1, nn.Linear)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/tgrigoryan/rs_foundation_models/rs_finetune
./run_tests.sh tests/reliable/test_last_n_lora.py::test_attach_last_n_with_oplora_adapter_class 2>&1 | tail -8
```

Expected: FAIL with `TypeError: attach_lora_to_last_n() got an unexpected keyword argument 'adapter_cls'`.

- [ ] **Step 3: Update `rs_finetune/reliable/last_n_placement.py`**

Replace the file with:

```python
"""#29 LastN-LoRA placement helper.

Attaches LoRA adapters to the last N transformer layers' MLP linears
(``linear1`` and ``linear2``) so earlier layers stay bit-identical to the
pretrained model.

The adapter class is injectable via ``adapter_cls``; defaults to
``reliable.lora_layer.LoRALayer``. Passing ``OPLoRALayer`` (with the
required ``preserve_k`` kwarg) places the OPLoRA variant via the same
code path.

Attention projections are not wrapped because ``nn.MultiheadAttention``
introspects ``out_proj.weight`` directly.
"""

from typing import Type

import torch.nn as nn

from reliable.lora_layer import LoRALayer


def _wrap_linear_with_adapter(
    linear: nn.Linear,
    rank: int,
    adapter_cls: Type[nn.Module],
    **adapter_kwargs,
) -> nn.Module:
    base_bias = (
        linear.bias.detach().clone() if linear.bias is not None else None
    )
    return adapter_cls(
        d_in=linear.in_features,
        d_out=linear.out_features,
        rank=rank,
        base_weight=linear.weight.detach().clone(),
        base_bias=base_bias,
        **adapter_kwargs,
    )


_LORA_TARGETS = ("linear1", "linear2")


def attach_lora_to_last_n(
    model: nn.Module,
    last_n: int,
    rank: int,
    adapter_cls: Type[nn.Module] = LoRALayer,
    **adapter_kwargs,
) -> None:
    """Attach adapters to the last ``last_n`` transformer layers' MLP
    projections.

    No-op when ``last_n <= 0``. Raises :class:`ValueError` when the model
    does not expose a ``transformer.layers`` attribute. Extra kwargs (e.g.
    ``preserve_k`` for OPLoRA) are forwarded to ``adapter_cls.__init__``.
    """
    if last_n <= 0:
        return
    transformer = getattr(model, "transformer", None)
    if transformer is None or not hasattr(transformer, "layers"):
        raise ValueError("Model has no `transformer.layers` to attach LoRA to")
    layers = transformer.layers
    total = len(layers)
    first = max(0, total - last_n)
    for idx in range(first, total):
        layer = layers[idx]
        for attr in _LORA_TARGETS:
            target = getattr(layer, attr, None)
            if isinstance(target, nn.Linear):
                setattr(
                    layer,
                    attr,
                    _wrap_linear_with_adapter(
                        target, rank=rank, adapter_cls=adapter_cls,
                        **adapter_kwargs,
                    ),
                )
        layer._lora_registry = {
            "rank": rank,
            "targets": list(_LORA_TARGETS),
            "adapter_cls": adapter_cls.__name__,
        }
```

- [ ] **Step 4: Run full regression, verify GREEN**

```bash
./run_tests.sh tests/reliable/ 2>&1 | tail -4
```

Expected: `71 passed` (70 prior + 1 new).

- [ ] **Step 5: Commit**

```bash
cd /home/tgrigoryan/rs_foundation_models
git add rs_finetune/reliable/last_n_placement.py rs_finetune/tests/reliable/test_last_n_lora.py
git commit -m "$(cat <<'EOF'
refactor(reliable): inject adapter_cls into attach_lora_to_last_n

Phase 1 final review flagged that _wrap_linear_with_lora hard-coded
LoRALayer, so OPLoRA could not reuse the placement helper. Injecting
adapter_cls (default LoRALayer) plus **adapter_kwargs forwarding lets
OPLoRA be placed via the same code path with `preserve_k=...` passed
through.

The _lora_registry now records adapter_cls.__name__ for downstream
introspection.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: CDSD — EMA teacher wrapper

`EMATeacher(student, momentum)` clones the student's parameters into a
frozen-grad teacher and exposes `update()` to do an exponential moving
average step. This is the iBOT-style teacher used by CDSD's distillation.

**Files:**
- Create: `rs_finetune/reliable/cdsd.py`
- Create: `rs_finetune/tests/reliable/test_cdsd.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for CDSD (#9 channel-dropout self-distillation)."""

import pytest
import torch
import torch.nn as nn

from reliable.cdsd import EMATeacher


def test_ema_teacher_clones_student_at_init(frozen_pretrained_weight):
    student = nn.Linear(8, 4)
    teacher = EMATeacher(student, momentum=0.99)
    # Same parameter values at init.
    for sp, tp in zip(student.parameters(), teacher.module.parameters()):
        assert torch.equal(sp, tp)
    # Teacher params have requires_grad=False.
    for tp in teacher.module.parameters():
        assert tp.requires_grad is False
```

- [ ] **Step 2: Run test, verify RED**

```bash
cd /home/tgrigoryan/rs_foundation_models/rs_finetune
./run_tests.sh tests/reliable/test_cdsd.py 2>&1 | tail -8
```

Expected: ERROR — `ModuleNotFoundError: No module named 'reliable.cdsd'`.

- [ ] **Step 3: Create `rs_finetune/reliable/cdsd.py`**

```python
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
```

- [ ] **Step 4: Run full regression, verify GREEN**

```bash
./run_tests.sh tests/reliable/ 2>&1 | tail -4
```

Expected: `72 passed` (71 prior + 1 new).

- [ ] **Step 5: Commit**

```bash
cd /home/tgrigoryan/rs_foundation_models
git add rs_finetune/reliable/cdsd.py rs_finetune/tests/reliable/test_cdsd.py
git commit -m "$(cat <<'EOF'
feat(reliable): CDSD EMATeacher clones student with frozen-grad params

EMATeacher(student, momentum) deep-copies the student at construction
and freezes the copy's grads. Exposes update() for the EMA step and
forward() under no_grad for soft-target generation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: CDSD — EMA momentum behaviour

Verify that `update()` actually moves the teacher toward the student under
typical momentum, and stays still at momentum=1.

**Files:**
- Modify: `rs_finetune/tests/reliable/test_cdsd.py`

- [ ] **Step 1: Write the failing test**

Append to `rs_finetune/tests/reliable/test_cdsd.py`:

```python
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
```

- [ ] **Step 2: Run tests, verify they pass (protective regressions)**

```bash
./run_tests.sh tests/reliable/test_cdsd.py 2>&1 | tail -5
```

Expected: 3 passed. Protective regressions of Task 2's EMA semantics.

- [ ] **Step 3: Commit**

```bash
git add rs_finetune/tests/reliable/test_cdsd.py
git commit -m "$(cat <<'EOF'
test(reliable): EMATeacher momentum=1 freezes, momentum=0 copies

Locks the EMA semantics: at momentum=1.0 update() is a no-op (teacher
never tracks); at momentum=0.0 update() snaps teacher to student.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: CDSD — channel_dropout function

Drops `floor(p * n_channels)` channels uniformly at random per batch
element in train mode (subject to `min_keep`); no-op in eval or when p=0.

**Files:**
- Modify: `rs_finetune/reliable/cdsd.py`
- Modify: `rs_finetune/tests/reliable/test_cdsd.py`

- [ ] **Step 1: Write the failing test**

Append to `rs_finetune/tests/reliable/test_cdsd.py`:

```python
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
```

- [ ] **Step 2: Run, verify RED**

```bash
./run_tests.sh tests/reliable/test_cdsd.py -k channel_dropout 2>&1 | tail -8
```

Expected: 3 ERROR — `ImportError: cannot import name 'channel_dropout'`.

- [ ] **Step 3: Append `channel_dropout` to `rs_finetune/reliable/cdsd.py`**

```python
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
```

- [ ] **Step 4: Verify GREEN**

```bash
./run_tests.sh tests/reliable/ 2>&1 | tail -4
```

Expected: `75 passed` (72 prior + 3 new).

- [ ] **Step 5: Commit**

```bash
git add rs_finetune/reliable/cdsd.py rs_finetune/tests/reliable/test_cdsd.py
git commit -m "$(cat <<'EOF'
feat(reliable): CDSD channel_dropout zeros random channel subset in train

channel_dropout(x, p, min_keep, training) zeros ``floor(p*C)`` channels
per batch element in training mode while always keeping at least
``min_keep``. No-op in eval mode or when p=0. Validates p ∈ (0, 1].

Used inside CDSD as the input-perturbation half of the channel-closure
prior; the EMA-teacher distillation loss (next task) supplies the
matching learning signal.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: CDSD — distillation loss

`cdsd_loss(student_tokens, teacher_tokens, lambda_distill)` returns a
scalar cosine-distance loss between matched patch-token tensors,
multiplied by `lambda_distill`.

**Files:**
- Modify: `rs_finetune/reliable/cdsd.py`
- Modify: `rs_finetune/tests/reliable/test_cdsd.py`

- [ ] **Step 1: Write the failing test**

Append to `rs_finetune/tests/reliable/test_cdsd.py`:

```python
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
```

- [ ] **Step 2: Run, verify RED**

```bash
./run_tests.sh tests/reliable/test_cdsd.py -k cdsd_loss 2>&1 | tail -8
```

Expected: 3 ERROR — `ImportError: cannot import name 'cdsd_loss'`.

- [ ] **Step 3: Append `cdsd_loss` to `rs_finetune/reliable/cdsd.py`**

```python
import torch.nn.functional as F


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
```

- [ ] **Step 4: Verify GREEN**

```bash
./run_tests.sh tests/reliable/ 2>&1 | tail -4
```

Expected: `78 passed` (75 prior + 3 new).

- [ ] **Step 5: Commit**

```bash
git add rs_finetune/reliable/cdsd.py rs_finetune/tests/reliable/test_cdsd.py
git commit -m "$(cat <<'EOF'
feat(reliable): CDSD cosine-distance distillation loss

cdsd_loss(student_tokens, teacher_tokens, lambda_distill) returns a
scalar cosine-distance loss over matched (B, N, D) patch tokens, scaled
by the user-provided lambda. Validates shape match.

Together with channel_dropout (Task 4) and EMATeacher (Task 2) this
completes the CDSD primitives. The training loop integration that wires
them into a step is Phase 7.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: CDSD — teacher forward has no grad

Protective regression: pulling samples from `EMATeacher.forward()` must
not register grad ops on the student's graph.

**Files:**
- Modify: `rs_finetune/tests/reliable/test_cdsd.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_ema_teacher_forward_has_no_grad():
    student = nn.Linear(8, 4)
    teacher = EMATeacher(student, momentum=0.99)
    x = torch.randn(2, 8, requires_grad=True)
    y = teacher(x)
    assert y.requires_grad is False
    assert y.grad_fn is None
```

- [ ] **Step 2: Run, verify it passes (protective regression)**

```bash
./run_tests.sh tests/reliable/test_cdsd.py::test_ema_teacher_forward_has_no_grad 2>&1 | tail -5
```

Expected: PASS — `EMATeacher.forward` is `@torch.no_grad`-decorated.

- [ ] **Step 3: Commit**

```bash
git add rs_finetune/tests/reliable/test_cdsd.py
git commit -m "$(cat <<'EOF'
test(reliable): EMATeacher forward produces grad-free outputs

Protective regression locking the @torch.no_grad contract on
EMATeacher.forward — soft targets must not couple to the student's
autograd graph.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: LSMM — head with frozen endmember dictionary

`LSMMHead(D, n_endmembers, n_bands, srf_matrix, endmembers)` registers
endmembers and SRF as buffers (frozen) and produces non-negative
abundances from input features.

**Files:**
- Create: `rs_finetune/reliable/lsmm_aux_head.py`
- Create: `rs_finetune/tests/reliable/test_lsmm_aux_head.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for LSMM auxiliary head (#23)."""

import pytest
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
```

- [ ] **Step 2: Run, verify RED**

```bash
./run_tests.sh tests/reliable/test_lsmm_aux_head.py 2>&1 | tail -8
```

Expected: ERROR — `ModuleNotFoundError: No module named 'reliable.lsmm_aux_head'`.

- [ ] **Step 3: Create `rs_finetune/reliable/lsmm_aux_head.py`**

```python
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
```

- [ ] **Step 4: Verify GREEN**

```bash
./run_tests.sh tests/reliable/ 2>&1 | tail -4
```

Expected: `79 passed` (78 prior + 1 new).

- [ ] **Step 5: Commit**

```bash
git add rs_finetune/reliable/lsmm_aux_head.py rs_finetune/tests/reliable/test_lsmm_aux_head.py
git commit -m "$(cat <<'EOF'
feat(reliable): LSMMHead registers endmembers and SRF as frozen buffers

Aux head holds a frozen (n_bands, K) endmember dictionary and a frozen
(3, n_bands) SRF matrix as buffers — never parameters, never in optimizer
groups. Exposes predict_abundances() returning softplus-positive
abundance vectors over input features.

Validates input shapes against (n_bands, n_endmembers) and (3, n_bands)
contracts.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: LSMM — abundances are non-negative

Protective regression for the `softplus` activation contract.

**Files:**
- Modify: `rs_finetune/tests/reliable/test_lsmm_aux_head.py`

- [ ] **Step 1: Write the failing test**

```python
def test_lsmm_head_abundances_are_non_negative():
    K, D, n_bands = 8, 32, 12
    head = LSMMHead(
        d=D, n_endmembers=K, n_bands=n_bands,
        srf_matrix=torch.randn(3, n_bands),
        endmembers=torch.randn(n_bands, K),
    )
    feats = torch.randn(4, D) * 5.0  # large negative inputs possible
    alpha = head.predict_abundances(feats)
    assert alpha.shape == (4, K)
    assert (alpha >= 0).all()
```

- [ ] **Step 2: Run — protective regression, expect PASS**

```bash
./run_tests.sh tests/reliable/test_lsmm_aux_head.py::test_lsmm_head_abundances_are_non_negative 2>&1 | tail -5
```

Expected: PASS — softplus output ≥ 0 by construction.

- [ ] **Step 3: Commit**

```bash
git add rs_finetune/tests/reliable/test_lsmm_aux_head.py
git commit -m "$(cat <<'EOF'
test(reliable): LSMM abundance predictions are non-negative

Protective regression locking the softplus contract on
LSMMHead.predict_abundances. Future refactors that accidentally swap to
linear / tanh / similar would be caught.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: LSMM — reconstruction loss

`reconstruction_loss(features, x_rgb)` predicts abundances, projects via
`SRF · E · α`, and returns squared-error vs `x_rgb`.

**Files:**
- Modify: `rs_finetune/reliable/lsmm_aux_head.py`
- Modify: `rs_finetune/tests/reliable/test_lsmm_aux_head.py`

- [ ] **Step 1: Write the failing test**

```python
def test_lsmm_reconstruction_loss_finite_and_lambda_zero_kills():
    K, D, n_bands = 8, 32, 12
    head = LSMMHead(
        d=D, n_endmembers=K, n_bands=n_bands,
        srf_matrix=torch.randn(3, n_bands),
        endmembers=torch.randn(n_bands, K),
    )
    feats = torch.randn(4, D)
    x_rgb = torch.randn(4, 3)
    loss = head.reconstruction_loss(feats, x_rgb, lambda_lsmm=0.5)
    assert torch.isfinite(loss)
    loss_zero = head.reconstruction_loss(feats, x_rgb, lambda_lsmm=0.0)
    assert loss_zero.item() == 0.0
```

- [ ] **Step 2: Run, verify RED**

```bash
./run_tests.sh tests/reliable/test_lsmm_aux_head.py::test_lsmm_reconstruction_loss_finite_and_lambda_zero_kills 2>&1 | tail -5
```

Expected: FAIL — `AttributeError: 'LSMMHead' object has no attribute 'reconstruction_loss'`.

- [ ] **Step 3: Append to `LSMMHead`**

Add this method inside the `LSMMHead` class:

```python
    def reconstruction_loss(
        self,
        features: torch.Tensor,
        x_rgb: torch.Tensor,
        lambda_lsmm: float,
    ) -> torch.Tensor:
        """Squared error between ``x_rgb`` and the SRF-projected mixture
        ``SRF · E · α``, scaled by ``lambda_lsmm``."""
        if lambda_lsmm == 0.0:
            return torch.zeros((), device=features.device, dtype=features.dtype)
        alpha = self.predict_abundances(features)            # (B, K)
        spectra = alpha @ self.endmembers.T                  # (B, n_bands)
        x_recon = spectra @ self.srf_matrix.T                # (B, 3)
        return lambda_lsmm * F.mse_loss(x_recon, x_rgb)
```

- [ ] **Step 4: Verify GREEN**

```bash
./run_tests.sh tests/reliable/ 2>&1 | tail -4
```

Expected: `81 passed` (80 prior + 1 new).

- [ ] **Step 5: Commit**

```bash
git add rs_finetune/reliable/lsmm_aux_head.py rs_finetune/tests/reliable/test_lsmm_aux_head.py
git commit -m "$(cat <<'EOF'
feat(reliable): LSMM reconstruction_loss = ||SRF·E·α − x_rgb||²

LSMMHead.reconstruction_loss(features, x_rgb, lambda_lsmm) computes the
mean-squared error between input RGB and the linearly-mixed
reconstruction SRF · E · α, scaled by lambda_lsmm. lambda_lsmm == 0
short-circuits to a zero scalar so the loss term can be cheaply disabled.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: LSMM — endmember cache roundtrip

Verify the head accepts endmembers loaded from disk via `tmp_artifact_dir`
(the production path: VCA-precomputed endmembers saved as a `.pt` and
loaded at fine-tune startup).

**Files:**
- Modify: `rs_finetune/tests/reliable/test_lsmm_aux_head.py`

- [ ] **Step 1: Write the failing test**

```python
def test_lsmm_head_loads_endmembers_from_artifact_cache(tmp_artifact_dir):
    K, n_bands = 8, 12
    endmembers = torch.randn(n_bands, K)
    artifact_path = tmp_artifact_dir / "endmembers.pt"
    torch.save(endmembers, artifact_path)

    loaded = torch.load(artifact_path, weights_only=True)
    head = LSMMHead(
        d=32, n_endmembers=K, n_bands=n_bands,
        srf_matrix=torch.randn(3, n_bands),
        endmembers=loaded,
    )
    assert torch.equal(head.endmembers, endmembers)
```

- [ ] **Step 2: Run — protective regression on top of Task 7's frozen-buffer contract**

```bash
./run_tests.sh tests/reliable/test_lsmm_aux_head.py::test_lsmm_head_loads_endmembers_from_artifact_cache 2>&1 | tail -5
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add rs_finetune/tests/reliable/test_lsmm_aux_head.py
git commit -m "$(cat <<'EOF'
test(reliable): LSMM endmember cache round-trip

Loads a synthetic endmember dictionary from tmp_artifact_dir, constructs
an LSMMHead from it, and verifies the buffer matches bit-for-bit. Locks
the offline-artifact production path: VCA endmembers will live on disk in
real deployments.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: CH-RS-FT — `smooth_channel_tokens` in train mode

Adds Gaussian noise to a Bernoulli-selected fraction of channel tokens.
Training-only; no-op in eval, σ=0, or p_smooth=0.

**Files:**
- Create: `rs_finetune/reliable/ch_rs_ft.py`
- Create: `rs_finetune/tests/reliable/test_ch_rs_ft.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for Channel-Token Hierarchical Randomized Smoothing FT (#15)."""

import pytest
import torch

from reliable.ch_rs_ft import smooth_channel_tokens


def test_smooth_channel_tokens_train_perturbs_subset():
    tokens = torch.zeros(2, 12, 8)
    torch.manual_seed(0)
    smoothed = smooth_channel_tokens(
        tokens, sigma=1.0, p_smooth=0.5, training=True
    )
    # Some channels were noised (non-zero), some kept clean (zero).
    nonzero_per_batch = (smoothed.abs().sum(dim=-1) > 0).sum(dim=-1)
    assert (nonzero_per_batch > 0).all()
    assert (nonzero_per_batch < 12).all()


def test_smooth_channel_tokens_eval_passthrough():
    tokens = torch.randn(2, 12, 8)
    out = smooth_channel_tokens(
        tokens, sigma=1.0, p_smooth=0.5, training=False
    )
    assert torch.equal(tokens, out)


def test_smooth_channel_tokens_sigma_zero_passthrough():
    tokens = torch.randn(2, 12, 8)
    out = smooth_channel_tokens(
        tokens, sigma=0.0, p_smooth=0.5, training=True
    )
    assert torch.equal(tokens, out)


def test_smooth_channel_tokens_p_zero_passthrough():
    tokens = torch.randn(2, 12, 8)
    out = smooth_channel_tokens(
        tokens, sigma=1.0, p_smooth=0.0, training=True
    )
    assert torch.equal(tokens, out)
```

- [ ] **Step 2: Run, verify RED**

```bash
./run_tests.sh tests/reliable/test_ch_rs_ft.py 2>&1 | tail -8
```

Expected: 4 ERROR — `ModuleNotFoundError: No module named 'reliable.ch_rs_ft'`.

- [ ] **Step 3: Create `rs_finetune/reliable/ch_rs_ft.py`**

```python
"""#15 Channel-Token Hierarchical Randomized Smoothing Fine-Tuning.

Two halves:

- :func:`smooth_channel_tokens` — adds Gaussian noise to a Bernoulli-
  selected fraction of channel tokens during training. No-op at eval.
- :func:`mc_smooth_predict` — eval-time Monte Carlo aggregation: forward
  the model many times under noise, return the majority-vote class and a
  vote-count confidence statistic (placeholder for proper Cohen-style
  certificates added later).
"""

import torch


def smooth_channel_tokens(
    tokens: torch.Tensor,
    sigma: float,
    p_smooth: float,
    training: bool,
) -> torch.Tensor:
    """Add Gaussian-σ noise to a Bernoulli-p_smooth subset of channel tokens.

    Args:
        tokens: ``(B, C, D)`` channel-token features.
        sigma: noise standard deviation.
        p_smooth: probability per channel of being noised.
        training: only perturb when True.

    Returns the same shape as ``tokens``. Pure passthrough when
    ``training is False``, ``sigma == 0``, or ``p_smooth == 0``.
    """
    if not training or sigma == 0.0 or p_smooth == 0.0:
        return tokens
    if not 0.0 < p_smooth <= 1.0:
        raise ValueError(f"p_smooth must be in (0, 1], got {p_smooth}")
    B, C, _D = tokens.shape
    mask = (torch.rand(B, C, device=tokens.device) < p_smooth).to(tokens.dtype)
    noise = torch.randn_like(tokens) * sigma
    return tokens + mask.unsqueeze(-1) * noise
```

- [ ] **Step 4: Verify GREEN**

```bash
./run_tests.sh tests/reliable/ 2>&1 | tail -4
```

Expected: `85 passed` (81 prior + 4 new).

- [ ] **Step 5: Commit**

```bash
git add rs_finetune/reliable/ch_rs_ft.py rs_finetune/tests/reliable/test_ch_rs_ft.py
git commit -m "$(cat <<'EOF'
feat(reliable): smooth_channel_tokens adds Gaussian noise to a token subset

smooth_channel_tokens(tokens, sigma, p_smooth, training) selects a
Bernoulli-p_smooth fraction of channel tokens and adds independent
Gaussian-σ noise to the selected ones. Pure passthrough when training is
False, sigma=0, or p_smooth=0. Validates p_smooth ∈ (0, 1].

Three protective regressions cover the passthrough modes; one driving
test asserts that train-mode smoothing produces a partial-perturbation
pattern (some channels noised, some kept).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: CH-RS-FT — Monte Carlo predict + vote-count statistic

`mc_smooth_predict(model, tokens_factory, n_mc, sigma, p_smooth)` calls
the model many times under fresh smoothings, returns `(majority_class,
vote_count_for_majority)`.

**Files:**
- Modify: `rs_finetune/reliable/ch_rs_ft.py`
- Modify: `rs_finetune/tests/reliable/test_ch_rs_ft.py`

- [ ] **Step 1: Write the failing test**

```python
def test_mc_smooth_predict_returns_majority_and_count():
    from reliable.ch_rs_ft import mc_smooth_predict

    # Toy classifier: returns deterministic class 3 regardless of input.
    class _Const(torch.nn.Module):
        def forward(self, x):  # x: (B, C, D) → (B, num_classes)
            return torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0]] * x.shape[0])

    model = _Const()
    tokens = torch.zeros(1, 4, 8)
    pred, votes = mc_smooth_predict(
        model, tokens, n_mc=10, sigma=0.1, p_smooth=0.5,
    )
    # Returns class with shape (B,) and a vote count (B,) for the majority.
    assert pred.shape == (1,)
    assert votes.shape == (1,)
    assert pred.item() == 3
    assert votes.item() == 10  # constant model → 10/10 votes


def test_mc_smooth_predict_n_mc_one_runs_once():
    from reliable.ch_rs_ft import mc_smooth_predict

    call_count = {"n": 0}

    class _Counter(torch.nn.Module):
        def forward(self, x):
            call_count["n"] += 1
            return torch.zeros(x.shape[0], 5)

    pred, votes = mc_smooth_predict(
        _Counter(), torch.zeros(1, 4, 8),
        n_mc=1, sigma=0.1, p_smooth=0.5,
    )
    assert call_count["n"] == 1
```

- [ ] **Step 2: Run, verify RED**

```bash
./run_tests.sh tests/reliable/test_ch_rs_ft.py -k mc_smooth_predict 2>&1 | tail -8
```

Expected: 2 ERROR — `ImportError: cannot import name 'mc_smooth_predict'`.

- [ ] **Step 3: Append to `rs_finetune/reliable/ch_rs_ft.py`**

```python
import torch.nn as nn


@torch.no_grad()
def mc_smooth_predict(
    model: nn.Module,
    tokens: torch.Tensor,
    n_mc: int,
    sigma: float,
    p_smooth: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Monte-Carlo aggregate logits over ``n_mc`` smoothed forwards.

    Returns ``(majority_class, vote_count_for_majority)`` per batch
    element. ``vote_count_for_majority`` is the integer number of MC
    samples whose argmax agreed with the final majority — a simple
    confidence statistic. (A proper Clopper-Pearson lower bound for
    Cohen-style certified radii is left for Phase 7.)
    """
    if n_mc < 1:
        raise ValueError(f"n_mc must be >= 1, got {n_mc}")
    B = tokens.shape[0]
    votes_per_class: dict[int, torch.Tensor] = {}
    all_preds = []
    for _ in range(n_mc):
        smoothed = smooth_channel_tokens(
            tokens, sigma=sigma, p_smooth=p_smooth, training=True,
        )
        logits = model(smoothed)
        all_preds.append(logits.argmax(dim=-1))
    stacked = torch.stack(all_preds, dim=0)              # (n_mc, B)
    majority = torch.mode(stacked, dim=0).values         # (B,)
    vote_count = (stacked == majority.unsqueeze(0)).sum(dim=0)  # (B,)
    return majority, vote_count
```

- [ ] **Step 4: Verify GREEN**

```bash
./run_tests.sh tests/reliable/ 2>&1 | tail -4
```

Expected: `87 passed` (85 prior + 2 new).

- [ ] **Step 5: Commit**

```bash
git add rs_finetune/reliable/ch_rs_ft.py rs_finetune/tests/reliable/test_ch_rs_ft.py
git commit -m "$(cat <<'EOF'
feat(reliable): mc_smooth_predict aggregates n_mc smoothed forwards

mc_smooth_predict(model, tokens, n_mc, sigma, p_smooth) runs the model
under fresh per-call channel-token smoothings and returns
(majority_class, vote_count_for_majority) per batch element. Decorated
@torch.no_grad. Validates n_mc >= 1.

vote_count_for_majority is a simple confidence statistic; the proper
Cohen-style Clopper-Pearson lower bound and certified radius land in
Phase 7 alongside the paper-table integration.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: CH-RS-FT — n_mc bounds validation

Protective regression on the `n_mc < 1` guard.

**Files:**
- Modify: `rs_finetune/tests/reliable/test_ch_rs_ft.py`

- [ ] **Step 1: Write the failing test**

```python
def test_mc_smooth_predict_rejects_n_mc_zero():
    from reliable.ch_rs_ft import mc_smooth_predict

    class _Stub(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 5)

    with pytest.raises(ValueError, match="n_mc"):
        mc_smooth_predict(
            _Stub(), torch.zeros(1, 4, 8),
            n_mc=0, sigma=0.1, p_smooth=0.5,
        )
```

- [ ] **Step 2: Run — protective regression, expect PASS**

```bash
./run_tests.sh tests/reliable/test_ch_rs_ft.py::test_mc_smooth_predict_rejects_n_mc_zero 2>&1 | tail -5
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add rs_finetune/tests/reliable/test_ch_rs_ft.py
git commit -m "$(cat <<'EOF'
test(reliable): mc_smooth_predict rejects n_mc < 1

Protective regression on the validation guard. matcher 'n_mc' is the
unique substring identifying the right ValueError path.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Phase 2 integration smoke

Verify CDSD, LSMM, and CH-RS-FT compose without type errors on a single
synthetic forward.

**Files:**
- Create: `rs_finetune/tests/reliable/test_phase2_integration.py`

- [ ] **Step 1: Write the failing test**

```python
"""Integration smoke test for Phase 2 training regularizers."""

import pytest
import torch
import torch.nn as nn

from reliable.cdsd import EMATeacher, channel_dropout, cdsd_loss
from reliable.ch_rs_ft import smooth_channel_tokens
from reliable.lsmm_aux_head import LSMMHead


def test_phase2_regularizers_compose_on_synthetic_forward():
    # Tiny encoder: per-channel feature in, per-channel-pooled out.
    class _Encoder(nn.Module):
        def __init__(self, c, d):
            super().__init__()
            self.proj = nn.Linear(c, d)

        def forward(self, x):                     # x: (B, C, H, W)
            B, C, H, W = x.shape
            tokens = x.mean(dim=(2, 3))           # (B, C)
            return self.proj(tokens.unsqueeze(-1) * torch.ones(1, 1, 4))  # (B, C, D=4)

    encoder = _Encoder(c=12, d=4)
    teacher = EMATeacher(encoder, momentum=0.99)

    x = torch.randn(2, 12, 8, 8)
    x_dropped = channel_dropout(x, p=0.3, min_keep=1, training=True)

    student_tokens = encoder(x_dropped)            # (B, 12, 4)
    teacher_tokens = teacher(x)                    # (B, 12, 4) under no_grad
    cdsd = cdsd_loss(student_tokens, teacher_tokens, lambda_distill=0.5)
    assert torch.isfinite(cdsd)

    # CH-RS-FT smoothing on the channel-token stream.
    smoothed = smooth_channel_tokens(
        student_tokens, sigma=0.1, p_smooth=0.5, training=True,
    )
    assert smoothed.shape == student_tokens.shape

    # LSMM aux loss on a pooled feature.
    pooled = student_tokens.mean(dim=1)            # (B, 4)
    head = LSMMHead(
        d=4, n_endmembers=8, n_bands=12,
        srf_matrix=torch.randn(3, 12),
        endmembers=torch.randn(12, 8),
    )
    rgb = x[:, :3].mean(dim=(2, 3))                # (B, 3) pooled RGB
    lsmm = head.reconstruction_loss(pooled, rgb, lambda_lsmm=0.3)
    assert torch.isfinite(lsmm)
```

- [ ] **Step 2: Run, verify it passes**

```bash
./run_tests.sh tests/reliable/test_phase2_integration.py 2>&1 | tail -5
```

Expected: PASS — all components from Phase 2 plumb together.

- [ ] **Step 3: Commit**

```bash
git add rs_finetune/tests/reliable/test_phase2_integration.py
git commit -m "$(cat <<'EOF'
test(reliable): Phase 2 integration — regularizers compose on synthetic forward

Smoke-tests that EMATeacher + channel_dropout + cdsd_loss + LSMMHead +
smooth_channel_tokens all plumb together on a single synthetic forward
through a toy encoder. Locks in the documented shape contracts:
- CDSD takes (B, C, D) student/teacher tokens
- LSMM takes (B, D) pooled features and (B, 3) RGB target
- CH-RS-FT smoothing takes (B, C, D) channel tokens

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: Mark Phase 2 complete in design docs

**Files:**
- Modify: `.cursor/plans/reliable-solutions.md`

- [ ] **Step 1: Run the full test suite, confirm green**

```bash
cd /home/tgrigoryan/rs_foundation_models/rs_finetune
./run_tests.sh 2>&1 | tail -4
```

Expected: `88 passed, 83 warnings`.

- [ ] **Step 2: Append to `.cursor/plans/reliable-solutions.md`**

Find the existing "## Implementation progress" section (added at the end
of Phase 1) and append a Phase 2 bullet UNDER it (do NOT touch any other
content):

```markdown
- **Phase 2 (Training regularizers) — COMPLETE (2026-04-25).**
  Shipped: `reliable/cdsd.py` (EMATeacher + channel_dropout + cdsd_loss),
  `reliable/lsmm_aux_head.py` (LSMMHead with frozen endmembers),
  `reliable/ch_rs_ft.py` (smooth_channel_tokens + mc_smooth_predict). 17
  Phase-2 tests green (88 total in the reliable suite). Phase-1 follow-up
  also landed: `attach_lora_to_last_n` accepts an `adapter_cls` kwarg so
  OPLoRA reuses the placement helper. Plan:
  `.cursor/plans/2026-04-25-reliable-training-regularizers-plan.md`.
```

- [ ] **Step 3: Commit**

```bash
cd /home/tgrigoryan/rs_foundation_models
git add .cursor/plans/reliable-solutions.md
git commit -m "$(cat <<'EOF'
docs(reliable): mark Phase 2 training regularizers complete

Phase 2 shipped: 3 production modules (cdsd, lsmm_aux_head, ch_rs_ft) and
17 tests, covering #9 CDSD, #23 LSMM, #15 CH-RS-FT, plus the Phase-1
follow-up adapter_cls injection in attach_lora_to_last_n. 88 tests green
in the reliable suite total (70 prior + 17 Phase-2 + 1 Phase-1 follow-up).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-review

**1. Spec coverage.** All three Phase-2 techniques from
`reliable-solutions.md` §A (CDSD) and §B (LSMM, CH-RS-FT) are covered:
- #9 CDSD → Tasks 2 (EMATeacher), 3 (momentum semantics), 4 (channel_dropout),
  5 (cdsd_loss), 6 (no-grad teacher).
- #23 LSMM → Tasks 7 (frozen-buffer head), 8 (non-negative abundances),
  9 (reconstruction loss), 10 (cache round-trip).
- #15 CH-RS-FT → Tasks 11 (smooth_channel_tokens), 12 (mc_smooth_predict),
  13 (n_mc validation).

Phase-1 follow-up (`adapter_cls` injection for OPLoRA) → Task 1.

Integration smoke → Task 14. Docs → Task 15.

Heads (#32 APH, #14 NCI-PIH, #28 MCSE) and memory modules (#11 Hopfield,
#24 SRF-bias) are explicitly out of scope — those are Phases 3 and 4.

**2. Placeholder scan.** No TBD / TODO / "implement later" / "similar to
Task N" / vague-validation patterns. Every code block is complete.

**3. Type consistency.** Across tasks:
- `EMATeacher(student, momentum)`; `.module` attribute holds the cloned
  net; `.update(student)` and `.forward(*args, **kwargs)` — consistent
  Tasks 2, 3, 6, 14.
- `channel_dropout(x, p, min_keep, training) -> Tensor` — Tasks 4, 14.
- `cdsd_loss(student_tokens, teacher_tokens, lambda_distill) -> Tensor`
  — Tasks 5, 14.
- `LSMMHead(d, n_endmembers, n_bands, srf_matrix, endmembers)` — Tasks 7,
  8, 9, 10, 14.
- `LSMMHead.predict_abundances(features) -> Tensor (..., n_endmembers)`
  — Tasks 7, 8.
- `LSMMHead.reconstruction_loss(features, x_rgb, lambda_lsmm) -> Tensor`
  — Tasks 9, 14.
- `smooth_channel_tokens(tokens, sigma, p_smooth, training) -> Tensor`
  — Tasks 11, 14.
- `mc_smooth_predict(model, tokens, n_mc, sigma, p_smooth) -> (Tensor,
  Tensor)` — Tasks 12, 13.
- `attach_lora_to_last_n(model, last_n, rank, adapter_cls=LoRALayer,
  **adapter_kwargs)` — Task 1.

**4. TDD discipline.** Driving-new-code tasks (RED first): 1, 2, 4, 5,
7, 9, 11, 12. Protective-regression tasks (pass on first run): 3, 6,
8, 10, 13. Smoke + docs: 14, 15. Pattern alternates `feat:` ↔ `test:`
just like Phase 1.

---

## Plan execution

Plan complete and saved to
`.cursor/plans/2026-04-25-reliable-training-regularizers-plan.md`. Two
execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task with two-
stage review (spec compliance + code quality). Same flow as Phase 1.

**2. Inline Execution** — run through tasks in this session via
`superpowers:executing-plans`.

Phases that follow Phase 2:

- **Phase 3** — Heads: #32 APH + #14 NCI-PIH + #28 MCSE + the head-
  agnostic null-invariance loss.
- **Phase 4** — Memory / attention bias: #11 Hopfield + #24 SRF-biased
  attention.
- **Phase 5** — Post-training: #21 MERA.
- **Phase 6** — Eval-time safety: #5 ReAct + #7 TC-CAF + #18 BPSG +
  #27 ADAPT + #12 / #26 imputation.
- **Phase 7** — Integration: R-grid + portability tests + CLI flag wiring
  into `train_classifier.py` / `train_segmenter.py` / `train_change.py`
  and the matching eval scripts.
