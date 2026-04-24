# Reliable-Core Phase 1 — LoRA Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the four foundation LoRA techniques from
`.cursor/plans/reliable-solutions.md` as composable modules: #29 LastN-LoRA
placement, #16 OPLoRA projection, #20 LoRA-Null initialization, #2 Hard
Channel Mask. Every piece ships with strict TDD tests and one dedicated CLI
flag can be wired into training scripts in a later phase.

**Architecture:** New package `rs_finetune/reliable/`, one focused module per
technique. LoRA adapters replace the MLP `nn.Linear` projections (`linear1`
and `linear2`) of the last `N` transformer blocks — attention `out_proj` is
*not* wrapped because `nn.MultiheadAttention` reads `out_proj.weight`
directly and module-replacement breaks that contract. Base LoRA keeps a
frozen weight and bias as buffers and zero-inits `B` so the pre-training
forward is preserved exactly. OPLoRA adds double-sided SVD projectors over
the LoRA delta. LoRA-Null initialises `A` in the null-space of subset
activations. Hard Mask gates the residual by a non-learnable per-channel
indicator.

**Tech stack:** PyTorch 2.x, pytest, uv. Existing mock backbones and
fixtures live in `rs_finetune/tests/reliable/conftest.py`.

---

## Progress marker (as of 2026-04-25)

Committed:

- ✅ `reliable/__init__.py` — package marker (commit `c6e527d`).
- ✅ `reliable/lora_layer.py` — base LoRA with zero-init B and frozen
  base_weight buffer (commit `31608f9`). *Does not yet support a frozen
  base bias; Task 2 adds that.*
- ✅ `reliable/last_n_placement.py` v0 — marker + contract behaviours
  (`last_n=0` noop, `N>depth` attaches all, rejects non-transformer
  models) and tail `self_attn.out_proj` replacement (commits `56d39b2`,
  `3fa7064`). *Tail target is wrong*; Task 3 retargets to `linear1` /
  `linear2` and Task 4 verifies forward preservation.
- ✅ `tests/reliable/test_lora_layer.py` — zero-init forward test.
- ✅ `tests/reliable/test_last_n_lora.py` — 5 structural tests.

Remaining work — Tasks 2 through 15 below.

---

## File structure

**Production modules** (all under `rs_finetune/reliable/`):

```
reliable/
    __init__.py              # package marker                    ✅ Task 0
    lora_layer.py            # LoRALayer (+bias)                  ✅ v0 / Task 2 adds bias
    last_n_placement.py      # #29 attach_lora_to_last_n          ✅ v0 / Task 3 retargets
    oplora.py                # #16 build_oplora_projectors, OPLoRALayer       Task 6-10
    lora_null_init.py        # #20 compute_activation_null_basis, init_lora_A Task 11-13
    channel_mask.py          # #2  build_hard_channel_mask, apply_hard_channel_mask Task 14-16
```

**Test modules** (all under `rs_finetune/tests/reliable/`):

```
test_lora_layer.py              ✅ zero-init test / Task 2 adds bias test
test_last_n_lora.py             ✅ marker + contract tests / Task 3-4 update
test_oplora.py                  Task 6-10
test_lora_null_init.py          Task 11-13
test_hard_channel_mask.py       Task 14-16
test_foundation_integration.py  Task 17
```

**Modified existing files:** none in this phase. CLI flag wiring into
`train_classifier.py` / `train_segmenter.py` / `train_change.py` lands in
Phase 7.

---

## Execution rules

- **TDD iron law:** one behaviour per red-green cycle. Write one test, watch
  it RED for a unique reason (`ModuleNotFoundError`, `AttributeError`,
  `AssertionError: values mismatch`), write the minimum code that flips it
  GREEN, commit.
- **If a test passes on its first run, back it out.** Either fold its
  assertions into an existing green test, or drop the test entirely. The
  only tests that ship are ones that caught a real regression or drove new
  code.
- **Run `./run_tests.sh` (full regression) after every commit.** Prior 54
  tests must stay green.
- **All commands run from `rs_finetune/` (cwd).** The repo uses uv; the
  test runner is `./run_tests.sh` which invokes `.venv/bin/python -m pytest`.

---

## Task 1: LoRALayer — zero-init forward

**Status: ✅ done (commit `31608f9`).**

Verify green with:

```bash
./run_tests.sh tests/reliable/test_lora_layer.py
```

Expected: `1 passed` (shared with the infra suite's 48 → full regression
49 passed total).

---

## Task 2: LoRALayer — frozen base bias

**Files:**
- Modify: `rs_finetune/tests/reliable/test_lora_layer.py`
- Modify: `rs_finetune/reliable/lora_layer.py`

- [ ] **Step 1: Write the failing test**

Append to `rs_finetune/tests/reliable/test_lora_layer.py`:

```python
def test_lora_with_base_bias_applies_bias(frozen_pretrained_weight):
    """When a frozen base_bias is passed, zero-init forward equals
    ``x @ W.T + b`` — bias from the wrapped Linear is preserved."""
    w = frozen_pretrained_weight(d_out=8, d_in=4)
    b = torch.randn(8)
    b.requires_grad_(False)
    lora = LoRALayer(d_in=4, d_out=8, rank=2, base_weight=w, base_bias=b)
    assert "base_bias" in dict(lora.named_buffers())
    assert "base_bias" not in dict(lora.named_parameters())
    x = torch.randn(3, 4)
    expected = x @ w.T + b
    assert torch.allclose(lora(x), expected, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
./run_tests.sh tests/reliable/test_lora_layer.py::test_lora_with_base_bias_applies_bias
```

Expected: `FAIL` with `TypeError: LoRALayer.__init__() got an unexpected keyword argument 'base_bias'`.

- [ ] **Step 3: Write minimal implementation**

Replace `rs_finetune/reliable/lora_layer.py` with:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
./run_tests.sh tests/reliable/
```

Expected: `55 passed` (54 prior + 1 new).

- [ ] **Step 5: Commit**

```bash
git add rs_finetune/reliable/lora_layer.py rs_finetune/tests/reliable/test_lora_layer.py
git commit -m "feat(reliable): LoRALayer supports frozen base bias"
```

---

## Task 3: LastN-LoRA — retarget from attention out_proj to MLP linears

**Why this task exists.** The current `attach_lora_to_last_n` replaces
`self_attn.out_proj` with `LoRALayer`. This breaks forward because
`nn.MultiheadAttention` reads `self.out_proj.weight` directly rather than
calling the module — so the `LoRALayer` (which has no `.weight` attribute)
raises `AttributeError`. The correct LoRA-for-transformers target on an
`nn.TransformerEncoderLayer` is the MLP projections `linear1` and
`linear2`, which are plain `nn.Linear` modules called as `self.linear1(x)`.

**Files:**
- Modify: `rs_finetune/tests/reliable/test_last_n_lora.py`
- Modify: `rs_finetune/reliable/last_n_placement.py`

- [ ] **Step 1: Write the failing test**

Replace the existing `test_attach_last_n_replaces_linears_on_tail` in
`rs_finetune/tests/reliable/test_last_n_lora.py` with:

```python
def test_attach_last_n_replaces_mlp_linears_on_tail(tiny_mock_multispec_backbone):
    """Tail layer's MLP linears (linear1, linear2) become LoRALayers
    wrapping the original weights; earlier layers keep plain nn.Linear.

    Attention out_proj is intentionally left alone because
    nn.MultiheadAttention reads out_proj.weight directly rather than
    calling the module."""
    from reliable.lora_layer import LoRALayer

    model = tiny_mock_multispec_backbone(n_channels=4, embed_dim=32)
    attach_lora_to_last_n(model, last_n=1, rank=4)

    tail = model.transformer.layers[-1]
    assert isinstance(tail.linear1, LoRALayer)
    assert isinstance(tail.linear2, LoRALayer)

    head = model.transformer.layers[0]
    assert isinstance(head.linear1, nn.Linear)
    assert isinstance(head.linear2, nn.Linear)
    assert not isinstance(head.linear1, LoRALayer)
    assert not isinstance(head.linear2, LoRALayer)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
./run_tests.sh tests/reliable/test_last_n_lora.py::test_attach_last_n_replaces_mlp_linears_on_tail
```

Expected: `FAIL` with `assert isinstance(tail.linear1, LoRALayer)` — the
current impl wraps `self_attn.out_proj`, not `linear1`.

- [ ] **Step 3: Rewrite implementation**

Replace `rs_finetune/reliable/last_n_placement.py` with:

```python
"""#29 LastN-LoRA placement helper.

Attaches LoRA adapters to the last N transformer layers' MLP linears
(``linear1`` and ``linear2``) so earlier layers stay bit-identical to the
pretrained model. Attention projections are not wrapped because
``nn.MultiheadAttention`` introspects ``out_proj.weight`` directly.
"""

import torch.nn as nn

from reliable.lora_layer import LoRALayer


def _wrap_linear_with_lora(linear: nn.Linear, rank: int) -> LoRALayer:
    base_bias = (
        linear.bias.detach().clone() if linear.bias is not None else None
    )
    return LoRALayer(
        d_in=linear.in_features,
        d_out=linear.out_features,
        rank=rank,
        base_weight=linear.weight.detach().clone(),
        base_bias=base_bias,
    )


_LORA_TARGETS = ("linear1", "linear2")


def attach_lora_to_last_n(model: nn.Module, last_n: int, rank: int) -> None:
    """Attach LoRA adapters to the last ``last_n`` transformer layers' MLP
    projections.

    No-op when ``last_n <= 0``. Raises :class:`ValueError` when the model
    does not expose a ``transformer.layers`` attribute.
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
                setattr(layer, attr, _wrap_linear_with_lora(target, rank=rank))
        layer._lora_registry = {"rank": rank, "targets": list(_LORA_TARGETS)}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
./run_tests.sh tests/reliable/
```

Expected: `55 passed`. If any prior test (`test_attach_last_n_wraps_only_tail`,
`test_attach_last_n_greater_than_depth_attaches_all`,
`test_attach_last_n_zero_is_noop`,
`test_attach_last_n_rejects_non_transformer_model`) now fails, it is
because those tests probed `_lora_registry`, which is still set. If
`test_attach_last_n_wraps_only_tail` fails due to attention-target
expectation, confirm it was asserting only `_lora_registry` membership
(it was) and adjust if needed.

- [ ] **Step 5: Commit**

```bash
git add rs_finetune/reliable/last_n_placement.py rs_finetune/tests/reliable/test_last_n_lora.py
git commit -m "fix(reliable): LastN-LoRA targets MLP linear1/linear2 instead of attention out_proj

nn.MultiheadAttention reads out_proj.weight directly, which breaks when
the module is replaced with a LoRALayer that has no .weight attribute.
Standard LoRA-for-transformers targets are the MLP projections anyway."
```

---

## Task 4: LastN-LoRA — zero-init forward preservation

**Files:**
- Modify: `rs_finetune/tests/reliable/test_last_n_lora.py`

- [ ] **Step 1: Write the failing test**

Append to `rs_finetune/tests/reliable/test_last_n_lora.py`:

```python
def test_attach_last_n_preserves_forward_at_init(
    tiny_mock_multispec_backbone, synthetic_multispec_batch
):
    """With zero-init B and bias preserved, wrapping tail MLP linears must
    not change the model's forward output."""
    import copy

    model = tiny_mock_multispec_backbone(n_channels=4, embed_dim=32)
    reference = copy.deepcopy(model)
    attach_lora_to_last_n(model, last_n=1, rank=4)

    x = synthetic_multispec_batch(n_channels=3)
    channel_ids = [0, 1, 2]
    with torch.no_grad():
        ref_out = reference(x, channel_ids=channel_ids)
        mod_out = model(x, channel_ids=channel_ids)
    assert torch.allclose(ref_out, mod_out, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it passes**

```bash
./run_tests.sh tests/reliable/test_last_n_lora.py::test_attach_last_n_preserves_forward_at_init
```

Expected: `PASS`. If it fails with an `AssertionError` mismatch, run the
test in verbose mode (`-vv`) and inspect the numerical diff. Typical
causes and fixes:

- **Base bias not propagated.** The wrapping helper in Task 3 passes
  `linear.bias` into `LoRALayer`, which Task 2 accepts as `base_bias`.
  If either link is broken, bias falls off and forward diverges by the
  bias vector. Check `tail.linear1.base_bias` equals the original
  `reference.transformer.layers[-1].linear1.bias`.
- **Weight not detached.** `linear.weight.detach().clone()` must be
  detached so its gradient graph isn't shared with the original.
- **Dropout.** `nn.TransformerEncoderLayer` has dropout, but `torch.no_grad()` + `model.eval()` idiom (which we use via absence of dropout in the mock; the mock doesn't set eval) could still flip dropout. Verify with `model.eval()` wrapping if the test is flaky.

Even though this test is expected to pass against Task 3's code, it is the
*reason* Task 3 required the bias retargeting — so we keep it as a
protective regression.

- [ ] **Step 3: Commit**

```bash
git add rs_finetune/tests/reliable/test_last_n_lora.py
git commit -m "test(reliable): LastN-LoRA preserves forward at zero-init"
```

---

## Task 5: OPLoRA — orthogonal projector builders

**Files:**
- Create: `rs_finetune/reliable/oplora.py`
- Create: `rs_finetune/tests/reliable/test_oplora.py`

- [ ] **Step 1: Write the failing test**

Create `rs_finetune/tests/reliable/test_oplora.py`:

```python
"""Tests for OPLoRA (#16)."""

import pytest
import torch

from reliable.oplora import build_oplora_projectors


def test_build_oplora_projectors_shapes_and_idempotence(frozen_pretrained_weight):
    w = frozen_pretrained_weight(d_out=16, d_in=8)
    P_L, P_R = build_oplora_projectors(w, preserve_k=3)
    assert P_L.shape == (16, 16)
    assert P_R.shape == (8, 8)
    # Orthogonal projector must be idempotent: P @ P ≈ P
    assert torch.allclose(P_L @ P_L, P_L, atol=1e-5)
    assert torch.allclose(P_R @ P_R, P_R, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
./run_tests.sh tests/reliable/test_oplora.py
```

Expected: `ERROR` with `ModuleNotFoundError: No module named 'reliable.oplora'`.

- [ ] **Step 3: Write minimal implementation**

Create `rs_finetune/reliable/oplora.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
./run_tests.sh tests/reliable/
```

Expected: `57 passed` (56 prior + 1 new).

- [ ] **Step 5: Commit**

```bash
git add rs_finetune/reliable/oplora.py rs_finetune/tests/reliable/test_oplora.py
git commit -m "feat(reliable): OPLoRA orthogonal projectors from frozen weight SVD"
```

---

## Task 6: OPLoRA — projector validates `preserve_k` range

**Files:**
- Modify: `rs_finetune/tests/reliable/test_oplora.py`

- [ ] **Step 1: Write the failing test**

Append to `rs_finetune/tests/reliable/test_oplora.py`:

```python
def test_build_oplora_projectors_rejects_out_of_range_k(frozen_pretrained_weight):
    w = frozen_pretrained_weight(d_out=8, d_in=4)
    with pytest.raises(ValueError, match="preserve_k"):
        build_oplora_projectors(w, preserve_k=100)
    with pytest.raises(ValueError, match="preserve_k"):
        build_oplora_projectors(w, preserve_k=-1)
```

- [ ] **Step 2: Run test to verify it passes**

```bash
./run_tests.sh tests/reliable/test_oplora.py::test_build_oplora_projectors_rejects_out_of_range_k
```

Expected: `PASS` — Task 5's implementation already raises on both cases.
This is a protective regression against future refactors.

- [ ] **Step 3: Commit**

```bash
git add rs_finetune/tests/reliable/test_oplora.py
git commit -m "test(reliable): OPLoRA rejects invalid preserve_k"
```

---

## Task 7: OPLoRALayer — zero-init forward equals base

**Files:**
- Modify: `rs_finetune/reliable/oplora.py`
- Modify: `rs_finetune/tests/reliable/test_oplora.py`

- [ ] **Step 1: Write the failing test**

Append to `rs_finetune/tests/reliable/test_oplora.py`:

```python
def test_oplora_layer_zero_init_forward_matches_base(frozen_pretrained_weight):
    from reliable.oplora import OPLoRALayer

    w = frozen_pretrained_weight(d_out=16, d_in=8)
    layer = OPLoRALayer(d_in=8, d_out=16, rank=4, base_weight=w, preserve_k=3)
    x = torch.randn(4, 8)
    base_out = x @ w.T
    assert torch.allclose(layer(x), base_out, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
./run_tests.sh tests/reliable/test_oplora.py::test_oplora_layer_zero_init_forward_matches_base
```

Expected: `ERROR` with `ImportError: cannot import name 'OPLoRALayer'`.

- [ ] **Step 3: Append `OPLoRALayer` to `rs_finetune/reliable/oplora.py`**

```python
import torch.nn as nn


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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
./run_tests.sh tests/reliable/
```

Expected: `59 passed` (58 prior + 1 new).

- [ ] **Step 5: Commit**

```bash
git add rs_finetune/reliable/oplora.py rs_finetune/tests/reliable/test_oplora.py
git commit -m "feat(reliable): OPLoRALayer with zero-init double-sided projection"
```

---

## Task 8: OPLoRALayer — preserves top-k singular triples after arbitrary update

**Files:**
- Modify: `rs_finetune/tests/reliable/test_oplora.py`

- [ ] **Step 1: Write the failing test**

Append to `rs_finetune/tests/reliable/test_oplora.py`:

```python
def test_oplora_layer_preserves_top_k_after_arbitrary_update(
    frozen_pretrained_weight,
):
    from reliable.oplora import OPLoRALayer

    w = frozen_pretrained_weight(d_out=16, d_in=8)
    layer = OPLoRALayer(d_in=8, d_out=16, rank=4, base_weight=w, preserve_k=3)

    # Force a large non-zero update to A and B.
    with torch.no_grad():
        layer.A.copy_(torch.randn_like(layer.A))
        layer.B.copy_(torch.randn_like(layer.B))

    delta = layer.B @ layer.A
    effective = w + layer.P_L @ delta @ layer.P_R

    _U_pre, S_pre, _Vh_pre = torch.linalg.svd(w, full_matrices=False)
    _U_eff, S_eff, _Vh_eff = torch.linalg.svd(effective, full_matrices=False)
    # Top-k singular values preserved to numerical precision.
    assert torch.allclose(S_pre[:3], S_eff[:3], rtol=1e-4)
```

- [ ] **Step 2: Run test to verify it passes**

```bash
./run_tests.sh tests/reliable/test_oplora.py::test_oplora_layer_preserves_top_k_after_arbitrary_update
```

Expected: `PASS`. If it fails, the projector math in Task 5 is wrong —
investigate the SVD indexing.

- [ ] **Step 3: Commit**

```bash
git add rs_finetune/tests/reliable/test_oplora.py
git commit -m "test(reliable): OPLoRA preserves top-k singular triples after arbitrary update"
```

---

## Task 9: LoRA-Null — compute activation null-space basis

**Files:**
- Create: `rs_finetune/reliable/lora_null_init.py`
- Create: `rs_finetune/tests/reliable/test_lora_null_init.py`

- [ ] **Step 1: Write the failing test**

Create `rs_finetune/tests/reliable/test_lora_null_init.py`:

```python
"""Tests for LoRA-Null initialization (#20)."""

import pytest
import torch

from reliable.lora_null_init import compute_activation_null_basis


def test_compute_activation_null_basis_shape_and_orthonormal():
    # 100 samples of 8-dim activations; null rank 5.
    activations = torch.randn(100, 8)
    U_null = compute_activation_null_basis(activations, null_rank=5)
    assert U_null.shape == (8, 5)
    # Columns orthonormal.
    gram = U_null.T @ U_null
    assert torch.allclose(gram, torch.eye(5), atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
./run_tests.sh tests/reliable/test_lora_null_init.py
```

Expected: `ERROR` with `ModuleNotFoundError: No module named 'reliable.lora_null_init'`.

- [ ] **Step 3: Write minimal implementation**

Create `rs_finetune/reliable/lora_null_init.py`:

```python
"""#20 LoRA-Null initialization.

Returns an orthonormal basis of a rank-``null_rank`` direction that is
orthogonal to the dominant principal directions of a sample of subset-
forward activations. Used to initialise ``LoRALayer.A`` so that, at the
start of training, ``A @ x ≈ 0`` for any ``x`` in the span of observed
subset activations — the adapter has zero initial effect on the subset
forward.
"""

import torch


def compute_activation_null_basis(
    activations: torch.Tensor, null_rank: int
) -> torch.Tensor:
    if activations.ndim != 2:
        raise ValueError(
            f"activations must be 2D (N, D), got shape {tuple(activations.shape)}"
        )
    _N, D = activations.shape
    if null_rank < 0 or null_rank > D:
        raise ValueError(
            f"null_rank must be in [0, {D}], got {null_rank}"
        )
    _U, _S, Vh = torch.linalg.svd(activations, full_matrices=True)
    V = Vh.T
    return V[:, D - null_rank :].contiguous()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
./run_tests.sh tests/reliable/
```

Expected: `61 passed`.

- [ ] **Step 5: Commit**

```bash
git add rs_finetune/reliable/lora_null_init.py rs_finetune/tests/reliable/test_lora_null_init.py
git commit -m "feat(reliable): LoRA-Null activation null-space basis via SVD"
```

---

## Task 10: LoRA-Null — validate rank arguments

**Files:**
- Modify: `rs_finetune/tests/reliable/test_lora_null_init.py`

- [ ] **Step 1: Write the failing test**

Append to `rs_finetune/tests/reliable/test_lora_null_init.py`:

```python
def test_compute_activation_null_basis_rejects_bad_rank():
    activations = torch.randn(20, 8)
    with pytest.raises(ValueError, match="null_rank"):
        compute_activation_null_basis(activations, null_rank=-1)
    with pytest.raises(ValueError, match="null_rank"):
        compute_activation_null_basis(activations, null_rank=9)


def test_compute_activation_null_basis_rejects_non_2d():
    x3d = torch.randn(2, 3, 4)
    with pytest.raises(ValueError, match="must be 2D"):
        compute_activation_null_basis(x3d, null_rank=2)
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
./run_tests.sh tests/reliable/test_lora_null_init.py
```

Expected: `PASS` for both (Task 9 already validates inputs). Keep as
protective regressions.

- [ ] **Step 3: Commit**

```bash
git add rs_finetune/tests/reliable/test_lora_null_init.py
git commit -m "test(reliable): LoRA-Null validates null_rank range and input dim"
```

---

## Task 11: LoRA-Null — init LoRA.A so subset activations map to null-space

**Files:**
- Modify: `rs_finetune/reliable/lora_null_init.py`
- Modify: `rs_finetune/tests/reliable/test_lora_null_init.py`

- [ ] **Step 1: Write the failing test**

Append to `rs_finetune/tests/reliable/test_lora_null_init.py`:

```python
def test_init_lora_a_in_null_space_kills_subset_activations(
    frozen_pretrained_weight,
):
    """After init, (B @ A) @ x ≈ 0 for any x ∈ span(subset activations).
    Since B is zero at init, the assertion reduces to checking A @ x ≈ 0."""
    from reliable.lora_layer import LoRALayer
    from reliable.lora_null_init import init_lora_a_in_null_space

    # Structured activations: signal on first 4 dims, null on last 4 dims.
    activations = torch.cat(
        [torch.randn(50, 4), torch.zeros(50, 4)], dim=1
    )
    w = frozen_pretrained_weight(d_out=16, d_in=8)
    lora = LoRALayer(d_in=8, d_out=16, rank=4, base_weight=w)
    init_lora_a_in_null_space(lora, activations, null_rank=4)

    projected = activations @ lora.A.T  # (50, 4)
    assert projected.abs().mean() < 1e-4
```

- [ ] **Step 2: Run test to verify it fails**

```bash
./run_tests.sh tests/reliable/test_lora_null_init.py::test_init_lora_a_in_null_space_kills_subset_activations
```

Expected: `ERROR` with `ImportError: cannot import name 'init_lora_a_in_null_space'`.

- [ ] **Step 3: Append implementation to `rs_finetune/reliable/lora_null_init.py`**

```python
from reliable.lora_layer import LoRALayer


def init_lora_a_in_null_space(
    lora: LoRALayer, activations: torch.Tensor, null_rank: int
) -> None:
    """Initialise ``lora.A`` so that ``A @ x ≈ 0`` for ``x`` in the span
    of ``activations``. LoRA ``rank`` must equal ``null_rank``.
    """
    if lora.A.shape[0] != null_rank:
        raise ValueError(
            f"LoRA rank ({lora.A.shape[0]}) must equal null_rank ({null_rank})"
        )
    U_null = compute_activation_null_basis(activations, null_rank=null_rank)
    with torch.no_grad():
        lora.A.copy_(U_null.T)  # (null_rank, D)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
./run_tests.sh tests/reliable/
```

Expected: `64 passed` (63 prior + 1 new).

- [ ] **Step 5: Commit**

```bash
git add rs_finetune/reliable/lora_null_init.py rs_finetune/tests/reliable/test_lora_null_init.py
git commit -m "feat(reliable): init_lora_a_in_null_space zeros subset activations"
```

---

## Task 12: LoRA-Null — rank mismatch raises

**Files:**
- Modify: `rs_finetune/tests/reliable/test_lora_null_init.py`

- [ ] **Step 1: Write the failing test**

Append to `rs_finetune/tests/reliable/test_lora_null_init.py`:

```python
def test_init_lora_a_rank_mismatch_raises(frozen_pretrained_weight):
    from reliable.lora_layer import LoRALayer
    from reliable.lora_null_init import init_lora_a_in_null_space

    activations = torch.randn(50, 8)
    w = frozen_pretrained_weight(d_out=16, d_in=8)
    lora = LoRALayer(d_in=8, d_out=16, rank=3, base_weight=w)
    with pytest.raises(ValueError, match="rank"):
        init_lora_a_in_null_space(lora, activations, null_rank=5)
```

- [ ] **Step 2: Run test**

```bash
./run_tests.sh tests/reliable/test_lora_null_init.py::test_init_lora_a_rank_mismatch_raises
```

Expected: `PASS` (Task 11 validates). Protective regression.

- [ ] **Step 3: Commit**

```bash
git add rs_finetune/tests/reliable/test_lora_null_init.py
git commit -m "test(reliable): LoRA-Null rank/null_rank mismatch raises"
```

---

## Task 13: Hard Channel Mask — non-learnable frozen buffer

**Files:**
- Create: `rs_finetune/reliable/channel_mask.py`
- Create: `rs_finetune/tests/reliable/test_hard_channel_mask.py`

- [ ] **Step 1: Write the failing test**

Create `rs_finetune/tests/reliable/test_hard_channel_mask.py`:

```python
"""Tests for Hard Channel Mask (#2)."""

import pytest
import torch

from reliable.channel_mask import build_hard_channel_mask


def test_hard_channel_mask_is_non_learnable_and_correct():
    mask = build_hard_channel_mask(
        training_channel_ids=[0, 1, 2], n_channels=12
    )
    assert mask.requires_grad is False
    assert mask.shape == (12,)
    expected = torch.zeros(12)
    expected[[0, 1, 2]] = 1.0
    assert torch.equal(mask, expected)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
./run_tests.sh tests/reliable/test_hard_channel_mask.py
```

Expected: `ERROR` with `ModuleNotFoundError: No module named 'reliable.channel_mask'`.

- [ ] **Step 3: Write minimal implementation**

Create `rs_finetune/reliable/channel_mask.py`:

```python
"""#2 Hard Channel Mask — non-learnable per-channel gate for LoRA residuals."""

from collections.abc import Iterable

import torch


def build_hard_channel_mask(
    training_channel_ids: Iterable[int], n_channels: int
) -> torch.Tensor:
    ids = list(training_channel_ids)
    for c in ids:
        if not 0 <= c < n_channels:
            raise ValueError(
                f"Channel id {c} out of range [0, {n_channels})"
            )
    mask = torch.zeros(n_channels)
    mask[ids] = 1.0
    mask.requires_grad_(False)
    return mask
```

- [ ] **Step 4: Run test to verify it passes**

```bash
./run_tests.sh tests/reliable/
```

Expected: `66 passed`.

- [ ] **Step 5: Commit**

```bash
git add rs_finetune/reliable/channel_mask.py rs_finetune/tests/reliable/test_hard_channel_mask.py
git commit -m "feat(reliable): Hard Channel Mask builder as frozen buffer"
```

---

## Task 14: Hard Channel Mask — rejects out-of-range ids

**Files:**
- Modify: `rs_finetune/tests/reliable/test_hard_channel_mask.py`

- [ ] **Step 1: Write the failing test**

Append to `rs_finetune/tests/reliable/test_hard_channel_mask.py`:

```python
def test_hard_channel_mask_rejects_out_of_range_ids():
    with pytest.raises(ValueError, match="out of range"):
        build_hard_channel_mask(training_channel_ids=[0, 12], n_channels=12)
    with pytest.raises(ValueError, match="out of range"):
        build_hard_channel_mask(training_channel_ids=[-1, 0], n_channels=12)
```

- [ ] **Step 2: Run test**

```bash
./run_tests.sh tests/reliable/test_hard_channel_mask.py::test_hard_channel_mask_rejects_out_of_range_ids
```

Expected: `PASS`. Protective regression.

- [ ] **Step 3: Commit**

```bash
git add rs_finetune/tests/reliable/test_hard_channel_mask.py
git commit -m "test(reliable): Hard Channel Mask rejects out-of-range ids"
```

---

## Task 15: Hard Channel Mask — apply zeros unseen-channel residuals

**Files:**
- Modify: `rs_finetune/reliable/channel_mask.py`
- Modify: `rs_finetune/tests/reliable/test_hard_channel_mask.py`

- [ ] **Step 1: Write the failing test**

Append to `rs_finetune/tests/reliable/test_hard_channel_mask.py`:

```python
def test_apply_hard_channel_mask_zeros_unseen_channels():
    from reliable.channel_mask import apply_hard_channel_mask

    residual = torch.ones(2, 12, 64)
    mask = build_hard_channel_mask(
        training_channel_ids=[0, 1, 2], n_channels=12
    )
    gated = apply_hard_channel_mask(residual, mask, channel_ids=list(range(12)))
    # Training channels unchanged; unseen zeroed.
    assert torch.equal(gated[:, :3, :], residual[:, :3, :])
    assert torch.equal(gated[:, 3:, :], torch.zeros_like(gated[:, 3:, :]))


def test_apply_hard_channel_mask_shape_mismatch_raises():
    from reliable.channel_mask import apply_hard_channel_mask

    residual = torch.ones(2, 4, 32)
    mask = build_hard_channel_mask(
        training_channel_ids=[0, 1, 2], n_channels=12
    )
    with pytest.raises(ValueError, match="channel"):
        apply_hard_channel_mask(residual, mask, channel_ids=[0, 1, 2])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
./run_tests.sh tests/reliable/test_hard_channel_mask.py -k apply_hard_channel_mask
```

Expected: both `ERROR` with `ImportError: cannot import name 'apply_hard_channel_mask'`.

- [ ] **Step 3: Append implementation to `rs_finetune/reliable/channel_mask.py`**

```python
def apply_hard_channel_mask(
    residual: torch.Tensor,
    mask: torch.Tensor,
    channel_ids: list[int],
) -> torch.Tensor:
    """Multiply per-channel residual by the hard mask.

    Args:
        residual: ``(B, C, ...)`` where ``C == len(channel_ids)``.
        mask: ``(n_channels,)`` frozen buffer from ``build_hard_channel_mask``.
        channel_ids: which channels each of the ``C`` positions corresponds to.
    """
    if residual.shape[1] != len(channel_ids):
        raise ValueError(
            f"residual has {residual.shape[1]} channel positions but "
            f"channel_ids has {len(channel_ids)}"
        )
    gate = mask[torch.tensor(channel_ids, device=residual.device)]
    shape = [1, len(channel_ids)] + [1] * (residual.ndim - 2)
    return residual * gate.view(*shape)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
./run_tests.sh tests/reliable/
```

Expected: `69 passed` (67 prior + 2 new).

- [ ] **Step 5: Commit**

```bash
git add rs_finetune/reliable/channel_mask.py rs_finetune/tests/reliable/test_hard_channel_mask.py
git commit -m "feat(reliable): apply_hard_channel_mask gates per-channel residuals"
```

---

## Task 16: Foundation integration — mask zeros the adapter contribution on unseen channels

**Files:**
- Create: `rs_finetune/tests/reliable/test_foundation_integration.py`

- [ ] **Step 1: Write the failing test**

Create `rs_finetune/tests/reliable/test_foundation_integration.py`:

```python
"""Integration tests verifying #29 + #2 compose.

Full LoRA-on-attention-with-channel-gating integration lives in Phase 2
once the head and channel-aware LoRA wrapper ship. This file only asserts
the utilities plumb together without type errors."""

import torch

from reliable.channel_mask import apply_hard_channel_mask, build_hard_channel_mask


def test_mask_gates_a_synthetic_per_channel_residual():
    mask = build_hard_channel_mask(
        training_channel_ids=[0, 1, 2], n_channels=12
    )
    # Pretend we have a per-channel residual for channels [0, 1, 2, 6, 10, 11]
    # where 6, 10, 11 are unseen (NIR + SAR).
    channel_ids = [0, 1, 2, 6, 10, 11]
    residual = torch.ones(2, len(channel_ids), 64)
    gated = apply_hard_channel_mask(residual, mask, channel_ids=channel_ids)
    # Training positions (0, 1, 2) unchanged.
    assert torch.equal(gated[:, :3, :], residual[:, :3, :])
    # Unseen positions zeroed.
    assert torch.equal(gated[:, 3:, :], torch.zeros_like(gated[:, 3:, :]))
```

- [ ] **Step 2: Run test to verify it passes**

```bash
./run_tests.sh tests/reliable/test_foundation_integration.py
```

Expected: `PASS`. The utilities are already implemented; this test locks in
their composed shape.

- [ ] **Step 3: Commit**

```bash
git add rs_finetune/tests/reliable/test_foundation_integration.py
git commit -m "test(reliable): foundation integration — mask gates unseen-channel residuals"
```

---

## Task 17: Mark Phase 1 complete in design docs

**Files:**
- Modify: `rs_finetune/../.cursor/plans/reliable-solutions.md`

- [ ] **Step 1: Run the full test suite**

```bash
./run_tests.sh
```

Expected: `70 passed` (48 pre-Phase-1 infra + 22 Phase 1 tests) and no new
warnings beyond the pre-existing `torch.jit` deprecations.

- [ ] **Step 2: Append progress note to `reliable-solutions.md`**

Add the following new section at the end of
`.cursor/plans/reliable-solutions.md`:

```markdown
## Implementation progress

- **Phase 1 (LoRA foundation) — COMPLETE (2026-04-25).**
  Shipped: `reliable/lora_layer.py`, `reliable/last_n_placement.py`,
  `reliable/oplora.py`, `reliable/lora_null_init.py`,
  `reliable/channel_mask.py`. 22 Phase-1 tests green (70 total in
  reliable suite). Plan:
  `.cursor/plans/2026-04-25-reliable-lora-foundation-plan.md`.
```

- [ ] **Step 3: Commit**

```bash
git add .cursor/plans/reliable-solutions.md
git commit -m "docs(reliable): mark Phase 1 LoRA foundation complete"
```

---

## Self-review

**1. Spec coverage.** All four foundation techniques from
`reliable-solutions.md` §A "Universally reliable core" are covered:
- #29 LastN-LoRA → Tasks 3 + 4 (target, preservation)
- #16 OPLoRA → Tasks 5–8 (projectors, layer, guarantee)
- #20 LoRA-Null → Tasks 9–12 (basis, validation, init_A, mismatch)
- #2 Hard Channel Mask → Tasks 13–15 (builder, validation, apply)

Shared prerequisite `LoRALayer` (zero-init + bias) → Tasks 1, 2.

Non-LoRA core techniques (#9 CDSD, #32 APH, #5 ReAct, #11, #23, #24, #21)
are explicitly out of scope and will each get their own plan in Phases 2–7.

**2. Placeholder scan.** No TBD / TODO / "implement later" markers. Every
code block is complete. No "similar to Task N" — each task's code is
self-contained.

**3. Type consistency.** Across tasks:
- `LoRALayer.__init__(d_in, d_out, rank, base_weight, base_bias=None)` —
  same signature used in Tasks 1, 2, 3, 11.
- `LoRALayer.A` shape `(rank, d_in)`, `LoRALayer.B` shape `(d_out, rank)`,
  `register_buffer("base_weight", …)` / `register_buffer("base_bias", …)`
  — consistent throughout.
- `build_oplora_projectors(weight, preserve_k) -> (P_L, P_R)` — Tasks 5, 7.
- `OPLoRALayer` accepts `(d_in, d_out, rank, base_weight, preserve_k,
  base_bias=None)` — consistent with Tasks 7, 8.
- `compute_activation_null_basis(activations, null_rank) -> Tensor` —
  Tasks 9, 11, 12.
- `init_lora_a_in_null_space(lora, activations, null_rank)` — Tasks 11, 12.
- `build_hard_channel_mask(training_channel_ids, n_channels) -> Tensor` —
  Tasks 13, 14, 15, 16.
- `apply_hard_channel_mask(residual, mask, channel_ids) -> Tensor` —
  Tasks 15, 16.

**4. TDD discipline.** Tasks 2, 3, 5, 7, 9, 11, 13, 15 each introduce a
genuine RED (missing module, missing function, missing argument, or
numerical mismatch). Tasks 4, 6, 8, 10, 12, 14, 16 are *protective
regressions* that pass immediately — each is explicitly called out as
such and is justified by being a different contractual assertion than
the driving test. If strict interpretation requires only failing-first
tests, Tasks 6, 8, 10, 12, 14, 16 can be skipped; the driving tests
(2, 3, 5, 7, 9, 11, 13, 15) still cover the main contracts.

---

## Plan execution

Plan complete and saved to
`.cursor/plans/2026-04-25-reliable-lora-foundation-plan.md`. Two execution
options:

**1. Subagent-Driven (recommended if subagents available)** — dispatch a
fresh subagent per task, review between tasks.

**2. Inline Execution** — run through tasks in this session via
`superpowers:executing-plans`.

Phases that follow this one (each will get its own plan after this one
ships):

- Phase 2 — Training regularizers: #9 CDSD + #23 LSMM + #15 CH-RS-FT.
- Phase 3 — Heads: #32 APH + #14 NCI-PIH + #28 MCSE + the head-agnostic
  null-invariance loss.
- Phase 4 — Memory / attention bias: #11 Hopfield + #24 SRF-biased
  attention (the latter pairs with #32 APH).
- Phase 5 — Post-training: #21 MERA.
- Phase 6 — Eval-time safety: #5 ReAct + #7 TC-CAF + #18 BPSG + #27 ADAPT
  + #12 / #26 imputation.
- Phase 7 — Integration (R-grid + portability) + CLI flag wiring into
  `train_classifier.py` / `train_segmenter.py` / `train_change.py` and the
  matching eval scripts.
