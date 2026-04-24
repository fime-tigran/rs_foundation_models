# Stack G — Cross-Band Fine-Tuning Plan

**Status (2026-04-23).** Active design. Not yet locked for implementation.
Stack G+ additions from 2026 SOTA literature sweep: see §12 at the bottom.

**Siblings.**
- `cross-band-finetune-catalog.md` — full library of 14 candidate approaches
  with mechanism, plug-in notes, arxiv refs. This file selects from that
  library and specifies how the selected pieces compose.
- `powerful_chivit_fine-tuning_07a25de9.plan.md` — the existing E0–E18 fix
  flags (HCS, channel-mean pool, embed freeze, channel gate, dropout,
  curriculum, spectral init) that Stack G builds on top of.
- `TRAINING_PLAN.md` — the current ablation matrix and recipe conventions.

---

## 1. Problem & constraints (condensed)

**Problem.**
χViT is fine-tuned on a strict subset of input channels (e.g. RGB only).
At eval, the model must handle:

- **Priority B — monotonic superset.** RGB → RGBN, S2 → S2+S1. Eval on the
  superset must be **provably ≥** eval on the training subset via
  *architectural* design.
- **Priority A — no-overlap.** RGB → S1, S2 → S1, RGB → N'S1S2. Strict
  monotonicity isn't applicable; design should still deliver real gains.

**Constraints (hard, enforced by CLAUDE.md).**

1. Fine-tune only on the training subset. No live 12-band forward during the
   fine-tune loop.
2. Architectural monotonicity preferred over calibrated gates.
3. No re-pretraining.
4. Offline artifacts from the pretraining corpus are allowed (frozen).
5. External pretrained foundation models are allowed at eval.

---

## 2. Architecture overview

```
                ┌──────────────────────────────────────────────┐
                │  Eval-time safety (optional, stackable)      │
                │  ─ #5 ReAct activation clipping              │
                │  ─ #7 Offline TC-CAF conformal fusion        │
                └───────────────────┬──────────────────────────┘
                                    │
    ┌───────────────────────────────┴───────────────────────────┐
    │                    χViT  encoder                          │
    │                                                           │
    │   ┌───────────────────────────────────────────────────┐   │
    │   │   #1  NSP-FT — null-space projection on every     │   │
    │   │       gradient update; subset forward preserved    │   │
    │   │       byte-identical (linear approx.)              │   │
    │   └───────────────────────────────────────────────────┘   │
    │   ┌───────────────────────────────────────────────────┐   │
    │   │   #2  Identity-Init LoRA + Hard Channel Mask —    │   │
    │   │       adapter gated by non-learnable channel-ID   │   │
    │   │       indicator; unseen-band tokens bypass the    │   │
    │   │       adapter entirely                            │   │
    │   └───────────────────────────────────────────────────┘   │
    │   ┌───────────────────────────────────────────────────┐   │
    │   │   #11  Hopfield Channel-Prototype Memory —        │   │
    │   │        frozen bank of pretrained per-channel      │   │
    │   │        vectors; zero-init cross-attention         │   │
    │   │        retrieves at eval (no-overlap lifeline)    │   │
    │   └───────────────────────────────────────────────────┘   │
    │                                                           │
    │   Training-only layers:                                   │
    │   ─ #9  CDSD — within-subset self-distillation prior      │
    │   ─ #4  OFT — optional optimizer for non-χViT baselines   │
    │                                                           │
    └───────────────────┬───────────────────────────────────────┘
                        │
    ┌───────────────────┴───────────────────────────────────────┐
    │   Head                                                    │
    │   ─ default: linear (cls) / UPerNet (seg, CD)             │
    │   ─ optional: #14  NCI-PIH  (set-transformer head with    │
    │      null-channel invariance loss)                        │
    └───────────────────────────────────────────────────────────┘
```

---

## 3. Core layers (all four required)

### Layer 1 — #1 NSP-FT (Null-Space Projected Fine-Tuning)

**Role.** The headline theorem. Guarantees the subset forward is preserved
exactly (linear approximation tight). New-band signal at eval can only
live in the untouched null-space of subset features.

**Mechanism.**

1. *Calibration phase* — run pretrained χViT over the training set, collect
   per-layer activation statistics, compute per-layer orthonormal basis
   `U_ℓ` of the activation subspace, store projector `P_ℓ = I − U_ℓ U_ℓᵀ`.
2. *Fine-tune phase* — every gradient update is projected:
   `W_ℓ ← W_ℓ − η · P_ℓ · ∇L(W_ℓ)`.

**Files to touch.**
- `rs_finetune/classifier_utils.py` — projector calibration, projected-
  optimizer wrapper.
- `rs_finetune/train_classifier.py` / `train_segmenter.py` / `train_change.py`
  — one-time calibration pass at startup; wire the projected optimizer.
- New module: `rs_finetune/nsp_ft.py`.

**New CLI flags.**

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--nsp_ft` | flag | off | Enable null-space projected updates |
| `--nsp_rank` | int | 256 | Rank of `U_ℓ` per layer |
| `--nsp_calib_fraction` | float | 1.0 | Fraction of training set for calibration |
| `--nsp_calib_cache` | path | auto | Cache path for computed projectors |

**Guarantee.** Strict worst-case in linear approximation; first-order tight
for the full non-linear transformer. Empirical drift:
`O(‖ΔW‖² · ‖x‖²)`.

**Cost.** ~5–10 % training overhead from the extra projection; ~9 MB cache
per ViT-B model (rank 256, 12 layers).

**Interaction with HCS.** Open question (see §7). Default: calibrate `U` with
HCS *disabled* (use the fixed training-band set). Alternatives explored in
ablation.

---

### Layer 2 — #2 Identity-Init LoRA + Hard Channel Mask

**Role.** Protects unseen-channel tokens. At eval, NIR / VV / VH tokens flow
through the *frozen pretrained* MLP+attention path, byte-identical to what
iBOT pretraining produced.

**Mechanism.** Attach a LoRA adapter in parallel with each attention and MLP
block. The adapter's residual is multiplied by a **hard, non-learnable**
indicator `m_c ∈ {0, 1}` — 1 iff channel `c` was in the training set, 0
otherwise. Zero-init on `B`.

```python
class ChannelMaskedAdapter(nn.Module):
    def __init__(self, dim, rank=8, training_channels=(0, 1, 2), n_ch=12):
        super().__init__()
        self.lora_A = nn.Linear(dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.lora_B.weight)
        mask = torch.zeros(n_ch)
        mask[list(training_channels)] = 1.0
        self.register_buffer("channel_mask", mask)

    def forward(self, x, channel_ids):
        delta = self.lora_B(self.lora_A(x))
        m = self.channel_mask[channel_ids]
        return x + delta * m[None, :, None]
```

**Files to touch.**
- `rs_finetune/change_detection_pytorch/encoders/chi_vit.py` — wrap each
  block's output with the adapter; pass `channel_idxs` through.

**New CLI flags.**

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--channel_mask_adapter` | flag | off | Enable identity-init + hard mask |
| `--adapter_rank` | int | 8 | LoRA rank |
| `--training_channel_ids` | int list | auto | Channel indices in pretrained embedding table (auto-derived from `--bands`) |

**Guarantee.** Strict worst-case on the N-path (pretrained bit-identical).
RGB path has no standalone guarantee — relies on NSP-FT for that.

**Cost.** `2 × rank × dim` per block, < 0.5 % of model. Negligible train/eval
overhead.

---

### Layer 3 — #9 CDSD (Channel-Dropout Self-Distillation)

**Role.** Training-time signal. Trains the *encoder* to handle variable
channel counts by distilling "dropped student" → "full EMA teacher" on the
same subset of bands. Builds the "channel closure" prior without ever
touching superset bands.

**Mechanism.** iBOT-style EMA teacher; student sees a randomly dropped
version of the subset, distilled against the full-subset teacher's tokens.

```python
# RGB fine-tune step
x = batch.rgb
d = random.randint(0, 2)
keep = torch.ones(3); keep[d] = 0
x_drop = x * keep[None, :, None, None]

t_drop = student.encoder(x_drop)                # (B, n', D)
with torch.no_grad():
    t_full = teacher_ema.encoder(x)             # (B, n, D)

# Cosine distillation on patch tokens
loss_distill = (1 - F.cosine_similarity(
    t_drop[:, 1:], t_full[:, 1:], dim=-1,
)).mean()

loss_ce = F.cross_entropy(student.head(t_drop[:, 0]), y)
loss = loss_ce + lambda_d * loss_distill
```

**Files to touch.**
- `rs_finetune/train_classifier.py` / `train_segmenter.py` / `train_change.py`
  — add EMA teacher, dropout + distillation loss, EMA momentum schedule.
- New module: `rs_finetune/cdsd.py`.

**New CLI flags.**

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--cdsd` | flag | off | Enable channel-dropout self-distillation |
| `--cdsd_lambda` | float | 0.5 | Distillation loss weight |
| `--cdsd_ema_momentum` | float | 0.996 | Teacher EMA momentum |
| `--cdsd_min_keep` | int | 1 | Min channels kept after dropout |

**Guarantee.** Expected-value only. Training signal directly matches the
eval shift — a model trained to fill in missing channels handles extra
channels robustly.

**Cost.** ~1.5× training time (EMA forward). Zero eval overhead.

**Overlap with existing E-grid flags.** CDSD generalises the existing
`--channel_dropout_rate` (which only drops channels, no distillation). If
`--cdsd` is on, prefer it over `--channel_dropout_rate`.

---

### Layer 4 — #11 Hopfield Channel-Prototype Memory

**Role.** The no-overlap (Priority A) lifeline. A frozen memory of per-channel
prototype vectors (from the pretrained χViT's `channel_embed` table). A
zero-init cross-attention head retrieves the closest prototype(s) for each
token, injecting "memory of pretraining" at eval — particularly useful when
eval bands are fully disjoint from training bands.

**Mechanism.**

```python
class ChannelMemoryHead(nn.Module):
    def __init__(self, D, prototypes):     # prototypes: (12, D), frozen
        super().__init__()
        self.register_buffer('memory', prototypes)
        self.cross_attn = nn.MultiheadAttention(D, 8, batch_first=True)
        nn.init.zeros_(self.cross_attn.out_proj.weight)
        nn.init.zeros_(self.cross_attn.out_proj.bias)

    def forward(self, channel_tokens):     # (B, n_ch_present, D)
        B = channel_tokens.shape[0]
        mem = self.memory.unsqueeze(0).expand(B, -1, -1)
        retrieved, _ = self.cross_attn(channel_tokens, mem, mem)
        return channel_tokens + retrieved  # zero at init
```

**Prototype source.** Three options (pick in §7 open questions):
- **P-a** simplest — use `pretrained_chivit.channel_embed` directly
  (already 12 × D, lives in the checkpoint). No pass over data needed.
- **P-b** richer — compute per-channel mean of layer-0 tokens on the iBOT
  pretraining corpus.
- **P-c** richest — per-layer prototypes (one memory per transformer block,
  retrieved at each layer).

**Files to touch.**
- `rs_finetune/change_detection_pytorch/encoders/chi_vit.py` — inject the
  retrieval head; forward channel tokens through it.
- New module: `rs_finetune/hopfield_memory.py`.
- Offline script (only for P-b / P-c): `rs_finetune/build_channel_prototypes.py`.

**New CLI flags.**

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--hopfield_memory` | flag | off | Enable retrieval head |
| `--hopfield_prototype_source` | {`channel_embed`, `layer0_mean`, `per_layer`} | `channel_embed` | Prototype variant |
| `--hopfield_prototype_path` | path | — | Cached prototypes (P-b, P-c) |
| `--hopfield_num_heads` | int | 8 | Cross-attention heads |

**Guarantee.** Worst-case **no-op at initialisation** (zero output
projection). After training the residual may become non-zero; if trained
with small LR and short schedule, retrieval stays a gentle refinement.

**Cost.** ~0.5 M params. +37 KB frozen memory. Offline prototype build
(P-b, P-c): minutes on one GPU.

---

## 4. Optional layers

### #5 ReAct Activation Clipping — eval-time safety net

**Role.** Training-free bounded-drift guarantee. Clip feature activations
at eval to the 95th-percentile seen during training.

**Mechanism.**

1. One pass over the training set with the fine-tuned model; record per-
   layer feature-norm distribution; set `τ_ℓ = Percentile_{95}(norms_ℓ)`.
2. At eval, apply `torch.clamp(x, max=τ_ℓ)` after each transformer block
   output.

**Files.** Forward hook in `chi_vit.py`; active only when `self.training is
False`. New flag `--react_clip --react_percentile 95`.

**Guarantee.** Superset-eval feature at layer `ℓ` differs from subset-eval
by at most `τ_ℓ · n_new_tokens`. Lipschitz-bounded propagation → bounded
output change.

**Cost.** Training-free. ~1 % eval overhead.

**When to include.** Always worth having — free insurance. Ship with
Stack G as a default.

---

### #7 Offline TC-CAF — PAC coverage at eval

**Role.** Adds a PAC-style coverage statement to the paper.

**Mechanism (leak-free, calibrated offline on the pretraining corpus).**

1. *Offline*: for each sample in the pretraining corpus,
   ```
   p_s = pretrained_chivit(x_sub).softmax(-1)
   p_t = pretrained_chivit(x_full).softmax(-1)
   nonconf = ‖p_s − p_t‖₁
   tau = Quantile_{1-α}(nonconf)
   ```
   Cache `tau`.
2. *At eval*: replace per-sample output with a conformal fusion rule —
   teacher if disagreement < `tau`, student otherwise.

**Files.** New script `rs_finetune/calibrate_conformal_tau.py`. Eval
plumbing in all three `eval_bands_*.py`.

**New CLI flags.**

| Flag | Meaning |
|---|---|
| `--conformal_fusion` | Enable fusion |
| `--tau_path` | Cached `tau` from offline calibration |
| `--teacher_ckpt` | Path to frozen pretrained χViT teacher |
| `--conformal_alpha` | Coverage parameter (e.g. 0.05) |

**Guarantee.** PAC coverage `≥ 1 − α − O(1/√n)` relative to pretraining
distribution.

**Cost.** One offline pass over the pretraining corpus. +1 forward per eval.

**When to include.** When you want a distribution-free guarantee in the
paper. Adds a Lemma + an eval-time forward.

---

### #4 OFT / BOFT — portability to non-χViT baselines

**Role.** Drop-in replacement for AdamW + weight decay that adds a
Lipschitz bound on the fine-tuning delta. Portable across iBOT,
DINOv2/v3, ViT-B, DOFA, TerraFM — the full comparison table.

**Mechanism.** Parameterise `W_ft = R · W_pre` with `RᵀR = I` via
`R = exp(Skew(θ))`. BOFT uses block-sparse butterfly factorisation for
efficiency.

**Files.** Either
- use `peft.OFT` directly (preferred), or
- implement wrapper in `rs_finetune/oft_opt.py`.

**New CLI flags.** `--oft_finetune --oft_rank 32 --oft_schedule butterfly`.

**Guarantee.** Feature-space drift bounded by the learned rotation angle.
In practice < 0.1 feature-norm change under typical angles.

**Cost.** Similar to AdamW. Works out-of-the-box on all ViT-family models in
the benchmark.

**When to include.** When running the comparison table — gives the "fair
Lipschitz-controlled fine-tune" baseline for every non-χViT model.

---

### #14 NCI-PIH — null-channel invariance head

**Role.** Head-side architectural invariance. Replace linear classifier
with a Set Transformer trained to be invariant under adding "null channel"
tokens.

**Mechanism.** Per-channel mean pool → Set Transformer → learnable pool
query → classifier. Training augmentation appends `1–3` zero-valued
tokens with a learnable "null" embedding; loss enforces
`‖head(set) − head(set ∪ null)‖² = 0`.

**Files.** `rs_finetune/train_classifier.py`; new module
`rs_finetune/nci_head.py`. For seg/CD: applies only to the global pooled
feature path (UPerNet dense decoder unchanged).

**New CLI flags.** `--nci_head --null_invariance_lambda 0.5`.

**Guarantee.** Expected-value (the invariance claim is distributional, but
the loss trains it directly — in practice it's a strong bias).

**Cost.** ~1 M params. Minor.

**When to include.** When you want an architectural claim at the head, not
just the encoder. Only lightly additive over the existing `channel_mean`
pool — include if the paper's story demands head-side invariance; skip
otherwise.

---

## 5. Composition & interactions

|                   | NSP-FT | Mask Adapter | CDSD   | Hopfield | ReAct  | TC-CAF | OFT     | NCI-PIH |
|-------------------|--------|--------------|--------|----------|--------|--------|---------|---------|
| **NSP-FT**        | —      | ✓            | ✓      | ✓        | ✓      | ✓      | ⚠       | ✓       |
| **Mask Adapter**  | ✓      | —            | ✓      | ✓        | ✓      | ✓      | ✓       | ✓       |
| **CDSD**          | ✓      | ✓            | —      | ✓        | ✓      | ✓      | ✓       | ✓       |
| **Hopfield**      | ✓      | ✓            | ✓      | —        | ✓      | ✓      | ✓       | ✓       |
| **ReAct**         | ✓      | ✓            | ✓      | ✓        | —      | ✓      | ✓       | ✓       |
| **TC-CAF**        | ✓      | ✓            | ✓      | ✓        | ✓      | —      | ✓       | ✓       |
| **OFT**           | ⚠      | ✓            | ✓      | ✓        | ✓      | ✓      | —       | ✓       |
| **NCI-PIH**       | ✓      | ✓            | ✓      | ✓        | ✓      | ✓      | ✓       | —       |

⚠ **NSP-FT × OFT** — both constrain gradient updates. NSP-FT projects into
activation null-space; OFT constrains to orthogonal rotations. Their
intersection is a strict subspace and may leave no useful update direction.
Use one or the other: NSP-FT for χViT headline, OFT for non-χViT baselines.

---

## 6. Workflow

### Offline (one-time, before fine-tuning)

1. **NSP-FT calibration** — single forward pass over training set with
   pretrained χViT; compute and cache `U_ℓ` per layer.
2. **Hopfield prototypes** — if P-b or P-c variant: one pass over iBOT
   pretraining corpus to compute per-channel means. If P-a: no pass, read
   `channel_embed` from checkpoint.
3. **Offline conformal calibration** (only if TC-CAF enabled) — pass over
   pretraining corpus computing student/teacher disagreement nonconformity;
   cache `tau`.

### Fine-tune (per experiment)

Standard existing loop (`train_classifier.py` / `train_segmenter.py` /
`train_change.py`) with:

- Channel-masked adapters wrapped around each χViT block (#2).
- NSP-FT projection applied at each `optimizer.step()` (#1).
- CDSD distillation loss added to the total loss (#9).
- Hopfield retrieval head forward — active, but zero residual until trained
  (#11).
- (Optional) OFT optimizer instead of AdamW for non-χViT baselines (#4).
- (Optional) NCI-PIH head replaces linear classifier (#14).

### Eval (per experiment × per band combo)

Standard `eval_bands_*.py` with:

- (Optional) ReAct clipping forward hook active (#5).
- (Optional) Conformal fusion gate active (#7).

---

## 7. Open design questions

These must resolve before implementation locks in.

1. **NSP-FT ↔ HCS.** χViT's HCS samples random channel subsets during
   fine-tune, so the "subset-forward activation subspace" itself varies.
   Options: (a) calibrate `U` over the union of HCS-subsampled forwards;
   (b) disable HCS during the calibration pass, use the fixed full training-
   band set; (c) per-subset-size family `U^{(S)}` selected at each batch.
   *Default proposal: (b); revisit in ablation.*

2. **Hopfield prototype source.** P-a (use `channel_embed` directly) vs
   P-b (layer-0 mean over pretraining corpus) vs P-c (per-layer banks).
   *Default proposal: start with P-a (zero engineering cost); graduate to
   P-b if P-a underperforms on no-overlap.*

3. **Seg/CD portability of each layer.**
   - NSP-FT: unchanged (operates on weights, not task outputs).
   - Mask Adapter: unchanged (χViT block wrapper).
   - CDSD: the EMA teacher needs to handle UPerNet decoder — dense prediction
     distillation rather than CLS-token distillation. Non-trivial.
   - Hopfield: retrieval head must emit patch-aligned tokens that feed into
     UPerNet; architectural surgery required.
   - NCI-PIH: applies only to the global feature path; UPerNet's dense
     decoder bypasses it.
   *Default proposal: start classification-only to iterate fast; port to
   seg/CD after classification results are clean.*

4. **Eval protocol for the monotonicity claim.** The paper claim "superset
   eval ≥ subset eval" needs a matched-pair test: same checkpoint, same
   sample, eval once with subset bands and once with superset bands,
   compare metric. Need to add a per-sample subset-band eval mode alongside
   the existing `--bands` JSON interface.

5. **Compute budget for the experiment matrix.** Net training overhead ≈
   1.6× baseline (NSP-FT ~10 %, CDSD ~50 %, others negligible). Full G-grid
   would cost 1.6 × (E0–E18 cost). Decide whether to run the full E-grid
   on m_eurosat first before touching BigEarthNet / So2Sat / Harvey / OSCD.

---

## 8. Experiment matrix

Proposed ablation grid — suffix `G` to distinguish from existing E0–E18.
Primary dataset: `m_eurosat` (match the E-grid protocol).

| ID | NSP-FT | Mask Adapter | CDSD | Hopfield | ReAct | TC-CAF | NCI-PIH | Purpose |
|----|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| G0 | — | — | — | — | — | — | — | Baseline: χViT E9 recipe, no Stack-G additions |
| G1 | ✓ | — | — | — | — | — | — | NSP-FT alone — headline theorem isolation |
| G2 | ✓ | ✓ | — | — | — | — | — | +Identity-Init Mask — protects N-path |
| G3 | ✓ | ✓ | ✓ | — | — | — | — | +CDSD — training-time closure prior |
| **G4** | ✓ | ✓ | ✓ | ✓ | — | — | — | **Stack G core (all four layers)** |
| G5 | ✓ | ✓ | ✓ | ✓ | ✓ | — | — | +ReAct safety net |
| G6 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | +TC-CAF PAC guarantee |
| G7 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Full stack incl. NCI-PIH head |

**Knock-out ablations** (run only if G4 beats G0):
- G4 − NSP-FT: measure contribution of the provable layer.
- G4 − Mask Adapter: measure contribution of N-path protection.
- G4 − CDSD: measure contribution of the closure prior.
- G4 − Hopfield: measure contribution of the retrieval memory (especially
  on no-overlap eval).

**Eval scope per ID.** Same as E-grid: RGB (in-dist), RGBN (superset-B),
S2+S1 (if S2-trained), S1 (no-overlap-A). See
`ablation_classification_common.sh` for the canonical flag wiring.

---

## 9. What each layer delivers

| Layer | Priority B (superset) | Priority A (no-overlap) | Paper claim |
|---|---|---|---|
| #1 NSP-FT | Core guarantee | Indirect (subset forward preserved → fallback works) | Theorem: subset eval preserved; superset ≥ subset on linear approx. |
| #2 Mask Adapter | Supports the guarantee | Strict: N-path is pretrained | Lemma: unseen-band output is pretrained-equivalent |
| #9 CDSD | Expected lift | Expected lift via closure prior | Empirical: RGBN-RGB delta increase on E-grid datasets |
| #11 Hopfield | Modest lift | Main driver | Empirical: S1-only eval lift via retrieved prototypes |
| #5 ReAct | Drift bound | Drift bound | Proposition: output-drift Lipschitz bound |
| #7 TC-CAF | PAC coverage | PAC coverage | Theorem: distribution-free coverage via training-conditional conformal |
| #4 OFT | Portability | Portability | Empirical: Lipschitz-safe FT on every benchmark model |
| #14 NCI-PIH | Head invariance | Head invariance | Architectural claim on head (loss-verified) |

---

## 10. Key arxiv references

- **NSP-FT**: Saha et al., ICLR 2021, [2103.09762](https://arxiv.org/abs/2103.09762); Qiu et al., NeurIPS 2023, [2306.07280](https://arxiv.org/abs/2306.07280).
- **Identity-Init Adapters**: AdaptFormer, Chen et al., NeurIPS 2022, [2205.13535](https://arxiv.org/abs/2205.13535); DoRA, Liu et al., ICML 2024, [2402.09353](https://arxiv.org/abs/2402.09353); VeRA, Kopiczko et al., ICLR 2024, [2310.11454](https://arxiv.org/abs/2310.11454).
- **CDSD (self-distillation)**: iBOT, Zhou et al., ICLR 2022, [2111.07832](https://arxiv.org/abs/2111.07832); MultiMAE, Bachmann et al., ECCV 2022, [2204.01678](https://arxiv.org/abs/2204.01678); I-JEPA, Assran et al., CVPR 2024, [2301.08243](https://arxiv.org/abs/2301.08243).
- **Hopfield retrieval**: Ramsauer et al., ICLR 2021, [2008.02217](https://arxiv.org/abs/2008.02217); RA-ViT, Iscen et al., CVPR 2023, [2304.01195](https://arxiv.org/abs/2304.01195).
- **ReAct**: Sun et al., NeurIPS 2021, [2111.12797](https://arxiv.org/abs/2111.12797); ASH, Djurisic et al., ICLR 2023, [2209.09858](https://arxiv.org/abs/2209.09858).
- **TC-CAF**: Angelopoulos & Bates 2023, [2107.07511](https://arxiv.org/abs/2107.07511); Bian & Barber 2022, [2205.14301](https://arxiv.org/abs/2205.14301); Conformal Risk Control, Angelopoulos et al., ICLR 2024, [2208.02814](https://arxiv.org/abs/2208.02814).
- **OFT / BOFT**: Qiu et al., NeurIPS 2023, [2306.07280](https://arxiv.org/abs/2306.07280); Liu et al., ICLR 2024, [2311.06243](https://arxiv.org/abs/2311.06243).
- **NCI-PIH**: Set Transformer, Lee et al., ICML 2019, [1810.00825](https://arxiv.org/abs/1810.00825); DeepSets, Zaheer et al., NeurIPS 2017, [1703.06114](https://arxiv.org/abs/1703.06114).

---

## 11. Next steps

1. Resolve the five open questions in §7 — either inline here (user review)
   or as a design-meeting follow-up.
2. Resolve the Stack G+ selection questions in §12.5.
3. Write the full implementation plan (files, CLI, tests, ablation scripts)
   via the `superpowers:writing-plans` skill — *after* Stack G / Stack G+ is
   locked.
4. Begin implementation under `rs_finetune/` following the E-grid conventions.

---

## 12. Stack G+ — 2026 SOTA additions

Surfaced by a four-thread literature sweep (variable-modality ViTs / modality-
incremental, DRO / IRM / Bayesian, feature-space + TTA + contrastive, spectral
physics). Thirteen new techniques (#15–#27) added to
`cross-band-finetune-catalog.md`. This section folds the highest-leverage ones
into an updated recommended stack and experiment matrix.

### 12.1 What changed

- **NSP-FT gets a concrete implementation.** #16 OPLoRA (AAAI 2026) is the
  published, validated realization of Stack G's #1 NSP-FT idea — SVD the
  frozen weights, double-sided-project LoRA updates onto the orthogonal
  complement of the top-k singular subspace. Top-k singular triples
  preserved exactly. Pair with #20 LoRA-Null for init.
- **New certified guarantee.** #15 CH-RS-FT gives a *certified* probabilistic
  l₀ guarantee over channel-token additions — the first (to our knowledge)
  formal subset→superset certificate achievable from subset-only training.
- **Cheap high-leverage add-on.** #17 HP-Freeze — offline Head Pursuit probe,
  freeze 5 % of specialized attention heads.
- **Principled safety gate.** #18 BPSG — Bayesian posterior credible-interval
  fallback; structurally distinct from the rejected option (c).
- **Physics prior.** #24 SRF-Biased Attention — 12×12 physical similarity
  matrix from published Sentinel metadata, free parameter-less bias.
- **Training-time regularizers.** #22 Channel-Embed Diffusion, #23 LSMM Head,
  #25 DEO Dual-Teacher (RGB-only).
- **Post-training recovery.** #21 MERA — task-arithmetic merge of pretrained
  and fine-tuned for no-overlap recovery.
- **Eval-time.** #26 TerraMind TiM (alternative to #12 DiffusionSat for
  no-overlap), #27 ADAPT (closed-form Gaussian alignment, eval-time).

### 12.2 Stack G+ architecture (proposed)

```
                ┌──────────────────────────────────────────────────┐
                │  Eval-time safety (stack freely)                 │
                │  ─ #5 ReAct clipping                             │
                │  ─ #7 Offline TC-CAF conformal fusion            │
                │  ─ #15 CH-RS-FT Monte-Carlo smoothing + cert.    │
                │  ─ #18 BPSG Bayesian posterior safety gate       │
                │  ─ #27 ADAPT closed-form Gaussian alignment      │
                │  ─ #26 TerraMind TiM synthesis (no-overlap only) │
                └───────────────────┬──────────────────────────────┘
                                    │
    ┌───────────────────────────────┴────────────────────────────────┐
    │                         χViT encoder                           │
    │                                                                │
    │   ┌────────────────────────────────────────────────────────┐   │
    │   │  Offline probe (one-time, pretraining corpus):          │   │
    │   │    ─ #17 HP-Freeze — identify + freeze specialized heads │   │
    │   └────────────────────────────────────────────────────────┘   │
    │   ┌────────────────────────────────────────────────────────┐   │
    │   │  #1 NSP-FT  =  #16 OPLoRA  +  #20 LoRA-Null init        │   │
    │   │       SVD-based projection; top-k singular preservation │   │
    │   └────────────────────────────────────────────────────────┘   │
    │   ┌────────────────────────────────────────────────────────┐   │
    │   │  #2 Identity-Init LoRA + Hard Channel Mask               │   │
    │   │       optionally upgraded to #19 SI-LoRA (Bayesian)      │   │
    │   └────────────────────────────────────────────────────────┘   │
    │   ┌────────────────────────────────────────────────────────┐   │
    │   │  #24 SRF-Biased Attention — 12×12 physics bias           │   │
    │   └────────────────────────────────────────────────────────┘   │
    │   ┌────────────────────────────────────────────────────────┐   │
    │   │  #11 Hopfield Channel-Prototype Memory                   │   │
    │   └────────────────────────────────────────────────────────┘   │
    │                                                                │
    │   Training-only layers:                                        │
    │   ─ #9 CDSD (or #22 ChEmbed Diffusion as alternative)          │
    │   ─ #23 LSMM auxiliary reconstruction head                     │
    │   ─ #25 DEO dual-teacher (RGB-only: χViT-EMA + DINOv3)         │
    │   ─ #4 OFT optimizer (portability to non-χViT baselines)       │
    │                                                                │
    └───────────────────┬────────────────────────────────────────────┘
                        │
    ┌───────────────────┴────────────────────────────────────────────┐
    │   Head                                                         │
    │   ─ default linear (cls) / UPerNet (seg, CD)                    │
    │   ─ optional: #14 NCI-PIH                                      │
    └────────────────────────────────────────────────────────────────┘

    Post-training (optional):
      ─ #21 MERA task-arithmetic merge + short subset-realign
```

### 12.3 Recommended Stack G+ core (4 required layers, upgraded)

| Layer | Stack G (current) | Stack G+ (proposed) | Why upgrade |
|-------|-------------------|---------------------|-------------|
| Gradient constraint | #1 NSP-FT (theory) | **#16 OPLoRA + #20 LoRA-Null init** | Published concrete implementation with exact top-k preservation theorem |
| Unseen-band protection | #2 Identity-Init + Hard Mask | #2 (unchanged) — optionally #19 SI-LoRA for Bayes | Optional Bayesian variant adds calibrated uncertainty |
| Training closure prior | #9 CDSD | #9 **or #22 ChEmbed Diffusion** | Alternative with manifold-based invariance |
| Memory / retrieval | #11 Hopfield | #11 (unchanged) | |
| **NEW: Specialized head freezing** | — | **#17 HP-Freeze** | Cheap worst-case addition — freezes 5% of heads specialized for SAR/NIR |

### 12.4 Optional layer expansion

Grouped by when they fire:

- **Offline, before fine-tune:**
  - #20 LoRA-Null init
  - #17 HP-Freeze probe + freeze
  - Prototypes for #11
  - VCA endmember dictionary for #23
  - Source-class Gaussians for #27
  - Conformal τ for #7 (TC-CAF)

- **During fine-tune:**
  - Core: #16 OPLoRA, #2 Mask, #9 CDSD (or #22), #11
  - Training regularizers: #23 LSMM aux head, #24 SRF-biased attention
  - Dual teachers: #25 DEO (RGB-only)

- **Post-training:**
  - #21 MERA — task-arithmetic merge + realign

- **Eval-time:**
  - #5 ReAct, #7 TC-CAF, #15 CH-RS-FT MC smoothing, #18 BPSG, #27 ADAPT
  - No-overlap fallback: #26 TerraMind TiM synthesis or #12 DiffusionSat

### 12.5 Open selection questions

1. **Which NSP-FT realization?** #16 OPLoRA (per-weight SVD, top-k preservation,
   AAAI 2026) vs pure gradient projection (original NSP-FT, Saha et al.).
   *Default:* go with #16 — concrete formulation + published validation.

2. **CDSD vs Channel-Embed Diffusion (#22)?** CDSD is the simpler baseline;
   ChEmbed Diffusion is more principled (on-manifold invariance) but requires
   fitting a tiny diffusion model offline. *Default:* start with CDSD; swap in
   #22 as ablation.

3. **Which Bayesian variant (if any) to adopt?** #18 BPSG as eval-time gate,
   #19 SI-LoRA as Bayesian adapter, or both? *Default:* BPSG (more valuable,
   eval-only, no training changes).

4. **Include #15 CH-RS-FT?** It's the strongest formal guarantee of anything
   on the menu but requires smoothing-compatible training + MC eval. *Default:*
   include as optional add-on; run G10 ablation to measure cost.

5. **Include #21 MERA?** Cheap post-training recovery step; minimal risk.
   *Default:* include; may be a big gain for no-overlap Priority A.

### 12.6 Updated experiment matrix

Drop-in expansion of §8. Keep G0–G7 as defined; add G8–G13 for new techniques.

| ID | New Stack G+ additions |
|----|----|
| G0–G7 | as §8 (Stack G core experiments) |
| G8 | G4 + **#16 OPLoRA** (replaces NSP-FT implementation) |
| G9 | G8 + **#17 HP-Freeze** |
| G10 | G9 + **#15 CH-RS-FT** (certified smoothing at train + eval) |
| G11 | G9 + **#24 SRF-biased attention** |
| G12 | G9 + **#22 Channel-Embed Diffusion** (replaces #9 CDSD) |
| G13 | G11 + **#23 LSMM aux head** + **#25 DEO dual-teacher** |
| G14 | Full Stack G+ (G9 + #15 + #24 + #23 + #25 + #18 + #21) |

**Knock-out ablations** (only if G14 beats G4):
- G14 − OPLoRA, G14 − HP-Freeze, G14 − CH-RS-FT, G14 − SRF, G14 − LSMM,
  G14 − DEO, G14 − MERA.

**Primary paper-eval configurations:**
- **Monotonicity theorem:** G10 (with certified CH-RS-FT guarantee).
- **Strongest expected score:** G14 (everything combined).
- **Most portable to non-χViT baselines:** #4 OFT applied to iBOT / DINOv2 /
  DINOv3 / ViT-B / DOFA / TerraFM in the comparison table.

### 12.7 Updated key references (2025–2026 additions)

- **OPLoRA**: Lin et al., AAAI 2026, [2510.13003](https://arxiv.org/abs/2510.13003).
- **LoRA-Null**: Mar 2025, [2503.02659](https://arxiv.org/abs/2503.02659).
- **CH-RS-FT / Hierarchical Randomized Smoothing**: Schuchardt et al., [2310.16221](https://arxiv.org/abs/2310.16221); AdaptDel NeurIPS 2025, [2511.09316](https://arxiv.org/pdf/2511.09316).
- **HP-Freeze / Head Pursuit**: Milan-Valverde et al., NeurIPS 2025, [2510.21518](https://arxiv.org/abs/2510.21518); Causal Head Gating, [2505.13737](https://arxiv.org/abs/2505.13737).
- **BPSG / SBA / Bayesian-LoRA**: SBA Feb 2026, [2602.17809](https://arxiv.org/abs/2602.17809); Bayesian-LoRA Jan 2026, [2601.21003](https://arxiv.org/abs/2601.21003); BayesLoRA Jun 2025, [2506.22809](https://arxiv.org/abs/2506.22809).
- **MERA**: Huang et al., Mar 2025, [2503.07663](https://arxiv.org/abs/2503.07663).
- **Channel-Embed Diffusion (inspired by)**: N-JEPA [2507.15216](https://arxiv.org/abs/2507.15216); MADCL [2509.20048](https://arxiv.org/abs/2509.20048); Manifold Diffusion [2510.02305](https://arxiv.org/abs/2510.02305).
- **LSMM Head / KARMA**: Dec 2025, [2512.12445](https://arxiv.org/abs/2512.12445).
- **SRF-Biased Attention / STARS**: [2411.05714](https://arxiv.org/abs/2411.05714).
- **DEO Dual-Teacher**: Feb 2026, [2602.19863](https://arxiv.org/abs/2602.19863).
- **TerraMind**: ICCV 2025, [2504.11171](https://arxiv.org/abs/2504.11171).
- **ADAPT Backprop-Free TTA**: Aug 2025, [2508.15568](https://arxiv.org/abs/2508.15568); Tilting the Latent Distribution, Feb 2026, [2602.02633](https://arxiv.org/abs/2602.02633).

---

## 13. Stack G++ — Head-only / LoRA-only scope

**Added 2026-04-23.** Active scope restriction: fine-tuning is now limited to
**head modifications** (classifier-head architecture, training, and
regularization) and **LoRA-family adapters** only. No full-backbone weight
updates. This changes which catalog techniques apply and promotes head/LoRA
specialists to the core.

### 13.1 What changed from §3–§4

- **Backbone weights are frozen by construction.** This makes several
  "preserve subset forward" techniques automatic.
- **LoRA constraint techniques become central.** OPLoRA, LoRA-Null init,
  SI-LoRA, Identity-Init+Mask all plug into the LoRA adapter directly.
- **Head architecture matters more** since it's the primary capacity lever.
- **Five new approaches added to the catalog:** #28 MCSE, #29 LastN-LoRA,
  #30 ChE-LoRA, #31 VPT, #32 APH.

### 13.2 Compatibility audit under head-only / LoRA-only

| # | Technique | Status |
|---|---|---|
| 1 | NSP-FT | **Subsumed.** Under LoRA → use #16 OPLoRA. Under head-only → trivially satisfied (backbone frozen). |
| 2 | Identity-Init LoRA + Hard Mask | **★ Core** |
| 9 | CDSD | Keep — loss-level |
| 11 | Hopfield Memory | Keep — head-adjacent cross-attention |
| 5 | ReAct clipping | Keep — eval-only |
| 7 | TC-CAF | Keep — eval-only |
| 4 | OFT / BOFT | Optional — applies to LoRA factors (orthogonal LoRA) |
| 14 | NCI-PIH | **★ Core** (head) |
| 15 | CH-RS-FT | Keep — input-token smoothing, backbone-weight-free |
| 16 | OPLoRA | **★ Core** (LoRA) |
| 17 | HP-Freeze | **Retired** — backbone heads already frozen |
| 18 | BPSG | Keep — Bayesian posterior over LoRA |
| 19 | SI-LoRA | Optional — Bayesian variant of #2 |
| 20 | LoRA-Null Init | **★ Core** (LoRA) |
| 21 | MERA | Keep — LoRA task-arithmetic merge |
| 22 | ChEmbed Diffusion | Keep — forward-time input augmentation |
| 23 | LSMM Aux Head | **★ Core** (head) |
| 24 | SRF-Biased Attention | Keep — parameter-free forward bias |
| 25 | DEO Dual-Teacher | Optional — loss-level |
| 26 | TerraMind TiM | Keep — eval-only (no-overlap) |
| 27 | ADAPT | Keep — eval-only, closed-form |
| 28 | MCSE | **★ Core** (head, new) |
| 29 | LastN-LoRA | **★ Core** (LoRA placement, new) |
| 30 | ChE-LoRA | **★ Core** (χViT-specific LoRA, new) |
| 31 | VPT | Optional — backbone-frozen prompts |
| 32 | APH | Optional — attention-pooled head |

### 13.3 Stack G++ architecture (head-only / LoRA-only)

```
                ┌──────────────────────────────────────────────────┐
                │  Eval-time safety (stack freely)                 │
                │  ─ #5 ReAct ─ #7 TC-CAF ─ #15 CH-RS-FT           │
                │  ─ #18 BPSG ─ #27 ADAPT                          │
                │  ─ #26 TerraMind TiM (no-overlap only)           │
                └───────────────────┬──────────────────────────────┘
                                    │
    ┌───────────────────────────────┴────────────────────────────────┐
    │  χViT encoder — BACKBONE FROZEN                                │
    │                                                                │
    │   Parameter-free forward modifications:                         │
    │   ─ #24 SRF-Biased Attention (12×12 physics bias)              │
    │   ─ #31 VPT learnable prompts (optional)                       │
    │                                                                │
    │   LoRA adapters — LAST-N BLOCKS ONLY (#29 LastN-LoRA):         │
    │   ┌────────────────────────────────────────────────────────┐   │
    │   │  #16 OPLoRA + #20 LoRA-Null init                        │   │
    │   │   (or alt: #19 SI-LoRA Bayesian variant)                │   │
    │   │  Hard channel mask per #2                               │   │
    │   └────────────────────────────────────────────────────────┘   │
    │                                                                │
    │   χViT-specific LoRA target:                                    │
    │   ─ #30 ChE-LoRA on channel_embed (training channels only)     │
    │                                                                │
    │   ┌────────────────────────────────────────────────────────┐   │
    │   │  #11 Hopfield Channel-Prototype Memory                   │   │
    │   │       (zero-init cross-attention head)                   │   │
    │   └────────────────────────────────────────────────────────┘   │
    │                                                                │
    │   Training regularizers (loss-level, backbone-frozen):         │
    │   ─ #9 CDSD  or  #22 ChEmbed Diffusion                         │
    │   ─ #23 LSMM Auxiliary Head (spectral reconstruction prior)    │
    │   ─ #25 DEO Dual-Teacher (optional)                            │
    │                                                                │
    └───────────────────┬────────────────────────────────────────────┘
                        │
    ┌───────────────────┴────────────────────────────────────────────┐
    │  Head (pick one)                                                │
    │   ─ #14 NCI-PIH  —  permutation-invariant set-transformer       │
    │   ─ #32 APH      —  attention-pooled head                       │
    │   ─ #28 MCSE     —  multi-head channel-subset ensemble          │
    │   ─ (baseline linear head)                                      │
    └────────────────────────────────────────────────────────────────┘

    Post-training (optional):
      ─ #21 MERA: task-arithmetic merge of LoRA weights
                  (pretrained = α·0 + (1−α)·trained_LoRA, plus realign)
```

### 13.4 Stack G++ core — recommended configuration

| Role | Technique | Notes |
|------|-----------|-------|
| **LoRA placement** | #29 LastN-LoRA (N=4) | Keep early spectral blocks untouched |
| **LoRA constraint** | #16 OPLoRA, preserve_k = 32 | Top-32 singular triples preserved exactly |
| **LoRA init** | #20 LoRA-Null init | Updates start in subset-activation null-space |
| **LoRA channel gating** | #2 Identity-Init + Hard Channel Mask | N-path pretrained bit-identical |
| **LoRA on embeddings** | #30 ChE-LoRA, rank = 4 | χViT-specific channel_embed adaptation |
| **Forward-time priors** | #24 SRF-Biased Attention | Physics-grounded, parameter-free |
| **Training regularizer** | #9 CDSD *or* #22 ChEmbed Diffusion | Pick one; #22 is the newer, more principled variant |
| **Auxiliary signal** | #23 LSMM Aux Head | Spectral unmixing prior from pretraining corpus |
| **Memory head** | #11 Hopfield Prototypes (P-a variant) | Uses pretrained `channel_embed` directly |
| **Classifier head** | #14 NCI-PIH *or* #32 APH *or* #28 MCSE | Variable-channel-aware; start with APH |

### 13.5 Stack G++ optional add-ons

- **Parameter-free backbone-frozen:** #31 VPT (deep, 10 prompts/layer).
- **Alternative LoRA:** #19 SI-LoRA (Stiefel-Bayesian) replaces #16; gives
  posterior variance as calibration signal.
- **Dual teacher:** #25 DEO with χViT-EMA + DINOv3, RGB-only.
- **Eval-time safety layer:** combine #5 + #18 + #27; optionally #15 for
  certified-radius claim.
- **Eval-time no-overlap booster:** #26 TerraMind TiM or #12 DiffusionSat.
- **Post-training:** #21 MERA (LoRA task-arithmetic merge + short realign).

### 13.6 Experiment matrix — Stack G++ grid

Drop-in replacement for §8 and §12.6. Adjusted to LoRA-only scope.

| ID | LoRA placement | LoRA constraint | Head | Extras | Purpose |
|----|---|---|---|---|---|
| H0 | none (head-only linear) | — | linear | — | Baseline: χViT E9 head-only recipe |
| H1 | LastN=4 | vanilla LoRA | linear | — | Pure LoRA baseline |
| H2 | LastN=4 | #16 OPLoRA + #20 LoRA-Null | linear | — | LoRA-constrained baseline |
| H3 | LastN=4 | #16+#20 | linear | #2 hard mask | +N-path preservation |
| H4 | LastN=4 | #16+#20 | linear | #2 + #30 ChE-LoRA | +channel-embed LoRA |
| H5 | LastN=4 | #16+#20 | APH (#32) | #2 + #30 | +attention-pooled head |
| H6 | LastN=4 | #16+#20 | APH | #2+#30 + #11 Hopfield | +prototype memory |
| H7 | LastN=4 | #16+#20 | APH | #2+#30+#11 + #23 LSMM | +spectral aux head |
| H8 | LastN=4 | #16+#20 | APH | H7 + #24 SRF-bias + #9 CDSD | +forward prior + closure |
| **H9** | LastN=4 | #16+#20 | APH | H8 + eval-time: #5+#18 | **Stack G++ core** |
| H10 | LastN=4 | #16+#20 | APH | H9 + #15 CH-RS-FT | +certified radius |
| H11 | LastN=4 | #16+#20 | APH | H9 + #21 MERA post-train | +merge-realign |
| H12 | LastN=4 | #16+#20 | **#28 MCSE** | H8 + MCSE ensemble head | Head-ensemble variant |
| H13 | LastN=4 | **#19 SI-LoRA** | APH | H8 | Bayesian Stiefel variant |
| H14 | LastN=4 | #16+#20 | APH | H9 + all optional add-ons | Full Stack G++ |

Knock-out ablations (only if H14 beats H0):
- H14 − OPLoRA, H14 − ChE-LoRA, H14 − Hopfield, H14 − LSMM,
  H14 − SRF-bias, H14 − CDSD, H14 − APH (→ linear), H14 − LastN (→ full-depth LoRA).

### 13.7 Open selection questions (§12.5 revisions)

1. **N for LastN-LoRA?** N=2 (minimal, only decoder-ish blocks) vs N=4
   (balanced) vs N = all blocks (full-depth LoRA). *Default:* N=4.
2. **LoRA constraint variant?** #16 OPLoRA (deterministic) vs #19 SI-LoRA
   (Bayesian). *Default:* #16 for the headline run; #19 as H13 ablation.
3. **Head architecture:** #14 NCI-PIH (null-invariance) vs #32 APH (attention
   pooling) vs #28 MCSE (subset ensemble) vs baseline linear. *Default:* APH
   for breadth; MCSE for the uncertainty story.
4. **Training regularizer:** #9 CDSD vs #22 ChEmbed Diffusion. *Default:*
   CDSD for simplicity; swap to #22 as ablation.
5. **Should any full-fine-tune techniques be kept as a point of comparison?**
   Stack G (full FT) could be one row in the final paper table to quantify
   how much performance is lost by restricting to head+LoRA. *Default:* yes
   — include Stack G G4 as a non-scope-compliant comparison row.
