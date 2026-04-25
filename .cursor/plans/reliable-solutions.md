# Reliable Cross-Band Solutions — Portable Stack

**Status (2026-04-23).** Curated subset of the 32-approach catalog in
`cross-band-finetune-catalog.md`, filtered against a *tightened* set of
requirements. Each technique here is expected to work predictably well **for
every model in the comparison table**.

## Scope

**Training mode.** Head-only + LoRA-family adapters. No full-backbone
fine-tuning.

**Task coverage.** Both Priority B (monotonic superset: RGB→RGBN, S2→S2+S1)
and Priority A (no-overlap: RGB→S1, S2→S1, RGB→N'S1S2).

**Target models.** The solutions must be reliable on:

| Model | Pretraining | Input | Notes |
|---|---|---|---|
| **χViT** | iBOT, 12-band S2+S1 | per-channel tokens | primary model; has `channel_embed` table |
| **TerraFM** | multisensor multi-modal S1+S2 | per-modality patch embeddings | multispectral-aware |
| **DOFA** | wavelength-hypernet, broad spectra | hypernet-generated embeddings | multispectral-aware |
| **DINOv2** | ImageNet, RGB only | stacked RGB | **no multispectral knowledge** |
| **DINOv3** | multi-modal, RGB-dominant | stacked RGB | **no multispectral knowledge** |

**Embedding generation is handled separately** (user-owned). This file only
cares about head / LoRA / loss / eval-time components that sit *after* each
model produces a feature representation.

## Portability filter

A technique is "reliable" iff all of the following hold:

1. **Frozen-backbone compatible.** Only head or LoRA parameters are updated.
2. **Architecture-agnostic.** Does not rely on a specific internal structure
   (e.g. `channel_embed` table, per-channel tokens *inside* the encoder). A
   technique that uses per-channel features *at the head level* is fine —
   the user-owned embedding generator produces those.
3. **Graceful degradation on DINOv2/v3.** DINOv2 and DINOv3 are RGB-only
   pretrained and are in the comparison table specifically as "what happens
   without multispectral pretraining." A technique that extracts value from
   the backbone's multispectral prior (e.g. retrieves pretrained per-band
   prototypes) is **reliable** as long as it **reduces to a no-op on DINOv2/
   v3 rather than hurting them**. Techniques that would actively damage
   DINOv2/v3 (by injecting noise or breaking the RGB pathway) are still
   excluded.
4. **Addresses B and/or A.** Contributes to at least one priority, preferably
   both.
5. **Guarantee well-defined across models.** Strict / bounded / PAC /
   expected — tier clearly stated.
6. **Complexity budget respected.** A simpler core technique covers the same
   job? Drop the complex alternative (keep it in the catalog for reference).

---

## Reliable techniques

Two categories. Both are "reliable" under the revised filter — the difference
is whether a technique draws on the backbone's multispectral prior.

### A — Universally reliable core

Uniformly reliable across all five models. Pick one from each role.

| Role | Technique | Catalog # | Tier | What it does |
|------|-----------|-----------|------|--------------|
| **LoRA placement** | LastN-LoRA (N=4) | #29 | — | Restrict LoRA to last 4 transformer blocks. |
| **LoRA constraint** | OPLoRA + LoRA-Null init | #16, #20 | 1 | Top-k singular triples of frozen weights preserved exactly. |
| **Unseen-band protection** | Identity-Init LoRA + Hard Channel Mask | #2 | 1 | Unseen-channel features bypass the adapter via non-learnable indicator. |
| **Training regularizer** | CDSD | #9 | 4 | Channel dropout + EMA self-distillation. Input-level loss. |
| **Classifier head** | APH (attention-pooled head) | #32 | 4 | Learnable query attends over per-channel features; variable-count-aware. |
| **Eval-time safety** | ReAct activation clipping | #5 | 2 | Clip feature norms to training-set 95-pct. Training-free. |

### B — Multispectral-reliable core (strong for χViT/TerraFM/DOFA, no-op on DINOv2/v3)

These techniques extract value from a multispectral pretrained backbone and
**degrade gracefully** (zero-init pathway, optional regularizer, etc.) on
RGB-only-pretrained DINOv2/v3.

| Role | Technique | Catalog # | Tier | What it does | Behavior on DINOv2/v3 |
|------|-----------|-----------|------|--------------|------------------------|
| **Priority-A lifeline** | Hopfield channel-prototype memory | #11 | 1 | Frozen memory of pretrained per-band prototypes; zero-init cross-attention retrieves at eval. Primary no-overlap lever. | Cross-attention stays at zero → no-op. No harm. |
| **Spectral regularizer** | LSMM auxiliary head (KARMA-lite) | #23 | 4 | Offline VCA endmember dictionary; aux reconstruction loss `RGB = SRF_RGB · E · α`. Head discarded at eval. | Still a physics-grounded regularizer on RGB features; weaker signal but not harmful. |
| **Physics attention bias** | SRF-biased attention in APH | #24 | 2 | Frozen 12×12 Sentinel SRF-overlap matrix added as pre-softmax bias inside APH attention. | Still a principled attention prior; physics holds regardless of pretraining. |

**Combined Reliable-Core = A ∪ B.** Nine techniques.

### Reliable optional add-ons

Stack on top without reducing portability.

| # | Name | Role | Why reliable |
|---|------|------|-------------|
| **#7** | TC-CAF (offline conformal fusion) | Eval-time PAC guarantee | Uses frozen teacher + offline calibration on pretraining corpus. |
| **#21** | MERA (merge-then-realign) | Post-training recovery | LoRA task-arithmetic blend; model-agnostic. Strong lever for Priority A. |
| **#18** | BPSG (Bayesian LoRA posterior gate) | Eval-time credible-interval check | Any LoRA admits Laplace / KFAC posterior. |
| **#27** | ADAPT (closed-form Gaussian alignment) | Eval-time feature alignment | Offline Gaussians from pretraining corpus. |
| **#14** | NCI-PIH (null-channel invariance head) | Alternative/stacked head | Pure set-transformer + invariance loss. |
| **#28** | MCSE (multi-head channel-subset ensemble) | Alternative head with uncertainty | K linear heads, ensemble variance = epistemic uncertainty. |
| **#15** | CH-RS-FT (channel-token randomized smoothing) | Eval-time certified radius | Feature-level Gaussian noise smoothing. |
| **#12** / **#26** | Diffusion / TerraMind imputation | No-overlap fallback | External generative FM imputes bands at eval. |

---

## Techniques removed from the reliable list (with reasons)

Promoted from caveat → core in this revision: **#11 Hopfield, #23 LSMM, #24
SRF-biased attention** (graceful on DINOv2/v3; strong on the multispectral
three).

Staying out of the reliable list:

| # | Name | Reason it's not in reliable-core |
|---|------|---|
| #1 | NSP-FT (full-FT gradient projection) | Subsumed by #16 OPLoRA under LoRA mode — same idea, tighter theorem. |
| #3 | Frozen-Expert MoE | Requires encoder architecture surgery — not frozen-backbone compatible. |
| #4 | OFT / BOFT | Replaces optimizer; #16 OPLoRA already gives a stronger LoRA-specific bound. Added complexity, no gain. |
| #6 | DOFA wavelength hypernet | Rejected — requires re-pretraining, redundant for GeoCrossBench. |
| #8 | PC-MS | Rejected — runs live 12-band teacher, violates subset-only rule. |
| #10 | Evidential per-channel | Known failure modes (evidence collapse). Not reliably reproducible across 5 models. Uncertainty signal already covered by #28 MCSE or #18 BPSG. |
| #13 | Perceiver-IO read-in | Requires encoder architecture surgery. |
| #17 | HP-Freeze | Backbone already frozen → nothing to freeze. Spirit of "preserve specialized heads" achieved by #29 LastN + #16 OPLoRA. Not a DINOv2/v3 issue — redundancy applies to all 5 models. |
| #19 | SI-LoRA (Stiefel Bayesian) | Portable, but #16 OPLoRA + #18 BPSG gives the same benefit without Stiefel parameterization complexity. |
| #22 | ChEmbed Diffusion | χViT-specific (needs `channel_embed` table). Can appear as a χViT-only bonus row. |
| #25 | DEO dual-teacher (RGB-only) | 2–3× training cost for marginal improvement over single-teacher #9 CDSD. |
| #30 | ChE-LoRA | χViT-specific. Can appear as a χViT-only bonus row. |
| #31 | VPT | Adds prompt-token management complexity for limited gain on this problem; prompts are most useful when pretrained knowledge is misaligned with the task, which isn't our core issue. |

---

## Recommended minimal reliable stack

```
                ┌──────────────────────────────────────────────────┐
                │  Eval-time (stack freely)                        │
                │  ─ #5  ReAct feature clipping       (always)     │
                │  ─ #7  TC-CAF conformal fusion      (optional)   │
                │  ─ #18 BPSG Bayesian posterior gate (optional)   │
                │  ─ #27 ADAPT Gaussian alignment     (optional)   │
                │  ─ #12 / #26 imputation             (no-overlap) │
                └───────────────────┬──────────────────────────────┘
                                    │
    ┌───────────────────────────────┴────────────────────────────────┐
    │  User-owned embedding generator → per-channel features         │
    │  (model-specific: χViT / TerraFM / DOFA / DINOv2 / DINOv3)     │
    └───────────────────┬────────────────────────────────────────────┘
                        │
    ┌───────────────────┴────────────────────────────────────────────┐
    │  LoRA adapters — last 4 blocks only (#29)                      │
    │                                                                 │
    │  For each LoRA:                                                 │
    │   ─ #20 LoRA-Null init                                          │
    │   ─ #16 OPLoRA — orthogonal projection on update                │
    │   ─ #2  Hard channel mask — unseen-channel features see Δ=0    │
    │                                                                 │
    │  Training-only losses:                                          │
    │   ─ #9  CDSD — channel dropout + EMA self-distillation          │
    │   ─ #23 LSMM aux head — VCA-endmember reconstruction (B)        │
    └───────────────────┬────────────────────────────────────────────┘
                        │
    ┌───────────────────┴────────────────────────────────────────────┐
    │  Feature-level memory (before the head):                        │
    │   ─ #11 Hopfield — frozen per-band prototype memory             │
    │                    (retrieval head, zero-init cross-attention;  │
    │                     on DINOv2/v3 → stays at zero, no harm)      │
    └───────────────────┬────────────────────────────────────────────┘
                        │
    ┌───────────────────┴────────────────────────────────────────────┐
    │  Head: #32 APH — Attention-Pooled Head                          │
    │   ─ Learnable query attends over per-channel features           │
    │   ─ #24 SRF-biased attention — 12×12 Sentinel SRF overlap as    │
    │        pre-softmax bias inside the APH attention                │
    │   ─ MLP produces class logits                                   │
    └────────────────────────────────────────────────────────────────┘

    Post-training:
      ─ #21 MERA: blend pretrained-backbone-plus-zero-LoRA with
                  pretrained-backbone-plus-trained-LoRA, pick α,
                  then short subset-only realign. Strong no-overlap
                  recovery for all 5 models.
```

### One-line summary per technique (in stack order)

| # | Name | What it buys | Works best on |
|---|------|--------------|----------------|
| 29 | LastN-LoRA (N=4) | Early blocks untouched | all 5 |
| 20 | LoRA-Null Init | Updates start in null-space of subset activations | all 5 |
| 16 | OPLoRA | Top-k singular triples of frozen weights preserved exactly | all 5 |
| 2 | Hard Channel Mask | Unseen-channel features bypass adapter | all 5 |
| 9 | CDSD | Training-time closure prior | all 5 |
| 23 | LSMM aux head | Physics-grounded spectral reconstruction regularizer | χViT/TerraFM/DOFA strong; DINOv2/v3 weak-but-safe |
| 11 | Hopfield prototype memory | Pretrained per-band prototypes retrieved at eval | χViT/TerraFM/DOFA strong; DINOv2/v3 no-op |
| 32 | APH | Variable-count-aware attention-pooled head | all 5 |
| 24 | SRF-biased attention (in APH) | Physics-based attention prior | all 5 (useful whenever APH attends over per-channel features) |
| 5 | ReAct | Eval-time feature-norm clipping | all 5 |
| 21 | MERA | Post-training merge recovers pretrained behavior on unseen bands | all 5 (especially DINOv2/v3) |

### Optional add-ons (layered safely)

- **Add #7 TC-CAF** for a distribution-free PAC coverage statement.
- **Add #18 BPSG** for per-sample Bayesian safety.
- **Add #27 ADAPT** for deterministic eval-time feature alignment.
- **Add #28 MCSE** as a head-ensemble alternative to #32 APH (explicit
  epistemic uncertainty).
- **Add #14 NCI-PIH** as a secondary stage after APH for null-invariance.
- **Add #12 DiffusionSat or #26 TerraMind TiM** for no-overlap (RGB→S1) eval
  — especially helpful for DINOv2/v3 which have no SAR prior to retrieve.

---

## Experiment matrix

Drop-in replacement for the H-grid in `stack-g-design.md` §13.6. Suffix `R`
denotes the *Reliable-Core* grid. Each row is runnable for each of
{χViT, TerraFM, DOFA, DINOv2, DINOv3} without architectural changes.

| ID | LoRA | Head | Training losses | Memory | Eval-time | Post-train | Purpose |
|----|---|---|---|---|---|---|---|
| R0 | none (head-only) | linear | CE | — | — | — | Baseline per model |
| R1 | #29 LastN=4 vanilla | linear | CE | — | — | — | Pure LoRA baseline |
| R2 | #29 + #16 + #20 | linear | CE | — | — | — | LoRA-constrained baseline |
| R3 | R2 + #2 hard mask | linear | CE | — | — | — | +N-path preservation |
| R4 | R3 | **#32 APH** | CE | — | — | — | +attention-pooled head |
| R5 | R3 | APH | CE + #9 CDSD | — | — | — | +closure prior |
| R6 | R3 | APH | CE + CDSD | **#11 Hopfield** | — | — | +prototype memory |
| R7 | R3 | APH | CE + CDSD + **#23 LSMM** | #11 | — | — | +spectral aux head |
| R8 | R3 | APH + **#24 SRF-bias** | CE + CDSD + #23 LSMM | #11 | — | — | +physics attn bias |
| **R9** | R3 | APH + #24 | CE + CDSD + #23 | #11 | **#5 ReAct** | — | **Reliable-Core (recommended)** |
| R10 | R3 | APH + #24 | CE + CDSD + #23 | #11 | ReAct + **#7 TC-CAF** | — | +PAC guarantee |
| R11 | R3 | APH + #24 | CE + CDSD + #23 | #11 | ReAct + #18 BPSG | — | +Bayesian gate |
| R12 | R3 | APH + #24 | CE + CDSD + #23 | #11 | ReAct | **#21 MERA** | +merge recovery |
| R13 | R3 | APH + #24 | CE + CDSD + #23 | #11 | ReAct + #7 + #27 | #21 | Full reliable stack |
| R14 | R3 | **#28 MCSE** | CE + CDSD + #23 | #11 | ReAct | #21 | Ensemble-head variant |
| R15 | R3 | APH + #24 | CE + CDSD + #23 | #11 | ReAct + #15 CH-RS-FT | #21 | +certified radius |
| R16-nov | R3 | APH + #24 | CE + CDSD + #23 | #11 | ReAct + **#26 TerraMind** | #21 | No-overlap booster |

**Primary comparisons to run across all five models:**
- **R0, R1, R2, R9 (Reliable-Core), R13 (Full reliable).**
- Each evaluated on: in-dist (training bands), superset (B), no-overlap (A).

**Per-model expectations for R9 vs R0 (% points, rough intuition):**
- χViT / TerraFM / DOFA — all of {R9 additions} contribute; expect large
  superset lift + meaningful no-overlap lift.
- DINOv2 / DINOv3 — #11 and #23 contribute less (backbone doesn't have
  multispectral prior to surface), but the LoRA-constraint stack and #21
  MERA still fire. Expect modest superset lift; no-overlap gain primarily
  through imputation (R16 row).

**Ablation (only once R9 beats R0):** knock out each core component from R9
(R9 − #16, R9 − Mask, R9 − CDSD, R9 − LSMM, R9 − Hopfield, R9 − SRF-bias,
R9 − APH, R9 − LastN, R9 − ReAct). For DINOv2/v3, the R9 − Hopfield and
R9 − LSMM ablations are expected to be ≈ no-change (those techniques are
no-ops on those backbones).

---

## Expected behavior per model

What each model should get from the Reliable-Core (R9):

| Model | Priority B (superset) | Priority A (no-overlap) | Which core layers fire strongly |
|---|---|---|---|
| **χViT** | Large gain | Real gain | All 9 core layers fire; Hopfield pulls from iBOT-pretrained channel_embed; LSMM leverages 12-band feature richness. |
| **TerraFM** | Large gain | Real gain | All 9; Hopfield uses multisensor-pretrained prototypes. |
| **DOFA** | Large gain | Real gain | All 9; Hopfield prototypes come from wavelength-hypernet outputs. |
| **DINOv2** | Modest gain | Small gain (add imputation) | LoRA stack + APH + CDSD + #21 MERA + #5 ReAct fire normally; **#11 Hopfield cross-attn stays near zero (no-op, harmless)**; **#23 LSMM aux loss is weaker but still a regularizer**; **#24 SRF-bias is a well-defined attention prior regardless**. For Priority A, add **#26 TerraMind TiM imputation** at eval. |
| **DINOv3** | Modest-to-good gain | Small gain (add imputation) | Same as DINOv2; multimodal pretraining makes #23 LSMM slightly stronger. |

**Key design invariant: no step hurts any model.**

- Multispectral-pretrained backbones (χViT, TerraFM, DOFA): everything fires
  at full strength; multispectral-prior techniques (#11, #23, #24) surface
  pretrained knowledge to the head.
- RGB-only-pretrained backbones (DINOv2/v3): multispectral-prior techniques
  degrade to no-op or to a regularizer signal that's consistent with the
  RGB feature space. The universal layers (LoRA-constraint stack, CDSD,
  APH, ReAct, MERA) still fire at full strength.

This is deliberate — χViT/TerraFM/DOFA get the best of the multispectral
story; DINOv2/v3 get the RGB-baseline story cleanly.

---

## Open design questions specific to the reliable scope

1. **N for LastN-LoRA per model?** ViT-B has 12 blocks; LastN=4 is the default.
   For ViT-L / S it may need to scale proportionally. *Default: 33 % of total
   depth, rounded up.*
2. **OPLoRA `k` per model?** Top-k singular triples preserved. Default: 32.
   Larger models may benefit from k=64.
3. **APH vs MCSE?** APH gives one head with attention; MCSE is K ensemble
   heads. APH is the simpler reliable default; MCSE only if you want
   per-sample uncertainty.
4. **MERA α search.** Task-arithmetic coefficient α needs a held-out *training
   distribution* val split (RGB → RGB, not RGB → RGBN) to pick α. Constraint-
   compliant. *Default: α ∈ {0.3, 0.5, 0.7}, pick by RGB-val accuracy.*
5. **Per-model-specific calibrations?** ReAct percentile (95 % default) and
   CDSD dropout rate (0.3 default) may want per-model tuning. *Default: use
   the same hyperparameters everywhere and report any tuning in appendix.*

---

## Files to create during implementation

```
rs_finetune/
    # Universal core (A — all 5 models)
    lora_reliable_core.py       # #29 LastN placement + #16 OPLoRA + #20 LoRA-Null + #2 Hard Mask
    cdsd.py                     # #9  Training-time channel-dropout self-distillation
    aph_head.py                 # #32 Attention-pooled head (optional #24 SRF-bias slot)
    react_clip.py               # #5  Eval-time feature-norm clipping

    # Multispectral core (B — strong on χViT/TerraFM/DOFA, no-op on DINOv2/v3)
    hopfield_memory.py          # #11 Frozen channel-prototype memory + zero-init cross-attention
    lsmm_aux_head.py            # #23 VCA endmember dictionary + auxiliary reconstruction head
    srf_bias.py                 # #24 12×12 Sentinel SRF overlap (frozen buffer, used inside APH)

    # Post-training (all 5 models)
    mera_merge.py               # #21 LoRA task-arithmetic merge + subset-realign

    # Optional add-ons
    calibrate_tc_caf.py         # #7  Offline conformal tau calibration
    bpsg_gate.py                # #18 Bayesian posterior safety gate
    adapt_align.py              # #27 Closed-form Gaussian alignment
    mcse_head.py                # #28 Multi-head channel-subset ensemble
    nci_head.py                 # #14 Null-channel invariance head
    ch_rs_smoothing.py          # #15 Channel-token randomized smoothing
    external_imputation.py      # #12 DiffusionSat / #26 TerraMind wrapper
```

Each module self-contained; no cross-module hacks. Plug-in points in the three
existing training scripts (`train_classifier.py`, `train_segmenter.py`,
`train_change.py`) + the matching eval scripts.

**Offline artifacts to precompute (one-time, from the pretraining corpus):**
- LoRA-Null activation SVD per LoRA'd weight (#20).
- Per-layer top-k singular vectors for OPLoRA (#16).
- Per-band prototype vectors for Hopfield memory (#11).
- VCA endmember dictionary for LSMM (#23).
- 12×12 SRF overlap matrix from Sentinel-2/S1 metadata (#24) — this is
  published physics, not data-derived.
- (Optional) τ for conformal fusion (#7); source-class Gaussians for ADAPT
  (#27); ReAct percentile thresholds (#5); feature-norm training statistics.

---

## Flag reference — one primary toggle per solution

Each technique gets a dedicated CLI flag so it can be turned on/off
independently for parallel ablation runs. Primary flags default **off**;
enable explicitly. Sub-flags are for tuning.

### Training-time flags

| # | Technique | Primary flag | Default | Sub-flags |
|---|-----------|--------------|---------|-----------|
| 29 | LastN-LoRA | `--lora_last_n N` | `0` (= off) | `--lora_rank 8` |
| 16 | OPLoRA | `--enable_oplora` | off | `--oplora_preserve_k 32` |
| 20 | LoRA-Null Init | `--enable_lora_null_init` | off | `--lora_null_rank 256`, `--lora_null_calib_path` |
| 2 | Hard Channel Mask | `--enable_hard_channel_mask` | off | training channel IDs derived from `--bands` |
| 9 | CDSD | `--enable_cdsd` | off | `--cdsd_lambda 0.5`, `--cdsd_ema_momentum 0.996`, `--cdsd_min_keep 1` |
| 11 | Hopfield Memory | `--enable_hopfield_memory` | off | `--hopfield_prototype_source {channel_embed,layer0_mean,per_layer}`, `--hopfield_prototype_path`, `--hopfield_num_heads 8` |
| 23 | LSMM Aux Head | `--enable_lsmm_aux_head` | off | `--lsmm_endmembers_path`, `--lsmm_n_endmembers 16`, `--lsmm_lambda 0.3` |
| 24 | SRF-Biased Attention | `--enable_srf_bias` | off | `--srf_bias_scale 1.0`, `--srf_overlap_path` |
| 15 | CH-RS-FT | `--enable_ch_rs_ft` | off | `--ch_rs_sigma 0.1`, `--ch_rs_p_smooth 0.3`, `--ch_rs_n_mc 50` |
| 22 | ChEmbed Diffusion | `--enable_chembed_diffusion` | off | `--chembed_diff_noise_steps 3`, `--chembed_diff_model_path` |
| 30 | ChE-LoRA | `--enable_che_lora` | off | `--che_lora_rank 4` |
| 25 | DEO Dual-Teacher | `--enable_deo_dual_teacher` | off | `--deo_teacher_ckpts path1,path2`, `--deo_lambda 0.3` |

### Head architecture (mutually exclusive)

| # | Head type | Flag | Sub-flags |
|---|-----------|------|-----------|
| — | linear (baseline) | `--head_type linear` (default) | — |
| 32 | APH | `--head_type aph` | `--aph_num_heads 8` |
| 14 | NCI | `--head_type nci` | `--nci_invariance_lambda 0.5` (applies to any head_type if combined below) |
| 28 | MCSE | `--head_type mcse` | `--mcse_subsets power_set`, `--mcse_reduction mean` |

**Head-agnostic add-on loss:** `--enable_null_invariance_loss` (works on any
`--head_type`) — trains the chosen head to ignore null-channel additions;
enables the #14 training-time invariance objective without forcing a NCI-type
architecture.

### Eval-time flags

| # | Technique | Primary flag | Default | Sub-flags |
|---|-----------|--------------|---------|-----------|
| 5 | ReAct clipping | `--enable_react_clip` | off | `--react_percentile 95`, `--react_stats_path` |
| 7 | TC-CAF | `--enable_tc_caf` | off | `--tc_caf_tau_path`, `--tc_caf_alpha 0.05`, `--tc_caf_teacher_ckpt` |
| 18 | BPSG | `--enable_bpsg` | off | `--bpsg_n_mc 20`, `--bpsg_ci_width 2.0` |
| 27 | ADAPT align | `--enable_adapt_align` | off | `--adapt_gaussians_path` |
| 12/26 | Imputation | `--imputation {none,diffusionsat,terramind}` | `none` | `--imputation_ckpt`, `--imputation_steps 20` |

### Post-training flag

| # | Technique | Primary flag | Default | Sub-flags |
|---|-----------|--------------|---------|-----------|
| 21 | MERA merge | `--enable_mera_merge` | off | `--mera_alpha 0.5` (or comma-list to sweep), `--mera_realign_steps 1000` |

### R-grid → flag mapping

Each row is minimal; cumulative over the row above unless noted.

| Row | Invocation |
|-----|------------|
| R0 | `--head_type linear --only_head` (baseline; Lightning `--only_head`) |
| R1 | `--lora_last_n 4` |
| R2 | R1 + `--enable_oplora --enable_lora_null_init` |
| R3 | R2 + `--enable_hard_channel_mask` |
| R4 | R3 + `--head_type aph` |
| R5 | R4 + `--enable_cdsd` |
| R6 | R5 + `--enable_hopfield_memory` |
| R7 | R6 + `--enable_lsmm_aux_head` |
| R8 | R7 + `--enable_srf_bias` |
| **R9** | R8 + `--enable_react_clip` **← Reliable-Core** |
| R10 | R9 + `--enable_tc_caf` |
| R11 | R9 + `--enable_bpsg` |
| R12 | R9 + `--enable_mera_merge` |
| R13 | R9 + `--enable_tc_caf --enable_adapt_align --enable_mera_merge` |
| R14 | R13 but swap `--head_type aph` → `--head_type mcse` |
| R15 | R12 + `--enable_ch_rs_ft` |
| R16-nov | R12 + `--imputation terramind --imputation_ckpt ...` |

### Knock-out ablation matrix

Once R9 clears R0 on a given model, drop each flag individually to measure
contribution:

| Ablation label | Command diff from R9 |
|----------------|-----------------------|
| R9 − OPLoRA | drop `--enable_oplora` |
| R9 − LoRA-Null | drop `--enable_lora_null_init` |
| R9 − Mask | drop `--enable_hard_channel_mask` |
| R9 − CDSD | drop `--enable_cdsd` |
| R9 − Hopfield | drop `--enable_hopfield_memory` |
| R9 − LSMM | drop `--enable_lsmm_aux_head` |
| R9 − SRF | drop `--enable_srf_bias` |
| R9 − APH | set `--head_type linear` |
| R9 − LastN | set `--lora_last_n 12` (full-depth LoRA) |
| R9 − ReAct | drop `--enable_react_clip` |

For χViT-bonus rows: add `--enable_che_lora` (#30) or
`--enable_chembed_diffusion` (#22) on top of R9. These are χViT-specific so
not in the portable grid.

### Parallel execution pattern

The existing `rs_finetune/ablation_classification_common.sh` and
`ablation_cls_array.sh` already do one-row-per-SLURM-array-index dispatch.
Reuse the pattern:

```bash
# In ablation_reliable_common.sh, add one branch per R-row:
case "$EXPERIMENT_ID" in
    R0)  FLAGS="--head_type linear" ;;
    R1)  FLAGS="--lora_last_n 4" ;;
    R2)  FLAGS="--lora_last_n 4 --enable_oplora --enable_lora_null_init" ;;
    R3)  FLAGS="$(R2_FLAGS) --enable_hard_channel_mask" ;;
    ...
    R9)  FLAGS="$(R8_FLAGS) --enable_react_clip" ;;
    R9_minus_oplora) FLAGS="$(R9_FLAGS without --enable_oplora)" ;;
    ...
esac
```

Dispatch via SLURM array: `sbatch --array=0-16 ablation_r_array.sh`.
Each flag is independent, so the knock-out ablations trivially fall out
of the grid.

---

## Relationship to Stack G / G+ / G++

- **Stack G (original)** assumed full-FT → obsolete under new scope.
- **Stack G+** added 2026 SOTA but still mixed full-FT and LoRA → partially
  superseded.
- **Stack G++** restricted to head/LoRA but still included χViT-specific
  techniques (#30 ChE-LoRA, #22 ChEmbed Diffusion, #11 Hopfield with
  model-specific strength).
- **Reliable-Core (this file)** = Stack G++ further filtered to be
  *uniformly reliable across all five models*. Smaller, stricter, easier to
  ship.

For χViT specifically, Reliable-Core can still be *augmented* with χViT-only
techniques (#30 ChE-LoRA, #22 ChEmbed Diffusion, #11 richer Hopfield) as a
"χViT+" configuration in the paper table — but those are out of the portable
baseline.

## References

All approach details, code sketches, and arxiv refs live in
`cross-band-finetune-catalog.md`. This file exists to be the single source of
truth for *what to actually run* on the benchmark.

## Implementation progress

- **Phase 1 (LoRA foundation) — COMPLETE (2026-04-25).**
  Shipped: `reliable/lora_layer.py`, `reliable/last_n_placement.py`,
  `reliable/oplora.py`, `reliable/lora_null_init.py`,
  `reliable/channel_mask.py`. 22 Phase-1 tests green (70 total in the
  reliable suite). Plan:
  `.cursor/plans/2026-04-25-reliable-lora-foundation-plan.md`.
