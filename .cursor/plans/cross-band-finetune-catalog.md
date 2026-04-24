# Cross-Band Fine-Tuning Techniques — Catalog

**Status:** Brainstorming catalog (not an approved design). Companion to
`powerful_chivit_fine-tuning_07a25de9.plan.md` (existing E0–E18 ablation grid)
and `TRAINING_PLAN.md` (active training recipe).

**Problem statement.**
χViT (ChannelViT + iBOT, pretrained on 12 Sentinel-2+Sentinel-1 bands) is
fine-tuned on a **strict subset** of input channels (e.g. RGB only).
At evaluation the model must handle:

- **Superset** — RGB → RGBN, S2 → S2+S1 (strictly more bands than training)
- **No-overlap** — RGB → S1, S2 → S1, RGB → N'S1S2 (eval bands disjoint or
  almost disjoint from training bands)

**Priority.** B (monotonic superset guarantee) first, A (no-overlap) second.

**Design constraints (final, as of 2026-04-23).**

1. **Fine-tune only on the training subset.** No peeking at N/S1 channels during
   the fine-tune training loop. Running a teacher live on 12-band data during
   fine-tuning violates this (kills #8 PC-MS).
2. **Architectural monotonicity preferred over calibrated gates.** Calibrating
   on a held-out eval-distribution split is "self-deception" — it's just
   ensembling with a learned weight.
3. **No re-pretraining in this project.** Techniques that need iBOT-scale
   pretraining from scratch are out of scope (kills #6 DOFA full variant).
4. **Offline artifacts from the pretraining corpus *are* allowed.** Anything
   precomputed once from the full-band pretraining data (statistics,
   prototypes, covariances, projection bases, per-layer activation subspaces)
   and used as a frozen artifact during fine-tuning is fair game — that's
   caching existing knowledge, not doing pretraining.
5. **External pretrained foundation models are allowed** (e.g. DOFA,
   DiffusionSat). They were pretrained outside our pipeline; using them
   at eval or as frozen distillation targets is fine.
6. **Portability to other models** (iBOT, DINOv2/3, ViT-B, DOFA, TerraFM, …)
   is desirable but not required — χViT-first designs are acceptable.

**Rules of thumb derived from the above.**

- Reject any technique that runs a fresh 12-band forward during the fine-tune
  loop. #8 PC-MS is the canonical example and is **out**.
- A DOFA-style wavelength hypernetwork is **dead for this benchmark**:
  GeoCrossBench eval bands are always a subset of the 12 bands χViT was
  pretrained on, so `--freeze_unused_channel_embeds` already yields the optimal
  per-band embedding. A hypernet would only matter if the benchmark exposed
  wavelengths outside pretraining — it doesn't. **#6 is out.**
- The iBOT-pretrained χViT checkpoint is the fixed starting point of the
  problem. Its 12 learned `channel_embed[c]` vectors may be used as frozen
  prototypes (no new data needed).

---

## Glossary of non-obvious terms

- **Lipschitz-safe / Lipschitz bound.** `f` is Lipschitz with constant `L` if
  `‖f(x) − f(x')‖ ≤ L · ‖x − x'‖` for all inputs. "Lipschitz-safe" means we've
  proved the fine-tuned model can't move its output further than `L ·
  (change in input)` — a "no explosions" guarantee, not monotonicity.

- **PAC (Probably Approximately Correct).** Learning-theory framework. A PAC
  guarantee reads *"with probability ≥ 1 − δ over the training sample, the
  test error is ≤ ε"*. Here it means: *"with prob ≥ 95 %, deploying the fused
  predictor on superset gives accuracy ≥ subset accuracy − 1 %"*.

- **Null-space.** The null-space of matrix `M` is `{v : Mv = 0}`. If `U` spans
  the directions the subset forward uses, the null-space of `U` is everything
  orthogonal — updates in that subspace don't affect subset outputs.

- **Hypernetwork.** A network whose *output is the weights of another network*.
  A small MLP `H(c)` produces `W` as a function of some context `c` (e.g.
  channel wavelength).

- **LoRA / adapter.** PEFT: replace a `d × d` update with `W + BA` where
  `B ∈ ℝ^{d×r}`, `A ∈ ℝ^{r×d}`, `r ≪ d`. Zero-init (`B = 0`) → identity at
  start.

- **Conformal prediction.** Distribution-free prediction-set framework.
  Training-conditional conformal (Bian & Barber 2022) uses only the training
  set — no held-out split.

- **Soft-MoE / Mixture-of-Experts.** Several smaller MLPs ("experts"); each
  token routed to a weighted combination. Hard routing = deterministic rule
  from token identity. We'd use channel identity as the routing key.

- **Dempster-Shafer / subjective logic / evidential DL.** Machinery for
  combining probabilistic predictions *with uncertainty*. Predictions carry
  evidence mass; uncertain sources contribute less to the fusion.

- **Positive-congruent loss.** Loss term that specifically penalises
  regressions — cases where the new model is wrong on examples the old model
  got right. From Meta's model-deployment literature.

- **Modern Hopfield network / associative memory.** A frozen key-value bank
  queried by softmax-weighted retrieval. Mathematically equivalent to cross-
  attention.

- **Distillation.** Training a student to match a teacher's outputs (logits
  or features), beyond the label-only supervised loss.

- **Permutation-invariant / Set Transformer.** Architecture whose output is
  independent of input order. Naturally handles variable input size — matches
  "train on 3 channels, eval on 4".

---

## Tier 1 — True worst-case architectural monotonicity

*Provably `superset ≥ subset` by construction.*

### #1. Null-Space Projected Fine-Tuning (NSP-FT)

**Intuition.** Fine-tune the model only in directions that the subset forward
pass never uses. RGB output then cannot move (it only ever looks in directions
we left untouched). New bands at eval contribute exclusively in the *null-space*
of subset features — they can add signal, never disturb existing signal.

**Mechanism.**

1. Run **pretrained** χViT over the full training set, collect per-layer
   activations.
2. For each layer `ℓ`, compute `U_ℓ` = orthonormal basis of
   `span(activations_ℓ)`.
3. Projector `P_ℓ = I − U_ℓ U_ℓᵀ` = everything orthogonal to subset directions.
4. Every gradient update: `W_ℓ ← W_ℓ − η · P_ℓ · ∇L(W_ℓ)`.

Updates satisfy `ΔW_ℓ · v = 0` for every `v ∈ span(U_ℓ)` → subset activations
are preserved exactly: `W_ℓ · x = (W_pre + ΔW) · x = W_pre · x`.

**Plug-in.** Wrap optimizer in `classifier_utils.py::load_encoder`. Add
one-time calibration pass at start of `train_*.py`. CLI:
`--nullspace_projection --nullspace_rank 256`.

**Guarantee.** Strict worst-case in linear approximation. For a pure linear
layer it's exact. For full non-linear transformer: first-order exact; drift
is `O(‖ΔW‖² · ‖x‖²)`, small in practice.

**Cost.** One forward over training set; `O(d × rank)` per layer (~9 MB for
ViT-B, rank 256). ~5–10 % training overhead. Sweet spot:
`rank = 0.3–0.5 × d`.

**Refs.** Gradient Projection Memory (Saha et al., ICLR 2021,
[2103.09762](https://arxiv.org/abs/2103.09762)); OFT extension (Qiu et al.,
NeurIPS 2023, [2306.07280](https://arxiv.org/abs/2306.07280)).

---

### #2. Identity-Init Adapter + Hard Channel Mask

**Intuition.** Bolt a LoRA/AdaptFormer/DoRA adapter onto each block. The
adapter's contribution is multiplied by a *hard, non-learnable* indicator
`m_c ∈ {0,1}` that is 1 iff channel `c` was in the training set. At eval: NIR
tokens have `m=0` → adapter bypassed → NIR flows through the frozen pretrained
path, bit-identical to iBOT-pretraining output.

**Mechanism.**

```python
class ChannelMaskedAdapter(nn.Module):
    def __init__(self, dim, rank=8, training_channels=(0, 1, 2)):
        super().__init__()
        self.lora_A = nn.Linear(dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.lora_B.weight)          # identity at init
        mask = torch.zeros(12)
        mask[list(training_channels)] = 1.0
        self.register_buffer("channel_mask", mask)

    def forward(self, x, channel_ids):
        delta = self.lora_B(self.lora_A(x))
        m = self.channel_mask[channel_ids]          # (n_tokens,)
        return x + delta * m[None, :, None]
```

**Plug-in.** Wrap each block in `chi_vit.py`. χViT already tracks
`channel_idxs`; pass it through. CLI: `--adapter_type lora --adapter_rank 8
--hard_channel_mask`.

**Guarantee.** Strict worst-case on the N-path (untouched pretrained weights).
RGB path has no guarantee standalone — combine with #1.

**Cost.** `2 × rank × dim` per block — <0.5 % of model. Negligible training
overhead.

**Refs.** AdaptFormer (Chen et al., NeurIPS 2022,
[2205.13535](https://arxiv.org/abs/2205.13535)); DoRA (Liu et al., ICML 2024,
[2402.09353](https://arxiv.org/abs/2402.09353)); VeRA (Kopiczko et al., ICLR
2024, [2310.11454](https://arxiv.org/abs/2310.11454)).

---

### #3. Frozen-Expert MoE by Channel Identity

**Intuition.** Replace each MLP with 12 smaller MLPs — one per band. Hard-route
each channel's tokens to its dedicated expert. Fine-tune only the 3 RGB
experts; keep the 9 unused experts frozen at pretrained weights.

**Mechanism.**

```python
class ChannelExpertMLP(nn.Module):
    def __init__(self, dim, num_channels=12, mlp_ratio=4):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                          nn.Linear(hidden, dim))
            for _ in range(num_channels)
        ])

    def forward(self, x, channel_ids):
        out = torch.zeros_like(x)
        for c, expert in enumerate(self.experts):
            mask = (channel_ids == c)
            if mask.any():
                out[:, mask, :] = expert(x[:, mask, :])
        return out
```

Init all 12 experts by cloning the pretrained MLP (legitimate because iBOT
pretraining mixed channels — no per-channel specialisation yet). Freeze
gradients for untrained channels' experts.

**Plug-in.** Replace MLP blocks in `chi_vit.py`. CLI:
`--channel_experts --train_experts 0 1 2`.

**Guarantee.** Strict worst-case on MLP path for unseen channels. Attention
is still shared — combine with #1 for full coverage.

**Cost.** Params grow 12× (85M → ~1G for ViT-B). Mitigate with shared FFN +
per-channel LoRA middle ground.

**Refs.** Soft-MoE (Puigcerver et al., ICLR 2024,
[2308.00951](https://arxiv.org/abs/2308.00951)); Expert-Choice routing (Zhou
et al., NeurIPS 2022, [2202.09368](https://arxiv.org/abs/2202.09368)).

---

## Tier 2 — Bounded drift / Lipschitz-safe

*Worst-case bound, not zero.*

### #4. Orthogonal / Butterfly Fine-Tuning (OFT / BOFT)

**Intuition.** Constrain FT delta to be an *orthogonal rotation*:
`W_ft = R · W_pre`, `RᵀR = I`. Rotations preserve all lengths, all pairwise
angles, all distances between neurons. The Lipschitz constant of the change
equals the rotation angle — a tight, provable bound.

**Mechanism.** Parameterise `R = exp(Skew(θ))` where `Skew` produces a
skew-symmetric matrix from learnable parameters `θ`; `exp` is matrix
exponential. BOFT factorises `R` into block-sparse butterfly rotations for
parameter efficiency.

**Plug-in.** Replace AdamW + weight decay with OFT + angle regulariser.
Available as drop-in in `peft` library. Portable across all encoders in the
benchmark.

**Guarantee.** Bounded Lipschitz worst-case. Not monotonic, but provably
tight: in practice < 0.1 feature-norm change under typical fine-tuning
angles.

**Refs.** OFT (Qiu et al., NeurIPS 2023,
[2306.07280](https://arxiv.org/abs/2306.07280)); BOFT (Liu et al., ICLR 2024,
[2311.06243](https://arxiv.org/abs/2311.06243)).

---

### #5. ReAct-Style Token Activation Clipping

**Intuition.** Extreme activations from OOD channels (SAR, NIR) destabilise
downstream layers. Clip activations to ranges seen during training — like a
rev-limiter on an engine. Thresholds set from training set only, so eval-
distribution independent.

**Mechanism.**

1. Run fine-tuned model over training set; record feature-norm distribution
   per layer.
2. Set `τ_ℓ = Percentile_95(norms_ℓ)`.
3. At eval, apply `x = torch.clamp(x, max=τ_ℓ)` after each transformer block.

**Plug-in.** Encoder forward hook at each layer output. Active only at eval.
CLI: `--react_clip --react_percentile 95`.

**Guarantee.** Bounded worst-case drift: superset forward at layer ℓ differs
from subset forward by at most `τ_ℓ · n_new_tokens`. Propagates through
Lipschitz-bounded transformer to bounded output change.

**Cost.** Training-free. Eval overhead ~1 %.

**Refs.** ReAct (Sun et al., NeurIPS 2021,
[2111.12797](https://arxiv.org/abs/2111.12797)); ASH (Djurisic et al., ICLR
2023, [2209.09858](https://arxiv.org/abs/2209.09858)).

---

### #6. DOFA Wavelength Hypernetwork — ❌ DEAD FOR THIS BENCHMARK

> **Status: rejected.** Two reasons, either sufficient:
> 1. **Rule 3** — the full-power version needs re-pretraining χViT from
>    scratch with the hypernet, which is out of scope for this project.
> 2. **Benchmark redundancy** — GeoCrossBench's superset/no-overlap eval
>    bands are always a subset of the 12 bands χViT was pretrained on (10 S2
>    + VV + VH). For any eval band we care about, the pretrained χViT already
>    holds the exact optimal `channel_embed[c]`. The existing flag
>    `--freeze_unused_channel_embeds` gives us that directly. A hypernet
>    fitted to 12 wavelength points could only approximate those learned
>    embeddings — strictly worse than reading them off. It would only matter
>    if the benchmark exposed wavelengths outside pretraining (e.g. B01
>    coastal, B09 water-vapor, hyperspectral) and it doesn't.
>
> Kept here as reference in case a future benchmark extends the wavelength
> range, or re-pretraining becomes feasible.

**Intuition.** Replace χViT's learned per-channel embeddings (a lookup table)
with a *hypernetwork* `H(λ)` that takes a band's central wavelength in nm and
outputs an embedding. Unseen-band embedding is now a *physically grounded
interpolation* of wavelengths seen during pretraining.

**Mechanism.**

```python
class DOFAHyperNet(nn.Module):
    def __init__(self, embed_dim=768, hidden=256, num_freq=16):
        super().__init__()
        self.num_freq = num_freq
        self.mlp = nn.Sequential(
            nn.Linear(2 * num_freq, hidden), nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def wavelength_encoding(self, wavelengths_nm):
        freqs = torch.logspace(-2, 2, self.num_freq,
                               device=wavelengths_nm.device)
        angles = wavelengths_nm[:, None] * freqs[None, :]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    def forward(self, wavelengths_nm):
        return self.mlp(self.wavelength_encoding(wavelengths_nm))

channel_embeds = hypernet(
    torch.tensor([665., 560., 490., ...])  # (n_ch,) in nm
)
```

Initialise hypernet from pretrained DOFA checkpoint (DOFA trained on SAR, VIS,
SWIR — covers wavelengths we need). Freeze during fine-tuning.

**Plug-in.** Replace `self.channel_embed` parameter in `chi_vit.py` with
`self.channel_hypernet` + buffer of wavelengths. CLI:
`--dofa_hypernet --hypernet_weights /path/to/dofa.pth --freeze_hypernet`.

**Guarantee.** Worst-case on embedding layer: unseen-channel embedding is
exactly what pretraining produces. Expected elsewhere.

**Cost.** ~0.5 M params for hypernet. Vastly smaller than a 100-band lookup.

**Refs.** DOFA (Xiong et al., NeurIPS 2024,
[2403.15356](https://arxiv.org/abs/2403.15356)); SpectralGPT (Hong et al.,
TPAMI 2024, [2311.07113](https://arxiv.org/abs/2311.07113)).

---

## Tier 3 — Distribution-free PAC guarantee (no held-out data)

### #7. Training-Conditional Conformal Agreement Fusion (TC-CAF)

> **Status: acceptable under Rule 4, but requires the leak-free reformulation
> below.** The original formulation (calibrate nonconformity by running the
> teacher on the 12-band version of each training example) violates Rule 1 —
> it's a live 12-band teacher forward on fine-tune data. Fixable by moving
> calibration offline onto the pretraining corpus; guarantee is then coverage
> relative to the *pretraining* distribution, which is genuinely weaker but
> still PAC.

**Intuition.** Combine subset-trained student (RGB forward) with pretrained
12-band teacher (superset forward) via a *rule* — not a learned gate. The rule
uses a threshold derived entirely from the **pretraining corpus** (an offline
artifact, Rule 4), so no fine-tune-phase 12-band exposure is needed.

**Mechanism (leak-free).**

1. *Offline calibration on the pretraining corpus* (done once, before
   fine-tuning):
   ```python
   # Uses the pretraining unlabeled S2+S1 corpus; no labels needed if we
   # define nonconformity via agreement rather than correctness.
   for x_full in pretraining_loader:
       x_sub = restrict_to(x_full, SUBSET_BANDS)           # RGB slice
       p_s = pretrained_chivit(x_sub).softmax(-1)
       p_t = pretrained_chivit(x_full).softmax(-1)
       nonconformity.append((p_s - p_t).abs().sum(-1))
   tau = Quantile(nonconformity, q=1-alpha)                # frozen artifact
   ```
2. *At eval* (our fine-tuned student + same frozen pretrained teacher):
   ```python
   def fusion(x_sub, x_full, tau):
       p_s = finetuned_student(x_sub).softmax(-1)
       p_t = pretrained_teacher(x_full).softmax(-1)
       disagree = (p_s - p_t).abs().sum(-1)
       return torch.where(disagree < tau, p_t, p_s)
   ```

**Plug-in.** Offline calibration script `calibrate_conformal_tau.py`. Then all
three `eval_bands_*.py` gain `--conformal_fusion --teacher_ckpt ...
--tau_path tau.pt`. No changes to `train_*.py`.

**Guarantee.** PAC coverage ≥ 1 − α − O(1/√n) *relative to the pretraining
distribution*. If fine-tune distribution differs substantially from
pretraining, coverage may drift; document the assumption in the paper.

**Cost.** One offline pass over the pretraining corpus (cheap; embarrassingly
parallel). +1 forward per eval sample (teacher).

**Refs.** Conformal Prediction tutorial (Angelopoulos & Bates 2023,
[2107.07511](https://arxiv.org/abs/2107.07511)); Training-Conditional
Conformal (Bian & Barber 2022,
[2205.14301](https://arxiv.org/abs/2205.14301)); Conformal Risk Control
(Angelopoulos et al., ICLR 2024,
[2208.02814](https://arxiv.org/abs/2208.02814)).

---

## Tier 4 — Expected-value from training-time closure priors

### #8. Positive-Congruent Multispectral Teacher Loss (PC-MS) — ❌ REJECTED

> **Status: violates Rule 1.** Runs a live 12-band teacher forward on each
> fine-tune batch. Even though the *student* only sees RGB, the training loop
> as a whole touches N/S1 channels every step — that's precisely the peek the
> problem statement forbids. Kept here for reference; do **not** use in this
> project. The same teacher-knowledge effect can be achieved via offline
> artifacts from pretraining: see #11 (channel-prototype memory) and the
> leak-free reformulation of #7.

**Intuition.** Every training image has all 12 bands on disk — we just *choose*
to feed only RGB to the student. Use the pretrained 12-band teacher on the
*full* image to produce a soft label. The student, seeing only RGB, is
penalised *only* on examples where the teacher was right. This imports "what
would a 12-band oracle say" into RGB-only fine-tuning. Student never sees N/S1
during training; its decision boundary shifts toward the 12-band manifold.

**Mechanism.**

```python
x_rgb, x_full, y = batch

logits_student = student(x_rgb)

with torch.no_grad():
    logits_teacher = teacher(x_full)           # frozen pretrained 12-band

loss_ce = F.cross_entropy(logits_student, y)

teacher_pred = logits_teacher.argmax(-1)
teacher_correct = (teacher_pred == y).float()
loss_pc = (teacher_correct * F.kl_div(
    F.log_softmax(logits_student, -1),
    F.softmax(logits_teacher / T, -1),
    reduction='none',
).sum(-1)).mean()

loss = loss_ce + lambda_pc * loss_pc
```

**Plug-in.** Load teacher once in `train_*.py`. Modify dataset classes in
`change_detection_pytorch/datasets/` to return both RGB slice and full-12-band
tensor. CLI: `--pc_teacher_path ... --pc_lambda 0.3 --pc_temperature 2.0`.

**Guarantee.** Expected-value only. Empirically: 30–60 % reduction in negative
flips (PC literature).

**Cost.** +1 forward per batch through frozen teacher. ~2× training time.
~2 GB GPU for ViT-B teacher.

**Refs.** PC-Training (Yan et al., CVPR 2021,
[2011.09161](https://arxiv.org/abs/2011.09161)); LGM (Zhao et al., CVPR 2023,
[2305.11097](https://arxiv.org/abs/2305.11097)); Backward-Compatible Training
(Shen et al., CVPR 2020,
[2003.11942](https://arxiv.org/abs/2003.11942)).

---

### #9. Channel-Dropout Self-Distillation (CDSD)

**Intuition.** During RGB fine-tuning, randomly drop one of {R, G, B} and
force the dropped-student to match the full-RGB teacher's tokens. Trains an
internal *channel closure prior*: "given some channels, fill in the others".
At eval, extra NIR is seen as bonus evidence, not distribution shift.

**Mechanism.**

```python
x_rgb = batch.rgb
dropped = random.randint(0, 2)
keep_mask = torch.ones(3); keep_mask[dropped] = 0
x_dropped = x_rgb * keep_mask[None, :, None, None]

tokens_dropped = student.encoder(x_dropped)

with torch.no_grad():
    tokens_full = teacher_ema.encoder(x_rgb)

loss_distill = (1 - F.cosine_similarity(
    tokens_dropped[:, 1:], tokens_full[:, 1:], dim=-1,
)).mean()

loss_ce = F.cross_entropy(student.head(tokens_dropped[:, 0]), y)
loss = loss_ce + lambda_distill * loss_distill
```

Teacher = EMA of student (iBOT/DINO style), no external model.

**Plug-in.** `train_classifier.py` + equivalents. Add EMA teacher. CLI:
`--channel_dropout_distill --distill_lambda 0.5 --ema_momentum 0.996`.

**Guarantee.** Expected-value. The training signal directly matches the eval
shift — a model trained to fill in missing channels handles extra channels
robustly.

**Cost.** ~1.5× training (EMA forward). No extra eval cost.

**Refs.** iBOT (Zhou et al., ICLR 2022,
[2111.07832](https://arxiv.org/abs/2111.07832)); MultiMAE (Bachmann et al.,
ECCV 2022, [2204.01678](https://arxiv.org/abs/2204.01678)); I-JEPA (Assran
et al., CVPR 2024, [2301.08243](https://arxiv.org/abs/2301.08243)).

---

### #10. Evidential Per-Channel Aggregation (EW-PCA)

**Intuition.** Each channel produces a per-channel classification with
*evidence* (confidence mass). Fuse with Dempster-Shafer: high-evidence
channels dominate; uncertain channels contribute negligibly. Unseen bands at
eval default to high uncertainty → contribute near-nothing → no regression.

**Mechanism.** Dirichlet-concentration output + OOD-driven evidence shrinkage:

```python
class EvidentialPerChannelHead(nn.Module):
    def __init__(self, dim, K, num_channels=12):
        super().__init__()
        self.channel_pool = nn.Linear(dim, dim)
        self.evidence_head = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, K),
        )
        self.ood_detector = nn.Linear(dim, 1)

    def forward(self, channel_tokens):
        per_ch = self.channel_pool(channel_tokens)
        e = F.softplus(self.evidence_head(per_ch))          # ≥ 0
        ood = self.ood_detector(per_ch).sigmoid()
        e = e * (1 - ood) + ood                              # OOD → uniform
        total_alpha = e.sum(dim=1) + 1                       # (B, K)
        return total_alpha / total_alpha.sum(dim=-1, keepdim=True)
```

Evidential CE loss (Sensoy et al. 2018) + uniform-prior KL regulariser.

**Plug-in.** Replace classifier head in `train_classifier.py`. CLI:
`--evidential_head --evidential_reg 0.1`.

**Guarantee.** Soft worst-case via evidence bounds. Unseen-band contribution
is bounded below by `1/K` (uniform prior).

**Cost.** Moderate head size. Tuning is non-trivial — evidential collapse
(all α → 1) is a documented pitfall.

**Refs.** Evidential DL (Sensoy et al., NIPS 2018,
[1806.01768](https://arxiv.org/abs/1806.01768)); subjective-logic survey
[2110.13530](https://arxiv.org/abs/2110.13530); evidential-DL critique
[2402.10980](https://arxiv.org/abs/2402.10980).

---

## Tier 5 — Targeted at no-overlap (SAR / N'S1S2)

### #11. Hopfield / Retrieval-Augmented Channel Prototypes

> **Status: pure under Rule 4.** Prototypes are an offline statistic over the
> pretraining corpus — computed once from the *existing* pretrained χViT,
> frozen, used as a read-only memory bank during fine-tune and eval. No
> 12-band data touches the fine-tune loop. Even simpler variant: use the
> pretrained `channel_embed[c]` table directly as the memory (no fresh pass
> over pretraining data needed at all).

**Intuition.** Build a frozen memory of per-channel prototype vectors (averaged
over unlabeled S2+S1 archives using the pretrained χViT). Attach a zero-init
cross-attention head that retrieves from this memory. At eval on SAR-only,
SAR tokens retrieve their pretrained prototypes — injecting "memory of
pretraining" at inference.

**Mechanism.**

*Offline prototype construction:*

```python
with torch.no_grad():
    for c in range(12):
        acc = []
        for batch in unlabeled_loader:
            x = batch[:, c:c+1, :, :]
            tokens = pretrained_chivit.patch_embed(x)
            acc.append(tokens.mean(dim=[0, 1]))
        prototype[c] = torch.stack(acc).mean(0)
# shape (12, D), frozen, saved once
```

*Retrieval head:*

```python
class ChannelMemoryHead(nn.Module):
    def __init__(self, D, prototypes_12band):
        super().__init__()
        self.register_buffer('memory', prototypes_12band)
        self.cross_attn = nn.MultiheadAttention(D, 8, batch_first=True)
        nn.init.zeros_(self.cross_attn.out_proj.weight)
        nn.init.zeros_(self.cross_attn.out_proj.bias)

    def forward(self, channel_tokens):
        B = channel_tokens.shape[0]
        mem = self.memory.unsqueeze(0).expand(B, -1, -1)
        retrieved, _ = self.cross_attn(channel_tokens, mem, mem)
        return channel_tokens + retrieved
```

**Plug-in.** `chi_vit.py`. Offline script `build_channel_prototypes.py`.
CLI: `--channel_memory --prototype_path /path/to/prototypes.pt`.

**Guarantee.** Worst-case ≥ subset at init (zero output projection → no-op).
Expected gains on SAR eval via retrieval of pretrained SAR prototypes.

**Cost.** ~0.5 M params cross-attn + 37 KB frozen memory buffer. Offline
prototype build: minutes.

**Refs.** Modern Hopfield (Ramsauer et al., ICLR 2021,
[2008.02217](https://arxiv.org/abs/2008.02217)); RA-ViT (Iscen et al., CVPR
2023, [2304.01195](https://arxiv.org/abs/2304.01195)).

---

### #12. DiffusionSat RGB-Imputation at Eval

**Intuition.** For true no-overlap (train RGB → eval SAR-only), the model has
nothing familiar to work with. A pretrained multispectral↔SAR diffusion model
(DiffusionSat) can sample a plausible RGB conditioned on SAR; feed imputed
RGB into the RGB-fine-tuned χViT. Bridges the gap with a frozen pretrained
prior — no label leak since the imputer was trained on unlabeled data.

**Mechanism.**

```python
x_sar = batch.sar

diffusion = load_pretrained_diffsat().eval()

with torch.no_grad():
    x_rgb_hat = diffusion.sample(
        condition=x_sar,
        target_bands=["B04", "B03", "B02"],
        num_inference_steps=20,
    )

logits = rgb_finetuned_model(x_rgb_hat)
```

**Plug-in.** `eval_bands_*.py` preprocessing. Only active when
`eval_bands ∩ training_bands = ∅`. CLI:
`--diffusion_imputer /path/to/diffsat.ckpt --imputer_steps 20
--imputer_target_bands B04 B03 B02`.

**Guarantee.** Expected-value, dependent on imputer fidelity. Published
DiffusionSat benchmarks: SSIM > 0.85 on SAR→RGB — good enough for downstream
classification.

**Cost.** +0.5–5 s per sample; +4 GB GPU for DiffusionSat (~1 B params).

**Refs.** DiffusionSat (Khanna et al., ICLR 2024,
[2312.03606](https://arxiv.org/abs/2312.03606)); Sen12MS-CR-TS
[2305.15235](https://arxiv.org/abs/2305.15235); CMID
[2306.02744](https://arxiv.org/abs/2306.02744).

---

## Tier 6 — Input-arity-invariant architectures

### #13. Perceiver-IO Latent Read-in

**Intuition.** Current χViT emits `n_ch · HW` tokens — total token count
changes with channel count, which shifts attention distributions. Perceiver-IO
puts `K` fixed learnable latent queries between channel tokens and
transformer blocks. Latents cross-attend to channel tokens, produce exactly
`K` outputs. Transformer sees constant-size input regardless of channel count.

**Mechanism.**

```python
class ChannelReadIn(nn.Module):
    def __init__(self, dim, num_latents=256, num_heads=8):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads,
                                                batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

    def forward(self, channel_tokens):
        B = channel_tokens.shape[0]
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)
        out, _ = self.cross_attn(
            self.norm_q(latents),
            self.norm_kv(channel_tokens),
            self.norm_kv(channel_tokens),
        )
        return latents + out
```

**Plug-in.** Insert between patch-embed and transformer blocks in `chi_vit.py`.
For seg/CD: matching `WriteOut` cross-attention decodes back to spatial. CLI:
`--perceiver_readin --num_latents 256`.

**Guarantee.** Bounded expected drift: `O(1/n_ch)` per latent per added band,
via softmax denominator.

**Cost.** `K × D` latents (~200 K) + one cross-attention (~4 M). `K = 256`
for classification, 1024 for dense prediction.

**Refs.** Perceiver IO (Jaegle et al., ICLR 2022,
[2107.14795](https://arxiv.org/abs/2107.14795)); Meta-Transformer (Zhang
et al., 2023, [2307.10802](https://arxiv.org/abs/2307.10802)); AnySat
(Astruc et al., ECCV 2024,
[2405.15512](https://arxiv.org/abs/2405.15512)).

---

### #14. Null-Channel Invariance Head (NCI-PIH)

**Intuition.** Replace the linear classifier with a Set Transformer over
per-channel pooled features. Train with synthetic "null channel" tokens and a
loss that enforces invariance to their addition. At eval, uninformative new
channels look null-like → head ignores them → no regression. Informative ones
contribute.

**Mechanism.**

```python
patch = encoder_output[:, 1:]                     # (B, n_ch·HW, D)
per_channel = patch.reshape(B, n_ch, HW, D).mean(dim=2)   # (B, n_ch, D)

# Null augmentation during training
n_null = random.randint(1, 3)
null_vec = torch.zeros(B, n_null, D, device=per_channel.device)
augmented = torch.cat(
    [per_channel, null_vec + self.null_embed.expand(B, n_null, D)], dim=1,
)

class SetTransformerHead(nn.Module):
    def __init__(self, D, K, num_heads=8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(D, num_heads, batch_first=True)
        self.pool_query = nn.Parameter(torch.randn(1, D) * 0.02)
        self.cross_attn = nn.MultiheadAttention(D, num_heads, batch_first=True)
        self.cls_head = nn.Linear(D, K)

    def forward(self, channel_feats):
        x, _ = self.self_attn(channel_feats, channel_feats, channel_feats)
        B = x.shape[0]
        q = self.pool_query.unsqueeze(0).expand(B, -1, -1)
        pooled, _ = self.cross_attn(q, x, x)
        return self.cls_head(pooled.squeeze(1))

logits_orig = head(per_channel)
logits_aug = head(augmented)
loss_invariance = (logits_orig - logits_aug).pow(2).mean()
loss = loss_ce + lambda_null * loss_invariance
```

**Plug-in.** Classifier head in `train_classifier.py`. Seg/CD: apply to global
feature path only. CLI: `--nci_head --null_invariance_lambda 0.5`.

**Guarantee.** Distributional expected-value. Null-invariance loss literally
trains invariance under null-like additions → strong inductive bias toward
monotonicity.

**Cost.** ~1 M params. Minor.

**Refs.** Set Transformer (Lee et al., ICML 2019,
[1810.00825](https://arxiv.org/abs/1810.00825)); DeepSets (Zaheer et al.,
NeurIPS 2017, [1703.06114](https://arxiv.org/abs/1703.06114)).

---

## Summary matrix

Purity column: ✓ = pure under the final constraints; ◐ = pure only with
reformulation (noted in the approach entry); ✗ = rejected.

| # | Name | Guarantee | Purity | χViT | Other models | Train cost | Eval cost | Addresses |
|---|------|-----------|--------|------|--------------|------------|-----------|-----------|
| 1 | NSP-FT | Worst-case (linear) | ✓ | ★★★ | ★ | +5–10 % | 0 | B |
| 2 | Identity-Init + Hard Mask | Worst-case (N-path) | ✓ | ★★★ | ★★ | ~0 | 0 | B |
| 3 | Frozen-Expert MoE | Worst-case (MLP only) | ✓ | ★★★ | ★ | +params 12× | 0 | B |
| 4 | OFT / BOFT | Lipschitz bound | ✓ | ★★ | ★★★ | similar | 0 | B |
| 5 | ReAct Clipping | Bounded drift | ✓ | ★★★ | ★★ | 0 | +1 % | B |
| **6** | **DOFA Hypernet** | — | **✗** | — | — | — | — | **rejected** |
| 7 | TC-CAF | PAC coverage | ◐ (offline calib) | ★★★ | ★★ | 0 offline | +1 fwd | B |
| **8** | **PC-MS Teacher Loss** | — | **✗** | — | — | — | — | **rejected** |
| 9 | CDSD | Expected | ✓ | ★★★ | ★★ | ~1.5× | 0 | B |
| 10 | Evidential EW-PCA | Soft worst-case | ✓ | ★★★ | ★★ | moderate | 0 | B, A |
| 11 | Hopfield Memory | Worst-case at init | ✓ | ★★★ | ★ | ~0 | +0.5 M | A |
| 12 | DiffusionSat Impute | Expected | ✓ (external pretrained) | ★★ | ★★★ | 0 | +4 GB, +5s | A |
| 13 | Perceiver-IO | Bounded O(1/n_ch) | ✓ | ★★★ | ★★ | +arch | 0 | B |
| 14 | NCI-PIH | Distributional | ✓ | ★★ | ★★★ | +head | 0 | B |

**Guarantee tiers (active approaches only):**
- **Worst-case architectural** (#1, #2, #3, #11): strict non-regression.
- **Bounded drift / Lipschitz** (#4, #5, #13): tight output-change bound.
- **PAC / distribution-free** (#7 offline): probabilistic coverage guarantee.
- **Expected-value / soft** (#9, #10, #12, #14): training- or test-time signals
  that raise the expected value; no theorem.

**Rejected:** #6 DOFA (redundant with `--freeze_unused_channel_embeds` for this
benchmark; full variant requires re-pretraining → Rule 3). #8 PC-MS (runs
live 12-band teacher in fine-tune loop → Rule 1).

---

## Recommended stacks

### Current primary recommendation — Stack G (2026-04-23)

**#1 NSP-FT + #2 Identity-Init + Hard Mask + #9 CDSD + #11 Hopfield Memory.**
All four are pure under the final constraints. One provable layer + three
expected-value layers, composed orthogonally:

- **#1 NSP-FT** — gradient updates projected into the null-space of subset
  activations; subset forward preserved exactly. The headline theorem.
- **#2 Identity-Init + Hard Channel Mask** — LoRA adapter gated by a
  non-learnable channel-ID indicator; unseen bands routed through the frozen
  pretrained path, byte-identical to iBOT-pretraining output. Strict
  worst-case on the N-path.
- **#9 CDSD** — within-subset self-distillation (EMA teacher) trains a
  channel-closure prior without touching superset bands.
- **#11 Hopfield Memory** — offline prototypes from the pretrained `channel_
  embed` table, retrieved at eval via a zero-init cross-attention head.
  Primary tool for the no-overlap case.

Optional add-ons (orthogonal to the primary four):

- **#5 ReAct Clipping** — training-free eval-time drift bound. Costs ~1 %
  eval time; useful as a last-mile safety net.
- **#7-offline TC-CAF** — conformal threshold calibrated offline on the
  pretraining corpus, deployed at eval for a PAC coverage statement.
  Publishable angle.
- **#4 OFT / BOFT** — drop-in replacement for AdamW + weight decay that
  adds a Lipschitz bound, portable to iBOT / DINOv2/3 / ViT-B / DOFA /
  TerraFM in the comparison table.

### Retired / rejected stacks

Earlier drafts referenced #6 DOFA and #8 PC-MS (stacks E and F in previous
revisions). Both are **out** under the final constraints — see the rejection
notes on the individual approach entries. Do not resurrect these stacks
without revisiting the constraint rules first.

### Goal-oriented alternatives

| Goal | Stack | Why |
|------|-------|-----|
| **Strongest guarantee** | #1 + #2 + #5 + #7-offline | Null-space preserves subset; identity-init keeps unseen channels pretrained; ReAct bounds drift; offline conformal gives PAC eval guarantee. |
| **Best expected gains** | #1 + #9 + #11 + #13 | Null-space safety; closure-prior distillation; pretrained-prototype memory; Perceiver-IO latent read-in for arity invariance. |
| **No-overlap champion** | #11 + #2 + #12 | Pretrained channel prototypes retrieved at eval; identity-init routes unseen channels through pretrained path; DiffusionSat imputation as fallback for pure-SAR eval. |
| **Minimal-risk add-on to E9** | #5 + #7-offline | Pure eval-time. No retraining. Good baseline row for the paper table. |
| **Most portable across models** | #4 + #5 + #7-offline | OFT + ReAct + offline conformal fusion all work on iBOT / DINOv2/3 / ViT-B / DOFA / TerraFM with no χViT-specific machinery. |

---

## Open questions for design phase

Resolved so far:
- ~~#6 DOFA replace or complement `freeze_unused_channel_embeds`?~~ Rejected.
- ~~#8 PC-MS teacher choice?~~ Rejected.

Still open for the Stack G design spec:

1. **#1 NSP-FT ↔ HCS interaction.** HCS varies channel count during fine-tune,
   so the "subset-forward activation subspace" is itself variable. Options:
   (a) compute `U_ℓ` as the union of subspaces over all HCS-sampled subsets
   seen during the calibration pass; (b) disable HCS for the calibration
   pass, use fixed training-band set for `U_ℓ`; (c) maintain a family of
   `U_ℓ^{(S)}` projectors, one per subset size `S`, and apply the projector
   matching the current batch's subset.

2. **#11 prototype source.** Three options:
   (a) **Simplest:** use pretrained χViT `channel_embed[c]` directly (one
   vector per channel, already in the checkpoint); no fresh data pass needed.
   (b) **Richer:** compute per-channel mean of layer-0 post-patch-embed
   tokens on the iBOT pretraining corpus.
   (c) **Richest:** per-layer prototypes — one memory bank per transformer
   block, retrieved at each layer. Most expressive but largest engineering
   surface.

3. **Seg/CD portability.** The Stack G techniques need adaptation for dense
   tasks: #11's retrieval head must emit patch-aligned tokens (not global);
   #9's EMA teacher must handle the UPerNet decoder; #2's hard channel mask
   works unchanged on χViT's per-channel token path. Scope whether to target
   classification first (single task, faster iteration) or all three tasks
   in parallel.

4. **Eval protocol for monotonicity claim.** To claim "superset ≥ subset" in
   the paper we need a matched-pair test: same checkpoint, same sample, eval
   once with subset bands and once with superset bands, compare metric.
   Need to add a per-sample subset-bands eval mode to `eval_bands_*.py`.

5. **Compute budget.** Stack G's training overhead breakdown: NSP-FT ~5–10 %,
   CDSD ~1.5×, #2 and #11 negligible. Net: ~1.6× baseline training time.
   Decide whether to run Stack G on the full E-grid datasets (eurosat, ben,
   brick, so2sat + Harvey + OSCD) or start on eurosat only like E0–E18 did.

---

# 2026 SOTA additions (approaches #15–#27)

Surfaced by a 4-thread literature sweep (variable-modality ViTs, DRO/IRM/Bayesian,
feature-space + TTA + contrastive, spectral-physics priors). All of the following
pass the purity audit (subset-only training, no re-pretraining, offline pretraining-
corpus artifacts allowed, external pretrained FMs allowed at eval). See
`stack-g-design.md` for how they fold into Stack G+.

---

## Tier 1 additions — worst-case architectural guarantees

### #15. Channel-Hierarchical Randomized Smoothing Fine-Tuning (CH-RS-FT)

> **Status: pure.** Certified **l₀ radius over channel tokens** — the only technique
> in the 2024–2026 literature giving a formal probabilistic certificate that
> "adding up to `k` channel tokens does not change the prediction." Directly
> delivers the monotonicity claim Priority B wants.

**Intuition.** Randomized smoothing, originally for pixel-level l₂ robustness,
transplanted to the *channel-token* axis. During fine-tuning, ablate random
subsets of channel tokens (Gaussian noise in the token-embedding space). At eval,
Monte-Carlo sample many ablations and take the majority-vote prediction. Classical
smoothing theorems then certify an l₀ ball: with probability ≥ 1−α, the prediction
is invariant to any modification of ≤ k channel tokens — and adding a new channel
is a specific form of "modification."

**Mechanism.**

```python
# Training: smooth over channel-token Gaussian noise
for batch in loader:
    mask = torch.bernoulli(torch.full((n_channels,), p_smooth))
    noise = torch.randn_like(channel_tokens) * sigma
    tokens_smoothed = channel_tokens + mask[None, :, None] * noise
    loss = F.cross_entropy(model(tokens_smoothed), y)

# Eval: Monte-Carlo smoothed classifier with certificate
def smooth_predict(tokens, n_mc=100, sigma=sigma):
    votes = torch.zeros(num_classes)
    for _ in range(n_mc):
        eps = torch.randn_like(tokens) * sigma
        pred = model(tokens + eps).argmax(-1)
        votes[pred] += 1
    top = votes.argmax()
    # Certified l0 radius k via Neyman-Pearson bound (Cohen '19 extended to l0)
    return top, certified_radius(votes)
```

**Guarantee.** Probabilistic worst-case: with probability ≥ 1 − α, prediction
invariant to ≤ `k` channel-token perturbations. `k` depends on smoothing noise
σ and the majority-vote margin.

**Plug-in.** Training loop modification (`train_*.py`), eval-time MC wrapper in
`eval_bands_*.py`. No new architecture.

**Refs.** Hierarchical Randomized Smoothing (Schuchardt et al.,
[2310.16221](https://arxiv.org/abs/2310.16221)); AdaptDel, NeurIPS 2025
([2511.09316](https://arxiv.org/pdf/2511.09316)).

---

### #16. OPLoRA — Orthogonal Projection LoRA (concrete NSP-FT realization)

> **Status: pure.** Published concrete implementation of the #1 NSP-FT idea with
> closed-form projectors. AAAI 2026.

**Intuition.** Take the SVD of a pretrained weight `W_pre = U Σ Vᵀ`. Let `U_k`,
`V_k` be the top-`k` singular vectors. Compute left and right projectors
`P_L = I − U_k U_kᵀ`, `P_R = I − V_k V_kᵀ`. Every LoRA update is double-sided-
projected: `ΔW_projected = P_L · (AB) · P_R`. Result: the top-`k` singular
triples of `W_pre` are **exactly preserved** through fine-tuning.

**Mechanism.**

```python
# Offline, one-time per weight matrix
U, S, Vh = torch.linalg.svd(W_pretrained, full_matrices=False)
U_k, V_k = U[:, :k], Vh[:k, :].T
P_L = torch.eye(W.shape[0]) - U_k @ U_k.T
P_R = torch.eye(W.shape[1]) - V_k @ V_k.T

# In training
def oplora_forward(x):
    delta = self.lora_A.weight @ self.lora_B.weight       # (d, d)
    delta_projected = self.P_L @ delta @ self.P_R         # orthogonal to top-k
    return x @ (W_pre + delta_projected)
```

**Plug-in.** `rs_finetune/oplora.py` — wraps nn.Linear layers. Drop into every
χViT attention/MLP block. CLI:
`--oplora --oplora_rank 8 --oplora_preserve_k 32`.

**Guarantee.** Strict worst-case preservation of top-`k` singular triples of
every frozen weight. Combined with #2 (hard channel mask): NIR tokens see `W_pre`
identically (masked), RGB tokens see `W_pre + projected_delta` which preserves
top-`k`.

**Cost.** SVD per weight offline (~seconds). Training overhead: two matrix
multiplies per layer (~2–3 %).

**Refs.** OPLoRA (Lin et al., AAAI 2026,
[2510.13003](https://arxiv.org/abs/2510.13003)); companion SC-LoRA
([2505.23724](https://arxiv.org/html/2505.23724)), SplitLoRA
([2505.22370](https://arxiv.org/html/2505.22370)).

---

### #17. HP-Freeze — Head-Pursuit Guided Attention-Head Freezing

> **Status: pure.** Offline probe on pretraining corpus → freeze specialized
> heads. Offline artifact under Rule 4.

**Intuition.** Recent interpretability work (Head Pursuit, NeurIPS 2025) shows
that ~1–5 % of attention heads in a multimodal transformer specialize in
specific semantic concepts. For χViT specifically: some heads handle "SAR-band
features," some "NIR integration," some "cross-channel multispectral fusion."
Offline, probe these heads on the pretraining corpus. Freeze the top-k
specialized heads during RGB fine-tuning. At eval with superset bands, those
frozen heads process NIR/SAR tokens with the exact pretrained behavior.

**Mechanism.**

```python
# Offline: rank heads by specialization score on the pretraining corpus
# using Head Pursuit (signal-processing probe) and/or Causal Head Gating
ranked_heads = head_pursuit_probe(
    pretrained_chivit, pretraining_corpus,
    concepts=["SAR_polarization", "NIR_reflectance", "multispectral_fusion"],
)
# Pick top-k (e.g., k=5% of total heads)
frozen_head_ids = ranked_heads[: int(0.05 * total_heads)]

# During fine-tuning: freeze those heads' parameters
for block_idx, block in enumerate(chivit.blocks):
    for head_idx in range(block.attn.num_heads):
        if (block_idx, head_idx) in frozen_head_ids:
            block.attn.freeze_head(head_idx)
```

**Plug-in.** Offline probe: new script `rs_finetune/head_pursuit_probe.py`.
Freezing: `rs_finetune/change_detection_pytorch/encoders/chi_vit.py` (head-level
`requires_grad` toggling).

**Guarantee.** Strict worst-case on frozen-head subspace — frozen heads = bit-
identical to pretrained heads.

**Cost.** Offline probe: one pass over pretraining sample + SVD. Training
overhead: zero. Small capacity cost (frozen heads don't adapt to downstream
task), offset by preserved multispectral capability.

**Refs.** Head Pursuit (Milan-Valverde et al., NeurIPS 2025,
[2510.21518](https://arxiv.org/abs/2510.21518)); Causal Head Gating (Ren et al.,
2025, [2505.13737](https://arxiv.org/abs/2505.13737)); Two-Phase Head-Specific
LoRA ([OpenReview HS-LoRA](https://openreview.net/forum?id=rssPS1bagp)).

---

## Tier 2 additions — bounded drift / Bayesian posterior

### #18. BPSG — Bayesian Posterior Safety Gate

> **Status: pure.** Bayesian LoRA posterior fit on subset data; safety gate uses
> only subset-side posterior, *not* eval distribution. Structurally different
> from the rejected option (c) "calibrate on held-out superset."

**Intuition.** Fit a Bayesian LoRA (Laplace-approximation SBA, Kronecker-factored
Bayesian-LoRA, or rank-variational BayesLoRA) on subset-only fine-tuning. At
eval, this gives a posterior distribution over predictions, not a single
point. Check: does the superset-forward prediction lie within the 95 % credible
interval of the subset-forward posterior predictive? If yes → accept the
superset refinement (adding bands shifted prediction within the believed
envelope). If no → the superset forward is an outlier relative to the subset
model's beliefs → fall back to subset prediction. **Near-monotonic by
construction**, with the gate derived from Bayesian evidence rather than
calibration data.

**Mechanism.**

```python
# Training: standard SBA Bayesian LoRA
bayesian_lora = StiefelBayesianLoRA(model, rank=8, prior=MatrixLangevin())
train_loop(bayesian_lora, data_rgb_only)

# Eval: posterior credible interval check
def bpsg_predict(x_rgb, x_full):
    samples_subset = []
    for _ in range(n_mc):
        w_sample = bayesian_lora.sample_posterior()
        samples_subset.append(model_with(w_sample)(x_rgb).softmax(-1))
    p_subset_mean = torch.stack(samples_subset).mean(0)
    p_subset_std = torch.stack(samples_subset).std(0)

    p_superset = model(x_full).softmax(-1)   # or mean over posterior

    within_ci = ((p_superset - p_subset_mean).abs() <= 2 * p_subset_std).all(-1)
    return torch.where(within_ci, p_superset, p_subset_mean)
```

**Plug-in.** Replace standard LoRA with a Bayesian variant (SBA / BayesLoRA /
Bayesian-LoRA). Eval-time posterior sampling in `eval_bands_*.py`.

**Guarantee.** Probabilistic monotonic: with high posterior probability, fused
output is a refinement of subset prediction within the posterior envelope.

**Cost.** Training: ~1.5× for MC-posterior fit. Eval: `n_mc × forward`.

**Refs.** SBA Stiefel Bayesian Adaptation (arxiv
[2602.17809](https://arxiv.org/abs/2602.17809), Feb 2026); Bayesian-LoRA
([2601.21003](https://arxiv.org/abs/2601.21003), Jan 2026); BayesLoRA
variational ([2506.22809](https://arxiv.org/abs/2506.22809), Jun 2025).

---

### #19. SI-LoRA — Stiefel Identity-Init LoRA

> **Status: pure.** Bayesian variant of #2 with calibrated uncertainty.

**Intuition.** Replace the plain LoRA in #2 with a *Stiefel-manifold-constrained*
LoRA whose factors are on the orthonormal manifold; a Matrix-Langevin prior
concentrated at zero enforces identity-init. Training converges to a small
orthonormal perturbation; posterior variance scales per-sample confidence. The
hard channel mask (still non-learnable) remains intact — SI-LoRA just replaces
the *what* of the adapter with a Bayesian version.

**Plug-in.** Extends #2's LoRA block. CLI: `--si_lora --stiefel_rank 8
--langevin_temp 0.01`.

**Guarantee.** Worst-case on N-path via mask + Bayesian-calibrated posterior on
RGB path.

**Refs.** SBA (arxiv [2602.17809](https://arxiv.org/abs/2602.17809), Feb 2026).

---

### #24. SRF-Biased Attention — physics-grounded attention prior

> **Status: pure.** Uses only public Sentinel-2/S1 sensor metadata (SRF), not
> any pretraining data.

**Intuition.** Each Sentinel band has a published Spectral Response Function
(SRF) — the sensor's sensitivity curve over wavelength. The overlap integral
between band `i`'s SRF and band `j`'s SRF is a principled *physical similarity
measure* between the two bands (unit: dimensionless, symmetric, positive-
definite in a physics sense). Form the 12×12 SRF-overlap matrix `S` offline
from published metadata. Add `β · log(S_ij)` as a *pre-softmax attention bias*
in χViT's channel-attention. Bands that are spectrally similar (e.g., B04 red
and B8 NIR both via visible light) attend more; bands dissimilar (B8 vs VV SAR)
attend less.

**Mechanism.**

```python
# Offline: build SRF-overlap matrix from Sentinel-2 L2A + Sentinel-1 metadata
# (ESA-published, public, not from data)
S = compute_srf_overlap_matrix(bands=["B02","B03","B04",..., "VV","VH"])
# S[i,j] in [0,1], symmetric

# In χViT channel-attention
def channel_attn(self, q, k, v, channel_ids):
    logits = (q @ k.transpose(-2, -1)) / sqrt(d_k)
    bias = self.srf_bias_scale * torch.log(self.S[channel_ids][:, channel_ids] + ε)
    logits = logits + bias
    return softmax(logits) @ v
```

**Plug-in.** `chi_vit.py` — one new buffer `self.S_overlap`, bias added to
channel-attention logits. CLI: `--srf_attn_bias --srf_bias_scale 1.0`.

**Guarantee.** Bounded drift (Lipschitz). Physics-grounded prior — unseen
bands (at eval) inherit physically-motivated attention weights.

**Cost.** Zero training cost. Trivial eval cost.

**Refs.** STARS (Honigsberg et al.,
[2411.05714](https://arxiv.org/abs/2411.05714)); KARMA SAM loss
([2512.12445](https://arxiv.org/abs/2512.12445), AAAI 2026 workshop).

---

## Tier 4 additions — expected-value via offline artifacts

### #20. LoRA-Null — null-space initialization

> **Status: pure.** Pure initialization trick; zero training overhead.

**Intuition.** Initialize LoRA's B matrix in the null-space of representative-
activation singular vectors (computed offline from the pretraining corpus).
Updates start in a subspace that doesn't interfere with pretrained knowledge.
During training, SGD may leave the null-space, but the first epochs — when the
adapter is most disruptive — are protected.

**Plug-in.** Weight init function in `rs_finetune/lora_null_init.py`. Call once
during model construction. CLI: `--lora_null_init --nullspace_rank 256`.

**Guarantee.** Expected-value at init → worst-case if combined with OPLoRA's
projection during training.

**Refs.** LoRA-Null (Arxiv [2503.02659](https://arxiv.org/abs/2503.02659),
Mar 2025).

---

### #21. MERA — Merge-Then-Realign

> **Status: pure.** Post-training weight interpolation; uses no data beyond
> subset.

**Intuition.** The fine-tuned χViT is great on RGB but has drifted on unseen-band
processing. The pretrained χViT is the opposite. Task-arithmetic merging lets
us blend: `W_merged = (1 − α) · W_pre + α · W_ft`. Low α recovers more
pretrained behavior (better on unseen bands), high α keeps more fine-tune
gains. After merging, a short subset-only **realign** step nudges the merged
model back onto the RGB task distribution.

**Mechanism.**

```python
# After standard fine-tune
W_ft = load("finetuned_checkpoint.pth")
W_pre = load("pretrained_chivit.pth")

# Pick α on a held-out RGB val split (still training-distribution, no superset peek)
for α in [0.3, 0.5, 0.7]:
    W_merged = {k: (1 - α) * W_pre[k] + α * W_ft[k] for k in W_pre}
    rgb_val_acc = eval_rgb(load_model(W_merged), rgb_val_split)
    ...

# Short realign
model = load_model(W_best_merged)
train_loop(model, rgb_train_data, steps=1000)
```

**Plug-in.** Post-training merge script + short realign loop.

**Guarantee.** Expected-value. Empirical: MERA paper shows large gains on
modality-incremental benchmarks.

**Refs.** MERA (Huang et al., Mar 2025,
[2503.07663](https://arxiv.org/abs/2503.07663)); Task Arithmetic (Ilharco
et al., 2023, [2311.03099](https://arxiv.org/abs/2311.03099)).

---

### #22. Channel-Embed Diffusion Prior

> **Status: pure.** Pretraining-corpus artifact (Rule 4).

**Intuition.** χViT's pretrained `channel_embed` table has 12 vectors, one per
band. Treat them as 12 samples from some underlying "channel-embedding
distribution" and fit a tiny diffusion model offline to that distribution (or,
more richly, fit the diffusion to per-token channel embeddings over the
pretraining corpus). During RGB fine-tuning, occasionally replace a channel's
embedding with a noised version drawn from the diffusion's early steps. The
fine-tuned model becomes invariant along the on-manifold diffusion tangent. At
eval, unseen-band embeddings lie inside the diffusion support → bounded
deviation.

**Plug-in.** Offline diffusion fit script. Fine-tune augmentation in `chi_vit.py`
(stochastic substitution of channel_embed at batch construction). CLI:
`--chembed_diffusion --diff_noise_steps 3`.

**Guarantee.** Bounded expected-value along the diffusion manifold. Upgrades
to worst-case if combined with #1 NSP-FT (projection eliminates out-of-manifold
updates).

**Refs.** N-JEPA ([2507.15216](https://arxiv.org/abs/2507.15216)); MADCL
([2509.20048](https://arxiv.org/abs/2509.20048)); Manifold Diffusion
([2510.02305](https://arxiv.org/abs/2510.02305)).

---

### #23. LSMM Auxiliary Head (KARMA-lite)

> **Status: pure.** Offline endmember extraction from pretraining corpus (Rule
> 4) + public Sentinel-2 SRF.

**Intuition.** Classical spectral unmixing decomposes a pixel's spectrum as a
linear combination of material endmembers (water, vegetation, urban soil, …).
Offline, run VCA or NMF on the pretraining corpus's 12-band spectra to extract
K endmember signatures `E ∈ ℝ^{12 × K}` (frozen). Add an auxiliary reconstruction
head during fine-tune that predicts per-patch abundances `α ∈ ℝ^K`. Loss
applied only on the *RGB subset*:
`L_recon = ‖x_RGB − SRF_RGB · E · α‖²`. The auxiliary head is discarded at
eval. The encoder now carries an implicit 12-band prior: it's learned to
produce plausible abundance vectors over a dictionary that includes SAR-
distinguishable materials, even though SAR never appeared in the fine-tune
loop.

**Plug-in.** Offline VCA/NMF script → frozen `E` buffer. New aux head in
`chi_vit.py`. CLI: `--lsmm_head --n_endmembers 16 --lsmm_lambda 0.3`.

**Guarantee.** Expected-value. Particularly promising for no-overlap SAR eval
because abundance vectors are a material-identity bottleneck.

**Refs.** KARMA (Dec 2025,
[2512.12445](https://arxiv.org/abs/2512.12445)); PISM
([2508.21618](https://arxiv.org/html/2508.21618v2)); RTM survey
([2507.09081](https://arxiv.org/html/2507.09081)).

---

### #25. DEO Dual-Teacher (RGB-Only Variant)

> **Status: pure in RGB-only variant.** NOTE: naive DEO involves running a
> 12-band teacher and would violate Rule 1. The RGB-only variant below replaces
> the 12-band teacher with an external optical VFM (DINOv3).

**Intuition.** Two frozen external teachers, both on RGB: (i) χViT itself
(EMA-frozen) as a self-teacher, (ii) DINOv3 as a strong general-purpose
optical visual foundation model. The student is the fine-tuning χViT. Loss:
standard CE + KL to both teachers on RGB outputs. The teacher ensemble
regularizes against drift from both χViT's pretrained cross-band structure
*and* from the general optical manifold.

**Plug-in.** Add second teacher loader + KL-loss term in `train_*.py`. CLI:
`--deo_dual_teacher --dinov3_ckpt ... --deo_ema_lambda 0.3 --deo_dinov3_lambda 0.3`.

**Guarantee.** Expected-value.

**Refs.** DEO (Feb 2026, [2602.19863](https://arxiv.org/abs/2602.19863)).
**Caveat:** paper's default uses a 12-band teacher. Use only the RGB-only
variant to respect Rule 1.

---

## Tier 5 additions — eval-time only

### #26. TerraMind TiM Synthesis (no-overlap alternative to #12)

> **Status: pure if used only at eval; TerraMind is an external pretrained FM
> (Rule 5).**

**Intuition.** TerraMind (IBM×ESA, ICCV 2025) is an any-to-any generative
multimodal EO foundation model. Its "Thinking-in-Modalities" (TiM) mode
generates auxiliary modality tokens at inference as intermediate reasoning.
For our no-overlap case, feed RGB to TerraMind → generate plausible N / S1
channels → concatenate into a 12-band tensor → feed to χViT (which was
pretrained on 12 bands so it can process the synthesized bands natively).
Alternative to DiffusionSat (#12); TerraMind is newer and more specialized for
EO-task-oriented generation.

**Plug-in.** Eval-time preprocessing in `eval_bands_*.py` when
`training_bands ∩ eval_bands = ∅`. CLI:
`--terramind_tim --terramind_ckpt ... --tim_steps 10`.

**Guarantee.** Expected-value, dependent on TerraMind fidelity.

**Cost.** Large: TerraMind is a full generative model (>1B params). Several
seconds per sample.

**Refs.** TerraMind (Jakubik et al., ICCV 2025,
[2504.11171](https://arxiv.org/abs/2504.11171)).

---

### #27. ADAPT — Backprop-Free TTA via Gaussian Alignment

> **Status: pure.** Pre-computed source-class Gaussians on pretraining corpus
> (Rule 4); closed-form eval-time alignment; no gradients; no eval labels.

**Intuition.** Classical TTA methods (TENT, CoTTA) fine-tune LayerNorm params
on eval data → philosophically blurry under our constraint. ADAPT is different:
offline, compute per-class Gaussian models of features on the *pretraining
corpus*. At eval, align each feature to the nearest/most-likely class-
conditional Gaussian via a *closed-form* transform — no gradient steps. Pure
deterministic projection; architecturally indistinguishable from "frozen
linear head."

**Plug-in.** Offline feature-Gaussian fit script. Eval-time transform in
`eval_bands_*.py`. CLI: `--adapt_align --source_gaussians_path ...`.

**Guarantee.** Bounded under Gaussian assumptions.

**Refs.** ADAPT (arxiv
[2508.15568](https://arxiv.org/abs/2508.15568), Aug 2025); Tilting the Latent
Distribution (arxiv [2602.02633](https://arxiv.org/abs/2602.02633), Feb 2026).

---

## Updated summary matrix — 2026 additions

| # | Name | Tier | Purity | Year | Addresses |
|---|---|---|---|---|---|
| 15 | CH-RS-FT | 1 (certified l₀) | ✓ | 2025 | **B (certified)** |
| 16 | OPLoRA | 1 (worst-case) | ✓ | 2026 | B |
| 17 | HP-Freeze | 1 (worst-case on heads) | ✓ | 2025 | B, A |
| 18 | BPSG | 2 (Bayesian CI) | ✓ | 2026 | B |
| 19 | SI-LoRA | 2 (bounded + Bayes) | ✓ | 2026 | B |
| 20 | LoRA-Null Init | 4 (expected, free) | ✓ | 2025 | B |
| 21 | MERA | 4 (expected, post-train) | ✓ | 2025 | B, A |
| 22 | ChEmbed Diffusion | 4 (expected + artifact) | ✓ | 2025 | B |
| 23 | LSMM Head | 4 (expected + physics) | ✓ | 2025-26 | B, A |
| 24 | SRF-Biased Attn | 2 (Lipschitz bound) | ✓ | 2024-25 | B |
| 25 | DEO Dual-Teacher (RGB-only) | 4 (expected) | ✓ (RGB-only variant) | 2026 | B |
| 26 | TerraMind TiM | 5 (eval-only) | ✓ (external FM) | 2025 | A |
| 27 | ADAPT TTA | 5 (eval-only, closed-form) | ✓ | 2025-26 | B |

**Cross-agent consensus picks:** #16 (OPLoRA) + #20 (LoRA-Null) appear in
multiple research threads as the canonical concrete NSP-FT realization. #17
(HP-Freeze) is the highest-leverage *cheap* addition. #18 (BPSG) is the cleanest
path to a safety gate that isn't "self-deception."

---

# Head-only / LoRA-only techniques (approaches #28–#32)

Added 2026-04-23 to reflect the narrowed training scope: fine-tuning is
restricted to **head modifications + LoRA adapters only**; no full-backbone
fine-tuning. See `stack-g-design.md` §13 for the scope-restricted Stack G++.

---

## Tier 4 additions (head- or LoRA-specific)

### #28. Multi-Head Channel-Subset Ensemble (MCSE)

> **Status: pure.** Head-only. Trains `K` linear heads on `K` channel subsets;
> backbone entirely frozen.

**Intuition.** Instead of a single linear classifier trained on the full subset
(RGB), train `K` classifier heads on `K` different channel subsets drawn from
the training bands — e.g., heads for `{R}`, `{G}`, `{B}`, `{R,G}`, `{R,B}`,
`{G,B}`, `{R,G,B}`. Each head sees a different amount of spectral information.
At eval, each head processes the bands it was trained on (fetched from the
eval-time superset/no-overlap input), and outputs are averaged with variance
as epistemic uncertainty. A new band at eval (NIR) either (a) is processed by
a special "all channels" head that averages over the extra band-tokens via
channel-mean pooling, or (b) is discarded if no head was trained with it.
Since all heads are linear over the same frozen backbone features, this is
~`K × n_classes × D` parameters total — trivially cheap.

**Mechanism.**

```python
# Subsets of training bands
subsets = [[0], [1], [2], [0,1], [0,2], [1,2], [0,1,2]]
heads = nn.ModuleList([nn.Linear(D, num_classes) for _ in subsets])

# Training
for x, y in loader:
    feats = backbone(x).detach()    # (B, D) frozen
    loss = 0.0
    for s, head in zip(subsets, heads):
        feats_s = pool(feats, channel_subset=s)
        loss = loss + F.cross_entropy(head(feats_s), y)
    loss.backward()

# Eval — ensemble over heads compatible with eval bands
def predict(x_eval, eval_bands):
    feats = backbone(x_eval)
    logits_mean, logits_var = [], []
    for s, head in zip(subsets, heads):
        if set(s).issubset(eval_bands):
            logits_mean.append(head(pool(feats, channel_subset=s)).softmax(-1))
    return torch.stack(logits_mean).mean(0), torch.stack(logits_mean).var(0)
```

**Plug-in.** `rs_finetune/train_classifier.py` + new module
`rs_finetune/mcse_head.py`. CLI: `--mcse_head --mcse_subsets power_set`.

**Guarantee.** Expected-value. Ensemble variance gives epistemic uncertainty
naturally.

**Cost.** Training: `K × ` head forward/backward per batch — negligible since
heads are linear and backbone is frozen. Parameter cost: `K ≤ 2^n_train_bands`,
typically < 10 heads.

**Refs.** Deep Ensembles (Lakshminarayanan et al., NeurIPS 2017,
[1612.01474](https://arxiv.org/abs/1612.01474)); BatchEnsemble (Wen et al.,
ICLR 2020, [2002.06715](https://arxiv.org/abs/2002.06715)); Sub-model
sampling for MC-Dropout ensembles (Havasi et al., ICLR 2021,
[2010.06610](https://arxiv.org/abs/2010.06610)).

---

### #29. LastN-LoRA — localized LoRA placement

> **Status: pure.** LoRA adapters restricted to the last `N` transformer
> blocks; early blocks stay fully pretrained.

**Intuition.** Early ViT blocks carry more "low-level" spectral / textural
knowledge from pretraining; late blocks specialize more for the downstream
task. Surgical fine-tuning literature (Lee '22) shows that adapting only *late*
layers is often optimal for in-distribution transfer and better for OOD
generalization. For our problem: the channel-aware spectral computations in
early blocks are exactly what we *don't* want to disturb (they were trained on
all 12 bands during iBOT), while late-block task-head adaptation is the part
that actually needs to learn the downstream labels. LastN-LoRA embodies this
principle as LoRA placement.

**Mechanism.**

```python
last_n = 4  # e.g. apply LoRA only to blocks [-4:]
for block_idx, block in enumerate(chivit.blocks):
    if block_idx >= len(chivit.blocks) - last_n:
        wrap_block_with_lora(block, rank=8)
    # else: block is fully frozen, no LoRA
```

**Plug-in.** `chi_vit.py` wrapper construction. CLI:
`--lora_last_n 4 --lora_rank 8`. Composes with all other LoRA techniques (#2,
#16, #19, #20, #30) — they are simply placed only on the last N blocks.

**Guarantee.** Strict worst-case on all blocks *before* block `(L − N)`: those
blocks are bit-identical to pretrained. Expected on the last-N blocks (depends
on what LoRA constraint is used there).

**Cost.** `N / L` of the LoRA param budget of full-depth LoRA. Zero overhead
on skipped blocks.

**Refs.** Surgical Fine-Tuning (Lee et al., ICLR 2023,
[2210.11466](https://arxiv.org/abs/2210.11466)); LoRA-in-Few-Layers analyses
(2024 follow-ups).

---

### #30. Channel-Embed LoRA (ChE-LoRA)

> **Status: pure.** χViT-specific; trains a low-rank delta on
> `channel_embed[c]` for training channels only.

**Intuition.** The *one part* of χViT most directly relevant to cross-band
behavior is the `channel_embed` lookup table (`n_channels × D`, frozen to 12
slots from iBOT pretraining). Instead of LoRA on attention or MLP — which
affects all tokens regardless of channel origin — LoRA the channel embedding
directly for *training* channels only. Unseen-band embeddings stay exactly
pretrained; all backbone attention/MLP blocks stay exactly pretrained; only
the 3 RGB channel embeddings drift from their pretrained values by a small
low-rank correction.

**Mechanism.**

```python
class ChannelEmbedLoRA(nn.Module):
    def __init__(self, D, n_channels=12, rank=4,
                 training_channels=(0,1,2)):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(n_channels, rank))
        self.B = nn.Parameter(torch.zeros(rank, D))
        mask = torch.zeros(n_channels)
        mask[list(training_channels)] = 1.0
        self.register_buffer("channel_mask", mask)

    def __call__(self, channel_embed_pretrained):
        delta = self.A @ self.B                        # (n_channels, D)
        delta_masked = delta * self.channel_mask[:, None]
        return channel_embed_pretrained + delta_masked

# In chi_vit forward: replace channel_embed lookup
ce = self.che_lora(self.channel_embed)                 # (n_channels, D)
channel_tokens = patch_tokens + ce[channel_ids]
```

**Plug-in.** `chi_vit.py` — swap `self.channel_embed` usage through a
ChannelEmbedLoRA wrapper. CLI: `--che_lora --che_rank 4`.

**Guarantee.** Strict worst-case on unseen-channel embeddings (mask is
non-learnable).

**Cost.** `2 × rank × n_channels + rank × D` params — typically < 10K for
ViT-B.

**Refs.** LoRA family adapted to embedding tables (analogous to token-embed
PEFT in NLP, e.g. 2305.14152).

---

### #31. Visual Prompt Tuning (VPT) for χViT

> **Status: pure.** Backbone entirely frozen; only prompts + linear head
> trained.

**Intuition.** Add `P` learnable prompt tokens prepended to the patch-token
sequence at each block (deep-VPT) or just at input (shallow-VPT). Prompts
participate in attention but are pure learnable parameters — not tied to any
channel. Backbone weights are untouched. For channel-aware variants, make
prompts *channel-indexed* (one set per channel, used only when that channel is
present at eval).

**Mechanism (deep-VPT).**

```python
class DeepVPT(nn.Module):
    def __init__(self, depth, n_prompts, D):
        super().__init__()
        self.prompts = nn.ParameterList([
            nn.Parameter(torch.randn(n_prompts, D) * 0.02)
            for _ in range(depth)
        ])

    def forward(self, x, block, layer_idx):
        p = self.prompts[layer_idx].unsqueeze(0).expand(x.shape[0], -1, -1)
        x_aug = torch.cat([p, x], dim=1)
        out = block(x_aug)
        return out[:, self.prompts[layer_idx].shape[0]:]   # drop prompt outputs
```

**Plug-in.** `chi_vit.py` forward modification. CLI:
`--vpt --vpt_type deep --vpt_n_prompts 10`.

**Guarantee.** Strict worst-case — backbone is never touched. Prompts can
only add signal via cross-attention; at init with zero prompts, model reduces
exactly to pretrained χViT + linear head.

**Cost.** `depth × n_prompts × D` params — typically < 200K for ViT-B.

**Refs.** Visual Prompt Tuning (Jia et al., ECCV 2022,
[2203.12119](https://arxiv.org/abs/2203.12119)); VPT in Null-Space (NeurIPS
2024, [2406.05658](https://arxiv.org/abs/2406.05658)); Deep Prompt Tuning.

---

### #32. Attention-Pooled Head (APH)

> **Status: pure.** Head-only replacement for the linear classifier.

**Intuition.** χViT's current classifier is a linear layer over `cls` or
`channel_mean` pooled features. APH replaces it with a tiny attention module
that learns *what to attend to* across channel tokens: a learnable query
token cross-attends to the per-channel pooled features, producing a
weighted aggregation, then an MLP produces class logits. More expressive than
mean/cls pooling but still backbone-frozen.

**Mechanism.**

```python
class AttentionPooledHead(nn.Module):
    def __init__(self, D, num_classes, num_heads=8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, D) * 0.02)
        self.cross_attn = nn.MultiheadAttention(D, num_heads,
                                                batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(D, D), nn.GELU(), nn.Linear(D, num_classes)
        )

    def forward(self, channel_feats):
        # channel_feats: (B, n_ch, D) — per-channel pooled features
        B = channel_feats.shape[0]
        q = self.query.unsqueeze(0).expand(B, -1, -1)
        pooled, _ = self.cross_attn(q, channel_feats, channel_feats)
        return self.mlp(pooled.squeeze(1))
```

**Plug-in.** `train_classifier.py` head swap. CLI: `--aph_head
--aph_num_heads 8`.

**Guarantee.** Expected-value only. But: naturally handles variable channel
count at eval (attention over `n_ch` keys works for any `n_ch`), and the
learnable query gives a principled fusion weight.

**Cost.** ~1–2 M params (one attention block + MLP).

**Refs.** CaiT class-attention pooling (Touvron et al., ICCV 2021,
[2103.17239](https://arxiv.org/abs/2103.17239)); Set Transformer
([1810.00825](https://arxiv.org/abs/1810.00825)).
