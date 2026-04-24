# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

Codebase for the **GeoCrossBench / χViT** paper (YerevaNN / UBC / Vector Institute). The current working focus is designing new fine-tuning mechanisms for **χViT (ChannelViT + iBOT)** and vanilla ViTs so they generalize across **subset → superset** and **non-overlapping** band settings in GeoCrossBench. These companion docs are authoritative and should be consulted before making non-trivial changes:

- `.cursor/rules/project-knowledge.mdc` — full architecture, band orderings per model, evaluation protocol, metrics, common pitfalls.
- `TRAINING_PLAN.md` — the active ablation matrix (E0–E18 for χViT plus per-model plans), band-order cheatsheet, checkpoint path conventions.
- `.cursor/plans/powerful_chivit_fine-tuning_07a25de9.plan.md` — rationale for the cross-band fix flags (HCS, channel-mean pool, embed freeze, channel gate, dropout, curriculum, spectral init).
- `.cursor/plans/cross-band-finetune-catalog.md` — 14-approach brainstorming catalog for the next research round (beyond E0–E18). Glossary (Lipschitz, PAC, null-space, etc.), per-approach mechanism + arxiv refs, guarantee tiers, and candidate stacks.
- `.cursor/plans/stack-g-design.md` — Stack G / G+ / G++ design history. Contains the full-FT story (§1–§11), 2026 SOTA additions (§12), and the head/LoRA re-scoping (§13).
- `.cursor/plans/reliable-solutions.md` — **current plan.** Curated subset of the 32-approach catalog, filtered for: (a) head-only / LoRA-only training, (b) cross-model portability to χViT / TerraFM / DOFA / DINOv2 / DINOv3, (c) reliable for both superset (B) and no-overlap (A). Contains Reliable-Core recommendation, R0–R13 experiment matrix, per-model expected behavior, implementation file list, and **one dedicated CLI flag per technique** (§Flag reference) for independent ablation.
- `.cursor/plans/reliable-solutions-test-plan.md` — **TDD test plan.** ~130 named tests across universal/multispectral core + optional add-ons + integration + portability. Red-green-refactor discipline; no production code without a failing test first. Read this before writing any implementation code.

## Research goal — new subset→superset fine-tuning techniques

Active research thread beyond the E0–E18 grid. Goal: new techniques that satisfy a **strict problem definition**, not whatever mechanism maximizes benchmark numbers.

**Problem we are solving.**
- **Priority B — monotonic superset.** Evaluation on the superset (RGB → RGBN, S2 → S2+S1) must be **provably ≥** evaluation on the training subset, via *architectural* guarantee — not via a gate calibrated on held-out superset data.
- **Priority A — no-overlap.** Evaluation on RGB→S1, S2→S1, RGB→N'S1S2 still benefits from new techniques, but strict monotonicity isn't applicable (no subset of the training bands remains).

**Design constraints — enforce when evaluating any candidate technique.**

1. **Fine-tune only on the training subset.** No peeking at N/S1 channels during the fine-tune training loop. Running a teacher live on 12 bands during fine-tuning (e.g. positive-congruent multispectral distillation, PC-MS) **violates this** and must be rejected.
2. **Architectural monotonicity is preferred over calibrated gates.** Calibrating a gate on a held-out validation split from the eval distribution is treated as "self-deception" for this problem — it's just ensembling with a learned weight.
3. **No re-pretraining in this project.** Techniques that require running iBOT-scale pretraining from scratch are out of scope. A DOFA-style wavelength hypernetwork, for example, would need re-pretraining for full effect → rejected.
4. **Offline artifacts from the pretraining corpus *are* allowed.** Anything that can be precomputed once from the full-band pretraining data (statistics, prototypes, covariances, projection bases, per-layer activation subspaces) and then used as a frozen artifact during fine-tuning is fair game — that's caching existing knowledge, not pretraining.
5. **External pretrained foundation models are allowed** (e.g. DOFA, DiffusionSat). They were pretrained outside our pipeline. Using them at eval or as frozen distillation sources is fine, as long as no training on 12-band data happens inside our fine-tune loop.

**Rules of thumb derived from the above.**

- Reject any technique that runs a fresh 12-band forward during the fine-tune loop.
- A DOFA-style hypernet is **dead for this benchmark**: GeoCrossBench eval bands are always a subset of the 12 bands χViT was pretrained on, so `--freeze_unused_channel_embeds` already produces the optimal per-band embedding. A hypernet only matters when the benchmark exposes wavelengths outside pretraining — it doesn't.
- The iBOT-pretrained χViT checkpoint is fixed and treated as the starting point of the problem. Its 12 learned `channel_embed[c]` vectors may be used as-is as frozen prototypes.

**Additional scope restrictions (as of 2026-04-23).**

1. **Head + LoRA only.** Full-backbone fine-tuning is out of scope. Training touches only classifier / segmentation / CD head parameters and LoRA-family adapters; backbone weights are frozen.
2. **Cross-model portability.** The chosen stack must run on **χViT, TerraFM, DOFA, DINOv2, DINOv3** with the same code path (modulo a user-owned embedding generator that produces per-channel features per model). Techniques that depend on χViT-specific internals (e.g. `channel_embed` lookup table) are demoted or dropped.
3. **Reliability across both priorities.** The stack must help both Priority B (superset) and Priority A (no-overlap); if a step helps one and hurts the other, it's excluded.

**Current shortlist — Reliable-Core (R9 in the R-grid).** See `.cursor/plans/reliable-solutions.md` for the single source of truth.

**Universal core (reliable for all 5 models):**

- **#29 LastN-LoRA (N=4)** — LoRA restricted to last 4 transformer blocks.
- **#16 OPLoRA + #20 LoRA-Null Init** — subset-forward preservation on LoRA updates via SVD-based projection. Strict worst-case on top-k singular triples.
- **#2 Identity-Init + Hard Channel Mask** — unseen-channel features bypass the adapter by a non-learnable channel-ID indicator.
- **#9 CDSD** — channel-dropout self-distillation with an EMA teacher; training-time closure prior.
- **#32 APH** — attention-pooled head; variable-channel-count-aware.
- **#5 ReAct** — eval-time feature-norm clipping.
- **#21 MERA** (post-training, optional but recommended) — LoRA task-arithmetic merge + short subset-only realign. Primary lever for no-overlap on all 5 models.

**Multispectral core (strong for χViT/TerraFM/DOFA; graceful no-op on DINOv2/v3):**

- **#11 Hopfield Channel-Prototype Memory** — frozen per-band prototypes + zero-init cross-attention retrieval. On multispectral backbones: primary Priority-A lever. On DINOv2/v3: cross-attention stays near zero → no harm.
- **#23 LSMM Auxiliary Head** — VCA-endmember spectral reconstruction regularizer. Physics-grounded; works as a weaker-but-safe regularizer on DINOv2/v3.
- **#24 SRF-Biased Attention (inside #32 APH)** — 12×12 Sentinel SRF overlap matrix as pre-softmax bias in APH attention. Parameter-free physics prior; safe on all models.

Optional add-ons (all reliable/portable): **#7 TC-CAF** (PAC), **#18 BPSG** (Bayesian gate), **#27 ADAPT** (Gaussian alignment), **#28 MCSE** (ensemble-head variant), **#14 NCI-PIH** (alternative/secondary head), **#15 CH-RS-FT** (certified radius), **#12/#26 imputation** (no-overlap fallback — especially for DINOv2/v3).

χViT-specific bonuses (not in the portable stack; may appear as separate paper rows): **#30 ChE-LoRA**, **#22 ChEmbed Diffusion**.

See `.cursor/plans/cross-band-finetune-catalog.md` for the full 32-approach library, `.cursor/plans/stack-g-design.md` for earlier (full-FT / G++ / H-grid) design history, and `.cursor/plans/reliable-solutions.md` for the current portable R-grid and implementation file list.

## Environment (uv)

This repo uses **uv** exclusively (not pip/conda). Python is pinned to 3.11, Torch wheels come from the `pytorch-cu128` index declared in `pyproject.toml`.

```bash
./bootstrap_env.sh            # creates/refreshes .venv via `uv sync --python 3.11`
./bootstrap_env.sh --fresh    # wipe .venv first
uv run <script.py> ...        # preferred launcher; scripts expect .venv
```

Shell scripts under `rs_finetune/` all do `source "$REPO_ROOT/.venv/bin/activate"` before invoking `uv run` / `torchrun`; keep that pattern when adding new scripts.

## Common commands

All training/eval commands run from `rs_finetune/`.

```bash
# Tests (pytest, monkey-patches sys.path to rs_finetune in tests/conftest.py)
cd rs_finetune && ./run_tests.sh                        # all tests, quiet
cd rs_finetune && ./run_tests.sh -k channel_dropout     # single test / filter

# Lint / format (config in pyproject.toml; src = rs_finetune, rs_pretrain)
uv run ruff check rs_finetune rs_pretrain
uv run ruff format rs_finetune rs_pretrain
uv run pylint rs_finetune                               # heavier, optional

# Classification finetune (PyTorch Lightning, single-GPU via --device N)
uv run train_classifier.py --backbone cvit-pretrained --encoder_weights chi_vit \
  --dataset_name m_eurosat --image_size 64 --bands B04 B03 B02 \
  --shared_proj --add_ch_embed --pooling_mode cls --only_head \
  --experiment_name cvit_eurosat_rgb_E0_baseline

# Segmentation / Change detection (manual DDP via torchrun, NOT plain python)
torchrun --nproc_per_node=2 train_segmenter.py ...      # see ablation_seg_cvit_rgb_example.sh
torchrun --nproc_per_node=2 train_change.py    ...      # see ablation_cd_cvit_rgb_example.sh

# Cross-band evaluation (takes JSON list of band combos via --bands)
uv run eval_bands_cls.py --model_config configs/cvit-pretrained.json \
  --dataset_config configs/m_eurosat.json --checkpoint_path <ckpt> \
  --bands '[["B04","B03","B02","B08"]]' --shared_proj --add_ch_embed
uv run eval_bands_seg.py ...    # segmentation eval
uv run eval_bands_cd.py  ...    # change-detection eval

# χViT classification ablation grid (E0..E18) — SLURM array or local loop
sbatch rs_finetune/ablation_cls_array.sh                # ABLATION_CLS_DATASET=eurosat|ben|brick|so2sat
EXPERIMENT_ID=E7 ./ablation_cls_eval_only.sh            # rerun eval for one experiment
```

`ablation_classification_common.sh` is the single source of truth for the E0–E18 flag combinations plus the RGB and RGBN-superset eval sweeps; `ablation_cls_array.sh` dispatches it across a SLURM job array and `ablation_cls_eval_only.sh` reuses it for eval-only re-runs. Treat that file as the canonical mapping — do not re-derive flag sets elsewhere.

## Architecture in 60 seconds

**Two trees, one package layout:**
- `rs_pretrain/` — iBOT-style self-supervised pretraining that produces χViT teacher checkpoints (`main_ibot_it.py`, `chivit_pretrain.py`, models in `models/`, HDF5/NAIP/Sentinel loaders in `dataset/`). Rarely the target of day-to-day edits.
- `rs_finetune/` — **primary working area**. Task scripts (`train_classifier.py` / `train_segmenter.py` / `train_change.py`) and matching `eval_bands_*.py` / `eval_scale_*.py` live at the top; shared plumbing sits under `change_detection_pytorch/` (encoder wrappers, datasets, UPerNet/Unet decoders, losses).

**Task → framework matrix:**
| Task | Entry | Framework | Multi-GPU |
|------|-------|-----------|-----------|
| Classification | `train_classifier.py` | PyTorch Lightning | single-GPU (`--device`) |
| Segmentation | `train_segmenter.py` | Manual DDP loop | `torchrun` |
| Change detection | `train_change.py` | Manual DDP loop (siamese UPerNet) | `torchrun` |

**Encoder dispatch.** Every backbone flows through `classifier_utils.py::load_encoder(encoder_name, encoder_weights, …)`. Name prefixes route to encoder dicts defined under `change_detection_pytorch/encoders/` (ibot-*, cvit-pretrained / chi_vit, dinov2 via torch.hub, dinov3 via HuggingFace `AutoModel`, terrafm, dofa, prithvi, croma, anysat, clay, swin-B, timm_*). Adding a new encoder means registering it in that dispatch.

**Config system.** Every run points at two JSONs in `rs_finetune/configs/`: a **model config** (`backbone`, `encoder_weights`, `in_features`, `in_channels`, `fill_zeros`, …) and a **dataset config** (`dataset_name`, `base_dir`, `splits_dir`, `num_classes`, `image_size`, `batch_size`). Dataset configs store paths *relative* to `DATASETS_ROOT`; loaders must call `resolve_dataset_config_dict()` right after `json.load` so env overrides apply.

**Storage roots (`rs_finetune/storage_paths.py`).** Override any of these via env without editing code:
- `GEOCROSSBENCH_DATASETS_ROOT` → `DATASETS_ROOT` (default `/mnt/weka/akhosrovyan/geocrossbench/datasets`)
- `GEOCROSSBENCH_BASE_MODELS_ROOT` → pretrained weights
- `RS_FOUNDATION_RESULTS_ROOT` → logs + `finetune_ckpts/{classification,segmentation,change_detection}/<experiment_name>/`

**Cross-band adaptation.** `--enable_multiband_input` re-initializes the first conv/patch_embed with either averaged RGB (`adapt_rgb_conv_layer_to_multiband`) or RGB-preserving spectral init (`--preserve_rgb_weights --spectral_init_new_channels`). All χViT cross-band fix flags (`--enable_sample`, `--pooling_mode channel_mean|cls+channel_mean`, `--freeze_unused_channel_embeds`, `--channel_embed_reg_lambda`, `--enable_channel_gate`, `--channel_dropout_rate`, `--curriculum_sampling`) are defined in `classifier_utils.py` / `callbacks.py` and wired through all three task scripts — keep them in sync when adding new ones.

## Non-obvious gotchas

- **Band ordering differs per model.** Always go through `utils.get_band_orders(model_name)`; χViT expects R,G,B (`B04 B03 B02`) while most others expect B,G,R. Segmentation/CD scripts use short names (`B2 B3 B4`, `vh vv`); classification uses `B02..B12 VV VH`.
- **`--only_head` vs Lightning's `Trainable params`.** Lightning prints the full backbone as trainable when gradients flow through it for head loss; the optimizer in `configure_optimizers` still restricts to `classifier.parameters()`. Don't rename head-only experiments as "full fine-tuning".
- **Seg/CD must be launched with `torchrun`** (manual DDP). Plain `python train_segmenter.py` will silently run on rank 0 only.
- **DINOv2/v3 image size must be a multiple of 14.** Callers use `(size // 14) * 14` — preserve this when changing sizes.
- **ChannelViT `--cvit_channels`** indexes into the model's own channel dict; values differ between variants (e.g. so2sat duplicates SAR). Cross-check against `utils.py` before copying from another script.
- **SAR normalization** uses hard-coded stats (`VH: -19.3/5.46`, `VV: -12.62/5.12`); don't recompute from batch.
- **Classification saves `best-model.ckpt` (single-label) or `best-model-f1.ckpt` (m_ben).** Seg/CD save `best_model.pth`. The ablation helper picks the right filename via `checkpoint_filename` per dataset.
