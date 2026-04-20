# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

Codebase for the **GeoCrossBench / χViT** paper (YerevaNN / UBC / Vector Institute). The current working focus is designing new fine-tuning mechanisms for **χViT (ChannelViT + iBOT)** and vanilla ViTs so they generalize across **subset → superset** and **non-overlapping** band settings in GeoCrossBench. Two companion docs are authoritative and should be consulted before making non-trivial changes:

- `.cursor/rules/project-knowledge.mdc` — full architecture, band orderings per model, evaluation protocol, metrics, common pitfalls.
- `TRAINING_PLAN.md` — the active ablation matrix (E0–E18 for χViT plus per-model plans), band-order cheatsheet, checkpoint path conventions.
- `.cursor/plans/powerful_chivit_fine-tuning_07a25de9.plan.md` — rationale for the cross-band fix flags (HCS, channel-mean pool, embed freeze, channel gate, dropout, curriculum, spectral init).

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
