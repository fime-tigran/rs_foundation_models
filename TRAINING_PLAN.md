# Superset Fine-Tuning & Evaluation Plan

## What "Superset" Means

| Train input | Eval input | New bands |
|---|---|---|
| RGB (3 bands) | RGBN (4 bands) | B08 (NIR) |
| S2 (10 bands) | S2+S1 (12 bands) | VH, VV (SAR) |

Key metric: **RGBN − RGB delta** (and S2+S1 − S2 delta). Positive = model benefits from extra bands.

---

## Band-Order Reference

| Model | RGB | S2 (10 bands) |
|---|---|---|
| **χViT** | `B04 B03 B02` | `B04 B03 B02 B05 B06 B07 B08 B8A B11 B12` |
| iBOT, DINOv2/v3, ViT-B, DOFA, TerraFM | `B02 B03 B04` | `B02 B03 B04 B05 B06 B07 B08 B8A B11 B12` |

Segmentation/CD scripts use shorthand (`B2 B3 B4`, lowercase `vh vv`). Classification uses full names.

---

## Applicable Fixes per Model

| Fix | χViT | iBOT (multispectral-pretrained) | DINOv2 / DINOv3 / ViT-B (RGB-only) |
|---|---|---|---|
| A — HCS (`--enable_sample`) | **Yes** | No (no HCS arch) | No |
| B — Channel-mean pool (`--pooling_mode channel_mean`) | **Yes** | No | No |
| C — Embed freeze (`--freeze_unused_channel_embeds`) | **Yes** | No | No |
| D — Channel gate (`--enable_channel_gate`) | **Yes** | No | No |
| Channel dropout (`--channel_dropout_rate`) | Prefer HCS instead | **Yes** | **Yes** |
| Spectral init at eval (`--spectral_init_new_channels`) | **Yes** | **Yes** | Yes (limited effect) |

---

## Checkpoint Paths

| Task | Directory |
|---|---|
| Classification | `/mnt/weka/tgrigoryan/rs_foundation/finetune_ckpts/classification/<exp>/` |
| Segmentation | `/mnt/weka/tgrigoryan/rs_foundation/finetune_ckpts/segmentation/<exp>/` |
| Change detection | `/mnt/weka/tgrigoryan/rs_foundation/finetune_ckpts/change_detection/<exp>/` |

Classification saves `best-model.ckpt` (single-label) or `best-model-f1.ckpt` (m_ben).  
Segmentation/CD saves `best_model.pth`.

---

---

# PART 1 — χViT (Primary Model)

χViT is the primary focus. Run the full ablation grid first on `m_eurosat`, then propagate winners to all datasets and tasks.

## 1.1 Feature Flags Reference

| Fix | Flag(s) | Description |
|---|---|---|
| **A — HCS** | `--enable_sample --min_sample_channels 1` | Sample random channel subsets each forward pass — variable token count |
| **B — Channel pool** | `--pooling_mode channel_mean` | Per-channel mean pooling — count-invariant output (vs CLS token) |
| **B-combined** | `--pooling_mode cls+channel_mean` | Average of CLS and channel_mean |
| **C — Embed freeze** | `--freeze_unused_channel_embeds` | Freeze embeddings for bands not in training set |
| **C-soft** | `--channel_embed_reg_lambda 0.1` | L2 regularize toward pretrained embed values |
| **C-hard** | `--frozen_channel_embed` | Freeze ALL channel embeddings |
| **D — Channel gate** | `--enable_channel_gate` | Learnable per-channel gate scalar |
| **Dropout** | `--channel_dropout_rate 0.2 --min_drop_channels 1` | Random per-channel masking within training bands |
| **Curriculum** | `--curriculum_sampling` | Anneal HCS aggressiveness (requires `--enable_sample`) |

## 1.2 Ablation Matrix — Classification, RGB → RGBN

Primary dataset: `m_eurosat`. Propagate winners to `so2sat`, `m_ben`, `m_brick`.

| ID | HCS | Pool | Embed | Gate | Dropout | Curriculum | Experiment suffix |
|---|---|---|---|---|---|---|---|
| **E0** | — | cls | — | — | — | — | `E0_baseline` |
| **E1** | ✓ | cls | — | — | — | — | `E1_hcs` |
| **E2** | — | ch_mean | — | — | — | — | `E2_pool` |
| **E3** | — | cls | freeze | — | — | — | `E3_embedfreeze` |
| **E4** | — | cls | — | — | 0.2 | — | `E4_dropout` |
| **E5** | — | cls | — | ✓ | — | — | `E5_gate` |
| **E6** | ✓ | ch_mean | — | — | — | — | `E6_hcs_pool` |
| **E7** | ✓ | ch_mean | freeze | — | — | — | `E7_hcs_pool_embedfreeze` |
| **E8** | ✓ | ch_mean | freeze | — | 0.2 | — | `E8_hcs_pool_embedfreeze_dropout` |
| **E9** | ✓ | ch_mean | freeze | — | 0.2 | ✓ | `E9_hcs_pool_embedfreeze_dropout_curriculum` |
| **E10** | ✓ | ch_mean | freeze | ✓ | 0.2 | ✓ | `E10_fullstack` |
| **E11** | ✓ | ch_mean | C-soft | — | 0.2 | ✓ | `E11_soft_embedreg` |
| **E12** | ✓ | cls+ch | freeze | — | 0.2 | — | `E12_combined_pool` |

**Recommended run order:** E0 → E6 → E7 → E9 → E10 (each adds one fix; skip if previous showed no benefit).

---

## 1.3 χViT Classification Training Commands

```bash
cd rs_finetune
```

### E0 — Baseline

```bash
python train_classifier.py \
  --backbone cvit-pretrained --encoder_weights chi_vit \
  --dataset_name m_eurosat --image_size 64 \
  --batch_size 64 --epoch 50 --lr 1e-4 --weight_decay 0.05 \
  --warmup_steps 20 --scheduler cosine \
  --bands B04 B03 B02 --pooling_mode cls --shared_proj \
  --device 1 --seed 42 \
  --experiment_name cvit_eurosat_rgb_E0_baseline
```

### E1 — HCS only

```bash
python train_classifier.py \
  --backbone cvit-pretrained --encoder_weights chi_vit \
  --dataset_name m_eurosat --image_size 64 \
  --batch_size 64 --epoch 50 --lr 1e-4 --weight_decay 0.05 \
  --warmup_steps 20 --scheduler cosine \
  --bands B04 B03 B02 --pooling_mode cls --shared_proj \
  --enable_sample --min_sample_channels 1 \
  --device 1 --seed 42 \
  --experiment_name cvit_eurosat_rgb_E1_hcs
```

### E2 — Channel-mean pool only

```bash
python train_classifier.py \
  --backbone cvit-pretrained --encoder_weights chi_vit \
  --dataset_name m_eurosat --image_size 64 \
  --batch_size 64 --epoch 50 --lr 1e-4 --weight_decay 0.05 \
  --warmup_steps 20 --scheduler cosine \
  --bands B04 B03 B02 --pooling_mode channel_mean --shared_proj \
  --device 1 --seed 42 \
  --experiment_name cvit_eurosat_rgb_E2_pool
```

### E3 — Embed freeze only

```bash
python train_classifier.py \
  --backbone cvit-pretrained --encoder_weights chi_vit \
  --dataset_name m_eurosat --image_size 64 \
  --batch_size 64 --epoch 50 --lr 1e-4 --weight_decay 0.05 \
  --warmup_steps 20 --scheduler cosine \
  --bands B04 B03 B02 --pooling_mode cls --shared_proj \
  --freeze_unused_channel_embeds \
  --device 1 --seed 42 \
  --experiment_name cvit_eurosat_rgb_E3_embedfreeze
```

### E4 — Channel dropout only

```bash
python train_classifier.py \
  --backbone cvit-pretrained --encoder_weights chi_vit \
  --dataset_name m_eurosat --image_size 64 \
  --batch_size 64 --epoch 50 --lr 1e-4 --weight_decay 0.05 \
  --warmup_steps 20 --scheduler cosine \
  --bands B04 B03 B02 --pooling_mode cls --shared_proj \
  --channel_dropout_rate 0.2 --min_drop_channels 1 \
  --device 1 --seed 42 \
  --experiment_name cvit_eurosat_rgb_E4_dropout
```

### E5 — Channel gate only

```bash
python train_classifier.py \
  --backbone cvit-pretrained --encoder_weights chi_vit \
  --dataset_name m_eurosat --image_size 64 \
  --batch_size 64 --epoch 50 --lr 1e-4 --weight_decay 0.05 \
  --warmup_steps 20 --scheduler cosine \
  --bands B04 B03 B02 --pooling_mode cls --shared_proj \
  --enable_channel_gate \
  --device 1 --seed 42 \
  --experiment_name cvit_eurosat_rgb_E5_gate
```

### E6 — HCS + channel-mean pool

```bash
python train_classifier.py \
  --backbone cvit-pretrained --encoder_weights chi_vit \
  --dataset_name m_eurosat --image_size 64 \
  --batch_size 64 --epoch 50 --lr 1e-4 --weight_decay 0.05 \
  --warmup_steps 20 --scheduler cosine \
  --bands B04 B03 B02 --pooling_mode channel_mean --shared_proj \
  --enable_sample --min_sample_channels 1 \
  --device 1 --seed 42 \
  --experiment_name cvit_eurosat_rgb_E6_hcs_pool
```

### E7 — HCS + pool + embed freeze

```bash
python train_classifier.py \
  --backbone cvit-pretrained --encoder_weights chi_vit \
  --dataset_name m_eurosat --image_size 64 \
  --batch_size 64 --epoch 50 --lr 1e-4 --weight_decay 0.05 \
  --warmup_steps 20 --scheduler cosine \
  --bands B04 B03 B02 --pooling_mode channel_mean --shared_proj \
  --enable_sample --min_sample_channels 1 \
  --freeze_unused_channel_embeds \
  --device 1 --seed 42 \
  --experiment_name cvit_eurosat_rgb_E7_hcs_pool_embedfreeze
```

### E8 — HCS + pool + embed freeze + dropout

```bash
python train_classifier.py \
  --backbone cvit-pretrained --encoder_weights chi_vit \
  --dataset_name m_eurosat --image_size 64 \
  --batch_size 64 --epoch 50 --lr 1e-4 --weight_decay 0.05 \
  --warmup_steps 20 --scheduler cosine \
  --bands B04 B03 B02 --pooling_mode channel_mean --shared_proj \
  --enable_sample --min_sample_channels 1 \
  --freeze_unused_channel_embeds \
  --channel_dropout_rate 0.2 --min_drop_channels 1 \
  --device 1 --seed 42 \
  --experiment_name cvit_eurosat_rgb_E8_hcs_pool_embedfreeze_dropout
```

### E9 — HCS + pool + embed freeze + dropout + curriculum ⭐ recommended first full run

```bash
python train_classifier.py \
  --backbone cvit-pretrained --encoder_weights chi_vit \
  --dataset_name m_eurosat --image_size 64 \
  --batch_size 64 --epoch 50 --lr 1e-4 --weight_decay 0.05 \
  --warmup_steps 20 --scheduler cosine \
  --bands B04 B03 B02 --pooling_mode channel_mean --shared_proj \
  --enable_sample --min_sample_channels 1 --curriculum_sampling \
  --freeze_unused_channel_embeds \
  --channel_dropout_rate 0.2 --min_drop_channels 1 \
  --device 1 --seed 42 \
  --experiment_name cvit_eurosat_rgb_E9_hcs_pool_embedfreeze_dropout_curriculum
```

### E10 — Full stack (all fixes)

```bash
python train_classifier.py \
  --backbone cvit-pretrained --encoder_weights chi_vit \
  --dataset_name m_eurosat --image_size 64 \
  --batch_size 64 --epoch 50 --lr 1e-4 --weight_decay 0.05 \
  --warmup_steps 20 --scheduler cosine \
  --bands B04 B03 B02 --pooling_mode channel_mean \
  --enable_sample --min_sample_channels 1 --curriculum_sampling \
  --freeze_unused_channel_embeds \
  --channel_dropout_rate 0.2 --min_drop_channels 1 \
  --enable_channel_gate \
  --device 1 --seed 42 \
  --experiment_name cvit_eurosat_rgb_E10_fullstack
```

### E11 — HCS + pool + C-soft (L2 reg) + dropout + curriculum

```bash
python train_classifier.py \
  --backbone cvit-pretrained --encoder_weights chi_vit \
  --dataset_name m_eurosat --image_size 64 \
  --batch_size 64 --epoch 50 --lr 1e-4 --weight_decay 0.05 \
  --warmup_steps 20 --scheduler cosine \
  --bands B04 B03 B02 --pooling_mode channel_mean \
  --enable_sample --min_sample_channels 1 --curriculum_sampling \
  --channel_embed_reg_lambda 0.1 \
  --channel_dropout_rate 0.2 --min_drop_channels 1 \
  --device 1 --seed 42 \
  --experiment_name cvit_eurosat_rgb_E11_soft_embedreg
```

### E12 — HCS + cls+channel_mean pool + embed freeze + dropout

```bash
python train_classifier.py \
  --backbone cvit-pretrained --encoder_weights chi_vit \
  --dataset_name m_eurosat --image_size 64 \
  --batch_size 64 --epoch 50 --lr 1e-4 --weight_decay 0.05 \
  --warmup_steps 20 --scheduler cosine \
  --bands B04 B03 B02 --pooling_mode cls+channel_mean --shared_proj \
  --enable_sample --min_sample_channels 1 \
  --freeze_unused_channel_embeds \
  --channel_dropout_rate 0.2 --min_drop_channels 1 \
  --device 1 --seed 42 \
  --experiment_name cvit_eurosat_rgb_E12_combined_pool
```

---

## 1.4 χViT S2 Training (10 bands → S2+S1 eval)

Baseline and best configuration. Adjust `--min_sample_channels` and `--min_drop_channels` to 3 for 10-band inputs.

### E0-s2 — S2 baseline

```bash
python train_classifier.py \
  --backbone cvit-pretrained --encoder_weights chi_vit \
  --dataset_name m_eurosat --image_size 64 \
  --batch_size 64 --epoch 50 --lr 1e-4 --weight_decay 0.05 \
  --warmup_steps 20 --scheduler cosine \
  --enable_multiband_input --multiband_channel_count 10 \
  --bands B04 B03 B02 B05 B06 B07 B08 B8A B11 B12 \
  --pooling_mode cls --shared_proj \
  --device 1 --seed 42 \
  --experiment_name cvit_eurosat_s2_E0_baseline
```

### E9-s2 — recommended S2 run

```bash
python train_classifier.py \
  --backbone cvit-pretrained --encoder_weights chi_vit \
  --dataset_name m_eurosat --image_size 64 \
  --batch_size 64 --epoch 50 --lr 1e-4 --weight_decay 0.05 \
  --warmup_steps 20 --scheduler cosine \
  --enable_multiband_input --multiband_channel_count 10 \
  --bands B04 B03 B02 B05 B06 B07 B08 B8A B11 B12 \
  --enable_sample --min_sample_channels 3 --curriculum_sampling \
  --pooling_mode channel_mean \
  --freeze_unused_channel_embeds \
  --channel_dropout_rate 0.2 --min_drop_channels 3 \
  --shared_proj \
  --device 1 --seed 42 \
  --experiment_name cvit_eurosat_s2_E9_hcs_pool_embedfreeze_dropout_curriculum
```

---

## 1.5 χViT Evaluation Commands

`eval_bands_cls.py` `channel_vit_order`: `['B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12','VV','VH']`.  
`--bands` is a **JSON string**. Always match `--pooling_mode` to training.

### RGB → RGBN (two variants per experiment)

```bash
EXP=cvit_eurosat_rgb_E9_hcs_pool_embedfreeze_dropout_curriculum
CKPT=/mnt/weka/tgrigoryan/rs_foundation/finetune_ckpts/classification/${EXP}/best-model.ckpt

# mean-init for new B08 weight
python eval_bands_cls.py \
  --model_config configs/cvit-pretrained.json \
  --dataset_config configs/m_eurosat.json \
  --checkpoint_path "$CKPT" --img_size 64 \
  --enable_multiband_input --multiband_channel_count 4 \
  --preserve_rgb_weights \
  --training_bands '["B04","B03","B02"]' --new_bands '["B08"]' \
  --bands '[["B04","B03","B02","B08"]]' \
  --pooling_mode channel_mean --shared_proj \
  --filename eval_${EXP}_rgbn_meaninit

# spectral-init for new B08 weight
python eval_bands_cls.py \
  --model_config configs/cvit-pretrained.json \
  --dataset_config configs/m_eurosat.json \
  --checkpoint_path "$CKPT" --img_size 64 \
  --enable_multiband_input --multiband_channel_count 4 \
  --preserve_rgb_weights --spectral_init_new_channels \
  --training_bands '["B04","B03","B02"]' --new_bands '["B08"]' \
  --bands '[["B04","B03","B02","B08"]]' \
  --pooling_mode channel_mean --shared_proj \
  --filename eval_${EXP}_rgbn_spectralinit
```

Swap `EXP` and `--pooling_mode` per experiment. For E0–E5 use `cls`; for E12 use `cls+channel_mean`.

### S2 → S2+S1

```bash
EXP=cvit_eurosat_s2_E9_hcs_pool_embedfreeze_dropout_curriculum
CKPT=/mnt/weka/tgrigoryan/rs_foundation/finetune_ckpts/classification/${EXP}/best-model.ckpt

python eval_bands_cls.py \
  --model_config configs/cvit-pretrained.json \
  --dataset_config configs/m_eurosat.json \
  --checkpoint_path "$CKPT" --img_size 64 \
  --enable_multiband_input --multiband_channel_count 12 \
  --preserve_rgb_weights --spectral_init_new_channels \
  --training_bands '["B04","B03","B02","B05","B06","B07","B08","B8A","B11","B12"]' \
  --new_bands '["VV","VH"]' \
  --bands '[["B04","B03","B02","B05","B06","B07","B08","B8A","B11","B12","VV","VH"]]' \
  --pooling_mode channel_mean --shared_proj \
  --filename eval_${EXP}_s2s1_spectralinit
```

---

## 1.6 Propagate Best χViT Config to All Datasets

Once the winning experiment ID (e.g. E9) is confirmed on `m_eurosat`:

```bash
for DATASET in m_eurosat so2sat m_brick m_ben; do
  IMG=$([ "$DATASET" = "m_ben" ] && echo 120 || ([ "$DATASET" = "so2sat" ] && echo 32 || echo 64))
  python train_classifier.py \
    --backbone cvit-pretrained --encoder_weights chi_vit \
    --dataset_name $DATASET --image_size $IMG \
    --batch_size 64 --epoch 50 --lr 1e-4 --weight_decay 0.05 \
    --warmup_steps 20 --scheduler cosine \
    --bands B04 B03 B02 --pooling_mode channel_mean \
    --enable_sample --min_sample_channels 1 --curriculum_sampling \
    --freeze_unused_channel_embeds \
    --channel_dropout_rate 0.2 --min_drop_channels 1 \
    --shared_proj \
    --device 1 --seed 42 \
    --experiment_name cvit_${DATASET}_rgb_best
done
```

---

## 1.7 χViT Segmentation — Harvey & Sen1Floods11

Script: `torchrun`. Band format: shorthand no leading zero (`B4 B3 B2`, `vv vh`).

### RGB training

```bash
torchrun --nproc_per_node=2 train_segmenter.py \
  --backbone cvit-pretrained --encoder_weights chi_vit \
  --encoder_depth 12 --in_channels 3 \
  --dataset_name harvey \
  --dataset_path /mnt/weka/akhosrovyan/geocrossbench/datasets/x-harvey \
  --img_size 96 --classes 2 --loss_type ce \
  --batch_size 8 --max_epochs 100 \
  --lr 1e-4 --weight_decay 0.01 \
  --lr_sched warmup_cosine --warmup_steps 20 --warmup_lr 1e-6 \
  --upernet_width 256 --decoder upernet \
  --bands B4 B3 B2 --cvit_channels 2 1 0 \
  --enable_sample \
  --channel_dropout_rate 0.2 --min_drop_channels 1 \
  --num_workers 8 --seed 42 \
  --experiment_name cvit_harvey_seg_rgb_E9
```

### S2 training

```bash
torchrun --nproc_per_node=2 train_segmenter.py \
  --backbone cvit-pretrained --encoder_weights chi_vit \
  --encoder_depth 12 --in_channels 10 \
  --dataset_name harvey \
  --dataset_path /mnt/weka/akhosrovyan/geocrossbench/datasets/x-harvey \
  --img_size 96 --classes 2 --loss_type ce \
  --batch_size 8 --max_epochs 100 \
  --lr 1e-4 --weight_decay 0.01 \
  --lr_sched warmup_cosine --warmup_steps 20 --warmup_lr 1e-6 \
  --enable_multiband_input --multiband_channel_count 10 \
  --upernet_width 256 --decoder upernet \
  --bands B4 B3 B2 B5 B6 B7 B8 B8A B11 B12 \
  --cvit_channels 0 1 2 3 4 5 6 7 8 9 \
  --enable_sample \
  --channel_dropout_rate 0.2 --min_drop_channels 3 \
  --num_workers 8 --seed 42 \
  --experiment_name cvit_harvey_seg_s2_E9
```

### Eval — RGB → RGBN

`eval_bands_seg.py` `channel_vit_order`: `['B4','B3','B2','B5','B6','B7','B8','B8A','B11','B12','vv','vh']`.

```bash
CKPT=/mnt/weka/tgrigoryan/rs_foundation/finetune_ckpts/segmentation/cvit_harvey_seg_rgb_E9/best_model.pth

python eval_bands_seg.py \
  --model_config configs/cvit-pretrained.json \
  --dataset_config configs/harvey.json \
  --checkpoint_path "$CKPT" \
  --size 96 --classes 2 --upernet_width 256 \
  --enable_multiband_input --multiband_channel_count 4 \
  --preserve_rgb_weights --spectral_init_new_channels \
  --training_bands '["B4","B3","B2"]' --new_bands '["B8"]' \
  --bands '[["B4","B3","B2","B8"]]' \
  --filename superset_cvit_harvey_seg_rgbn
```

### Eval — S2 → S2+S1

```bash
CKPT=/mnt/weka/tgrigoryan/rs_foundation/finetune_ckpts/segmentation/cvit_harvey_seg_s2_E9/best_model.pth

python eval_bands_seg.py \
  --model_config configs/cvit-pretrained.json \
  --dataset_config configs/harvey.json \
  --checkpoint_path "$CKPT" \
  --size 96 --classes 2 --upernet_width 256 \
  --enable_multiband_input --multiband_channel_count 12 \
  --preserve_rgb_weights --spectral_init_new_channels \
  --training_bands '["B4","B3","B2","B5","B6","B7","B8","B8A","B11","B12"]' \
  --new_bands '["vv","vh"]' \
  --bands '[["B4","B3","B2","B5","B6","B7","B8","B8A","B11","B12","vv","vh"]]' \
  --filename superset_cvit_harvey_seg_s2s1
```

---

## 1.8 χViT Change Detection — Harvey & OSCD

Script: `torchrun`. Flag is `--enable_multiband` (not `--enable_multiband_input`).  
Harvey split lists are hardcoded inside the script.

### RGB training

```bash
torchrun --nproc_per_node=2 train_change.py \
  --backbone cvit-pretrained --encoder_weights chi_vit \
  --encoder_depth 12 --in_channels 3 \
  --dataset_name harvey \
  --dataset_path /mnt/weka/akhosrovyan/geocrossbench/datasets/x-harvey \
  --img_size 256 --fusion diff \
  --batch_size 8 --max_epochs 100 \
  --lr 1e-4 --weight_decay 0.01 \
  --lr_sched warmup_cosine --warmup_steps 20 --warmup_lr 1e-6 \
  --upernet_width 256 \
  --bands B4 B3 B2 --cvit_channels 2 1 0 \
  --enable_sample \
  --channel_dropout_rate 0.2 --min_drop_channels 1 \
  --num_workers 8 --seed 42 \
  --experiment_name cvit_harvey_cd_rgb_E9
```

### S2 training

```bash
torchrun --nproc_per_node=2 train_change.py \
  --backbone cvit-pretrained --encoder_weights chi_vit \
  --encoder_depth 12 --in_channels 10 \
  --dataset_name harvey \
  --dataset_path /mnt/weka/akhosrovyan/geocrossbench/datasets/x-harvey \
  --img_size 256 --fusion diff \
  --batch_size 8 --max_epochs 100 \
  --lr 1e-4 --weight_decay 0.01 \
  --lr_sched warmup_cosine --warmup_steps 20 --warmup_lr 1e-6 \
  --enable_multiband --multiband_channel_count 10 \
  --upernet_width 256 \
  --bands B4 B3 B2 B5 B6 B7 B8 B8A B11 B12 \
  --cvit_channels 0 1 2 3 4 5 6 7 8 9 \
  --enable_sample \
  --channel_dropout_rate 0.2 --min_drop_channels 3 \
  --num_workers 8 --seed 42 \
  --experiment_name cvit_harvey_cd_s2_E9
```

### OSCD RGB training

```bash
torchrun --nproc_per_node=2 train_change.py \
  --backbone cvit-pretrained --encoder_weights chi_vit \
  --encoder_depth 12 --in_channels 3 \
  --dataset_name OSCD \
  --dataset_path /mnt/weka/akhosrovyan/geocrossbench/datasets/x-oscd/ \
  --metadata_path /mnt/weka/akhosrovyan/geocrossbench/datasets/x-oscd/ \
  --img_size 192 --fusion diff \
  --batch_size 8 --max_epochs 100 \
  --lr 1e-4 --weight_decay 0.01 \
  --lr_sched warmup_cosine --warmup_steps 20 --warmup_lr 1e-6 \
  --upernet_width 256 \
  --bands B4 B3 B2 --cvit_channels 2 1 0 \
  --enable_sample --channel_dropout_rate 0.2 --min_drop_channels 1 \
  --num_workers 8 --seed 42 \
  --experiment_name cvit_oscd_cd_rgb_E9
```

### CD Eval — RGB → RGBN

`eval_bands_cd.py` (non-SAR main) `channel_vit_order`: `['B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12','VV','VH']` (uppercase, full names).

```bash
CKPT=/mnt/weka/tgrigoryan/rs_foundation/finetune_ckpts/change_detection/cvit_harvey_cd_rgb_E9/best_model.pth

python eval_bands_cd.py \
  --model_config configs/cvit-pretrained.json \
  --dataset_config configs/harvey.json \
  --checkpoint_path "$CKPT" \
  --size 256 --upernet_width 256 \
  --enable_multiband_input --multiband_channel_count 4 \
  --preserve_rgb_weights --spectral_init_new_channels \
  --training_bands '["B04","B03","B02"]' --new_bands '["B08"]' \
  --bands '[["B04","B03","B02","B08"]]' \
  --filename superset_cvit_harvey_cd_rgbn
```

### CD Eval — S2 → S2+S1 (Harvey)

```bash
CKPT=/mnt/weka/tgrigoryan/rs_foundation/finetune_ckpts/change_detection/cvit_harvey_cd_s2_E9/best_model.pth

python eval_bands_cd.py \
  --model_config configs/cvit-pretrained.json \
  --dataset_config configs/harvey.json \
  --checkpoint_path "$CKPT" \
  --size 256 --upernet_width 256 \
  --enable_multiband_input --multiband_channel_count 12 \
  --preserve_rgb_weights --spectral_init_new_channels \
  --training_bands '["B04","B03","B02","B05","B06","B07","B08","B8A","B11","B12"]' \
  --new_bands '["VV","VH"]' \
  --bands '[["B04","B03","B02","B05","B06","B07","B08","B8A","B11","B12","VV","VH"]]' \
  --filename superset_cvit_harvey_cd_s2s1
```

### OSCD Eval — S2 → S2+S1

```bash
CKPT=/mnt/weka/tgrigoryan/rs_foundation/finetune_ckpts/change_detection/cvit_oscd_cd_rgb_E9/best_model.pth

python eval_bands_cd.py \
  --model_config configs/cvit-pretrained.json \
  --dataset_config configs/oscd.json \
  --checkpoint_path "$CKPT" \
  --size 192 --upernet_width 256 \
  --s2_sar \
  --enable_multiband_input --multiband_channel_count 12 \
  --training_bands '["B04","B03","B02","B05","B06","B07","B08","B8A","B11","B12"]' \
  --new_bands '["VV","VH"]' \
  --filename superset_cvit_oscd_cd_s2s1
```

---

## 1.9 χViT Result Tracking Table

| ID | Pool | HCS | Embed | Dropout | Curr | Gate | EuroSAT RGB | EuroSAT RGBN | Δ | EuroSAT S2+S1 |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | cls | — | — | — | — | — | | | | |
| E1 | cls | ✓ | — | — | — | — | | | | |
| E2 | ch_mean | — | — | — | — | — | | | | |
| E3 | cls | — | freeze | — | — | — | | | | |
| E4 | cls | — | — | 0.2 | — | — | | | | |
| E5 | cls | — | — | — | — | ✓ | | | | |
| E6 | ch_mean | ✓ | — | — | — | — | | | | |
| E7 | ch_mean | ✓ | freeze | — | — | — | | | | |
| E8 | ch_mean | ✓ | freeze | 0.2 | — | — | | | | |
| E9 | ch_mean | ✓ | freeze | 0.2 | ✓ | — | | | | |
| E10 | ch_mean | ✓ | freeze | 0.2 | ✓ | ✓ | | | | |
| E11 | ch_mean | ✓ | reg | 0.2 | ✓ | — | | | | |
| E12 | cls+ch | ✓ | freeze | 0.2 | — | — | | | | |

Positive Δ (RGBN−RGB) = model leverages extra NIR band. **This is the primary success criterion.**

---

---

# PART 2 — Comparison Models

Two experiments per model: **baseline** (no new fixes) and **+dropout** (Generic Fix 2, only applicable fix for standard ViTs).  
These provide the numbers to compare against χViT in the superset table.

---

## 2.1 iBOT (multispectral-pretrained, million_aid checkpoint)

Applicable fix: `--channel_dropout_rate` (Generic 2). HCS/channel-pool/embed-freeze are χViT-only.

### Classification — RGB training

```bash
# Baseline
python train_classifier.py \
  --backbone ibot-B --encoder_weights million_aid \
  --dataset_name m_eurosat --image_size 64 \
  --batch_size 64 --epoch 50 --lr 1e-4 --weight_decay 0.05 \
  --warmup_steps 20 --scheduler cosine \
  --bands B02 B03 B04 \
  --device 1 --seed 42 \
  --experiment_name ibot_eurosat_rgb_baseline

# +channel dropout
python train_classifier.py \
  --backbone ibot-B --encoder_weights million_aid \
  --dataset_name m_eurosat --image_size 64 \
  --batch_size 64 --epoch 50 --lr 1e-4 --weight_decay 0.05 \
  --warmup_steps 20 --scheduler cosine \
  --bands B02 B03 B04 \
  --channel_dropout_rate 0.2 --min_drop_channels 1 \
  --device 1 --seed 42 \
  --experiment_name ibot_eurosat_rgb_dropout
```

### Classification — S2 training

```bash
# Baseline
python train_classifier.py \
  --backbone ibot-B --encoder_weights million_aid \
  --dataset_name m_eurosat --image_size 64 \
  --batch_size 64 --epoch 50 --lr 1e-4 --weight_decay 0.05 \
  --warmup_steps 20 --scheduler cosine \
  --enable_multiband_input --multiband_channel_count 10 \
  --bands B02 B03 B04 B05 B06 B07 B08 B8A B11 B12 \
  --device 1 --seed 42 \
  --experiment_name ibot_eurosat_s2_baseline

# +channel dropout
python train_classifier.py \
  --backbone ibot-B --encoder_weights million_aid \
  --dataset_name m_eurosat --image_size 64 \
  --batch_size 64 --epoch 50 --lr 1e-4 --weight_decay 0.05 \
  --warmup_steps 20 --scheduler cosine \
  --enable_multiband_input --multiband_channel_count 10 \
  --bands B02 B03 B04 B05 B06 B07 B08 B8A B11 B12 \
  --channel_dropout_rate 0.2 --min_drop_channels 3 \
  --device 1 --seed 42 \
  --experiment_name ibot_eurosat_s2_dropout
```

### Eval — RGB → RGBN

```bash
for EXP in ibot_eurosat_rgb_baseline ibot_eurosat_rgb_dropout; do
  CKPT=/mnt/weka/tgrigoryan/rs_foundation/finetune_ckpts/classification/${EXP}/best-model.ckpt
  python eval_bands_cls.py \
    --model_config configs/ibot-B.json \
    --dataset_config configs/m_eurosat.json \
    --checkpoint_path "$CKPT" --img_size 64 \
    --enable_multiband_input --multiband_channel_count 4 \
    --preserve_rgb_weights --spectral_init_new_channels \
    --training_bands '["B02","B03","B04"]' --new_bands '["B08"]' \
    --bands '[["B02","B03","B04","B08"]]' \
    --filename eval_${EXP}_rgbn
done
```

### Eval — S2 → S2+S1

```bash
for EXP in ibot_eurosat_s2_baseline ibot_eurosat_s2_dropout; do
  CKPT=/mnt/weka/tgrigoryan/rs_foundation/finetune_ckpts/classification/${EXP}/best-model.ckpt
  python eval_bands_cls.py \
    --model_config configs/ibot-B.json \
    --dataset_config configs/m_eurosat.json \
    --checkpoint_path "$CKPT" --img_size 64 \
    --enable_multiband_input --multiband_channel_count 12 \
    --preserve_rgb_weights --spectral_init_new_channels \
    --training_bands '["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12"]' \
    --new_bands '["VV","VH"]' \
    --bands '[["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12","VV","VH"]]' \
    --filename eval_${EXP}_s2s1
done
```

### Segmentation — RGB + S2

```bash
# RGB baseline
torchrun --nproc_per_node=2 train_segmenter.py \
  --backbone overlap_ibot-B --encoder_weights million_aid_overlap \
  --encoder_depth 12 --in_channels 3 \
  --dataset_name harvey --dataset_path /mnt/weka/akhosrovyan/geocrossbench/datasets/x-harvey \
  --img_size 96 --classes 2 --loss_type ce \
  --batch_size 8 --max_epochs 100 --lr 1e-4 --weight_decay 0.01 \
  --lr_sched warmup_cosine --warmup_steps 20 --warmup_lr 1e-6 \
  --upernet_width 256 --decoder upernet \
  --bands B2 B3 B4 --num_workers 8 --seed 42 \
  --experiment_name ibot_harvey_seg_rgb_baseline

# S2 baseline
torchrun --nproc_per_node=2 train_segmenter.py \
  --backbone overlap_ibot-B --encoder_weights million_aid_overlap \
  --encoder_depth 12 --in_channels 10 \
  --dataset_name harvey --dataset_path /mnt/weka/akhosrovyan/geocrossbench/datasets/x-harvey \
  --img_size 96 --classes 2 --loss_type ce \
  --batch_size 8 --max_epochs 100 --lr 1e-4 --weight_decay 0.01 \
  --lr_sched warmup_cosine --warmup_steps 20 --warmup_lr 1e-6 \
  --enable_multiband_input --multiband_channel_count 10 \
  --upernet_width 256 --decoder upernet \
  --bands B2 B3 B4 B5 B6 B7 B8 B8A B11 B12 --num_workers 8 --seed 42 \
  --experiment_name ibot_harvey_seg_s2_baseline
```

---

## 2.2 DINOv2 (ImageNet-pretrained, RGB-only)

Only fix applicable: `--channel_dropout_rate`. No spectral knowledge for new bands.  
Image size is auto-rounded to `(size//14)*14`.

### Classification — RGB training

```bash
# Baseline
python train_classifier.py \
  --backbone dinov2 --encoder_weights "" \
  --dataset_name m_eurosat --image_size 64 \
  --batch_size 64 --epoch 50 --lr 1e-4 --weight_decay 0.05 \
  --warmup_steps 20 --scheduler cosine \
  --bands B02 B03 B04 \
  --device 1 --seed 42 \
  --experiment_name dinov2_eurosat_rgb_baseline

# +channel dropout
python train_classifier.py \
  --backbone dinov2 --encoder_weights "" \
  --dataset_name m_eurosat --image_size 64 \
  --batch_size 64 --epoch 50 --lr 1e-4 --weight_decay 0.05 \
  --warmup_steps 20 --scheduler cosine \
  --bands B02 B03 B04 \
  --channel_dropout_rate 0.2 --min_drop_channels 1 \
  --device 1 --seed 42 \
  --experiment_name dinov2_eurosat_rgb_dropout
```

### Classification — S2 training

```bash
python train_classifier.py \
  --backbone dinov2 --encoder_weights "" \
  --dataset_name m_eurosat --image_size 64 \
  --batch_size 64 --epoch 50 --lr 1e-4 --weight_decay 0.05 \
  --warmup_steps 20 --scheduler cosine \
  --enable_multiband_input --multiband_channel_count 10 \
  --bands B02 B03 B04 B05 B06 B07 B08 B8A B11 B12 \
  --device 1 --seed 42 \
  --experiment_name dinov2_eurosat_s2_baseline
```

### Eval — RGB → RGBN

```bash
for EXP in dinov2_eurosat_rgb_baseline dinov2_eurosat_rgb_dropout; do
  CKPT=/mnt/weka/tgrigoryan/rs_foundation/finetune_ckpts/classification/${EXP}/best-model.ckpt
  python eval_bands_cls.py \
    --model_config configs/dinov2.json \
    --dataset_config configs/m_eurosat.json \
    --checkpoint_path "$CKPT" --img_size 64 \
    --enable_multiband_input --multiband_channel_count 4 \
    --preserve_rgb_weights \
    --training_bands '["B02","B03","B04"]' --new_bands '["B08"]' \
    --bands '[["B02","B03","B04","B08"]]' \
    --filename eval_${EXP}_rgbn
done
```

### Eval — S2 → S2+S1

```bash
CKPT=/mnt/weka/tgrigoryan/rs_foundation/finetune_ckpts/classification/dinov2_eurosat_s2_baseline/best-model.ckpt
python eval_bands_cls.py \
  --model_config configs/dinov2.json \
  --dataset_config configs/m_eurosat.json \
  --checkpoint_path "$CKPT" --img_size 64 \
  --enable_multiband_input --multiband_channel_count 12 \
  --preserve_rgb_weights \
  --training_bands '["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12"]' \
  --new_bands '["VV","VH"]' \
  --bands '[["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12","VV","VH"]]' \
  --filename eval_dinov2_eurosat_s2_baseline_s2s1
```

### Segmentation

```bash
# RGB baseline
torchrun --nproc_per_node=2 train_segmenter.py \
  --backbone dinov2 --encoder_weights "" \
  --encoder_depth 12 --in_channels 3 \
  --dataset_name harvey --dataset_path /mnt/weka/akhosrovyan/geocrossbench/datasets/x-harvey \
  --img_size 96 --classes 2 --loss_type ce \
  --batch_size 8 --max_epochs 100 --lr 1e-4 --weight_decay 0.01 \
  --lr_sched warmup_cosine --warmup_steps 20 --warmup_lr 1e-6 \
  --upernet_width 256 --decoder upernet \
  --bands B2 B3 B4 --num_workers 8 --seed 42 \
  --experiment_name dinov2_harvey_seg_rgb_baseline
```

---

## 2.3 DINOv3 (dinov3_vitb16)

Identical structure to DINOv2. Swap `--backbone dinov3_vitb16` and `--model_config configs/dinov3.json`.

### Classification — RGB training

```bash
# Baseline
python train_classifier.py \
  --backbone dinov3_vitb16 --encoder_weights "" \
  --dataset_name m_eurosat --image_size 64 \
  --batch_size 64 --epoch 50 --lr 1e-4 --weight_decay 0.05 \
  --warmup_steps 20 --scheduler cosine \
  --bands B02 B03 B04 \
  --device 1 --seed 42 \
  --experiment_name dinov3_eurosat_rgb_baseline

# +channel dropout
python train_classifier.py \
  --backbone dinov3_vitb16 --encoder_weights "" \
  --dataset_name m_eurosat --image_size 64 \
  --batch_size 64 --epoch 50 --lr 1e-4 --weight_decay 0.05 \
  --warmup_steps 20 --scheduler cosine \
  --bands B02 B03 B04 \
  --channel_dropout_rate 0.2 --min_drop_channels 1 \
  --device 1 --seed 42 \
  --experiment_name dinov3_eurosat_rgb_dropout
```

### Classification — S2 training

```bash
python train_classifier.py \
  --backbone dinov3_vitb16 --encoder_weights "" \
  --dataset_name m_eurosat --image_size 64 \
  --batch_size 64 --epoch 50 --lr 1e-4 --weight_decay 0.05 \
  --warmup_steps 20 --scheduler cosine \
  --enable_multiband_input --multiband_channel_count 10 \
  --bands B02 B03 B04 B05 B06 B07 B08 B8A B11 B12 \
  --device 1 --seed 42 \
  --experiment_name dinov3_eurosat_s2_baseline
```

### Eval — RGB → RGBN

```bash
for EXP in dinov3_eurosat_rgb_baseline dinov3_eurosat_rgb_dropout; do
  CKPT=/mnt/weka/tgrigoryan/rs_foundation/finetune_ckpts/classification/${EXP}/best-model.ckpt
  python eval_bands_cls.py \
    --model_config configs/dinov3.json \
    --dataset_config configs/m_eurosat.json \
    --checkpoint_path "$CKPT" --img_size 64 \
    --enable_multiband_input --multiband_channel_count 4 \
    --preserve_rgb_weights \
    --training_bands '["B02","B03","B04"]' --new_bands '["B08"]' \
    --bands '[["B02","B03","B04","B08"]]' \
    --filename eval_${EXP}_rgbn
done
```

### Segmentation

```bash
torchrun --nproc_per_node=2 train_segmenter.py \
  --backbone dinov3_vitb16 --encoder_weights "" \
  --encoder_depth 12 --in_channels 3 \
  --dataset_name harvey --dataset_path /mnt/weka/akhosrovyan/geocrossbench/datasets/x-harvey \
  --img_size 96 --classes 2 --loss_type ce \
  --batch_size 8 --max_epochs 100 --lr 1e-4 --weight_decay 0.01 \
  --lr_sched warmup_cosine --warmup_steps 20 --warmup_lr 1e-6 \
  --upernet_width 256 --decoder upernet \
  --bands B2 B3 B4 --num_workers 8 --seed 42 \
  --experiment_name dinov3_harvey_seg_rgb_baseline
```

---

## 2.4 ViT-B (timm_vit-b, ImageNet)

Same as DINOv2/v3. Only channel dropout applies.

### Classification — RGB training

```bash
# Baseline
python train_classifier.py \
  --backbone timm_vit-b --encoder_weights "" \
  --dataset_name m_eurosat --image_size 64 \
  --batch_size 64 --epoch 50 --lr 1e-4 --weight_decay 0.05 \
  --warmup_steps 20 --scheduler cosine \
  --bands B02 B03 B04 \
  --device 1 --seed 42 \
  --experiment_name vitb_eurosat_rgb_baseline

# +channel dropout
python train_classifier.py \
  --backbone timm_vit-b --encoder_weights "" \
  --dataset_name m_eurosat --image_size 64 \
  --batch_size 64 --epoch 50 --lr 1e-4 --weight_decay 0.05 \
  --warmup_steps 20 --scheduler cosine \
  --bands B02 B03 B04 \
  --channel_dropout_rate 0.2 --min_drop_channels 1 \
  --device 1 --seed 42 \
  --experiment_name vitb_eurosat_rgb_dropout
```

### Classification — S2 training

```bash
python train_classifier.py \
  --backbone timm_vit-b --encoder_weights "" \
  --dataset_name m_eurosat --image_size 64 \
  --batch_size 64 --epoch 50 --lr 1e-4 --weight_decay 0.05 \
  --warmup_steps 20 --scheduler cosine \
  --enable_multiband_input --multiband_channel_count 10 \
  --bands B02 B03 B04 B05 B06 B07 B08 B8A B11 B12 \
  --device 1 --seed 42 \
  --experiment_name vitb_eurosat_s2_baseline
```

### Eval — RGB → RGBN

```bash
for EXP in vitb_eurosat_rgb_baseline vitb_eurosat_rgb_dropout; do
  CKPT=/mnt/weka/tgrigoryan/rs_foundation/finetune_ckpts/classification/${EXP}/best-model.ckpt
  python eval_bands_cls.py \
    --model_config configs/timm_vit-b.json \
    --dataset_config configs/m_eurosat.json \
    --checkpoint_path "$CKPT" --img_size 64 \
    --enable_multiband_input --multiband_channel_count 4 \
    --preserve_rgb_weights \
    --training_bands '["B02","B03","B04"]' --new_bands '["B08"]' \
    --bands '[["B02","B03","B04","B08"]]' \
    --filename eval_${EXP}_rgbn
done
```

### Eval — S2 → S2+S1

```bash
CKPT=/mnt/weka/tgrigoryan/rs_foundation/finetune_ckpts/classification/vitb_eurosat_s2_baseline/best-model.ckpt
python eval_bands_cls.py \
  --model_config configs/timm_vit-b.json \
  --dataset_config configs/m_eurosat.json \
  --checkpoint_path "$CKPT" --img_size 64 \
  --enable_multiband_input --multiband_channel_count 12 \
  --preserve_rgb_weights \
  --training_bands '["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12"]' \
  --new_bands '["VV","VH"]' \
  --bands '[["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12","VV","VH"]]' \
  --filename eval_vitb_eurosat_s2_baseline_s2s1
```

### Segmentation

```bash
torchrun --nproc_per_node=2 train_segmenter.py \
  --backbone timm_vit-b --encoder_weights "" \
  --encoder_depth 12 --in_channels 3 \
  --dataset_name harvey --dataset_path /mnt/weka/akhosrovyan/geocrossbench/datasets/x-harvey \
  --img_size 96 --classes 2 --loss_type ce \
  --batch_size 8 --max_epochs 100 --lr 1e-4 --weight_decay 0.01 \
  --lr_sched warmup_cosine --warmup_steps 20 --warmup_lr 1e-6 \
  --upernet_width 256 --decoder upernet \
  --bands B2 B3 B4 --num_workers 8 --seed 42 \
  --experiment_name vitb_harvey_seg_rgb_baseline
```

---

## 2.5 Comparison Model Summary Table

Fill in after each eval. The goal: χViT (with best E-ID) shows larger positive Δ than all comparison models.

| Model | Variant | EuroSAT RGB | EuroSAT RGBN | Δ RGB→RGBN | EuroSAT S2 | EuroSAT S2+S1 | Δ S2→S2+S1 |
|---|---|---|---|---|---|---|---|
| χViT | E0 baseline | | | | | | |
| χViT | E9 best | | | | | | |
| χViT | E10 full | | | | | | |
| iBOT | baseline | | | | | | |
| iBOT | +dropout | | | | | | |
| DINOv2 | baseline | | | | | | |
| DINOv2 | +dropout | | | | | | |
| DINOv3 | baseline | | | | | | |
| DINOv3 | +dropout | | | | | | |
| ViT-B | baseline | | | | | | |
| ViT-B | +dropout | | | | | | |

---

---

# PART 3 — Critical Pitfalls

| # | Pitfall | Fix |
|---|---|---|
| 1 | Band-name casing differs per script | cls: `B02 VV VH`; seg: `B2 vv vh`; cd: `B02 VV VH` |
| 2 | χViT bands reversed | `B04 B03 B02` for χViT; `B02 B03 B04` for all others |
| 3 | Segmenter: `--enable_multiband_input`; CD: `--enable_multiband` | Never mix these flags |
| 4 | Seg/CD require `torchrun`; Classification requires plain `python` | |
| 5 | DINOv2/v3 round image size to `(size//14)*14` automatically | Pass `64` or `120`; script adjusts |
| 6 | `--preserve_rgb_weights` requires both `--training_bands` and `--new_bands` | Always pass all three together |
| 7 | `--multiband_channel_count` at eval = total bands in eval list | RGB→RGBN: 4; S2→S2+S1: 12 |
| 8 | Harvey CD split lists are hardcoded in `train_change.py` | No `--metadata_path` needed for Harvey CD |
| 9 | OSCD SAR-only eval uses `--sar`; S2+SAR superset uses `--s2_sar` | Two different code paths |
| 10 | Classification checkpoint: `.ckpt` (Lightning); Seg/CD: `.pth` (state_dict) | |
| 11 | `--pooling_mode` at eval must match training | Pass same value used in training command |

---

# PART 4 — Result Files

All eval scripts write to `cwd` and `./eval_outs/`:

```
eval_<exp>_rgbn.txt            ← RGB→RGBN per-band accuracy/F1/mIoU/bIoU
eval_<exp>_s2s1.txt            ← S2→S2+S1
./eval_outs/<exp>/results.npy  ← full dict keyed by checkpoint path
```

Superset AVG = `mean(RGBN score, S2+S1 score)` — matches the paper table column.
