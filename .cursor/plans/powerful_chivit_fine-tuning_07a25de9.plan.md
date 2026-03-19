---
name: Cross-Band Robust Fine-tuning
overview: Improve cross-band generalization when fine-tuning on a SUBSET of bands and evaluating on a SUPERSET or non-overlapping set. Focus on χViT first, with generic fixes for all models.
todos:
  - id: zero-init
    content: "Generic Fix 1: Zero-initialize new channel weights in adapt_*_preserve_rgb — guarantees no degradation at eval (all models)"
    status: completed
  - id: channel-dropout
    content: "Generic Fix 2: Add ChannelDropout within training bands — model robust to variable channel counts"
    status: completed
  - id: hcs-finetune
    content: "χViT Fix A: Re-enable HCS during fine-tuning on training bands — enable_sample=True"
    status: completed
  - id: channel-pool
    content: "χViT Fix B: Per-channel mean pooling for classification — channel-count-invariant output"
    status: completed
  - id: embed-preserve
    content: "χViT Fix C: Freeze/regularize unused channel embeddings during subset fine-tuning"
    status: completed
  - id: channel-gate
    content: "χViT Fix D: Learnable per-channel gates for controlled new-channel contribution at eval"
    status: completed
  - id: curriculum
    content: "Enhancement: Curriculum channel sampling — anneal dropout aggressiveness over epochs"
    status: completed
isProject: false
---

# Cross-Band Robust Fine-tuning

## Constraint

**You fine-tune on a SUBSET of bands. You evaluate on a SUPERSET or non-overlapping set. You NEVER train on the eval bands.**

## Goal

A model fine-tuned on RGB should perform BETTER on RGBN than on RGB alone. RGBN has strictly more information. The model must be able to leverage the extra NIR band at eval time, even though it never saw NIR during fine-tuning.

## Focus Example

Train on RGB (3 bands) -> Eval on RGBN (4 bands). All fixes are generic and cover:

- **Superset**: RGB->RGBN, RGB->S2, S2->S2+S1
- **No-overlap**: RGB->S1, S2->S1, RGB->non-RGB S2 bands

---

## Root Cause: Fine-tuning Destroys Pretrained Multispectral Knowledge

### The problem

Many evaluated models are pretrained on multispectral data:

- **χViT**: iBOT on 12 bands (S2+S1), shared Conv3d + 12 channel embeddings
- **iBOT** (some checkpoints): pretrained on S2 multispectral
- **Prithvi**: pretrained on 6 HLS bands
- **CROMA**: pretrained on S2+S1
- **DOFA, TerraFM, AnySat**: pretrained on various multispectral data

These models already have meaningful pretrained weights for NIR and other bands. The pretrained Conv2d/Conv3d learned to process NIR, the pretrained channel embeddings encode NIR identity.

But when fine-tuning on RGB, `adapt_rgb_conv_layer_to_multiband` (line 27-30 of [classifier_utils.py](rs_finetune/classifier_utils.py)) **destroys all of this**:

```python
averaged_weights = old_weights.mean(dim=1, keepdim=True)  # average ALL channel weights
new_weights = averaged_weights.repeat(1, new_in_channels, 1, 1)  # identical copies
new_weights = new_weights / new_in_channels
```

This replaces 12 distinct, pretrained channel-specific filters with 3 identical averaged copies. The NIR filter, the SAR filters — all gone.

At eval on RGBN, `adapt_rgb_conv_layer_to_multiband_preserve_rgb` (line 74) initializes the NIR weight as `mean(fine-tuned R, G, B weights)` — a meaningless average that has nothing to do with NIR.

**Critical**: New channel weights must be **non-zero** so the model can use their information. Zero-init would ignore the data entirely. We need spectrally-informed initialization instead of mean or zero.

### What SHOULD happen

During RGB fine-tuning, keep the conv at pretrained size (e.g., 12 channels). Only train the RGB channel weights. Freeze the rest. At eval on RGBN, the pretrained NIR weight is still there — intact, meaningful, ready to process NIR data.

---

## Part 1: Generic Fixes (ALL multispectral-pretrained models)

### Generic Fix 1: Preserve pretrained conv weights during fine-tuning

**Files:** [classifier_utils.py](rs_finetune/classifier_utils.py), [train_classifier.py](rs_finetune/train_classifier.py)

**Instead of resizing the conv to match training bands, keep it at pretrained size and manage channels via input construction + weight freezing.**

For standard ViTs (Conv2d patch embed) fine-tuned on RGB:

1. Keep `patch_embed.proj` at pretrained channel count (e.g., 12)
2. During training, construct full-sized input: place RGB data in the correct channel slots, zero the rest
3. Freeze conv weights for non-training channels (prevent weight decay from degrading them)
4. At eval on RGBN: place RGB+NIR data in correct channel slots. The NIR weight is the pretrained one.

```python
def freeze_non_training_conv_weights(conv: nn.Conv2d, training_channel_indices: list[int]):
    mask = torch.ones(conv.in_channels, dtype=torch.bool)
    mask[training_channel_indices] = False
    conv.weight.register_hook(
        lambda grad: grad * (~mask)[None, :, None, None].to(grad.device).float()
    )
```

This approach:

- Preserves ALL pretrained channel weights (not just RGB)
- RGB weights get gradients and fine-tune normally
- NIR/SAR/etc. weights stay at pretrained values — ready for eval
- No conv resizing, no weight averaging, no destruction

**For RGB-pretrained models** (ImageNet iBOT, DINOv2, DINOv3, ViT-B — 3-channel conv):

These models have no pretrained weight for the new channel(s). When expanding, we need **non-zero weights** so the model can use their information. Use spectrally-informed initialization:

- **Optical→optical** (e.g., RGB→RGBN): weighted average with higher weight to spectrally closest bands — e.g. NIR: `0.6*W_red + 0.3*W_green + 0.1*W_blue`
- **S2→S2+S1** (optical→SAR): VV, VH have no spectral proximity to optical — use **equal weights** over all 10 S2 bands
- **No-overlap** (e.g., RGB→S1): equal weights over training bands

The real power of this fix is for multispectral-pretrained models where preserving the pretrained NIR weight means the model CAN process NIR meaningfully at eval.

**Eval-time adaptation** (when conv expansion is needed):

Modify `adapt_rgb_conv_layer_to_multiband_preserve_rgb` to accept spectral weights for new channels. For each new channel, compute a **weighted average** of training band weights:

- **Optical→optical** (e.g., RGB→RGBN): weights proportional to spectral proximity (NIR/B08: 0.6*red + 0.3*green + 0.1*blue)
- **S2→S2+S1** (optical→SAR): VV and VH have no spectral proximity to optical bands — use **equal weights** (mean of all 10 S2 bands) as the best available proxy
- **RGB→S1** (no-overlap): same — equal weights over training bands

Define band proximity weights in [utils.py](rs_finetune/utils.py); for SAR bands, fall back to uniform weights over the training set.

**CLI args:**

- `--preserve_pretrained_conv` (flag): keep conv at pretrained size, freeze non-training weights
- `--training_channel_indices` (int list): which channels of the pretrained conv correspond to training bands
- `--spectral_init_new_channels` (flag): use spectrally-informed init for new channels (vs mean)

### Generic Fix 2: Channel dropout within training bands

**Files:** [train_classifier.py](rs_finetune/train_classifier.py), [train_segmenter.py](rs_finetune/train_segmenter.py), [train_change.py](rs_finetune/train_change.py)

During fine-tuning on the TRAINING SUBSET (e.g., RGB), randomly zero-out channels:

```python
class ChannelDropout(nn.Module):
    def __init__(self, p: float = 0.2, min_channels: int = 1):
        super().__init__()
        self.p = p
        self.min_channels = min_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x
        B, C, H, W = x.shape
        keep = max(self.min_channels, int(C * (1 - self.p)))
        mask = torch.zeros(B, C, 1, 1, device=x.device)
        for b in range(B):
            idx = torch.randperm(C, device=x.device)[:keep]
            mask[b, idx] = C / keep
        return x * mask
```

Why this helps:

- Model sees variable channel counts during training (1, 2, or 3 when training on RGB)
- Features/CLS token become robust to channel count changes
- At eval on RGBN (4 channels), the transition from 3 to 4 is smoother
- Prevents the model from building rigid dependencies on all training channels being present

**CLI args:**

- `--channel_dropout_rate` (float, default 0.0): dropout probability per channel
- `--min_drop_channels` (int, default 1): minimum channels to keep

---

## Part 2: χViT-Specific Enhancements

χViT has the strongest position for cross-band generalization because its architecture natively supports variable channels. These fixes unlock that potential.

### χViT Fix A: Re-enable HCS during fine-tuning on training bands

**Files:** [chi_vit.py](rs_finetune/change_detection_pytorch/encoders/chi_vit.py), [train_classifier.py](rs_finetune/train_classifier.py)

HCS is χViT's native channel dropout. Currently disabled at fine-tuning time (`enable_sample` commented out in `chi_vit_encoders`, line 415).

Re-enable it. When fine-tuning on RGB (3 bands), HCS samples subsets of {R, G, B}:

- Batch 1: model sees {R, B} (2 channels, 2*H'*W' tokens)
- Batch 2: model sees {G} (1 channel, 1*H'*W' tokens)
- Batch 3: model sees {R, G, B} (3 channels, 3*H'*W' tokens)

This is more powerful than Generic Fix 2 because:

- Sequence length actually changes (not zero-masking within fixed-length sequence)
- Channel embeddings are properly re-indexed
- CLS token learns to produce consistent features for variable token counts
- At eval with 4 tokens (RGBN), the CLS token handles it — it's trained for variable counts

Existing infrastructure in `PatchEmbedPerChannel.forward()` (lines 107-117):

```python
if self.training and self.enable_sample:
    Cin_new = random.randint(1, Cin)
    channels = random.sample(range(Cin), k=Cin_new)
    x = x[:, channels, :, :]
    channel_idxs = channels
```

**CLI args:**

- `--enable_sample` (flag, already exists): enables HCS
- `--min_sample_channels` (int, default 1): minimum channels sampled

### χViT Fix B: Channel-count-invariant classification pooling

**File:** [chi_vit.py](rs_finetune/change_detection_pytorch/encoders/chi_vit.py), `forward()` method

The CLS token is the bottleneck for superset eval. Train on RGB (588 tokens), eval on RGBN (784 tokens) — 33% more tokens shift the CLS representation.

Segmentation already handles this correctly (line 362-363):

```python
feat = norm_x[:, 1:].reshape(B, -1, hw_shape[0], hw_shape[1], Cout).mean(dim=1)
```

Apply the same per-channel mean pooling to classification:

```python
# Replace: return x[:, 0, :]
n_ch = len(channel_idxs) if isinstance(channel_idxs, (list, tuple)) else channel_idxs.shape[-1]
patch_tokens = x[:, 1:]
B, N, D = patch_tokens.shape
spatial = N // n_ch
per_channel = patch_tokens.reshape(B, n_ch, spatial, D)
return per_channel.mean(dim=2).mean(dim=1)  # B, D — channel-count-invariant
```

Adding NIR at eval adds one more element to the outer mean. The spatial features per channel are stable; the mean over channels normalizes the count change.

**CLI args:**

- `--pooling_mode` (str): `cls` (default, backward compat), `channel_mean`, `cls+channel_mean`

### χViT Fix C: Preserve unused channel embeddings

**Files:** [chi_vit.py](rs_finetune/change_detection_pytorch/encoders/chi_vit.py), [train_classifier.py](rs_finetune/train_classifier.py)

During RGB fine-tuning, only 3 of 12 channel embeddings receive gradients. The other 9 (including NIR) are subject to weight decay, which pushes them toward zero — destroying their pretrained values.

At eval on RGBN, the NIR embedding has degraded. The transformer can't properly identify NIR tokens.

**Fix options (from most to least restrictive):**

1. **Selective freeze** (recommended): freeze embeddings for bands NOT in the training set:

```python
   for idx in range(self.in_chans):
       if idx not in training_channel_idxs:
           self.channel_embed.data[:, :, idx].requires_grad_(False)
   

```

1. **Freeze all** (`--frozen_channel_embed` already exists): all embeddings stay at pretrained values
2. **L2 regularization toward pretrained values** (softest):

```python
   embed_reg = (self.channel_embed - self.pretrained_channel_embed).pow(2).mean()
   loss += lambda_reg * embed_reg
   

```

**CLI args:**

- `--freeze_unused_channel_embeds` (flag): freeze non-training-band embeddings
- `--channel_embed_reg_lambda` (float, default 0.0): L2 regularization strength
- `--frozen_channel_embed` (flag, already exists): freeze ALL embeddings

### χViT Fix D: Learnable per-channel gates

**File:** [chi_vit.py](rs_finetune/change_detection_pytorch/encoders/chi_vit.py)

Add a scalar gate per channel that controls contribution strength:

```python
self.channel_gate = nn.Parameter(torch.zeros(in_chans))

# In forward, after patch embedding + channel embedding:
gates = self.channel_gate[channel_idxs].sigmoid()
x = x * gates[None, None, :, None, None]
```

Initialize all gates to large positive (sigmoid ~ 1.0). During RGB fine-tuning, only R/G/B gates get gradients. At eval, NIR gate stays at pretrained value (~1.0), giving full contribution.

**CLI args:**

- `--enable_channel_gate` (flag): add per-channel gates

### Enhancement: Curriculum channel sampling

Anneal HCS/channel dropout aggressiveness over epochs:

```python
min_ch = max(1, int(n_channels * epoch_progress * 0.5))
sampled = random.randint(min_ch, n_channels)
```

Early: aggressive (1 to C channels) — forces per-channel independence.
Late: mild (C//2 to C) — fine-tunes cross-channel synergy.

**CLI args:**

- `--curriculum_sampling` (flag): enable annealed sampling

---

## Summary: How RGBN > RGB is Achieved

### For χViT (primary focus):

1. Shared Conv3d already processes NIR correctly (same filter as RGB)
2. NIR channel embedding is pretrained (iBOT on 12 bands) and PRESERVED during fine-tuning (Fix C)
3. HCS during fine-tuning (Fix A) teaches the transformer to handle variable token counts
4. Per-channel pooling (Fix B) makes the output structurally invariant to channel count
5. At eval: NIR tokens carry real information, transformer attends to them, and the pooled output integrates them cleanly

### For multispectral-pretrained standard ViTs:

1. Pretrained conv has meaningful NIR weights from multispectral pretraining
2. During RGB fine-tuning, NIR weights are PRESERVED (frozen, not destroyed) (Generic Fix 1)
3. Channel dropout (Generic Fix 2) makes features robust to channel count changes
4. At eval: NIR data passes through the pretrained NIR filter, producing meaningful activations

### For RGB-pretrained standard ViTs (ImageNet DINOv2, etc.):

1. No pretrained NIR weight exists — no basis for processing NIR
2. Channel dropout helps with count-robustness
3. Realistically: RGBN ~ RGB (these models cannot leverage NIR without retraining)
4. This is an inherent architectural limitation, not something fine-tuning can fix

---

## Applicability Matrix


| Fix                               | χViT            | Multispectral ViTs                | RGB-only ViTs          |
| --------------------------------- | --------------- | --------------------------------- | ---------------------- |
| Preserve conv weights (Generic 1) | N/A (native)    | Yes — keeps pretrained NIR filter | N/A (no NIR filter)    |
| Channel dropout (Generic 2)       | Use HCS instead | Yes                               | Yes — count robustness |
| HCS (χViT A)                      | Yes             | N/A                               | N/A                    |
| Channel-mean pool (χViT B)        | Yes             | N/A                               | N/A                    |
| Embed preservation (χViT C)       | Yes             | N/A                               | N/A                    |
| Channel gating (χViT D)           | Yes             | N/A                               | N/A                    |
| Curriculum (Enhancement)          | Yes             | Yes                               | Yes                    |


---

## Configuration Interface

Every fix is opt-in via CLI arguments. All defaults preserve backward compatibility.

### New arguments (all training scripts)


| Argument                       | Type     | Default | Fix       | Description                                                                                          |
| ------------------------------ | -------- | ------- | --------- | ---------------------------------------------------------------------------------------------------- |
| `--preserve_pretrained_conv`   | flag     | off     | Generic 1 | Keep conv at pretrained size, freeze non-training weights                                            |
| `--training_channel_indices`   | int list | auto    | Generic 1 | Which pretrained conv channels map to training bands                                                 |
| `--spectral_init_new_channels` | flag     | off     | Generic 1 | Use spectrally-informed init for new channels at eval (weighted avg, higher weight to closest bands) |
| `--channel_dropout_rate`       | float    | 0.0     | Generic 2 | Probability of dropping each channel (0 = disabled)                                                  |
| `--min_drop_channels`          | int      | 1       | Generic 2 | Min channels to keep when dropout active                                                             |


### New arguments (χViT-specific)


| Argument                         | Type  | Default | Fix         | Description                                      |
| -------------------------------- | ----- | ------- | ----------- | ------------------------------------------------ |
| `--pooling_mode`                 | str   | `cls`   | χViT B      | `cls`, `channel_mean`, `cls+channel_mean`        |
| `--freeze_unused_channel_embeds` | flag  | off     | χViT C      | Freeze embeddings for non-training bands         |
| `--channel_embed_reg_lambda`     | float | 0.0     | χViT C      | L2 regularization toward pretrained embed values |
| `--enable_channel_gate`          | flag  | off     | χViT D      | Add learnable per-channel gates                  |
| `--curriculum_sampling`          | flag  | off     | Enhancement | Anneal sampling aggressiveness over epochs       |
| `--min_sample_channels`          | int   | 1       | χViT A      | Min channels when HCS is active                  |


### Existing arguments (already in codebase)


| Argument                 | Fix    | Notes                                             |
| ------------------------ | ------ | ------------------------------------------------- |
| `--enable_sample`        | χViT A | Enables HCS (already exists, line 296)            |
| `--frozen_channel_embed` | χViT C | Freezes ALL embeddings (already exists, line 279) |
| `--bands`                | all    | Training band set                                 |


### Example: Maximum cross-band robustness (χViT)

```bash
python train_classifier.py \
  --backbone cvit-pretrained \
  --bands B04 B03 B02 \
  --enable_sample \
  --min_sample_channels 1 \
  --pooling_mode channel_mean \
  --freeze_unused_channel_embeds
```

### Example: Multispectral-pretrained iBOT with preserved conv

```bash
python train_classifier.py \
  --backbone ibot-B \
  --encoder_weights million_aid_fa \
  --bands B04 B03 B02 \
  --preserve_pretrained_conv \
  --channel_dropout_rate 0.2
```

---

## Files to Modify

- [classifier_utils.py](rs_finetune/classifier_utils.py) — `ChannelDropout` module, preserve-conv logic, weight freezing
- [chi_vit.py](rs_finetune/change_detection_pytorch/encoders/chi_vit.py) — pooling mode, channel gating, min_sample_channels
- [train_classifier.py](rs_finetune/train_classifier.py) — new CLI args, wire fixes into `Classifier`
- [train_segmenter.py](rs_finetune/train_segmenter.py) — new CLI args, channel dropout, preserve-conv
- [train_change.py](rs_finetune/train_change.py) — new CLI args, channel dropout, preserve-conv
- [eval_bands_cls.py](rs_finetune/eval_bands_cls.py) — `--pooling_mode`
- [eval_bands_seg.py](rs_finetune/eval_bands_seg.py) — eval-time channel mapping
- [eval_bands_cd.py](rs_finetune/eval_bands_cd.py) — eval-time channel mapping

---

## Implementation Order

1. **χViT Fix A** (HCS) — enable existing `--enable_sample`, add `--min_sample_channels`
2. **χViT Fix B** (per-channel pooling) — `--pooling_mode` in chi_vit.py
3. **χViT Fix C** (embed preservation) — `--freeze_unused_channel_embeds`
4. **Generic Fix 1** (preserve conv) — `--preserve_pretrained_conv` + weight freezing
5. **Generic Fix 2** (ChannelDropout) — `--channel_dropout_rate` module
6. **χViT Fix D** (channel gating) — `--enable_channel_gate`
7. **Curriculum** — `--curriculum_sampling`

