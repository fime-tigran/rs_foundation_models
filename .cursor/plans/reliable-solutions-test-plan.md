# Reliable-Solutions Test Plan (TDD)

**Status (2026-04-25).** Test-first specification for the Reliable-Core +
optional add-ons in `reliable-solutions.md`. Every technique gets a dedicated
test suite; every test is written and watched-to-fail **before** any
production code lands.

**Discipline.** Follows `superpowers:test-driven-development`:

1. RED — write one minimal test for one behavior.
2. Verify failure (must fail for the *right* reason: feature missing).
3. GREEN — minimal code to pass, nothing more.
4. Verify pass.
5. Refactor if helpful; keep green.
6. Next test.

**Iron law.** No production code (module, class, flag wiring) without a
failing test first. If a commit contains new production code without a test
that failed on the previous commit, we back it out and redo it.

## 1. Infrastructure to add

### 1.1 Directory layout

```
rs_finetune/tests/reliable/
    __init__.py
    conftest.py                      # shared fixtures (see §1.2)
    # universal core
    test_last_n_lora.py              # #29
    test_oplora.py                   # #16
    test_lora_null_init.py           # #20
    test_hard_channel_mask.py        # #2
    test_cdsd.py                     # #9
    test_aph_head.py                 # #32
    test_react_clip.py               # #5
    # multispectral core
    test_hopfield_memory.py          # #11
    test_lsmm_aux_head.py            # #23
    test_srf_bias.py                 # #24
    # post-training
    test_mera_merge.py               # #21
    # optional add-ons
    test_tc_caf.py                   # #7
    test_bpsg.py                     # #18
    test_adapt_align.py              # #27
    test_mcse_head.py                # #28
    test_nci_invariance.py           # #14 (head + loss)
    test_ch_rs_ft.py                 # #15
    test_imputation.py               # #12 / #26
    # integration + portability
    test_r_grid.py                   # R0, R1, R2, R9, R13 smoke + matched-pair
    test_portability.py              # χViT / TerraFM / DOFA / DINOv2 / DINOv3 shim
```

Run them with `./run_tests.sh -k reliable` (extends existing runner).

### 1.2 Shared fixtures (`conftest.py`)

Keep the existing top-level `tests/conftest.py` fixtures (seed), and add
reliable-specific fixtures here.

| Fixture | Provides | Notes |
|---|---|---|
| `tiny_vit` | A small ViT with `depth=4, embed_dim=64, num_heads=4` | Fast TDD cycles (< 50ms / forward) |
| `tiny_chivit` | Existing `chivit_tiny(n_channels=4)` helper | Re-export for convenience |
| `tiny_mock_multispec_backbone` | A tiny ViT wrapper that accepts `(B, C, H, W)` and returns `(B, C, D)` per-channel features | Stand-in for χViT / TerraFM / DOFA pipeline output |
| `tiny_mock_rgb_only_backbone` | A tiny ViT accepting `(B, 3, H, W)` → `(B, D)` global feature | Stand-in for DINOv2 / DINOv3 |
| `training_channel_ids` | `[0, 1, 2]` (RGB) | Default subset |
| `eval_superset_channel_ids` | `[0, 1, 2, 6]` (+B08 index) | Superset case |
| `eval_no_overlap_channel_ids` | `[10, 11]` (VV, VH) | No-overlap case |
| `synthetic_multispec_batch` | `(B=4, C, H=32, W=32)` torch.randn; parameterized on C | |
| `synthetic_per_channel_features` | `(B=4, C, D=64)` torch.randn; parameterized on C | Represents post-embedding-generator output |
| `synthetic_labels` | `(B=4,)` int64 in `[0, num_classes)` | |
| `frozen_pretrained_weight` | `torch.randn(d_out, d_in, requires_grad=False)` | For OPLoRA / LoRA-Null tests |
| `tmp_artifact_dir` | `tmp_path` wrapper for caching offline artifacts (SVD, Gaussians, prototypes) | Uses pytest's `tmp_path` |

### 1.3 Tolerances

- `atol=1e-6` for "exactly preserved" claims on float32
- `atol=1e-5` for LayerNorm'd features
- `atol=1e-4` for outputs after multiple matmuls
- SVD checks: compare singular values with `rtol=1e-5`
- Monte-Carlo tests use explicit `torch.manual_seed` + `rtol=0.05` with
  `n_mc ≥ 50`

### 1.4 Mock models for portability

Rather than import TerraFM / DOFA / DINOv2 weights (slow, network-dependent),
create **tiny parametric stand-ins** that mimic each model's I/O contract:

- `MockChiViT`: `forward(x, channel_ids)` → per-channel tokens
- `MockTerraFM`: `forward(x)` with per-modality patches → per-channel features
- `MockDOFA`: `forward(x, wavelengths)` → per-channel features via hypernet
- `MockDINOv2`: `forward(x)` → global CLS feature (RGB-only)
- `MockDINOv3`: same as DINOv2

Each has ~50 K params and < 10 ms forward. Portability tests run all
techniques against all five mocks.

## 2. Ordering strategy — what to TDD first

Build the test suite and implementation in dependency order so each new
component has working pieces below it:

1. **Fixtures + mock models** (`conftest.py`) — no production code, but tests
   for fixtures themselves (smoke forwards, shape checks).
2. **#29 LastN-LoRA** (pure LoRA placement; foundational for most other LoRA
   techniques).
3. **#16 OPLoRA** (projection on top of LastN-LoRA).
4. **#20 LoRA-Null Init** (init strategy; composes with #16).
5. **#2 Hard Channel Mask** (gate on top of LoRA).
6. **#9 CDSD** (loss + EMA teacher; independent of above but uses fixtures).
7. **#32 APH** (head).
8. **#5 ReAct** (eval-only hook; no training coupling).
9. **#11 Hopfield Memory** (head-adjacent module).
10. **#23 LSMM Aux Head** (aux head).
11. **#24 SRF-Biased Attention** (bias inside #32 APH — needs APH first).
12. **#21 MERA** (post-training; operates on trained LoRA).
13. Optional add-ons: #7, #18, #27, #28, #14, #15, #12/#26 — in any order.
14. **Integration tests** (`test_r_grid.py`) — last; depends on everything.
15. **Portability tests** (`test_portability.py`) — last.

## 3. Per-technique test suites

Each test below is one `def test_*` in its respective file. Each has a
"Fails because" hint making the TDD red-phase check explicit.

### 3.1 `test_last_n_lora.py` (#29)

| Test | Behavior | Fails because |
|---|---|---|
| `test_last_n_zero_attaches_no_adapters` | With `lora_last_n=0`, zero new params added to model. | `LastNLoRAWrap` doesn't exist or always attaches. |
| `test_last_n_4_attaches_to_blocks_8_to_11_only` | With `depth=12, N=4`, `trainable_parameters` on blocks 0–7 unchanged; blocks 8–11 have added LoRA params. | Wrapper attaches everywhere or wrong range. |
| `test_last_n_attaches_in_reverse_index_order` | Using indices [-N:] (not [:N]) → block 0 stays pretrained. | Slicing bug. |
| `test_last_n_preserves_forward_when_lora_zero_init` | At init, `model_with_lora(x)` == `model(x)` (within atol). | LoRA not zero-init. |
| `test_last_n_greater_than_depth_attaches_all` | `N=12` on 12-block model attaches everywhere (no IndexError). | Bad slicing. |
| `test_last_n_rank_respected` | With `lora_rank=8`, each LoRA has A shape (d,8), B shape (8,d). | Rank misconfigured. |
| `test_last_n_gradient_flows_only_to_attached_blocks` | After one backward, `grad is None` on block 0 params; `grad is not None` on block 11 LoRA params. | Backbone not frozen. |

### 3.2 `test_oplora.py` (#16)

| Test | Behavior | Fails because |
|---|---|---|
| `test_oplora_zero_init_matches_base` | At init, `OPLoRA(W_base)(x) == W_base @ x`. | LoRA B not zero. |
| `test_oplora_projector_shapes` | `P_L: (d_out,d_out)`, `P_R: (d_in,d_in)` from SVD of a `(d_out,d_in)` weight. | Shape mismatch. |
| `test_oplora_projector_is_idempotent` | `P @ P == P` (within atol). | Projector built wrong. |
| `test_oplora_projector_orthogonal_to_top_k` | `P_L @ U_k ≈ 0` and `V_kᵀ @ P_R ≈ 0` where (U_k, V_k) are top-k singular vectors. | SVD misused. |
| `test_oplora_preserves_top_k_after_arbitrary_update` | After a forced large update to `A, B`, top-k singular triples of `W_base + P_L · B @ A · P_R` equal those of `W_base` (within rtol=1e-5). | Projection not applied on forward. |
| `test_oplora_allows_change_in_non_top_k_directions` | Apply `x` in null-space direction; output diff before/after updates is non-zero. Sanity that projection isn't killing everything. | Projector wipes all updates. |
| `test_oplora_integrates_with_last_n` | Stack `#16` inside blocks attached by `#29`; forward runs. | Interface mismatch. |
| `test_oplora_k_out_of_range` | `preserve_k > min(d_out, d_in)` raises `ValueError`. | No validation. |

### 3.3 `test_lora_null_init.py` (#20)

| Test | Behavior | Fails because |
|---|---|---|
| `test_lora_null_calibrates_basis_from_activations` | Given a batch of synthetic activations, returned `U_null` has `shape=(d, null_rank)` and orthonormal columns. | Calibration function missing. |
| `test_lora_null_init_B_in_null_space` | After init, `B.T @ U_activations ≈ 0`. | B isn't projected into null space. |
| `test_lora_null_init_zero_effect_on_subset` | `(B @ A) @ subset_activations ≈ 0`. | Direct behavior fails if init bug. |
| `test_lora_null_init_nonzero_on_orthogonal_direction` | `(B @ A) @ v_orth` is non-zero for `v_orth ⊥ span(U)`. Sanity. | B is globally zero. |
| `test_lora_null_init_rank_respected` | `rank(B) == null_rank` (allowing for numerical zero tolerance). | Rank mismatch. |
| `test_lora_null_init_cache_roundtrip` | Compute SVD → save to `tmp_artifact_dir` → load → same basis within atol. | Serialization bug. |

### 3.4 `test_hard_channel_mask.py` (#2)

| Test | Behavior | Fails because |
|---|---|---|
| `test_mask_is_non_learnable` | `mask_buffer.requires_grad == False`; not in optimizer's param groups. | Mask stored as Parameter. |
| `test_mask_training_channels_get_one` | For `training_channels=[0,1,2]`, `mask[[0,1,2]] == 1`, others `0`. | Mask misbuilt. |
| `test_forward_training_channel_adapter_contributes` | For a per-channel feature at channel 0 (training), output differs from bypass after LoRA update. | Adapter gated off. |
| `test_forward_unseen_channel_adapter_zero_contribution` | For a per-channel feature at channel 7 (unseen), `output == frozen_base_output`. | Mask not applied. |
| `test_mask_composition_with_oplora` | Stack `#2` + `#16`; unseen channel still gets zero adapter contribution. | Order-of-ops bug. |
| `test_mask_from_band_list` | `build_channel_mask(bands=["B04","B03","B02"], all_bands=[...12...])` returns mask with 1s at the right indices. | Band-to-index mapping broken. |

### 3.5 `test_cdsd.py` (#9)

| Test | Behavior | Fails because |
|---|---|---|
| `test_cdsd_flag_off_no_teacher_loaded` | With `enable_cdsd=False`, no `ema_teacher` attribute is on the wrapped module. | Loader ignores flag. |
| `test_cdsd_training_drops_at_least_one_channel_per_step` | With `train()` mode, ≥ 1 of the training channels is zeroed in each forward. | Drop not applied. |
| `test_cdsd_min_keep_respected` | With `min_keep=2`, never drops more than `n_train_channels - 2`. | Bad sampling. |
| `test_cdsd_eval_mode_no_drop` | In `eval()` mode, input passes through unchanged. | Drop fires at eval. |
| `test_cdsd_loss_has_distillation_term` | `total_loss` includes `λ · distill_term`; swap `λ → 0` and loss equals pure CE. | Loss not composed. |
| `test_cdsd_ema_teacher_updates_after_step` | After `teacher_update()`, teacher params differ from previous (momentum < 1). | EMA not implemented. |
| `test_cdsd_ema_momentum_one_freezes_teacher` | With momentum=1, teacher never changes. | Convention reversed. |
| `test_cdsd_teacher_forward_no_grad` | `teacher(x)` does not create grad history. | Missing `torch.no_grad`. |

### 3.6 `test_aph_head.py` (#32)

| Test | Behavior | Fails because |
|---|---|---|
| `test_aph_forward_3_channels` | Input `(B=2, 3, D=64)` → output `(B=2, num_classes)`. | Head crashes on 3. |
| `test_aph_forward_4_channels` | Same module, input `(B=2, 4, D=64)` → output same shape. | Not variable-count-aware. |
| `test_aph_forward_12_channels` | Input `(B=2, 12, D=64)` → works. | Attention shape hardcoded. |
| `test_aph_query_is_learnable` | `head.query.requires_grad == True`; is in `head.parameters()`. | Query stored as buffer. |
| `test_aph_attention_weights_sum_to_one` | Extract `attn_weights`; each row sums to 1 within atol. | Softmax missing. |
| `test_aph_permutation_invariance_via_query` | Permute channels along dim=1; `softmax(q·K)` reordered correspondingly but pooled output within atol. | Position leaks. |
| `test_aph_gradient_reaches_query` | After `loss.backward()`, `head.query.grad is not None`. | Detached. |

### 3.7 `test_react_clip.py` (#5)

| Test | Behavior | Fails because |
|---|---|---|
| `test_react_training_mode_passthrough` | In `train()`, `forward(x) == x`. | Clipping fires during training. |
| `test_react_eval_mode_clips_at_percentile` | In `eval()`, with threshold set to 1.0, `|output|.max() <= 1.0` + atol. | Clamp not applied. |
| `test_react_preserves_direction` | After clipping, `cosine_sim(output, input) ≥ 1 - small` for each token. | Clip distorts direction. |
| `test_react_threshold_derived_from_training_set` | Given a fixture's training-set percentile computation, returned τ matches `np.percentile(norms, 95)`. | Wrong percentile. |
| `test_react_stats_roundtrip_cache` | Compute → save → load → same τ. | Serialization bug. |
| `test_react_per_layer_thresholds` | Different τ per transformer block; hook applies the right one. | Global τ instead. |

### 3.8 `test_hopfield_memory.py` (#11)

| Test | Behavior | Fails because |
|---|---|---|
| `test_hopfield_memory_frozen` | `memory_buffer.requires_grad == False`. | Memory stored as Parameter. |
| `test_hopfield_zero_init_output_projection` | At init, `cross_attn.out_proj.weight.norm() == 0`. | Default init used. |
| `test_hopfield_forward_no_change_at_init` | `module(feat) == feat` within atol. | Init not respected. |
| `test_hopfield_forward_runs_with_variable_channel_count` | `module(feat_3ch)` and `module(feat_4ch)` both work. | Shape hardcoded. |
| `test_hopfield_memory_source_channel_embed` | With `source="channel_embed"`, memory shape `(n_channels, D)` matches χViT's pretrained vectors. | Source loader wrong. |
| `test_hopfield_memory_source_layer0_mean_from_cache` | With `source="layer0_mean"`, loads from `tmp_artifact_dir`. | Cache path ignored. |
| `test_hopfield_noop_on_rgb_only_backbone_simulation` | Given a mock RGB-only backbone whose memory is zeroed, retrieval output is zero. Ensures DINOv2/v3 graceful degradation. | Retrieval leaks noise. |

### 3.9 `test_lsmm_aux_head.py` (#23)

| Test | Behavior | Fails because |
|---|---|---|
| `test_lsmm_endmembers_frozen` | Dictionary buffer has `requires_grad == False`. | Parameter instead of buffer. |
| `test_lsmm_abundances_nonnegative` | For any input, predicted `α ≥ 0` (softplus / softmax). | Missing activation. |
| `test_lsmm_reconstruction_well_defined` | `loss = ||x_rgb - SRF_RGB @ E @ α||²` is finite. | Shape mismatch. |
| `test_lsmm_aux_head_discarded_at_eval` | Eval forward omits aux head; `hasattr(output, "lsmm")` is False. | Aux runs always. |
| `test_lsmm_flag_off_no_head_attached` | With `enable_lsmm_aux_head=False`, module tree has no `lsmm_*` submodule. | Loader ignores flag. |
| `test_lsmm_endmembers_from_vca_cache` | Given a VCA-computed dictionary on synthetic data, load from cache → same buffer. | Cache round-trip bug. |
| `test_lsmm_lambda_zero_removes_loss_term` | With `lsmm_lambda=0`, total loss equals pure CE. | Still adds term. |

### 3.10 `test_srf_bias.py` (#24)

| Test | Behavior | Fails because |
|---|---|---|
| `test_srf_matrix_shape_12_by_12` | Loaded bias has shape `(12, 12)`. | Wrong dims. |
| `test_srf_matrix_symmetric` | `S == S.T` within atol. | Asymmetric. |
| `test_srf_matrix_diagonal_is_one` | `S[i,i] == 1` for all i. | Self-overlap wrong. |
| `test_srf_matrix_values_in_unit_interval` | `0 ≤ S ≤ 1`. | Physics bug. |
| `test_srf_bias_applied_pre_softmax` | Patch APH to record pre-softmax logits; assert bias summed in. | Bias applied post-softmax. |
| `test_srf_bias_scale_zero_noop` | `srf_bias_scale=0` → attention weights identical to vanilla softmax. | Default includes bias. |
| `test_srf_bias_composes_with_aph_forward` | APH + SRF: forward on 3, 4, 12 channels all work. | Interface mismatch. |

### 3.11 `test_mera_merge.py` (#21)

| Test | Behavior | Fails because |
|---|---|---|
| `test_mera_alpha_zero_recovers_pretrained` | `merge(pre, ft, α=0)` state_dict equals pretrained within atol. | Arithmetic inverted. |
| `test_mera_alpha_one_recovers_finetuned` | `α=1` equals fine-tuned. | Same. |
| `test_mera_alpha_half_midpoint` | `α=0.5` equidistant from both in parameter space. | Blend formula wrong. |
| `test_mera_lora_arithmetic_consistent` | For a LoRA-only finetune, merged model's LoRA-on contribution equals `α · (B_ft @ A_ft)`. | LoRA not blended. |
| `test_mera_realign_updates_head_only` | After realign, backbone unchanged; head changed. | Realign leaks into backbone. |
| `test_mera_realign_zero_steps_noop` | `realign_steps=0` doesn't run any training. | Loop off-by-one. |
| `test_mera_alpha_search_returns_best_on_val` | Given a synthetic val curve, best α selected. | Selection bug. |

### 3.12 `test_tc_caf.py` (#7, optional)

| Test | Behavior | Fails because |
|---|---|---|
| `test_tc_caf_tau_loaded_from_cache` | `load_tau(path)` returns a scalar. | Serialization wrong. |
| `test_tc_caf_fusion_rule_agrees_uses_teacher` | Disagreement `< τ` → output == teacher logits. | Rule inverted. |
| `test_tc_caf_fusion_rule_disagrees_uses_student` | Disagreement `≥ τ` → output == student logits. | Same. |
| `test_tc_caf_student_equals_teacher_degenerate` | When student == teacher, output equals both. | Numerical error. |
| `test_tc_caf_alpha_zero_never_fuses` | With `α=0` (strict), never use teacher. | α misinterpreted. |

### 3.13 `test_bpsg.py` (#18, optional)

| Test | Behavior | Fails because |
|---|---|---|
| `test_bpsg_samples_from_posterior` | `sample_posterior()` returns different samples across calls. | Not stochastic. |
| `test_bpsg_ci_width_scales_with_std` | Larger `ci_width` → wider accept region. | Parameter ignored. |
| `test_bpsg_fallback_when_superset_outside_ci` | Given synthetic super/sub forwards and `ci_width=0`, all samples fall back to subset. | Fallback logic bug. |
| `test_bpsg_accept_when_inside_ci` | With `ci_width=10`, all samples accept superset. | Same. |
| `test_bpsg_n_mc_respected` | `n_mc=20` → exactly 20 MC samples drawn. | Hardcoded. |

### 3.14 `test_adapt_align.py` (#27, optional)

| Test | Behavior | Fails because |
|---|---|---|
| `test_adapt_deterministic_output` | Same input → same output. | Stochastic. |
| `test_adapt_gaussians_loaded` | Load from `tmp_artifact_dir`, shapes match `(num_classes, D)` for mean and `(num_classes, D, D)` (or diagonal `D`) for cov. | Bad cache format. |
| `test_adapt_closed_form_moves_toward_mean` | Input far from nearest mean → output closer to that mean. | Math bug. |
| `test_adapt_null_shift_when_input_is_mean` | Input exactly at a class mean → output identical. | Off-by-one. |

### 3.15 `test_mcse_head.py` (#28, optional)

| Test | Behavior | Fails because |
|---|---|---|
| `test_mcse_builds_k_heads_from_subsets` | With `subsets=["R","G","B","RG","RB","GB","RGB"]`, 7 heads built. | Subset parser broken. |
| `test_mcse_each_head_sees_its_subset_only` | Head `h_RG` gets features restricted to channels [0,1]. | Channel slicing wrong. |
| `test_mcse_ensemble_variance_computed` | Output includes `(mean, var)`; `var > 0` when heads disagree. | Var omitted. |
| `test_mcse_only_compatible_heads_used_at_eval` | With `eval_bands=[0,1]`, only heads whose subset ⊆ {0,1} vote. | Includes incompatible. |
| `test_mcse_power_set_expansion` | `subsets="power_set"` with 3 training channels → 7 non-empty subsets. | Enumeration wrong. |

### 3.16 `test_nci_invariance.py` (#14, optional)

| Test | Behavior | Fails because |
|---|---|---|
| `test_nci_loss_zero_without_null_augmentation` | `loss(set, set) == 0`. | Wrong metric. |
| `test_nci_loss_positive_with_null_addition` | `loss(set, set ∪ null)` > 0 before training. | Baseline error. |
| `test_nci_loss_compatible_with_any_head` | Run with `head_type=aph` and `head_type=linear`; loss is computable. | Head coupling. |
| `test_nci_invariance_lambda_zero_removes_term` | `λ=0` → loss equals pure CE. | Still adds term. |
| `test_nci_null_token_is_zero_plus_embed` | Null token = zero vector + learnable null-embedding. | Wrong construction. |

### 3.17 `test_ch_rs_ft.py` (#15, optional)

| Test | Behavior | Fails because |
|---|---|---|
| `test_ch_rs_training_adds_noise` | In `train()`, feature tokens have noise added; norms change. | No noise. |
| `test_ch_rs_eval_mc_smoothing_returns_majority` | With `n_mc=50`, majority vote over noised forwards returned. | No MC loop. |
| `test_ch_rs_certified_radius_computed` | Return value includes a certificate `k ≥ 0`. | Certificate omitted. |
| `test_ch_rs_sigma_zero_passthrough` | With σ=0, forward is deterministic. | Noise not scaled. |
| `test_ch_rs_p_smooth_zero_no_tokens_noised` | `p_smooth=0` → no channel tokens noised. | Always applied. |

### 3.18 `test_imputation.py` (#12 / #26, optional)

| Test | Behavior | Fails because |
|---|---|---|
| `test_imputation_none_passthrough` | `--imputation none` → input returned unchanged. | Wrong dispatch. |
| `test_imputation_invoked_when_no_overlap` | `training_bands ∩ eval_bands = ∅` → imputer called. | Condition reversed. |
| `test_imputation_not_invoked_when_overlap` | Overlap exists → imputer skipped. | Called spuriously. |
| `test_imputation_diffusionsat_ckpt_loaded` | Model loaded from ckpt path. | Wrong path. |
| `test_imputation_output_shape_matches_target_bands` | Output has `(B, len(target_bands), H, W)`. | Shape wrong. |
| `test_imputation_cached_result_reused` | Running twice on same input doesn't re-invoke diffusion. | No cache. |

## 4. Integration tests — `test_r_grid.py`

These depend on all unit tests passing. Each R-row row gets smoke + matched-
pair tests.

| Test | Behavior | Fails because |
|---|---|---|
| `test_r0_baseline_forward_backward` | With all flags off, run one step on synthetic data → loss scalar, gradients on head only. | Baseline broken. |
| `test_r9_reliable_core_forward_backward` | With all 9 core flags on, one step runs without errors. | Composition bug. |
| `test_r9_matched_pair_subset_forward_preserved` | Sample a random input, forward on training subset before vs after one fine-tune step. With OPLoRA + hard mask, should be within atol. Priority B guarantee. | Preservation fails. |
| `test_r9_minus_oplora_changes_output` | Drop OPLoRA from R9 → output differs from full R9. Sanity that each flag contributes. | Flag inert. |
| `test_r_grid_all_rows_smoke` | Parametrized over R0..R13; each row completes one train step + one eval step. | Any row broken. |
| `test_r9_no_regression_vs_r0_on_subset` | On training bands only (in-distribution), R9 ≥ R0 within noise on a toy 2-class task. | R9 worse than baseline. |

## 5. Portability tests — `test_portability.py`

Run Reliable-Core (R9) against each mock backbone. This is the key cross-
model guarantee check.

| Test | Behavior | Fails because |
|---|---|---|
| `test_r9_runs_on_mock_chivit` | Forward + one train step with `MockChiViT`. | Interface mismatch. |
| `test_r9_runs_on_mock_terrafm` | Same with `MockTerraFM`. | Same. |
| `test_r9_runs_on_mock_dofa` | Same with `MockDOFA`. | Same. |
| `test_r9_runs_on_mock_dinov2` | Same with `MockDINOv2`. | Same. |
| `test_r9_runs_on_mock_dinov3` | Same with `MockDINOv3`. | Same. |
| `test_hopfield_is_noop_on_dinov2_mock` | With `MockDINOv2` whose pretrained multispectral memory is zero, #11's forward returns input unchanged. Graceful degradation check. | Memory leaks garbage. |
| `test_lsmm_runs_with_rgb_only_backbone` | `MockDINOv2` + LSMM aux head — forward/backward runs (even if weak signal). | RGB-only breaks aux head. |
| `test_srf_bias_runs_in_aph_on_all_mocks` | APH + SRF for each of 5 mocks. | Dimension coupling. |
| `test_mera_merges_on_each_mock` | MERA post-training pass for each mock. | State dict mismatch. |

## 6. Completion criteria

A technique is "done" when:

- [ ] All tests in its suite file pass.
- [ ] Each test was watched to fail on the commit *before* its implementation.
- [ ] No test added in a follow-up "coverage" commit.
- [ ] `./run_tests.sh -k <technique_name>` produces green output with no
      warnings.
- [ ] Related integration test (`test_r_grid.py`) row incorporating this
      technique still green.
- [ ] Portability test for this technique passes on all five mock backbones.

A PR is "done" when:

- [ ] All unit, integration, and portability tests green.
- [ ] New CLI flag(s) documented in `reliable-solutions.md` §Flag reference.
- [ ] `./run_tests.sh` green end-to-end.
- [ ] No test-only methods added to production classes (see
      `testing-anti-patterns.md`).

## 7. Test implementation order per technique

For each technique's suite, TDD in this order (smallest / independent tests
first):

1. Flag-off passthrough (easiest).
2. Shape / type checks (small).
3. Learnability / frozen-ness assertions.
4. Forward behavior (single invocation, no gradients).
5. Backward behavior (gradients flow correctly).
6. Guarantee tests (the main claim — hardest; do these with care).
7. Integration with the prior technique in the dependency chain.

For every single test: **RED → verify fail → GREEN → verify pass → refactor**.
Never batch more than one test.

## 8. Where new code lives

Test files live under `rs_finetune/tests/reliable/`. Production code lives
under `rs_finetune/` (one new module per technique, as listed in
`reliable-solutions.md §Files to create`). CLI flag wiring goes into the
three existing training scripts and their eval counterparts.

No test-only hooks inside production modules. If a test needs visibility into
internal state, expose a clean observation API (e.g., `get_attention_weights`)
that production code can also benefit from — or use `pytest.MonkeyPatch` on
the public surface.

## 9. What's explicitly NOT in this plan

- Benchmark / accuracy tests on real datasets. Those are separate from unit
  TDD and belong in experiment tracking, not CI.
- End-to-end training runs. Use the R-grid ablation scripts once unit +
  integration + portability are green.
- Tests for techniques we deliberately dropped (NSP-FT, HP-Freeze, ChEmbed
  Diffusion, ChE-LoRA, DOFA hypernet, etc.). Those remain in the catalog
  for reference, not for implementation.

## 10. Estimated sizing

| Component | Tests | Production LOC (est.) |
|-----------|-------|----------------------|
| Universal core (7 techniques) | ~55 tests | ~900 LOC |
| Multispectral core (3 techniques) | ~20 tests | ~400 LOC |
| Post-training (#21 MERA) | 7 tests | ~150 LOC |
| Optional add-ons (7 techniques) | ~35 tests | ~700 LOC |
| Integration (#test_r_grid.py) | 6 tests | ~200 LOC |
| Portability (#test_portability.py) | 9 tests | ~300 LOC |
| Shared infra (conftest, mocks) | — | ~300 LOC |
| **Total** | **~130 tests** | **~2950 LOC + tests** |

At ~15–30 min per test (write, verify red, implement, verify green,
refactor), total budget ≈ 35–65 hours of focused TDD. Plan for 1–2 weeks
at realistic pace.
