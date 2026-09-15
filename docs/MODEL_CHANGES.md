# cbct_recon — Model Changes: Binned Aggregation, Wide-N Training, and Reference Alignment

This document records the changes made to the reconstruction model in June 2026,
starting from a summary of the original architecture. All new behavior is **opt-in**
(`--use_binned_aggr`); without the flag the original model runs unchanged.

---

## 1. The original model (`models/twod_to_threed_model.py`, `twoD2threeDModel`)

Sparse-view CBCT reconstruction: N 2D projections (+ gantry angles) → one 3D volume
(Nx×Ny×Nz = 256×256×128 at 2 mm voxels).

Pipeline per forward pass:

1. **2D encoding.** Each projection goes through `MultiScaleCNNEncoder`, producing
   feature maps at three scales: fine (C_a=128, H/8), mid (C_b=256, H/16), and
   coarse (C_f=768, H/32). Coarse tokens are modulated by Plücker ray embeddings
   (`ModLN`) and refined per-view by a 6-layer spatial transformer (`FusionViT`).
2. **2D→3D backprojection + per-voxel view aggregation.** For each voxel of a 3D
   grid, every view's feature map is sampled at the voxel's projected detector
   location (`map_3d_grid_to_2d_batch` + `grid_sample`), then pooled across views:
   - **Coarse (16×16×8):** `MultiImageAggregator` — a CLS token + the N view
     features through a 6-layer transformer, run per voxel (chunked, gradient
     checkpointed). Combined with separate mean and max pools via a 1×1 conv
     (`netrefine_coarse`).
   - **Mid (32×32×16) and fine (64×64×32):** `ViewWeightAggregator` — an O(N)
     learned-score softmax-weighted sum, computed one-shot over all voxels.
3. **Coverage mask.** Per coarse voxel: saturating in-FOV view count with a
   half-life scaled by the circular spread of the angles. Passed to the
   refinement transformer so poorly covered voxels lean on the reference prior.
4. **Reference path (paired batches only).** A second scan of the same patient
   (reference CT + its DRRs at the same angles) provides a prior:
   `RefVolumeEncoder` (3D CNN) encodes the reference volume; the refinement
   transformer (`RefinementBlock` ×6) runs Q = source backprojection,
   K = reference backprojection, V = reference 3D-CNN features, with a per-voxel
   learned gate boosted where coverage is low. A parallel reference decode path
   reconstructs the reference volume as an auxiliary task (`L_vol_ref`).
5. **Decoding.** `MultiScaleDecoder3D`: 16³ → 32³ (mid skip) → 64³ (fine skip)
   → 128³ → 256³ (no skips in the last two octaves). Source and reference are
   decoded in one batched call.
6. **Losses.** Volume L1 + 3-plane SSIM, multi-scale Scharr gradient loss,
   optional DRR render loss (`DifferentiableDRR`), optional perceptual loss using
   the (frozen) reference encoder. Two AdamW optimizers: base params, and
   ref-specific params (stepped on paired batches only, lr×2).
7. **Training.** `train_mixed.py` + accelerate (DDP, bf16): mostly single batches,
   ~30 % paired. N sampled uniformly from `[min_num_images, max_num_images]`
   (historically 10–50). Angle patterns: full circle, Halcyon 211°, TrueBeam 200°.

### Problems this work addressed

| Problem | Symptom |
|---|---|
| Aggregation does not scale in N | At N in the hundreds: coarse transformer cost grows ~linearly-to-quadratically; mid/fine materialize a `(B·total, N, C)` tensor (≈20 GB at N=300, fine scale) → OOM |
| Aggregation accuracy degrades at large N | Softmax weights dilute toward 1/N → pooling collapses to a mean over views, blurring exactly the detail extra projections should add |
| Train/test N mismatch | Model only ever saw N ∈ [10, 50]; aggregated feature statistics shift with N |
| Reference CT barely helps | Confirmed conduit bottleneck: even the reference decode path — which receives the exact volume as input — reconstructed no better than the source path |

---

## 2. Angular-binned two-stage aggregation (`models/twod_to_threed_model_binning.py`)

New model `twoD2threeDBinningModel(twoD2threeDModel)`, selected via
`--use_binned_aggr` (construction goes through `models.create_recon_model(opt)`;
default off = original class, byte-for-byte). Only `__init__`, the two
aggregation methods, `forward`, and `compute_volume_loss` are overridden —
encoder, FusionViT, refinement transformer, decoder, renderer, and all other
losses are inherited unchanged.

### Design

The gantry circle is split into **K = `--aggr_bins` (36) fixed 10° bins** in the
absolute gantry frame, shared by all three scales (constructors take K per scale
for future ablation, e.g. fine = 72).

**Stage 1 — within-bin (O(N) total).**
Views inside one bin are nearly redundant. Per voxel, a learned scalar score
(`score_view`) + masked softmax reduces each bin's views to one feature.
Implemented fully vectorised: one `grid_sample` over all views per voxel
chunk, then a segmented (scatter-based) softmax grouped by bin — an earlier
bin-by-bin loop was kernel-launch-bound (B×K small kernels per chunk, doubled
again by checkpoint recompute in backward; verified ≈10× slower even on CPU).
The (N, C, chunk) transient is bounded by an N-adaptive voxel chunk
(`--aggr_vox_chunk` applies up to N=64 and shrinks ∝1/N beyond) and is freed
per chunk under checkpointing. Two behavioral fixes over the original:
- Views whose projection of the voxel falls **outside the detector FOV are
  masked out** of the softmax (the original silently pooled zero-features via
  `padding_mode='zeros'`).
- The coarse **mean and max pools are accumulated in the same pass** (verified
  numerically identical to the parent's separate sweeps), so the original's
  three full sweeps over all N views became one.

**Stage 2 — across-bin (O(K), constant in N).**
The expressive aggregation always sees exactly K tokens whether N = 10 or 900,
so compute, memory, and feature statistics are identical across regimes:
- **Coarse:** `BinnedTransformerAggregator` — CLS + K bin tokens through the
  transformer. Empty bins keep their slot with a shared **learned empty-token**,
  so missing-wedge directions are encoded, not dropped.
- **Mid/fine:** `BinnedViewWeightAggregator` — scored softmax over **occupied
  bins only** (safe softmax; all-empty voxels output a learned constant).

Every bin token carries a 4-dim geometry embedding projected to C:
`[sin θₖ, cos θₖ, log1p(per-voxel in-FOV count), conjugate-covered]`.
The **conjugate flag** (bin k or bin k+K/2 occupied) distinguishes a Parker
short scan (211° ≈ data-complete via conjugate rays) from a genuine
limited-angle scan — K must be even for the pairing.

**Coverage-mask fix (2θ).** Angular spread is now computed on doubled angles:
tomographic directions are *lines*, so θ and θ+180° measure (nearly) the same
line integrals. Measured effect: a 211° arc scores spread ≈ 0.85 (the parent's
plain-θ statistic gave ≈ 0.48, under-crediting short scans); a true 90°
limited-angle scan stays at ≈ 0.37.

### Memory: per-chunk gradient checkpointing

Chunking alone bounded the *forward* peak but autograd still retained every
bin's sampled features for backward — O(total·N·C) across the loop (~35 GB at
N=100 in the paired path; OOMed an 80 GB GPU). Each voxel chunk's Stage-1 +
Stage-2 computation now runs under `torch.utils.checkpoint`
(`use_reentrant=False`): backward recomputes the sampling/softmax and stored
activations drop to **O(total·C), independent of N**. Chunk sizes:
`--aggr_chunk_size` (coarse, 1024) and `--aggr_vox_chunk` (mid/fine, 8192)
bound the recompute transient.

### DDP safety

`train_mixed.py` wraps each subnet individually in `DistributedDataParallel`
via `accelerator.prepare`, which hides custom attributes and breaks gradient
bookkeeping for parameters invoked outside the wrapped forward. Therefore:
- Stage 1 lives **inside** the aggregator modules' `forward()` (shared base
  class `_BinnedAggregatorBase`), so every parameter is exercised within a
  single wrapped call — the same pattern as the parent's
  `checkpoint(aggr_net, s)`.
- The model reads `.K` only through `getattr(net, 'module', net)`.
- A latent autograd bug found by the tests was fixed: the coarse mean/max
  accumulators used in-place indexed writes, and `torch.maximum`'s backward
  needs its unmodified inputs — gradients through `coarse_mean`/`coarse_max`
  raised "modified by an inplace operation". Accumulation is now out-of-place.

---

## 3. Wide-N training distribution (`ct_dataset.py`)

`--num_images_sampling {uniform, loguniform}` (default `uniform` = original
`random.randint`). `loguniform` draws N = ⌊exp(U(log min, log(max+1)))⌋ —
equal probability **per octave**, so with a range like 10–500, sparse counts
stay well represented (10–20 is as likely as 250–500; plain uniform would give
P(N ≤ 50) ≈ 8 % and the model would forget the sparse-view regime).

Implemented in `sample_num_images()` and wired through all three collate
functions (`single_batch_collate`, `paired_batch_collate`, `batch_collate`)
and their dataloader creators. N is additionally **clamped to the sparsest
batch item's available projection count** (previously an assert crash when
`max_num_images` exceeded a folder's contents).

Verified distribution (200 K draws, range 10–500): each octave band
[10,20), [20,40), …, [160,320) receives ≈ 17.7 %.

---

## 4. Reference alignment + multi-scale prior injection (binned model only)

### Diagnosis: the conduit problem

The reference's only route into the source reconstruction was the 16³ coarse
grid: `ref_f_1_16` (16·16·8 × 768 ≈ **1.57 M numbers for an 8.39 M-voxel
volume**, downstream of four stride-2 convs and a softmax-averaged attention
retrieval). The 1/4- and 1/8-scale reference features were computed and
**discarded**; the source decoder's mid/fine skips contained no reference
information, and the decoder's last two octaves (64³ → 256³) have no skips at
all. So high-frequency prior content — the thing sparse views cannot recover —
never reached the output. Decisive evidence: the reference *decode* path,
which receives the exact volume as input, reconstructed no better than the
source path. Both paths drew all fine-scale information from the same place:
the projections.

### Fix part 1 — attention-derived affine alignment (`AffineCorrespondenceAlign`, `netalign`)

A dedicated correspondence cross-attention on the coarse feature volumes
(source backprojection vs reference backprojection — same feature space):

1. Each source voxel's attention row over the 2048 reference voxels is
   **soft-argmaxed** into an expected matched reference coordinate.
2. A **confidence- and coverage-weighted ridge least-squares fit** turns the
   2048 noisy votes into one robust affine `theta` (B, 3, 4), regularized
   toward identity by `--align_ridge` (1.0): with unreliable early attention
   the transform stays ≈ identity; it earns deviation as attention sharpens.
3. The fit is in closed form (`fit_weighted_affine`, float32 solve under
   autocast) and **differentiable end-to-end**: both the alignment loss and
   the decoder backpropagate through the fit into the attention projections.
4. Inputs are **detached** so alignment objectives cannot distort the
   backprojected features and the reconstruction loss cannot game geometry.
5. Coordinates are generated by `F.affine_grid` itself, so the fit and every
   later warp share one convention by construction.

Supervision (added in `compute_volume_loss`, no train-script changes):
`L_align = 1 − NCC(warp(ref_vol, θ), real_vol)` — this is what makes the
attention *geometric* — plus 0.1 × the near-affine consistency residual
`Σ wᵢ‖T(pᵢ) − cᵢ‖²` (`L_align_consist`), weighted by `--lambda_align` (0.1).
Train-time only (needs GT); at test time `theta` is still predicted and used.

### Fix part 2 — multi-scale gated injection (`GatedRefSkipFusion`)

- `ref_f_1_8` (32³×256, capacity 0.5× the volume) and `ref_f_1_4` (64³×128,
  capacity **2.0×** — the first scale that can carry the full volume) are kept.
- The reference is **encoded once in its own frame**; the *feature maps* are
  affine-warped by `theta` (exact for translation, accurate for the small
  rotations in scope — no second encoder pass).
- **Source decoder skips** receive the *warped* features through per-voxel
  gated fusion: `src + sigmoid(gate(cat(src, ref))) · proj(ref)`, with `proj`
  and `gate` zero-initialized → the fusion starts as an **exact identity** and
  training starts from current behavior.
- The **reference decode path** receives its own *unwarped* features (its
  target is the reference frame). Source + ref are batched into a single
  fusion call per scale (one DDP forward per module).

**Built-in success criterion:** with the conduit open, `L_vol_ref` should drop
far below `L_vol` during training — the direct refutation of the original
symptom. Whatever gap remains between them afterward is, by construction, the
correspondence/gating part of the problem, cleanly separated from capacity.

New nets `align`, `skipfuse_mid`, `skipfuse_fine` are registered in
`model_names` (saved/loaded/prepared) and routed to the **ref optimizer**
(stepped on paired batches only).

---

## 5. Infrastructure changes

| Change | Where | Why |
|---|---|---|
| `create_recon_model(opt)` factory | `models/__init__.py` | Selects binning model on `--use_binned_aggr`; default path untouched. All 7 train/test scripts construct through it. |
| Generic `accelerator.prepare` | all 4 train scripts | Sub-networks prepared by iterating `model.model_names` instead of a hardcoded list — required for the new nets, and it fixed `train_paired`/`train_single`/`train_single_paired` silently omitting `netaggr_fine`/`netaggr_mid`. (`test.py` already iterated generically.) |
| Renderer vectorization | `models/renderer.py` | (Pre-existing user change, committed) `forward_` renders C angles per call with the volume shared; unused rigid-transform path removed. |
| `models/multiscale_cnn_encoder.py` committed | — | Was imported but never committed; fresh clones could not run. |

### New options (`options/base_options.py`)

| Option | Default | Meaning |
|---|---|---|
| `--use_binned_aggr` | off | Use `twoD2threeDBinningModel`; off = original model |
| `--aggr_bins` | 36 | Angular bins K (absolute gantry frame, all scales; must be even) |
| `--aggr_vox_chunk` | 8192 | Voxel chunk for binned mid/fine aggregation |
| `--num_images_sampling` | `uniform` | `loguniform` = equal probability per octave of N |
| `--lambda_align` | 0.1 | Weight of NCC alignment loss + consistency term (0 disables the loss; θ is still fitted and used) |
| `--align_ridge` | 1.0 | Pull of the affine fit toward identity |

### Typical launch

```bash
accelerate launch train_mixed.py @./options/datasets/all.ini @./options/geometry/hal.ini \
    --name hal_bin --angle_select_mode halcyon --use_rope \
    --use_binned_aggr \
    --min_num_images 8 --max_num_images 100 --num_images_sampling loguniform \
    --lambda_L1 -1.0 --lambda_ssim -1.0
```

---

## 6. Verification performed (CPU smoke tests, no GPU/training required)

- **Binning:** bin assignment edge cases (359.9°, negatives, wraparound);
  conjugate flag k ↔ k+K/2; per-bin in-FOV counts vs brute force; Stage-1
  weighting vs direct masked softmax; mean/max outputs **exactly equal** the
  parent's reductions; gradient flow to encoder features and scorers.
- **Checkpointing:** checkpointed outputs and input gradients bit-identical to
  the direct path; `no_grad` eval path works.
- **DDP:** behavior identical through a mock DDP wrapper; **every parameter of
  each aggregator receives grad within a single forward call** (the
  find-unused-parameters safety property); the in-place mean/max autograd bug
  was caught and fixed by these tests.
- **Alignment:** exact affine recovery at ridge 0; identity collapse at high
  ridge; untrained module stays near identity (max dev 0.25 at ridge 1.0);
  fit ↔ `affine_grid` convention consistency; fusion bit-exact identity at
  init with grads to `proj`; optimizer split (new nets in ref group, no
  overlap); `L_align ≈ 0` for an identical identity-aligned reference;
  single-path forward unaffected.
- **2θ spread:** 211° → 0.85, 90° → 0.37, 360° → 0.99.
- **Log-uniform sampler:** octave bands equal at ≈ 17.7 % over 200 K draws;
  bounds and degenerate ranges correct; `uniform` mode unchanged.

---

## 7. Compatibility notes

- **Checkpoints:** `aggr`, `aggr_fine`, `aggr_mid` have new shapes under the
  same filenames — not loadable from original-model checkpoints. `align`,
  `skipfuse_mid`, `skipfuse_fine` are new files. All other nets are
  weight-compatible (selective warm-start from e.g. `hal_rope` is possible).
- **Render loss:** still renders all N angles if re-enabled
  (`lambda_L1`/`lambda_ssim` currently −1 = off). Subsample ~10 random angles
  per iteration before turning it back on at large N.
- **Arc modes cap N physically:** with `halcyon`/`truebeam` selection only
  ~211/360 of a folder's projections lie inside the arc; realized N saturates
  there regardless of the drawn N.

## 8. What to monitor in the first training run / open items

Monitor: `L_vol_ref` vs `L_vol` (conduit-open check), `L_align` (alignment
becoming geometric), `L_align_consist` (attention converging to a coherent
global transform), and dense-vs-sparse validation at
N ∈ {10, 25, 50, 150, 300, 500} × {full arc, 211°}.

Open items, in rough priority order:

1. **Wider reference-only misalignment augmentation** (±15–20 mm, ±10°) if
   `L_align` plateaus — natural aug (±10 mm/10° relative) may let the head
   coast on "identity is good enough", and deployment setup errors are larger.
2. **Phase-2 full-resolution injection:** the warped reference volume itself
   (1 channel) into the skip-free `up3`/`up4` decoder stages, plus the
   deformable residual `cᵢ − T(pᵢ)` the fit already produces (breathing-scale
   deformation matters at 2–4 mm voxels).
3. **Per-scale K ablation** (fine = 72) if dense-N validation shows peripheral
   detail loss from 10° bins.
4. **FDK/filtered-backprojection hybrid branch** (Pillar 2, deliberately
   deferred): at dense N a ramp-filtered backprojection branch gated by the
   coverage mask would guarantee clinical quality and reduce the learned
   path's burden to artifact correction.
5. **`--align_ridge` decay** once `L_align_consist` is low (it is read at
   construction; set per run).
