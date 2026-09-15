# Real-data track — Halcyon HyperSight clinical scans

Testing the model (and every comparator) on **real** Halcyon projections with
the clinical iTools reconstruction as reference, alongside the synthetic DRR
benchmark.

## Data

15 de-identified Halcyon HyperSight scans (iTools Reconstruction 3.7 dump,
tapped after Analytical Spectrum Correction, before Filter & Decimation).

* The raw scans are institutional patient data and are **not distributed**.
  The scripts below document the exact processing chain so the same pipeline
  can be applied to your own Halcyon iTools dumps (`h5/`, `recon_vol/`).

Sites: pubic, thyroid, prostate bed ×5, prone rectum, uterus, esophagus, HN,
bladder, lt pelvic, tongue, tonsil. 335 projections (100 kV, head/neck) or
406–407 (125–140 kV) over a 211° arc.

## Geometry — verified, not assumed

All parameters come from the HDF5 attributes (`AcqParams` / `ProjParams`);
nothing is hard-coded.

| quantity | value | note |
|---|---|---|
| SAD / SID | 1000 / 1540 mm | real detector position (mag 1.54) |
| ImagerResY | 1.12 mm | → **0.7273 mm at iso = our psize exactly** (rows 1:1) |
| ImagerResX | 0.28 mm | → 0.1818 mm at iso = psize/4 (bin cols ×4) |
| arc | StartAngle −126° → StopAngle +85° | 211°, matches our halcyon mode |
| angles | `ProjParams.GantryRtn` | per projection |

**Angle convention: `theta = GantryRtn + 180`** (Parker gamma `+1`).
Established empirically, not by reading conventions: FDK reconstructions under
8 candidate mappings × both gamma signs were scored against the clinical recon
by NCC. `gantry+180` reaches **NCC 0.87 with zero residual shift**; the
negated-gantry and ±90° variants only fit by demanding large incoherent
translations. (`verify_prostate_geometry.py`, `refine_real_geometry.py`.)

**Detector row placement: `row_shift = -(ImagerLng / mag) / psize`.**
The dump stores only the illuminated rows, so the row count varies per scan
(212–384) as the blades crop the detector. The band's z centre is where the
detector sits — `ImagerLng/mag`, which is **per-scan (−9.5 … +29.1 mm)** and
agrees with the blade centre `0.5*(BladeY1+BladeY2)` to ≤0.16 mm on all 15
scans. The sign is negative because row 0 is superior (+z). Confirmed by
sweeping row offsets: residual z misalignment is 0 mm only at this value
(`calib_row_shift.py`).

**Per-scan gate** (FDK vs clinical recon, no post-hoc alignment): NCC 0.92–0.97
on pelvis/abdomen, 0.73–0.83 on head/neck (dense bone, dental metal, vendor
corrections our plain FDK does not apply). Residual misalignment ≤1 voxel on
every scan.

## Evaluation grid and the intersected region

Our grid is 256×256×128 @2 mm (512×512×256 mm). The clinical recon differs:

| | in-plane spacing | in-plane FOV | z spacing |
|---|---|---|---|
| pelvis | 1.051 mm | 538 mm | 2.0 mm |
| bladder | 0.862 mm | 441 mm | 2.0 mm |
| head/neck | 0.550 mm | 282 mm | 2.0 mm |

* **z spacing already matches** — no z resampling, only padding.
* **In-plane: downsample 1.9×–3.6×** to 2 mm, single pass; everything is
  compared on our grid (no upsampling anywhere in the scoring path).
* **The crop is free**: 0.000 % of anatomy (HU > −500) falls outside our
  512 mm FOV on any scan.
* An anti-alias prefilter was tested and **rejected** (it discards recoverable
  detail; plain linear interpolation is better here).

**Metrics use `validmask.npy` — the physically intersected region only**:
our grid ∩ clinical-recon support ∩ illuminated band. This matters a lot:
valid coverage ranges from **27.6 % (P07_HN**, whose 282 mm FOV covers only
140/256 voxels per in-plane axis**)** to 93.8 % (pelvis). Scoring the full
grid would let air-vs-air dominate and inflate every method equally while
compressing the differences between them. `eval/real_eval.py` (compare repo)
reports both the valid-region and an anatomy-only (body-masked) variant.

**Framing:** the clinical iTools reconstruction is a **reference, not ground
truth** — a full-scan, vendor-corrected result. Real-data numbers measure
*agreement with the clinical standard*; all methods inherit its bias equally.

## View draws

Each scan is a full acquisition, so it yields several sparse-view cases.
`make_view_draws.py` (compare repo) writes **one seeded manifest consumed by
every method**, so all see byte-identical subsets:

* **K = 3** draws per scan at 10/20/50/100 views → **180 runs/method**
* **SAX-NeRF: K = 3 at 10 v, K = 1 above** → 90 runs (draw variance is large
  when sparse, negligible at 100/407 views; it costs ~50 min–3.5 h per run)

## Scripts

| script | role |
|---|---|
| `prep_prostate_h5.py` | h5 + DICOM → `projs.npy`, `angles.npy`, `gt.npy`, `zmask.npy`, `validmask.npy`, `meta.json` |
| `verify_prostate_geometry.py` | 16-way angle-convention check vs clinical recon |
| `refine_real_geometry.py` | fine offset sweep + residual rigid-shift estimate |
| `calib_row_shift.py` | detector row placement (z registration) |
| `quantify_resample_error.py` | crop / downsample cost of our grid |
| `../../cbct_recon_compare/realdata/make_view_draws.py` | shared view manifest |
| `../../cbct_recon_compare/eval/real_eval.py` | metrics on the intersected region |

## Status

Prep, geometry verification and the view manifest are **done for all 15
scans**. Still to build: per-method inference adapters (each comparator
currently reads synthetic scan folders and must instead consume
`projs.npy`/`angles.npy` + `draws.json`), then submission of the run grid.
