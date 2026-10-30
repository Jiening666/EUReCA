# EUReCA: End-to-End Unified CBCT Reconstruction Across Clinical Acquisitions

[![Project page](https://img.shields.io/badge/project-page-blue)](https://jiening666.github.io/cbct_recon/)
[![arXiv](https://img.shields.io/badge/arXiv-2610.37740-b31b1b)](https://arxiv.org/abs/2610.37740)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20weights-jzhu35%2FEUReCA-yellow)](https://huggingface.co/jzhu35/EUReCA)
[![Code license](https://img.shields.io/badge/code-Apache--2.0-green)](LICENSE)
[![Weights license](https://img.shields.io/badge/weights-CC%20BY--NC%204.0-lightgrey)](WEIGHTS_LICENSE.md)

EUReCA is a single geometry-aware feed-forward network that reconstructs a
cone-beam CT volume from **2 to 100 projections** acquired on a clinical linac
(Varian Halcyon) geometry: full or partial arcs, uniform or arbitrary gantry
angles, without view-count-specific retraining. Multiscale 2D features are
back-projected along the exact cone-beam rays, pooled over angular
sectors into a fixed-size 3D representation, and fused at every scale with a
differentiable FDK reconstruction of the same projections. The model is trained
on digitally reconstructed radiographs (DRRs) of public CT datasets and
validated on real Halcyon HyperSight projections.

* Output grid: 256 × 256 × 128 voxels at 2 mm; detector 384 × 768 at 0.7273 mm
  (isocenter plane), SAD 1000 mm.
* One forward pass takes 0.3–1.3 s per volume on an A100, depending on N.

<p align="center"><img src="https://jiening666.github.io/cbct_recon/static/images/fig_method_overview.png" width="90%"></p>

## Interactive visualization

Reconstructions can be explored in the browser, with no install, on the project
page: **<https://jiening666.github.io/cbct_recon/>**

* **Interactive volume viewer** — step through axial, coronal and sagittal
  planes of a reconstructed volume.
* **Qualitative comparisons** — EUReCA against the feed-forward and
  measurement-driven baselines on the same case.
* **Reconstruction vs. number of views** — the same patient reconstructed from
  increasing numbers of projections.
* **Additional acquisition trajectories** — short arcs, orthogonal pairs and
  fixed-angle acquisitions.

The page also carries the abstract, method summary and a link to the [arXiv preprint](https://arxiv.org/abs/2610.37740).

---

## Installation

Linux + an NVIDIA GPU (inference fits in 16 GB at N = 100). Python 3.11.

```bash
git clone https://github.com/Jiening666/EUReCA.git && cd EUReCA
conda create -n eureca python=3.11 -y && conda activate eureca
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # pick your CUDA
pip install -r requirements.txt
```

`numba` (CUDA DRR renderer) is only needed to generate training data;
`itk-rtk` is only needed for the optional RTK cross-check in `realdata/`.

## Pretrained models

Weights are hosted on the Hugging Face Hub under **CC BY-NC 4.0**
(see [`WEIGHTS_LICENSE.md`](WEIGHTS_LICENSE.md)). Every model has the same
architecture (210 M parameters, 0.84 GB in fp32 safetensors) and differs only
in the view-sampling regime it was trained on.

| name | trained for | views | arc |
|---|---|---|---|
| `eureca_main` | general sparse-view CBCT | 8–100, log-uniform | 211° Halcyon half-scan, random start |
| `eureca_4view` | 4 fixed gantry angles | 4 | 0 / 45 / 90 / 135° |
| `eureca_2ortho` | 2 orthogonal views | 2 | 0 / 90° |
| `eureca_arc90` | limited arc | 15–45 | 90° |
| `eureca_arc120` | limited arc | 15–45 | 120° |

```bash
python download_weights.py                         # eureca_main  -> weights/eureca_main
python download_weights.py --model all --demo      # all five + the demo case
```

Each model folder contains `latest_net_<module>.safetensors` for the 17
sub-networks, `train_opt.ini` (full training configuration; `test*.py` replay
the architecture flags from it automatically) and `manifest.json` (parameter
counts and sha256).

## Quick start: reconstruct the demo case

The demo case is a reserved-test LUNA16 thorax CT (CC BY 3.0) rendered as 180
DRRs at 2° steps over 360°, with the 2 mm ground-truth volume in HU. It is
downloaded to `demo/data/luna16_27966/` by `download_weights.py --demo`.

```bash
bash demo/run_demo.sh                   # eureca_main, 20 views on a 211-degree arc
bash demo/run_demo.sh eureca_main 100   # 100 views
bash demo/run_demo.sh eureca_4view      # 4 views at 0/44/90/134 degrees
bash demo/run_demo.sh eureca_2ortho     # 0 and 90 degrees
```

The script writes `outputs/demo/<model>/luna16_27966_<N>views.nii.gz` (model,
values in [0,1] = HU −1000…1000), an analytic FDK from all 180 views for
comparison, and prints PSNR / SSIM against the ground truth. Expected scores on
this case (A40 GPU, bf16):

| command | views | PSNR (dB) | SSIM |
|---|---|---|---|
| `run_demo.sh eureca_main 20` | 20 on 211° | 27.59 | 0.849 |
| `run_demo.sh eureca_main 100` | 100 on 211° | 28.60 | 0.869 |
| `run_demo.sh eureca_4view` | 4 fixed | 27.53 | 0.838 |
| `run_demo.sh eureca_2ortho` | 2 orthogonal | 26.05 | 0.811 |
| `run_demo.sh eureca_arc90 30` | 30 on 90° | 27.86 | 0.855 |
| `run_demo.sh eureca_arc120 30` | 30 on 120° | 28.31 | 0.860 |

Under the hood it calls:

```bash
python test_npy.py @./options/geometry/hal.ini \
    --name eureca_main --checkpoints_dir weights \
    --npy_root demo/data --scan luna16_27966 --view_indices 0 6 11 ... \
    --angle_sign 1 --angle_offset 0 --fdk --output_dir outputs/demo
```

## Reconstruct your own projections

`test_npy.py` reads one folder per scan containing

* `projections.npy` — `(N, 384, 768)` float line integrals (−log of the
  flat-field-normalised intensity), resampled to 0.7273 mm at the isocenter
  plane and centre-cropped to 384 × 768. Any global scale is fine: the model
  normalises the whole stack by its maximum.
* `angles.npy` — `(N,)` gantry angles in degrees, 0–360, in the training
  convention (DRR convention of `ct_generate_drr.py`; for Halcyon XIM
  `KVSourceRtn` use `--angle_sign -1 --angle_offset -90`).

```bash
python test_npy.py @./options/geometry/hal.ini --name eureca_main --checkpoints_dir weights \
    --npy_root /path/to/scans --scan <scan_folder> --num_views 50 [--fdk] --output_dir outputs/nii
```

For Varian Halcyon data the `realdata/` folder contains the full pipeline we
used: XIM readers (`xim_fast.py`), preprocessing of raw frames
(`prepare_prostate_xim.py`: bowtie / blade / −log / magnification-corrected
resampling), geometry verification against the clinical reconstruction, and
`test_mat.py` for vendor-corrected iTools `Proj_*.mat` exports. See
[`realdata/README.md`](realdata/README.md). No patient data are distributed.

## Training

Training data are CT volumes; DRRs are rendered on the fly (CUDA, `numba`)
into cached per-scan projection libraries (`<scan>/hal_<k>/img/proj_<angle>.npy`).

1. **Download and arrange public CTs** with `scripts/download_*.sh` and
   `scripts/extract_*_ct.sh` (LUNA16, AMOS22, MELA, RibFrac, TotalSegmentator,
   AbdomenAtlas 3.0 Mini, RSNA 2023). Each dataset root should contain
   `<case_id>/<scan_id>.nii.gz`.
2. **Point `options/datasets/example.ini`** at your dataset roots
   (comma-separated).
3. **Launch** with `accelerate` (2 GPUs, bf16); the exact configuration of
   `eureca_main` is `scripts/train_main.slurm`, and the four specialists are
   `scripts/{fourview,twoortho,shortarc90,shortarc120}_hal.slurm`.

```bash
accelerate launch --mixed_precision=bf16 --multi_gpu --num_processes=2 train_single.py \
    @./options/datasets/example.ini @./options/geometry/hal.ini \
    --name my_run --checkpoints_dir ./checkpoints \
    --use_binned_aggr --use_rope --use_fdk \
    --encoder_backbone convnextv2_nano.fcmae_ft_in22k_in1k \
    --encoder_view_chunk 16 --backbone_lr_mult 0.1 --bin_ray_pe 1 --fdk_film 1 \
    --angle_select_mode halcyon --min_num_images 8 --max_num_images 100 --num_images_sampling loguniform \
    --lambda_L1 -1.0 --lambda_ssim -1.0 --blade_blocking --filter_max_res 5.0 --filter_min_phys_z 300.0
```

Cases are split 80 / 10 / 10 into train / val / reserved test (seed 42) and the
ids are written to `checkpoints/<run>/split_cases.json`; `test.py` evaluates on
the reserved split only. Add `--resume` to continue after a crash. Each run
writes about 3 GB (weights plus optimizer state). `smoke_test.py` runs one
forward/backward step on synthetic data to check memory before a full job.

Architecture flags (`--use_binned_aggr`, `--aggr_bins`, `--refine_depth`,
`--use_fdk`, `--fdk_film`, `--encoder_backbone`, grid sizes …) are saved to
`train_opt.ini` and replayed at test time, so `test*.py` only need `--name`.

## Repository layout

```
train.py / train_single.py    training (DDP via accelerate)
test.py                       DRR test harness on the reserved split (PSNR / SSIM / ST-MAE sweeps)
test_npy.py / test_mat.py     reconstruct from projections (.npy stack / iTools .mat) -> NIfTI
test_real.py                  reconstruct straight from a raw Halcyon scan folder
download_weights.py           fetch released weights + demo case from the Hugging Face Hub
demo/                         run_demo.sh, evaluate.py
models/                       the network (twod_to_threed_model*.py), encoder, aggregators, differentiable FDK (fdk.py)
options/                      argparse options; geometry presets (hal.ini, truebeam.ini); dataset preset example
ct_dataset.py                 manifests, CT resampling, DRR libraries, view sampling, collates
ct_generate_drr.py            CUDA DRR renderer
threeD_geometry.py            cone-beam geometry (Plücker rays, back-projection)
realdata/                     Halcyon XIM / iTools real-data pipeline
scripts/                      dataset download / extraction, training job scripts
freeze_model.py               snapshot code + weights + config into a self-contained bundle
docs/MODEL_CHANGES.md         design notes on the angular pooling and FDK fusion
```

## Citation

```bibtex
@article{zhu2026eureca,
  title   = {EUReCA: End-to-End Unified CBCT Reconstruction Across Clinical Acquisitions},
  author  = {Zhu, Jiening and Zhang, Chengzhu and Fan, Jason and Kuo, LiCheng and Cai, Weixing
             and He, Xiuxiu and Cervino, Laura and Moran, Jean and Li, Xiang and Li, Tianfang and Fu, Yabo},
  journal = {arXiv preprint arXiv:2610.37740},
  year    = {2026}
}
```

## License

Code: [Apache-2.0](LICENSE). Model weights: [CC BY-NC 4.0](WEIGHTS_LICENSE.md)
(the encoder is fine-tuned from Meta's ConvNeXt V2, itself CC BY-NC 4.0).
Demo data: derived from LUNA16 / LIDC-IDRI (CC BY 3.0).

## Acknowledgements

Public CT datasets used for training: LUNA16, AMOS22, MELA 2022, RibFrac,
TotalSegmentator, AbdomenAtlas 3.0 Mini and RSNA 2023 Abdominal Trauma
Detection. Analytic reconstruction cross-checks use [RTK](https://www.openrtk.org/).
