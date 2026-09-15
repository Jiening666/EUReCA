---
license: cc-by-3.0
task_categories:
  - image-to-3d
tags:
  - medical-imaging
  - cbct
  - cone-beam-ct
  - sparse-view-reconstruction
  - luna16
size_categories:
  - n<1K
pretty_name: EUReCA demo case
---

# EUReCA demo case

One simulated cone-beam acquisition for trying the
[EUReCA](https://github.com/Jiening666/EUReCA) CBCT reconstruction models
without any data preparation.

* **Source CT:** LUNA16 / LIDC-IDRI series
  `1.3.6.1.4.1.14519.5.2.1.6279.6001.154837327827713479309898027966` (thorax),
  CC BY 3.0. This case is in EUReCA's reserved test split and was never used
  for training or model selection.
* **Geometry:** Varian Halcyon on-board imager as simulated for training:
  SAD 1000 mm, detector 384 × 768 pixels at 0.7273 mm (isocenter plane),
  volume 256 × 256 × 128 voxels at 2 mm.

## Files (`luna16_27966/`)

| file | content |
|---|---|
| `projections.npy` | `(180, 384, 768)` float16, DRR line integrals at gantry angles 0, 2, …, 358°, divided by their global maximum (`scale` in `meta.json`) |
| `angles.npy` | `(180,)` float32 gantry angles in degrees, training convention |
| `gt_hu.nii.gz` | ground-truth CT resampled to the 2 mm reconstruction grid, int16 HU |
| `aug.ini` | resampling transform used (identity) |
| `meta.json` | provenance, geometry and scaling |

## Usage

```bash
git clone https://github.com/Jiening666/EUReCA.git && cd EUReCA
pip install -r requirements.txt
python download_weights.py --model eureca_main --demo
bash demo/run_demo.sh eureca_main 20        # -> PSNR / SSIM against gt_hu.nii.gz
```

The models expect views drawn from the arc they were trained on (a 211° arc for
`eureca_main`); `demo/run_demo.sh` selects them from this 360° library. Any
subset of views and angles can be passed to `test_npy.py --view_indices`.

## License and attribution

Derived from LUNA16 (https://luna16.grand-challenge.org/), which is based on
LIDC-IDRI, distributed under CC BY 3.0. Please cite LUNA16 and LIDC-IDRI when
using this case: Setio et al., *Medical Image Analysis* 42 (2017); Armato et
al., *Medical Physics* 38 (2011).
