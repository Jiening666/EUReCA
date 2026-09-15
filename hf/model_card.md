---
license: cc-by-nc-4.0
library_name: pytorch
pipeline_tag: image-to-3d
tags:
  - medical-imaging
  - cbct
  - cone-beam-ct
  - sparse-view-reconstruction
  - radiotherapy
  - safetensors
---

# EUReCA: End-to-End Unified CBCT Reconstruction Across Clinical Acquisitions

Pretrained weights for **EUReCA**, a single geometry-aware feed-forward network
that reconstructs a cone-beam CT volume (256 × 256 × 128 at 2 mm) from 2 to 100
projections on a clinical Varian Halcyon geometry: full or partial arcs, uniform
or arbitrary gantry angles, one set of weights per acquisition regime.

* **Code:** https://github.com/Jiening666/EUReCA
* **Project page / paper:** https://jiening666.github.io/cbct_recon/
* **Demo projections:** https://huggingface.co/datasets/jzhu35/EUReCA-demo

## Models

All five models share one architecture (210 M parameters, fp32 safetensors,
0.84 GB each) and differ only in the view-sampling regime they were trained on.

| folder | trained for | views | arc |
|---|---|---|---|
| `eureca_main` | general sparse-view CBCT | 8–100, log-uniform | 211° Halcyon half-scan, random start |
| `eureca_4view` | 4 fixed gantry angles | 4 | 0 / 45 / 90 / 135° |
| `eureca_2ortho` | 2 orthogonal views | 2 | 0 / 90° |
| `eureca_arc90` | limited arc | 15–45 | 90° |
| `eureca_arc120` | limited arc | 15–45 | 120° |

Each folder contains `latest_net_<module>.safetensors` for the 17 sub-networks
(encoder, ModLN, view aggregators, 3-D refinement transformer, FDK encoder and
gates, decoder), `train_opt.ini` with the complete training configuration and
`manifest.json` with per-module parameter counts and sha256 checksums.

## Usage

```bash
git clone https://github.com/Jiening666/EUReCA.git && cd EUReCA
pip install -r requirements.txt
python download_weights.py --model eureca_main --demo
bash demo/run_demo.sh eureca_main 20
```

Or fetch a folder directly:

```python
from huggingface_hub import snapshot_download
snapshot_download("jzhu35/EUReCA", allow_patterns=["eureca_main/*"], local_dir="weights")
```

Input: `(N, 384, 768)` line-integral projections resampled to 0.7273 mm at the
isocenter plane (SAD 1000 mm) plus gantry angles in degrees. Output: volume in
[0, 1] mapping to HU −1000 … 1000. See the GitHub README for the exact
conventions and for Varian XIM / iTools preprocessing.

## Training data

Trained on digitally reconstructed radiographs of public CT datasets (LUNA16,
AMOS22, MELA 2022, RibFrac, TotalSegmentator, AbdomenAtlas 3.0 Mini, RSNA 2023
Abdominal Trauma) and institutional planning CTs, validated on real Halcyon
HyperSight projections. No patient data are distributed.

## Intended use and limitations

Research use in CBCT reconstruction. The models are tuned to the Halcyon
on-board-imager geometry (SAD 1000 mm, SID 1540 mm, 384 × 768 detector at
0.7273 mm isocenter pitch, 2 mm voxels); other geometries need resampling to
this detector grid and may degrade. Not a medical device; not for clinical
decision-making.

## License

**CC BY-NC 4.0.** The 2D encoder is fine-tuned from Meta AI's ConvNeXt V2
(`convnextv2_nano.fcmae_ft_in22k_in1k`, CC BY-NC 4.0), so the derived weights
inherit the non-commercial restriction. The source code is Apache-2.0.

## Citation

```bibtex
@article{zhu2026eureca,
  title   = {EUReCA: End-to-End Unified CBCT Reconstruction Across Clinical Acquisitions},
  author  = {Zhu, Jiening and Zhang, Chengzhu and Fan, Jason and Kuo, LiCheng and Cai, Weixing
             and He, Xiuxiu and Cervino, Laura and Moran, Jean and Li, Xiang and Li, Tianfang and Fu, Yabo},
  journal = {IEEE Transactions on Medical Imaging (under review)},
  year    = {2026}
}
```
