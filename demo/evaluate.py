#!/usr/bin/env python
"""Score a reconstructed NIfTI against the demo ground truth.

    python demo/evaluate.py outputs/demo/luna16_27966_20views.nii.gz \
        --gt demo/data/luna16_27966/gt_hu.nii.gz

Uses the same definitions as test.py: volumes mapped to [0,1] (HU -1000..1000),
PSNR = -10 log10(MSE) over the whole volume, SSIM = 3-D SSIM (pytorch_msssim,
data_range 1). Model outputs are already in [0,1]; the GT NIfTI is in HU.
"""
import argparse

import nibabel as nib
import numpy as np
import torch
from pytorch_msssim import ssim


def to_unit(vol, is_hu):
    v = vol.astype(np.float32)
    if is_hu:
        v = (v + 1000.0) / 2000.0
    return np.clip(v, 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred", nargs="+", help="reconstructed .nii.gz (model output, [0,1])")
    ap.add_argument("--gt", required=True, help="ground-truth .nii.gz in HU")
    ap.add_argument("--pred_in_hu", action="store_true", help="set if the prediction is in HU")
    a = ap.parse_args()

    gt = to_unit(nib.load(a.gt).get_fdata(), is_hu=True)
    gt_t = torch.from_numpy(gt)[None, None]
    print(f"{'volume':<50} {'PSNR (dB)':>10} {'SSIM':>8}")
    for p in a.pred:
        pr = to_unit(nib.load(p).get_fdata(), is_hu=a.pred_in_hu)
        if pr.shape != gt.shape:
            raise SystemExit(f"{p}: shape {pr.shape} != gt {gt.shape}")
        mse = float(np.mean((pr - gt) ** 2))
        psnr = -10.0 * np.log10(mse + 1e-8)
        s = ssim(torch.from_numpy(pr)[None, None], gt_t, data_range=1.0, size_average=True).item()
        print(f"{p:<50} {psnr:>10.2f} {s:>8.4f}")


if __name__ == "__main__":
    main()
