"""
Reconstruct an iTools-preprocessed Halcyon projection stack with DifferentiableFDK.

The test folder produced by DataInspector/iTools contains:
    itools_projs_iso.npy     (N, H, W) line-integral projections
    Proj_*.mat               individual projection matrices, no angle metadata

So this script uses the known Halcyon short-scan arc by default and makes the
angle mapping explicit.  Use --angle_mode raw when the iTools projection order
already matches the model/renderer convention.  The old exploratory convention
was equivalent to --angle_mode legacy.

Example:
    python reconstruct_itools_fdk.py ^
        --input_dir D:/Jiening/prostate/processed/test ^
        --angle_mode raw ^
        --output D:/Jiening/prostate/processed/test/recon_fdk_raw.nii.gz
"""

import argparse
import os
from pathlib import Path

# Anaconda on Windows can load Intel OpenMP once through NumPy/MKL and again
# through PyTorch. Set this before importing either stack so the process can run.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import nibabel as nib
import numpy as np
import torch

from models.fdk import DifferentiableFDK


SAD = 1000.0
PSIZE = 0.7273
NX, NY, NZ, VOX = 256, 256, 128, 2.0


def map_angles(angles, mode):
    if mode == "raw":
        mapped = angles
    elif mode == "offset":
        mapped = angles - 90.0
    elif mode == "legacy":
        mapped = -angles - 90.0
    else:
        raise ValueError(f"unknown angle_mode {mode!r}")
    return np.mod(mapped, 360.0).astype(np.float32)


def save_nii(volume, path):
    img = nib.Nifti1Image(
        volume.astype(np.float32),
        np.diag([VOX, VOX, VOX, 1.0]).astype(np.float64),
    )
    img.header.set_xyzt_units("mm")
    nib.save(img, str(path))


def maybe_save_axial_png(volume, path):
    try:
        from PIL import Image
    except Exception:
        return

    sl = volume[:, :, volume.shape[2] // 2]
    lo, hi = np.percentile(sl, [1, 99.5])
    img = np.clip((sl - lo) / max(hi - lo, 1e-8), 0, 1)
    Image.fromarray((img * 255).astype(np.uint8)).save(path)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_dir", default="D:/Jiening/prostate/processed/test")
    parser.add_argument("--proj_file", default="itools_projs_iso.npy")
    parser.add_argument("--angles_file", default="",
                        help="Optional .npy angles. If absent, use --start_angle/--stop_angle.")
    parser.add_argument("--start_angle", type=float, default=-126.0)
    parser.add_argument("--stop_angle", type=float, default=85.0)
    parser.add_argument("--angle_mode", choices=["raw", "offset", "legacy"],
                        default="raw")
    parser.add_argument("--n_views", type=int, default=None,
                        help="Uniformly subsample to this many views.")
    parser.add_argument("--output", default="",
                        help="Output .nii.gz path. Defaults to _diagnostic/mine_itools_fdk_<mode>.nii.gz.")
    parser.add_argument("--filter", choices=["ramp", "hann"], default="ramp")
    parser.add_argument("--no_input_norm", action="store_true",
                        help="Feed projections at their original scale instead of dividing by the global max.")
    parser.add_argument("--no_output_norm", action="store_true",
                        help="Save raw FDK amplitudes instead of DifferentiableFDK's clip(0)/p99 normalized [0,1] volume.")
    parser.add_argument("--parker_gamma_sign", type=float, default=1.0,
                        choices=[-1.0, 1.0],
                        help="Fan-angle sign used by Parker short-scan weighting. Use -1 if one side stays over/under weighted.")
    parser.add_argument("--no_parker", action="store_true")
    parser.add_argument("--no_pad", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = input_dir / "_diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = Path(args.output) if args.output else out_dir / f"mine_itools_fdk_{args.angle_mode}.nii.gz"

    projs = np.load(input_dir / args.proj_file).astype(np.float32)
    if args.angles_file:
        angles = np.load(args.angles_file).astype(np.float32)
    else:
        angles = np.linspace(args.start_angle, args.stop_angle, projs.shape[0]).astype(np.float32)

    if projs.shape[0] != angles.shape[0]:
        raise ValueError(f"{projs.shape[0]} projections but {angles.shape[0]} angles")

    if args.n_views and args.n_views < projs.shape[0]:
        idx = np.linspace(0, projs.shape[0] - 1, args.n_views).round().astype(int)
        projs = projs[idx]
        angles = angles[idx]

    angles = map_angles(angles, args.angle_mode)
    if not args.no_input_norm:
        projs = projs / max(float(projs.max()), 1e-9)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"input: {projs.shape}, device={device}")
    print(f"angle_mode={args.angle_mode}: {angles[0]:.2f} -> {angles[-1]:.2f} deg")
    print(f"input_norm={not args.no_input_norm}, filter={args.filter}")

    fdk = DifferentiableFDK(
        NX, NY, NZ, VOX, projs.shape[1], projs.shape[2], PSIZE,
        SAD=SAD, filter=args.filter, parker=not args.no_parker,
        parker_gamma_sign=args.parker_gamma_sign,
        pad=not args.no_pad,
        normalize_output=not args.no_output_norm,
    ).to(device)

    with torch.no_grad():
        vol = fdk(
            torch.from_numpy(projs)[None].to(device),
            torch.from_numpy(angles)[None].to(device),
        ).squeeze().cpu().numpy()

    save_nii(vol, output)
    maybe_save_axial_png(vol, output.with_suffix("").with_suffix(".png"))
    print(f"saved {vol.shape} -> {output}")


if __name__ == "__main__":
    main()
