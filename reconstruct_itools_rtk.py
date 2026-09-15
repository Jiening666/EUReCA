"""
Reconstruct an iTools-preprocessed Halcyon projection stack with RTK FDK.

This mirrors reconstruct_itools_fdk.py for the DataInspector/iTools test folder:
    itools_projs_iso.npy     (N, H, W) line-integral projections

By default, angles are the known Halcyon short-scan arc (-126..85 deg) mapped
directly into RTK geometry after wrapping to 0..360.  Keep --rtk_angle_mode
explicit when comparing conventions.
"""

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import itk
import nibabel as nib
import numpy as np
from itk import RTK as rtk


SAD = 1000.0
SID = 1540.0
PSIZE = 0.7273
DET = PSIZE * SID / SAD
NX, NY, NZ, VOX = 256, 256, 128, 2.0
IT = itk.Image[itk.F, 3]


def map_angles(angles, mode):
    if mode == "raw":
        mapped = angles
    elif mode == "offset":
        mapped = angles - 90.0
    elif mode == "legacy":
        mapped = -angles - 90.0
    elif mode == "varian":
        mapped = 180.0 - angles
    else:
        raise ValueError(f"unknown rtk_angle_mode {mode!r}")
    return np.mod(mapped, 360.0).astype(np.float32)


def save_nii(volume, path):
    img = nib.Nifti1Image(
        volume.astype(np.float32),
        np.diag([VOX, VOX, VOX, 1.0]).astype(np.float64),
    )
    img.header.set_xyzt_units("mm")
    nib.save(img, str(path))


def normalize_clip_p99(volume):
    """Match DifferentiableFDK's display normalization: negatives to 0, p99 to 1."""
    vol = np.clip(volume, 0.0, None)
    hi = float(np.percentile(vol, 99.0))
    return np.clip(vol / max(hi, 1e-8), 0.0, 1.0).astype(np.float32)


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
    parser.add_argument("--rtk_angle_mode",
                        choices=["raw", "offset", "legacy", "varian"],
                        default="raw")
    parser.add_argument("--n_views", type=int, default=None,
                        help="Uniformly subsample to this many views.")
    parser.add_argument("--output", default="",
                        help="Output .nii.gz path. Defaults to _diagnostic/rtk_itools_fdk_<mode>.nii.gz.")
    parser.add_argument("--output_norm", action="store_true",
                        help="Save clip(0)/p99 normalized volume in [0,1], matching DifferentiableFDK display normalization.")
    parser.add_argument("--no_flip_detector_v", action="store_true",
                        help="Do not flip detector rows before RTK. By default rows are flipped to match the repo FDK/DRR convention where projection row 0 is +z.")
    parser.add_argument("--no_parker", action="store_true",
                        help="Skip RTK Parker short-scan weighting.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = input_dir / "_diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = Path(args.output) if args.output else out_dir / f"rtk_itools_fdk_{args.rtk_angle_mode}.nii.gz"

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

    angles = map_angles(angles, args.rtk_angle_mode)
    if not args.no_flip_detector_v:
        projs = projs[:, ::-1, :]
    print(f"input: {projs.shape}")
    print(f"rtk_angle_mode={args.rtk_angle_mode}: {angles[0]:.2f} -> {angles[-1]:.2f} deg")
    print(f"flip_detector_v={not args.no_flip_detector_v}")

    pim = itk.image_from_array(np.ascontiguousarray(projs))
    pim.SetSpacing([DET, DET, 1.0])
    pim.SetOrigin([-(projs.shape[2] - 1) * DET / 2.0,
                   -(projs.shape[1] - 1) * DET / 2.0,
                   0.0])

    geo = rtk.ThreeDCircularProjectionGeometry.New()
    for angle in angles:
        geo.AddProjection(SAD, SID, float(angle))

    proj_input = pim
    if not args.no_parker:
        parker = rtk.ParkerShortScanImageFilter[IT].New()
        parker.SetInput(pim)
        parker.SetGeometry(geo)
        parker.Update()
        proj_input = parker.GetOutput()

    # RTK rotation axis is y, so put the longitudinal dimension on size_y.
    src = rtk.ConstantImageSource[IT].New()
    src.SetSpacing([VOX, VOX, VOX])
    src.SetSize([NX, NZ, NY])
    src.SetOrigin([-(NX - 1) * VOX / 2.0,
                   -(NZ - 1) * VOX / 2.0,
                   -(NY - 1) * VOX / 2.0])
    src.SetConstant(0.0)

    fdk = rtk.FDKConeBeamReconstructionFilter[IT].New()
    fdk.SetInput(0, src.GetOutput())
    fdk.SetInput(1, proj_input)
    fdk.SetGeometry(geo)

    t0 = time.time()
    fdk.Update()
    raw = itk.array_from_image(fdk.GetOutput())  # (z=NY, y=NZ, x=NX)
    vol = np.transpose(raw, (0, 2, 1)).clip(0, None)
    if args.output_norm:
        vol = normalize_clip_p99(vol)
    print(f"RTK FDK done {raw.shape} -> {vol.shape} in {time.time() - t0:.0f}s")

    save_nii(vol, output)
    maybe_save_axial_png(vol, output.with_suffix("").with_suffix(".png"))
    print(f"saved -> {output}")


if __name__ == "__main__":
    main()
