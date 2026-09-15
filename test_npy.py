"""
test_npy.py — Reconstruct with the cbct_recon model from PREPROCESSED .npy.

Loads the projections/angles produced by prepare_prostate_xim.py
(<scan>/projections.npy + angles.npy), optionally sub-samples to a chosen
number of views, runs the model, and writes the reconstructed volume as
NIfTI (.nii.gz).  With --fdk it additionally reconstructs an analytic FDK
volume from ALL projections and saves it as <scan>_fdk_<N>views.nii.gz.

Examples
--------
# one scan, 30 views
python test_npy.py --name my_ckpt --checkpoints_dir ./checkpoints \
    --npy_root D:/Jiening/prostate/processed --scan anon007_hn --num_views 30

# every scan under the processed root, 60 views each
python test_npy.py --name my_ckpt --npy_root D:/Jiening/prostate/processed \
    --all --num_views 60 --output_dir ./outputs/nii

Input per scan folder (from prepare_prostate_xim.py):
    projections.npy            (N, H, W)  line integrals
    projections_modelnorm.npy  (N, H, W)  per-image max-normalised [optional]
    angles.npy                 (N,)       gantry angle [deg]

Subset / views:
    --num_views K   uniformly sub-sample to K views (default: all)
    --view_indices  explicit zero-based indices (overrides --num_views)
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import nibabel as nib
from accelerate import Accelerator

from frozen_model_utils import (
    add_frozen_model_args,
    maybe_inject_frozen_args,
    maybe_inject_train_opt_args,
    maybe_prepend_frozen_code_path,
)
maybe_prepend_frozen_code_path()
from options.test_options import TestOptions
from models import create_recon_model
from models.fdk import DifferentiableFDK


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_opt():
    base   = TestOptions()
    parser = base.initialize(
        argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            fromfile_prefix_chars='@',
        )
    )
    parser.add_argument('--npy_root', type=str, required=True,
                        help='Folder containing scan subfolders with '
                             'projections.npy / angles.npy')
    parser.add_argument('--scan', type=str, default=None,
                        help='Scan subfolder name (e.g. anon007_hn). '
                             'Omit with --all to process every scan.')
    parser.add_argument('--all', action='store_true',
                        help='Process every subfolder that has projections.npy')
    parser.add_argument('--num_views', type=int, default=None,
                        help='Uniformly sub-sample to this many views '
                             '(None = use all)')
    parser.add_argument('--view_indices', type=int, nargs='*', default=None,
                        help='Explicit zero-based view indices (overrides '
                             '--num_views)')
    parser.add_argument('--proj_file', type=str, default='projections.npy',
                        help='Projection .npy filename inside each scan folder')
    parser.add_argument('--fdk', action='store_true',
                        help='Also reconstruct an analytic FDK volume from ALL '
                             'projections (ignores --num_views) and save it as '
                             '<scan>_fdk_<N>views.nii.gz. Uses the base '
                             '--fdk_filter / --fdk_sad / --fdk_vox_chunk options.')
    parser.add_argument('--fdk_save_filter', type=str, default=None,
                        choices=['ramp', 'hann'],
                        help='Override the filter for the standalone --fdk volume only '
                             '(defaults to --fdk_filter). --fdk_pad / --no-fdk_pad and '
                             '--fdk_parker / --no-fdk_parker are shared base options.')
    parser.add_argument('--angle_sign', type=float, default=1.0,
                        help='Multiplies the stored angle BEFORE --angle_offset: '
                             'angle = (angle_sign*KVSourceRtn + angle_offset) % 360. '
                             'Use -1 to reverse rotation direction (mirror / '
                             'opposite chirality). NOTE the legacy convention '
                             '"angle = (180 - KVSourceRtn) % 360" is '
                             '--angle_sign -1 --angle_offset 180.')
    parser.add_argument('--angle_offset', type=float, default=-90.0,
                        help='Degrees added after --angle_sign to match the '
                             'training convention. Saved angles are the XIM '
                             'KVSourceRtn (= GantryRtn + 90). Default (sign=+1, '
                             'offset=-90) is a pure rotation; if the recon is '
                             'left-right MIRRORED vs the planning CT, you need a '
                             'reflection: --angle_sign -1 (e.g. -1/+180).')
    parser.add_argument('--output_dir', type=str, default='',
                        help='Where to write <scan>_<K>views.nii.gz '
                             '(default: checkpoints_dir/name/recon_nii)')
    add_frozen_model_args(parser)
    parser.set_defaults(phase='test')

    argv        = maybe_inject_frozen_args(parser)
    if argv is None:                       # not frozen: replay arch from train_opt.ini
        argv    = maybe_inject_train_opt_args(parser)
    opt         = parser.parse_args(argv)
    opt.isTrain = False
    base.opt    = opt
    if not opt.output_dir:
        opt.output_dir = str(Path(opt.checkpoints_dir) / opt.name / "recon_nii")
    return opt


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def select_views(N, num_views=None, view_indices=None):
    """Return sorted zero-based view indices."""
    if view_indices is not None:
        return np.asarray(sorted(view_indices), dtype=int)
    if num_views is not None and num_views < N:
        return np.linspace(0, N - 1, num_views).round().astype(int)
    return np.arange(N, dtype=int)


def load_scan(scan_dir, proj_file, num_views=None, view_indices=None,
              angle_offset=0.0, angle_sign=1.0):
    """
    Load + normalise projections and angles for one scan.

    Returns image_data dict (projections (1,K,H,W), angles (1,K)) and the
    selected view indices/angles.
    """
    scan_dir = Path(scan_dir)
    projs = np.load(scan_dir / proj_file).astype(np.float32)   # (N, H, W)
    angs  = np.load(scan_dir / "angles.npy").astype(np.float32)  # (N,)
    if projs.shape[0] != angs.shape[0]:
        raise ValueError(f"{scan_dir.name}: {projs.shape[0]} projections vs "
                         f"{angs.shape[0]} angles")

    # Map stored XIM KVSourceRtn to the training angle convention:
    #   angle = (angle_sign * KVSourceRtn + angle_offset) mod 360
    # angle_sign = -1 reverses rotation direction (mirror / chirality); e.g. the
    # legacy convention (180 - KVSourceRtn) is angle_sign=-1, angle_offset=180.
    if angle_sign != 1.0 or angle_offset:
        angs = np.mod(angle_sign * angs + angle_offset, 360.0).astype(np.float32)

    idx   = select_views(projs.shape[0], num_views, view_indices)
    projs = projs[idx]
    angs  = angs[idx]

    # Per-VOLUME global-max normalisation — matches single_batch_collate, which
    # divides the whole (N,H,W) stack by ONE max over all views. _load_proj does
    # NOT per-image normalise and DRRs are saved unnormalised, so this single
    # global scale (preserving relative inter-view intensity) is what the model
    # was trained on. (Per-image normalisation would flatten that and is wrong.)
    mx    = float(projs.max())
    projs = projs / max(mx, 1e-9)

    image_data = {
        'projections': torch.from_numpy(projs).unsqueeze(0),   # (1, K, H, W)
        'angles':      torch.from_numpy(angs).unsqueeze(0),    # (1, K)
        'N':           int(projs.shape[0]),
    }
    return image_data, idx, angs


# ---------------------------------------------------------------------------
# NIfTI output
# ---------------------------------------------------------------------------

def save_nii(vol_np, path, spacing):
    """Write (Nx,Ny,Nz) float volume as .nii.gz with isotropic `spacing` mm."""
    affine = np.diag([spacing, spacing, spacing, 1.0]).astype(np.float64)
    img = nib.Nifti1Image(vol_np.astype(np.float32), affine)
    img.header.set_xyzt_units('mm')
    nib.save(img, str(path))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    opt = build_opt()

    npy_root = Path(opt.npy_root)
    if opt.all:
        scans = sorted(d.name for d in npy_root.iterdir()
                       if (d / opt.proj_file).is_file())
    elif opt.scan:
        scans = [opt.scan]
    else:
        raise SystemExit("Specify --scan <folder> or --all")
    if not scans:
        raise SystemExit(f"No scans with {opt.proj_file} under {npy_root}")

    # Model
    accelerator = Accelerator(mixed_precision="bf16")
    device      = accelerator.device
    model = create_recon_model(opt)
    model.device = device
    model.setup(opt)
    for name in model.model_names:
        net = getattr(model, 'net' + name)
        setattr(model, 'net' + name, accelerator.prepare(net))
    model.eval()

    # Analytic FDK (optional) — same iso-plane geometry as the model.
    fdk = None
    if opt.fdk:
        fdk = DifferentiableFDK(
            opt.Nx, opt.Ny, opt.Nz, opt.vol_size, opt.H, opt.W, opt.psize,
            SAD=getattr(opt, 'fdk_sad', 1000.0),
            filter=(opt.fdk_save_filter or opt.fdk_filter),
            vox_chunk=getattr(opt, 'fdk_vox_chunk', 131072),
            parker=opt.fdk_parker, pad=opt.fdk_pad, cutoff=opt.fdk_cutoff,
        ).to(device)

    out_dir = Path(opt.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}\nVolume grid: {opt.Nx}x{opt.Ny}x{opt.Nz} "
          f"@ {opt.vol_size} mm\nAngle map: angle = ({opt.angle_sign:+.0f}*KVSourceRtn "
          f"{opt.angle_offset:+.1f}) mod 360"
          f"{'  | FDK: on (all views%s)' % (', Parker' if opt.fdk_parker else '') if fdk is not None else ''}\n")

    for scan in scans:
        scan_dir = npy_root / scan
        try:
            image_data, idx, angs = load_scan(
                scan_dir, opt.proj_file, opt.num_views, opt.view_indices,
                angle_offset=opt.angle_offset, angle_sign=opt.angle_sign)
        except FileNotFoundError as e:
            print(f"[skip] {scan}: {e}")
            continue

        K = image_data['N']
        print(f"=== {scan}: {K} views "
              f"(angles {angs.min():.1f}..{angs.max():.1f}°) ===")

        image_data['projections'] = image_data['projections'].to(device)
        with torch.no_grad():
            vol = model.forward(image_data, None, device=device)   # (1,Nx,Ny,Nz)
        vol_np = vol.squeeze(0).cpu().float().numpy()

        out_path = out_dir / f"{scan}_{K}views.nii.gz"
        save_nii(vol_np, out_path, opt.vol_size)
        # also keep the exact view selection for reproducibility
        np.save(out_dir / f"{scan}_{K}views_idx.npy", idx)
        print(f"    model  saved {vol_np.shape}  ->  {out_path}")

        # ── FDK over ALL projections ────────────────────────────────────────
        if fdk is not None:
            fdk_data, _, fdk_angs = load_scan(
                scan_dir, opt.proj_file, num_views=None, view_indices=None,
                angle_offset=opt.angle_offset, angle_sign=opt.angle_sign)
            N_all = fdk_data['N']
            with torch.no_grad():
                fvol = fdk(fdk_data['projections'].to(device),
                           fdk_data['angles'].to(device))   # (1,1,Nx,Ny,Nz)
            fvol_np = fvol.squeeze(0).squeeze(0).cpu().float().numpy()
            fdk_path = out_dir / f"{scan}_fdk_{N_all}views.nii.gz"
            save_nii(fvol_np, fdk_path, opt.vol_size)
            print(f"    fdk    saved {fvol_np.shape}  ({N_all} views)  ->  {fdk_path}")
        print()

    print("Done.")


if __name__ == '__main__':
    main()
