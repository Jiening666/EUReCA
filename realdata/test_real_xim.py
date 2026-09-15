"""
test_real_xim.py — Run cbct_recon model on raw Halcyon XIM projections.

Usage:
    python test_real_xim.py \\
        --real_data_root /path/to/scan_folder \\
        --name my_checkpoint \\
        --checkpoints_dir ./checkpoints \\
        [--output_dir ./outputs/real_xim] \\
        [--num_projections 30] \\
        [--no_bowtie_correction] \\
        [--xim_json subset.json] \\
        [--xim_indices 0 5 10 ...] \\
        [--xim_mask_file indicators.txt] \\
        [--use_wandb]

Scan folder must contain:
    Scan.xml
    Acquisitions/<id>/Proj_*.xim
    Calibrations/AIR-Bowtie-*/Current/FilterBowtie.xim   (optional)

Subset selection (at most one; priority: xim_json > xim_indices > xim_mask_file):
    --xim_json       JSON file with a list of int indices or filename strings
    --xim_indices    space-separated zero-based integer indices
    --xim_mask_file  path to a 0/1 file (.txt one per line, or .npy), one per XIM
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import numpy as np
import torch
from accelerate import Accelerator

from options.test_options import TestOptions
from models import create_recon_model
from xim_dataset import load_xim_projections


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _save_slice_png(tensor_2d, path):
    try:
        from PIL import Image
    except ImportError:
        print(f"  [warn] Pillow not installed; skipping {path}")
        return
    arr = tensor_2d.cpu().float().numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    Image.fromarray((arr * 255).astype(np.uint8)).save(path)


def _save_volume_slices(vol, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mx, my, mz = vol.shape[0] // 2, vol.shape[1] // 2, vol.shape[2] // 2
    _save_slice_png(vol[:, :, mz], out_dir / "axial.png")
    _save_slice_png(vol[:, my, :], out_dir / "coronal.png")
    _save_slice_png(vol[mx, :, :], out_dir / "sagittal.png")


def _to_wandb_image(tensor_2d):
    import wandb
    arr = tensor_2d.cpu().float().numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return wandb.Image((arr * 255).astype(np.uint8))


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

    # Data / output
    parser.add_argument('--real_data_root', type=str, required=True,
                        help='Scan folder containing Scan.xml, Acquisitions/, Calibrations/')
    parser.add_argument('--output_dir', type=str, default='',
                        help='Output directory (default: checkpoints_dir/name/real_test_xim)')
    parser.add_argument('--num_projections', type=int, default=None,
                        help='Uniformly sub-sample to this many projections '
                             'before reconstruction (None = use all loaded)')

    # Flat-field
    parser.add_argument('--no_bowtie_correction', action='store_true',
                        help='Skip bowtie flat-field correction '
                             '(fall back to per-image 99th-pct normalisation)')

    # Subset selection (priority: xim_json > xim_indices > xim_mask_file)
    parser.add_argument('--xim_json', type=str, default=None,
                        help='JSON file with a list of int indices or '
                             'filename strings (e.g. ["Proj_00010.xim", ...])')
    parser.add_argument('--xim_indices', type=int, nargs='*', default=None,
                        help='Zero-based indices of XIMs to load '
                             '(space-separated, e.g. --xim_indices 0 5 10)')
    parser.add_argument('--xim_mask_file', type=str, default=None,
                        help='Path to 0/1 indicator file (.txt one-per-line '
                             'or .npy), one entry per XIM in sorted order')

    parser.set_defaults(phase='test')
    opt         = parser.parse_args()
    opt.isTrain = False
    base.opt    = opt

    if not opt.output_dir:
        opt.output_dir = str(
            Path(opt.checkpoints_dir) / opt.name / "real_test_xim"
        )
    return opt


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

def reconstruct(model, image_data, device, num_projections=None):
    """Run one model forward pass; optionally uniformly sub-sample projections."""
    projs  = image_data['projections']   # (1, N, H, W)
    angles = image_data['angles']        # (1, N)

    if num_projections is not None and projs.shape[1] > num_projections:
        idx    = torch.linspace(0, projs.shape[1] - 1, num_projections).long()
        projs  = projs[:, idx]
        angles = angles[:, idx]

    batch = {'projections': projs, 'angles': angles, 'N': projs.shape[1]}
    with torch.no_grad():
        vol = model.forward(batch, None, device=device)   # (1, Nx, Ny, Nz)
    return vol.squeeze(0)                                 # (Nx, Ny, Nz)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def test_real_xim(opt):
    accelerator = Accelerator(mixed_precision="bf16")
    device      = accelerator.device

    # Model
    model = create_recon_model(opt)
    model.device = device
    model.setup(opt)
    for name in model.model_names:
        net = getattr(model, 'net' + name)
        setattr(model, 'net' + name, accelerator.prepare(net))
    model.eval()

    # wandb (optional)
    use_wandb = getattr(opt, 'use_wandb', False)
    if use_wandb and accelerator.is_main_process:
        import wandb
        from wandb import Settings
        wandb.init(
            project="CBCT-real-xim",
            name=opt.name,
            config=vars(opt),
            settings=Settings(init_timeout=300),
        )

    # Load XIM projections
    print(f"\nLoading XIM projections from: {opt.real_data_root}")
    image_data = load_xim_projections(
        scan_root=opt.real_data_root,
        H=opt.H,
        W=opt.W,
        psize=opt.psize,
        bowtie_correction=not opt.no_bowtie_correction,
        xim_indices=opt.xim_indices,
        xim_mask=opt.xim_mask_file,
        xim_json=opt.xim_json,
    )
    n_proj = image_data['N']
    print(f"Loaded {n_proj} projections → {image_data['projections'].shape}\n")

    # Move to device
    image_data['projections'] = image_data['projections'].to(device)
    image_data['angles']      = image_data['angles'].to(device)

    # Reconstruct
    print("Reconstructing …")
    vol = reconstruct(model, image_data, device,
                      num_projections=opt.num_projections)

    # Save outputs
    out_dir = Path(opt.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vol_np = vol.cpu().float().numpy()
    np.save(out_dir / "volume.npy", vol_np)
    print(f"Saved volume {vol_np.shape}  →  {out_dir / 'volume.npy'}")

    _save_volume_slices(vol, out_dir)
    print(f"Saved slices  →  {out_dir}/{{axial,coronal,sagittal}}.png")

    if use_wandb and accelerator.is_main_process:
        import wandb
        mx, my, mz = vol.shape[0] // 2, vol.shape[1] // 2, vol.shape[2] // 2
        wandb.log({
            "axial":    _to_wandb_image(vol[:, :, mz]),
            "coronal":  _to_wandb_image(vol[:, my, :]),
            "sagittal": _to_wandb_image(vol[mx, :, :]),
            "n_proj":   n_proj,
        })
        wandb.finish()

    print(f"\nDone. Outputs in: {out_dir}")


if __name__ == '__main__':
    opt = build_opt()
    test_real_xim(opt)
