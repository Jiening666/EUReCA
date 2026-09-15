"""
test_real.py  —  Run cbct_recon model on real Halcyon CBCT projections.

Usage:
    python test_real.py \
        --real_data_root /path/to/cbct_scan \
        --name my_checkpoint \
        --checkpoints_dir ./checkpoints \
        [--output_dir ./outputs/real] \
        [--time_values 0.0 0.5 1.0] \
        [--time_tol 0.05] \
        [--split train] \
        [--num_projections 30] \
        [--use_wandb]

The scan folder must contain:
    meta_data.json   — projection metadata (angle_rad, time, file_path)
    Scan.xml         — Halcyon geometry XML (optional but recommended)
    <projection .npy files referenced in meta_data.json>

Outputs (per phase) are saved under --output_dir:
    phase_<t>/volume.npy          — reconstructed 3-D volume (Nx × Ny × Nz)
    phase_<t>/axial.png
    phase_<t>/coronal.png
    phase_<t>/sagittal.png
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
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
from recon_dataset import create_real_cbct_dataloader, load_all_phases


# ── helpers ──────────────────────────────────────────────────────────────────

def _save_slice_png(tensor_2d, path):
    """Save a (H, W) float32 tensor as a greyscale PNG."""
    try:
        from PIL import Image
    except ImportError:
        print("  [warn] Pillow not installed; skipping PNG save.")
        return
    arr = tensor_2d.cpu().float().numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    arr = (arr * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _save_volume_slices(vol, out_dir):
    """Save axial/coronal/sagittal mid-slices from vol (Nx, Ny, Nz) tensor."""
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


# ── main ─────────────────────────────────────────────────────────────────────

def build_opt():
    """Build options: start from TestOptions then override/add real-data args."""
    base = TestOptions()
    # We call gather_options so TestOptions registers all its flags first,
    # then we parse again with our extra flags appended.
    parser = base.initialize(
        argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            fromfile_prefix_chars='@',
        )
    )
    # real-data specific
    parser.add_argument('--real_data_root', type=str, required=True,
                        help='Path to folder containing meta_data.json + projections')
    parser.add_argument('--output_dir', type=str, default='',
                        help='Where to save outputs (default: checkpoints_dir/name/real_test)')
    parser.add_argument('--time_values', type=float, nargs='*', default=None,
                        help='Respiratory-phase time values to reconstruct (default: all in metadata)')
    parser.add_argument('--time_tol', type=float, default=0.025,
                        help='±tolerance for matching time values')
    parser.add_argument('--split', type=str, default='train',
                        help='Which split key in meta_data.json to read (default: train)')
    parser.add_argument('--num_projections', type=int, default=None,
                        help='Max projections per phase (None = use all available)')
    add_frozen_model_args(parser)
    parser.set_defaults(phase='test')

    argv = maybe_inject_frozen_args(parser)
    if argv is None:                       # not frozen: replay arch from train_opt.ini
        argv = maybe_inject_train_opt_args(parser)
    opt = parser.parse_args(argv)
    opt.isTrain = False
    base.opt = opt

    if not opt.output_dir:
        opt.output_dir = str(
            Path(opt.checkpoints_dir) / opt.name / "real_test"
        )
    return opt


def reconstruct_phase(model, image_data, device, num_projections=None, n_iters=1):
    """Run one reconstruction; optionally sub-sample projections and iterate.

    Input contract the model expects:
      * projections (1, N, H, W): attenuation/line-integral images at pixel
        size opt.psize (at isocenter), cropped to opt.H x opt.W, normalized by
        ONE global max over the whole N-view stack (per-volume, NOT per-image).
      * angles (1, N): gantry angles in DEGREES.
    n_iters > 1 runs iterative refinement (forward_iterative) for the FDK/
    iterative model; 1 = single pass.
    """
    projs   = image_data['projections']   # (1, N, H, W)
    angles  = image_data['angles']        # (1, N), degrees

    if num_projections is not None and projs.shape[1] > num_projections:
        idx    = torch.linspace(0, projs.shape[1] - 1, num_projections).long()
        projs  = projs[:, idx]
        angles = angles[:, idx]

    # Per-volume normalization (one global max over the N-view stack).
    # Idempotent if the loader already normalized to [0,1].
    projs = projs / projs.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-9)

    img = {'projections': projs, 'angles': angles, 'N': projs.shape[1]}
    with torch.no_grad():
        if n_iters > 1 and hasattr(model, 'forward_iterative'):
            vol = model.forward_iterative(img, None, device,
                                          n_iters=n_iters, train=False)
        else:
            vol = model.forward(img, None, device=device)   # (1, Nx, Ny, Nz)
    return vol.squeeze(0)   # (Nx, Ny, Nz)


def test_real(opt):
    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device

    # ── model ────────────────────────────────────────────────────────────────
    model = create_recon_model(opt)
    model.device = device
    model.setup(opt)
    for name in model.model_names:
        net = getattr(model, 'net' + name)
        setattr(model, 'net' + name, accelerator.prepare(net))
    model.eval()

    # ── wandb (optional) ─────────────────────────────────────────────────────
    use_wandb = getattr(opt, 'use_wandb', False)
    if use_wandb and accelerator.is_main_process:
        import wandb
        from wandb import Settings
        wandb.init(
            project="CBCT-real-test",
            name=opt.name,
            config=vars(opt),
            settings=Settings(init_timeout=300),
        )

    # ── load projections for all phases ──────────────────────────────────────
    print(f"\nLoading real CBCT projections from: {opt.real_data_root}")
    phase_batches = load_all_phases(
        data_root=opt.real_data_root,
        split=opt.split,
        time_values=opt.time_values,
        time_tol=opt.time_tol,
        H=opt.H,
        W=opt.W,
        psize=opt.psize,
    )
    print(f"Phases to reconstruct: {sorted(phase_batches.keys())}\n")

    out_root = Path(opt.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # ── per-phase reconstruction ──────────────────────────────────────────────
    for t, image_data in sorted(phase_batches.items()):
        tag = f"phase_{t:.2f}"
        n_proj = image_data['projections'].shape[1]
        print(f"[{tag}]  {n_proj} projections → reconstructing ...")

        vol = reconstruct_phase(model, image_data, device,
                                num_projections=opt.num_projections)

        phase_dir = out_root / tag
        phase_dir.mkdir(parents=True, exist_ok=True)

        # save volume
        vol_np = vol.cpu().float().numpy()
        np.save(phase_dir / "volume.npy", vol_np)
        print(f"  saved volume  {vol_np.shape}  →  {phase_dir / 'volume.npy'}")

        # save slice images
        _save_volume_slices(vol, phase_dir)
        print(f"  saved slices  →  {phase_dir}/{{axial,coronal,sagittal}}.png")

        # wandb: log slices
        if use_wandb and accelerator.is_main_process:
            import wandb
            mx, my, mz = vol.shape[0] // 2, vol.shape[1] // 2, vol.shape[2] // 2
            wandb.log({
                f"{tag}/axial":    _to_wandb_image(vol[:, :, mz]),
                f"{tag}/coronal":  _to_wandb_image(vol[:, my, :]),
                f"{tag}/sagittal": _to_wandb_image(vol[mx, :, :]),
                f"{tag}/n_proj":   n_proj,
            })

    print(f"\nDone. All outputs in: {out_root}")
    if use_wandb and accelerator.is_main_process:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    opt = build_opt()
    test_real(opt)
