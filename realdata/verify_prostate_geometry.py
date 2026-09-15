"""Resolve the iTools -> our-convention angle mapping empirically.

For every candidate mapping in prep_prostate_h5.ANGLE_MODES, FDK-reconstruct
the real projections with the repo's DifferentiableFDK and score the result
against the clinical DICOM recon with NCC (scale/offset invariant, so the
unknown line-integral units do not matter). The winner is the convention.

Also reports a sanity check on the arc: our synthetic halcyon runs span 211
deg, and the dump's StartAngle/StopAngle should agree once mapped.

    python realdata/verify_prostate_geometry.py --prep_dir <prep out> \
        [--modes gantry neg_gantry ...] [--views 100]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from realdata.prep_prostate_h5 import ANGLE_MODES        # noqa: E402


def ncc(a, b, mask=None):
    if mask is not None:
        a = a[mask]; b = b[mask]
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prep_dir', required=True)
    p.add_argument('--modes', nargs='*', default=list(ANGLE_MODES))
    p.add_argument('--views', type=int, default=100,
                   help='subsample this many projections for the probe')
    p.add_argument('--Nx', type=int, default=256); p.add_argument('--Ny', type=int, default=256)
    p.add_argument('--Nz', type=int, default=128); p.add_argument('--vox', type=float, default=2.0)
    p.add_argument('--psize', type=float, default=0.7273)
    p.add_argument('--SAD', type=float, default=1000.0)
    p.add_argument('--parker', action='store_true', default=True)
    args = p.parse_args()

    import torch
    from models.fdk import DifferentiableFDK

    d = Path(args.prep_dir)
    projs = np.load(d / 'projs.npy')
    gantry = np.load(d / 'gantry.npy')
    gt = np.load(d / 'gt.npy')
    zmask = np.load(d / 'zmask.npy') if (d / 'zmask.npy').exists() else None
    meta = json.loads((d / 'meta.json').read_text())
    print(f'projs {projs.shape}  gantry {gantry.min():.1f}..{gantry.max():.1f}  '
          f'arc {abs(gantry.max()-gantry.min()):.1f} deg')

    n = projs.shape[0]
    sel = np.linspace(0, n - 1, min(args.views, n)).astype(int)
    P = projs[sel]
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    mask3d = None
    if zmask is not None:
        mask3d = np.zeros(gt.shape, bool); mask3d[:, :, zmask] = True
        print(f'scoring on {int(zmask.sum())}/{gt.shape[2]} covered slices')

    results = []
    for mode in args.modes:
        theta = np.asarray(ANGLE_MODES[mode](gantry[sel]), np.float32) % 360.0
        for gamma_sign in (-1.0, 1.0):
            fdk = DifferentiableFDK(
                Nx=args.Nx, Ny=args.Ny, Nz=args.Nz, res=args.vox,
                Np=projs.shape[1], Nq=projs.shape[2], psize=args.psize,
                SAD=args.SAD, filter='hann', parker=args.parker,
                parker_gamma_sign=gamma_sign, normalize_output=False).to(dev)
            with torch.no_grad():
                b = torch.from_numpy(P.astype(np.float32)).to(dev)[None]
                a = torch.from_numpy(theta).to(dev)[None]
                vol = fdk(b, a)[0, 0].clamp(min=0).cpu().numpy()
            s = ncc(vol, gt, mask3d)
            results.append((s, mode, gamma_sign))
            print(f'  {mode:16s} gamma{gamma_sign:+.0f}  NCC={s:.4f}')

    results.sort(reverse=True)
    best = results[0]
    print(f'\nBEST: angle_mode={best[1]}  parker_gamma_sign={best[2]:+.0f}  NCC={best[0]:.4f}')
    print('Runner-up:', results[1])
    (Path(args.prep_dir) / 'geometry_verdict.json').write_text(json.dumps(
        dict(best_mode=best[1], best_gamma_sign=best[2], best_ncc=best[0],
             ranking=[(m, g, s) for s, m, g in results]), indent=1))


if __name__ == '__main__':
    main()
