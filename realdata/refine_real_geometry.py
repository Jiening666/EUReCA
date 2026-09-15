"""Refine the real-scan geometry after the coarse convention check.

The coarse sweep (verify_prostate_geometry.py) fixes the ROTATION SIGN but
leaves two things open that both cap NCC:
  1. a constant angular offset that is not a multiple of 90 deg,
  2. a rigid shift between the clinical DICOM recon and our iso-centred grid
     (Halcyon writes couch coordinates: CouchVrt/-Lat/-Lng are non-zero, so
     the recon need not be centred on isocentre).

Stage A: fine offset sweep (default +-180 deg, 5 deg step) for the winning
         sign, NCC against the clinical recon on covered slices.
Stage B: at the best offset, estimate the residual rigid shift by 3-D
         cross-correlation and re-score after applying it; report the shift
         in mm so it can be folded into the prep (or into the GT resample).

    python realdata/refine_real_geometry.py --prep_dir <prep> [--sign -1]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def ncc(a, b, mask=None):
    if mask is not None:
        a, b = a[mask], b[mask]
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def best_shift(vol, ref, max_shift=20):
    """Integer voxel shift maximising correlation (FFT phase correlation)."""
    a = vol - vol.mean(); b = ref - ref.mean()
    F = np.fft.fftn(a) * np.conj(np.fft.fftn(b))
    cc = np.real(np.fft.ifftn(F / (np.abs(F) + 1e-9)))
    cc = np.fft.fftshift(cc)
    c = np.array(cc.shape) // 2
    sub = cc[c[0]-max_shift:c[0]+max_shift+1,
             c[1]-max_shift:c[1]+max_shift+1,
             c[2]-max_shift:c[2]+max_shift+1]
    idx = np.unravel_index(np.argmax(sub), sub.shape)
    return np.array(idx) - max_shift          # (dx,dy,dz) in voxels


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prep_dir', required=True)
    p.add_argument('--sign', type=float, default=-1.0, help='+1 or -1 on gantry')
    p.add_argument('--offsets', type=float, nargs=3, default=[-180, 180, 5],
                   metavar=('LO', 'HI', 'STEP'))
    p.add_argument('--views', type=int, default=80)
    p.add_argument('--gamma', type=float, default=1.0)
    p.add_argument('--vox', type=float, default=2.0)
    p.add_argument('--psize', type=float, default=0.7273)
    args = p.parse_args()

    import torch
    from models.fdk import DifferentiableFDK

    d = Path(args.prep_dir)
    projs = np.load(d / 'projs.npy'); gantry = np.load(d / 'gantry.npy')
    gt = np.load(d / 'gt.npy')
    zmask = np.load(d / 'zmask.npy')
    m3 = np.zeros(gt.shape, bool); m3[:, :, zmask] = True

    sel = np.linspace(0, projs.shape[0] - 1, min(args.views, projs.shape[0])).astype(int)
    P = torch.from_numpy(projs[sel].astype(np.float32))
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    P = P.to(dev)[None]
    fdk = DifferentiableFDK(Nx=256, Ny=256, Nz=128, res=args.vox,
                            Np=projs.shape[1], Nq=projs.shape[2], psize=args.psize,
                            SAD=1000.0, filter='hann', parker=True,
                            parker_gamma_sign=args.gamma,
                            normalize_output=False).to(dev)

    def recon(offset):
        th = (args.sign * gantry[sel] + offset) % 360.0
        with torch.no_grad():
            a = torch.from_numpy(th.astype(np.float32)).to(dev)[None]
            return fdk(P, a)[0, 0].clamp(min=0).cpu().numpy()

    lo, hi, st = args.offsets
    results = []
    for off in np.arange(lo, hi + 1e-6, st):
        v = recon(float(off))
        s = ncc(v, gt, m3)
        results.append((s, float(off)))
        print(f'  offset {off:+7.1f}  NCC={s:.4f}')
    results.sort(reverse=True)
    best_ncc, best_off = results[0]
    print(f'\nSTAGE A best: offset={best_off:+.1f} deg  NCC={best_ncc:.4f}')

    # Stage B: residual rigid shift at the best offset
    v = recon(best_off)
    sh = best_shift(v, gt)
    shifted = np.roll(v, -sh, axis=(0, 1, 2))
    s_after = ncc(shifted, gt, m3)
    print(f'STAGE B shift = {sh} voxels = {sh * args.vox} mm  ->  NCC={s_after:.4f}')

    (d / 'geometry_refined.json').write_text(json.dumps(dict(
        sign=args.sign, gamma=args.gamma, best_offset_deg=best_off,
        ncc_offset=best_ncc, shift_voxels=sh.tolist(),
        shift_mm=(sh * args.vox).tolist(), ncc_after_shift=s_after,
        top5=[(o, s) for s, o in results[:5]]), indent=1))
    print(f'wrote {d}/geometry_refined.json')


if __name__ == '__main__':
    main()
