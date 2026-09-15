"""Projection-domain verification of the real-data angle mapping.

The strongest possible check, and it involves neither FDK nor the network:
render a DRR of the scan's own clinical reconstruction at the angle we BELIEVE
each real projection was taken at, and compare it with that real projection.
If the mapping is right the two images must look like the same view of the
same patient (NCC high); if the mapping is wrong they cannot.

Also sweeps a constant angle offset over the full circle and reports the NCC
profile, so the correct offset is read off directly in the projection domain
-- independent of the volume-domain evidence (which FDK can satisfy even with
a 180 deg error, because backprojection is far more forgiving than a learned
prior).

    python realdata/verify_projection_match.py --prep_dir <prep2/scan> \
        [--sweep] [--n_probe 8]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def ncc(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prep_dir', required=True)
    p.add_argument('--n_probe', type=int, default=8, help='projections to test')
    p.add_argument('--sweep', action='store_true', help='sweep a constant offset')
    p.add_argument('--sweep_step', type=float, default=10.0)
    p.add_argument('--vox', type=float, default=2.0)
    p.add_argument('--psize', type=float, default=0.7273)
    p.add_argument('--save_png', default='')
    args = p.parse_args()

    import torch
    from models.renderer import DifferentiableDRR

    d = Path(args.prep_dir)
    projs = np.load(d / 'projs.npy')
    angles = np.load(d / 'angles.npy')       # already the mapped theta
    gt = np.load(d / 'gt.npy')
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    step = 0.5 * args.vox
    diag = (256 ** 2 + 256 ** 2 + 128 ** 2) ** 0.5 * args.vox
    drr = DifferentiableDRR(Nx=256, Ny=256, Nz=128, resx=args.vox, resy=args.vox,
                            resz=args.vox, Np=projs.shape[1], Nq=projs.shape[2],
                            psize_x=args.psize, psize_y=args.psize,
                            nSample=int(diag / step) + 2, sample_step=step,
                            SAD=1000.0).to(dev)
    xv = torch.from_numpy(gt).to(dev).permute(2, 1, 0).contiguous()[None]

    idx = np.linspace(0, len(angles) - 1, args.n_probe).astype(int)
    real = projs[idx]
    th = angles[idx].astype(np.float32)

    def render(offset):
        with torch.no_grad():
            a = torch.from_numpy(((th + offset) % 360.0).astype(np.float32)).to(dev)[None]
            return drr(xv, a)[0].cpu().numpy()

    syn = render(0.0)
    print(f'{"view":>5s} {"theta":>8s} {"NCC":>7s}   real[mean p99]      drr[mean p99]')
    per = []
    for j in range(len(idx)):
        c = ncc(syn[j], real[j]); per.append(c)
        print(f'{idx[j]:5d} {th[j]:8.1f} {c:7.3f}   '
              f'[{real[j].mean():.3g} {np.percentile(real[j],99):.3g}]  '
              f'[{syn[j].mean():.3g} {np.percentile(syn[j],99):.3g}]')
    print(f'\nmean NCC at current mapping: {np.mean(per):.4f}')
    if np.mean(per) > 0.9:
        print('  -> projections MATCH: angle mapping is right in projection domain')
    else:
        print('  -> MISMATCH: the loaded projections are not this view of the volume')

    if args.sweep:
        print('\noffset sweep (projection domain, independent of FDK/network):')
        best = []
        for off in np.arange(0, 360, args.sweep_step):
            s = render(float(off))
            m = float(np.mean([ncc(s[j], real[j]) for j in range(len(idx))]))
            best.append((m, float(off)))
            print(f'  offset {off:6.1f}  meanNCC={m:.4f}')
        best.sort(reverse=True)
        print(f'\nBEST offset {best[0][1]:+.1f} deg (meanNCC {best[0][0]:.4f}); '
              f'runner-up {best[1][1]:+.1f} ({best[1][0]:.4f})')
        print(f'current prep applies theta = GantryRtn + 180; this sweep is ON TOP '
              f'of that, so the total correction is 180 + {best[0][1]:.1f} deg')

    if args.save_png:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        n = min(4, len(idx))
        fig, ax = plt.subplots(2, n, figsize=(3.2 * n, 6))
        for j in range(n):
            ax[0, j].imshow(real[j], cmap='gray'); ax[0, j].set_title(f'real  {th[j]:.0f}°')
            ax[1, j].imshow(syn[j], cmap='gray'); ax[1, j].set_title('DRR(clinical)')
            for r in (0, 1):
                ax[r, j].set_xticks([]); ax[r, j].set_yticks([])
        fig.tight_layout(); fig.savefig(args.save_png, dpi=140)
        print(f'saved {args.save_png}')


if __name__ == '__main__':
    main()
