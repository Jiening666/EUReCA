"""Find the correct detector-row placement (z registration) empirically.

prep_prostate_h5 centres the physical detector band and then offsets it by
band_centre/psize rows. Whether that offset is + or - depends on the row
orientation convention of the renderer/FDK, which is easier to measure than
to argue about: rolling the projection rows by d is exactly equivalent to
changing the applied row_shift by d, so we sweep d, reconstruct, and pick
the placement whose residual z misalignment against the clinical recon is
zero (and NCC is maximal).

    python realdata/calib_row_shift.py --prep_dir <prep> --angle_offset 180
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from realdata.refine_real_geometry import ncc, best_shift    # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prep_dir', required=True)
    p.add_argument('--sign', type=float, default=1.0)
    p.add_argument('--angle_offset', type=float, default=180.0)
    p.add_argument('--gamma', type=float, default=1.0)
    p.add_argument('--rolls', type=int, nargs='*', default=None,
                   help='row rolls to test (default: 0, +-row_shift, +-2*row_shift)')
    p.add_argument('--views', type=int, default=80)
    p.add_argument('--vox', type=float, default=2.0)
    p.add_argument('--psize', type=float, default=0.7273)
    args = p.parse_args()

    import torch
    from models.fdk import DifferentiableFDK

    d = Path(args.prep_dir)
    projs = np.load(d / 'projs.npy'); gantry = np.load(d / 'gantry.npy')
    gt = np.load(d / 'gt.npy'); zmask = np.load(d / 'zmask.npy')
    meta = json.loads((d / 'meta.json').read_text())
    rs = int(meta['row_shift'])
    rolls = args.rolls if args.rolls else [0, -rs, rs, -2 * rs, 2 * rs]
    m3 = np.zeros(gt.shape, bool); m3[:, :, zmask] = True

    sel = np.linspace(0, projs.shape[0] - 1, min(args.views, projs.shape[0])).astype(int)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    theta = ((args.sign * gantry[sel] + args.angle_offset) % 360.0).astype(np.float32)
    ang = torch.from_numpy(theta).to(dev)[None]
    fdk = DifferentiableFDK(Nx=256, Ny=256, Nz=128, res=args.vox,
                            Np=projs.shape[1], Nq=projs.shape[2], psize=args.psize,
                            SAD=1000.0, filter='hann', parker=True,
                            parker_gamma_sign=args.gamma,
                            normalize_output=False).to(dev)

    print(f'row_shift currently applied by prep: {rs} rows '
          f'({rs * args.psize:.1f} mm); testing rolls {rolls}')
    out = []
    for roll in rolls:
        P = np.roll(projs[sel], roll, axis=1).astype(np.float32)
        with torch.no_grad():
            v = fdk(torch.from_numpy(P).to(dev)[None], ang)[0, 0].clamp(min=0).cpu().numpy()
        s0 = ncc(v, gt, m3)
        sh = best_shift(v, gt)
        s1 = ncc(np.roll(v, -sh, axis=(0, 1, 2)), gt, m3)
        print(f'  roll {roll:+5d} rows ({roll*args.psize:+7.1f} mm): NCC={s0:.4f}  '
              f'residual shift={sh} vox ({sh*args.vox} mm)  NCC_aligned={s1:.4f}')
        out.append(dict(roll=int(roll), ncc=s0, shift_vox=sh.tolist(),
                        shift_mm=(sh * args.vox).tolist(), ncc_aligned=s1,
                        abs_z_mm=abs(float(sh[2] * args.vox))))

    out.sort(key=lambda r: (r['abs_z_mm'], -r['ncc']))
    best = out[0]
    print(f"\nBEST placement: roll={best['roll']} rows -> effective row_shift="
          f"{rs + best['roll']} rows; residual z={best['abs_z_mm']:.0f} mm, "
          f"NCC={best['ncc']:.4f}")
    (d / 'row_shift_calib.json').write_text(json.dumps(
        dict(prep_row_shift=rs, best_roll=best['roll'],
             effective_row_shift=rs + best['roll'], results=out), indent=1))


if __name__ == '__main__':
    main()
