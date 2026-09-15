"""Control experiment: run SYNTHETIC DRRs of a real scan's clinical volume
through the SAME adapter path used for real projections.

Purpose. On real projections the main model scored 12.8 dB with NCC 0.44 while
a plain FDK of the same data reached 15.6 dB / NCC 0.78 -- structure disagreed
even though intensity statistics matched almost exactly. Two very different
causes are consistent with that:

  (A) the adapter feeds the network something mis-conventioned (row flip,
      transpose, angle handedness): FDK tolerates such errors far better than a
      learned prior does, so FDK can look fine while the network collapses;
  (B) a genuine real-vs-synthetic domain gap (scatter / beam hardening /
      detector response absent from the training DRRs).

This script discriminates them. It renders DRRs of the scan's own clinical
reconstruction with the repo's DifferentiableDRR, at the SAME angles as a real
draw, and pushes them through the identical adapter code path. Then:

  ~30 dB  -> adapter + orientation are correct; the real-data drop is (B).
  ~13 dB  -> the adapter is mis-feeding the network; (A), and fixable.

    python realdata/adapter_control.py --prep_dir <prep2/scan> \
        --draws draws.json --name hal_n2 --checkpoints_dir <ckpt>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from realdata.run_real_main import build_opt, save_nii     # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prep_dir', required=True)
    p.add_argument('--draws', required=True)
    p.add_argument('--scan', default=None)
    p.add_argument('--n_views', type=int, default=10)
    p.add_argument('--draw_k', type=int, default=0)
    p.add_argument('--name', default='hal_n2')
    p.add_argument('--checkpoints_dir', default='./checkpoints')
    p.add_argument('--epoch', default='latest')
    p.add_argument('--out_dir', default='')
    p.add_argument('--vox', type=float, default=2.0)
    p.add_argument('--psize', type=float, default=0.7273)
    args, unknown = p.parse_known_args()

    import torch
    from models import create_recon_model
    from models.renderer import DifferentiableDRR
    from models.fdk import DifferentiableFDK

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    prep = Path(args.prep_dir)
    scan = args.scan or prep.name
    gt = np.load(prep / 'gt.npy').astype(np.float32)
    valid = np.load(prep / 'validmask.npy')
    real_projs = np.load(prep / 'projs.npy')
    angles_all = np.load(prep / 'angles.npy')

    man = json.loads(Path(args.draws).read_text())
    d = man['scans'][scan]['draws'][str(args.n_views)][args.draw_k]
    idx = np.asarray(d['idx'], int)
    ang = angles_all[idx].astype(np.float32)
    print(f'{scan}: {args.n_views} views (draw {args.draw_k}), angles '
          f'{ang.min():.1f}..{ang.max():.1f}')

    # --- synthetic DRRs of the clinical volume, same angles ------------------
    step = 0.5 * args.vox
    diag = (256 ** 2 + 256 ** 2 + 128 ** 2) ** 0.5 * args.vox
    drr = DifferentiableDRR(Nx=256, Ny=256, Nz=128, resx=args.vox, resy=args.vox,
                            resz=args.vox, Np=384, Nq=768, psize_x=args.psize,
                            psize_y=args.psize, nSample=int(diag / step) + 2,
                            sample_step=step, SAD=1000.0).to(dev)
    A = torch.from_numpy(ang).to(dev)[None]
    with torch.no_grad():
        xv = torch.from_numpy(gt).to(dev).permute(2, 1, 0).contiguous()
        syn = drr(xv[None], A)                       # (1,N,384,768)

    # --- model on BOTH inputs, identical adapter path ------------------------
    argv = ['--name', args.name, '--checkpoints_dir', args.checkpoints_dir,
            '--epoch', str(args.epoch)] + list(unknown)
    opt = build_opt(argv)
    model = create_recon_model(opt)
    model.device = torch.device(dev)
    model.setup(opt)
    for nn in model.model_names:
        setattr(model, 'net' + nn, getattr(model, 'net' + nn).to(dev))
    model.eval()

    fdk = DifferentiableFDK(Nx=256, Ny=256, Nz=128, res=args.vox, Np=384, Nq=768,
                            psize=args.psize, SAD=1000.0, filter='hann',
                            parker=True, parker_gamma_sign=1.0,
                            normalize_output=True).to(dev)

    def norm(P):
        return P / torch.clamp(P.amax(dim=(1, 2, 3), keepdim=True), min=1e-9)

    def score(v):
        a = v[valid].astype(np.float64); b = gt[valid].astype(np.float64)
        mse = float(((a - b) ** 2).mean())
        am, bm = a - a.mean(), b - b.mean()
        ncc = float((am * bm).sum() / (np.linalg.norm(am) * np.linalg.norm(bm) + 1e-9))
        return 10 * np.log10(1.0 / max(mse, 1e-12)), ncc

    real = torch.from_numpy(real_projs[idx].astype(np.float32)).to(dev)[None]
    out = Path(args.out_dir or (prep / 'adapter_control'))
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'input':22s} {'model PSNR':>11s} {'model NCC':>10s} "
          f"{'FDK PSNR':>9s} {'FDK NCC':>8s}")
    for label, P in (('SYNTHETIC DRR(gt)', syn), ('REAL projections', real)):
        Pn = norm(P)
        with torch.no_grad():
            pv = model.forward({'projections': Pn, 'angles': A,
                                'N': int(len(idx))}, None, device=dev)
            fv = fdk(Pn, A)[0, 0].clamp(0, 1).cpu().numpy()
        mv = pv.clamp(0, 1)[0].cpu().numpy()
        mp, mn = score(mv); fp, fn_ = score(fv)
        print(f'{label:22s} {mp:11.2f} {mn:10.3f} {fp:9.2f} {fn_:8.3f}')
        save_nii(mv, out / f'model_{label.split()[0].lower()}.nii.gz', args.vox)
        save_nii(fv, out / f'fdk_{label.split()[0].lower()}.nii.gz', args.vox)
    save_nii(gt, out / 'gt.nii.gz', args.vox)
    print(f'\nvolumes -> {out}')
    print('READ: synthetic ~30 dB => adapter fine, real drop is a domain gap;'
          '  synthetic also low => adapter mis-feeds the network.')


if __name__ == '__main__':
    main()
