"""Run the real-data adapter on a SYNTHETIC case with a known test.py score.

If the adapter reproduces test.py's number (~28.6 dB for D0_mela_0005 @10v),
the adapter is correct and the poor real-data result is data-driven. If it
lands ~15 dB, the adapter itself mis-feeds the network -- and the printed
input diff shows exactly where it differs from what test.py hands the model.

    python realdata/adapter_synth_check.py --scan_root <case scan_root> \
        --name hal_n2 --checkpoints_dir ./checkpoints
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from realdata.run_real_main import build_opt, arch_argv_from_train_opt   # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scan_root', required=True)
    p.add_argument('--aug', default='hal_0')
    p.add_argument('--n_views', type=int, default=10)
    p.add_argument('--name', default='hal_n2')
    p.add_argument('--checkpoints_dir', default='./checkpoints')
    p.add_argument('--epoch', default='latest')
    p.add_argument('--geometry_ini', default='./options/geometry/hal.ini')
    args, unknown = p.parse_known_args()

    import torch
    from models import create_recon_model
    from ct_dataset import ImageDataset, single_batch_collate
    from torch.utils.data import DataLoader
    from functools import partial

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    argv = (['--name', args.name, '--checkpoints_dir', args.checkpoints_dir,
             '--epoch', str(args.epoch)]
            + arch_argv_from_train_opt(args.checkpoints_dir, args.name)
            + list(unknown))
    opt = build_opt(argv, args.geometry_ini)
    print(f'opt geometry: H={opt.H} W={opt.W} psize={opt.psize} '
          f'Nx={opt.Nx} Ny={opt.Ny} Nz={opt.Nz} vol_size={getattr(opt,"vol_size","?")}')

    # Diff our opt against the recorded training options: anything that differs
    # and is NOT replayed by model.setup() silently builds a different network
    # from the one the checkpoint was trained for.
    tini = Path(args.checkpoints_dir) / args.name / 'train_opt.ini'
    if tini.exists():
        toks = [t for t in tini.read_text().split() if t.strip()]
        train_opts = {}
        i = 0
        while i < len(toks):
            if toks[i].startswith('--'):
                k = toks[i][2:]
                v = toks[i + 1] if i + 1 < len(toks) and not toks[i + 1].startswith('--') else 'True'
                train_opts[k] = v
                i += 2
            else:
                i += 1
        diffs = []
        for k, v in sorted(train_opts.items()):
            if not hasattr(opt, k):
                continue
            ours = getattr(opt, k)
            if str(ours) != str(v) and str(ours).lower() != str(v).lower():
                diffs.append(f'    {k}: ours={ours!r}  train={v!r}')
        print(f'\nopt vs train_opt.ini: {len(diffs)} differing keys')
        for d in diffs[:25]:
            print(d)

    aug = Path(args.scan_root) / args.aug
    # ---- 1. build image_data EXACTLY as test.py does -----------------------
    ds = ImageDataset([str(aug / 'img')], img_type=getattr(opt, 'img_type', 'npy'))
    collate = partial(single_batch_collate, select_mode='halcyon',
                      H=opt.H, W=opt.W, resw=opt.resw, resh=opt.resh,
                      psize=opt.psize, min_num_images=args.n_views,
                      max_num_images=args.n_views, add_noise=False,
                      extra_angles=getattr(opt, 'extra_angles', ''))
    ref = next(iter(DataLoader(ds, batch_size=1, shuffle=False,
                               collate_fn=collate, num_workers=0)))
    print(f"\ntest.py-style input: projections {tuple(ref['projections'].shape)} "
          f"dtype={ref['projections'].dtype} "
          f"range[{ref['projections'].min():.4f},{ref['projections'].max():.4f}] "
          f"mean={ref['projections'].mean():.4f}")
    print(f"  angles {tuple(ref['angles'].shape)} "
          f"[{ref['angles'].min():.1f},{ref['angles'].max():.1f}]  N={ref['N']}")

    # ---- 2. rebuild the SAME views through the adapter's own path ----------
    ang = ref['angles'][0].numpy()
    projs, angles_found = [], []
    for a in ang:
        f = None
        for cand in (aug / 'img' / f'proj_{a:.2f}.npy',):
            if cand.exists():
                f = cand; break
        if f is None:
            cands = sorted((aug / 'img').glob('proj_*.npy'))
            key = lambda q: abs(float(q.stem.split('_')[1]) - float(a))
            f = min(cands, key=key)
        projs.append(np.load(f).astype(np.float32))
        angles_found.append(float(f.stem.split('_')[1]))
    P = torch.from_numpy(np.stack(projs)).to(dev)[None]
    if P.shape[-2:] != (opt.H, opt.W):
        print(f'  [note] raw file size {tuple(P.shape[-2:])} != model ({opt.H},{opt.W})')
    P = P / torch.clamp(P.amax(dim=(1, 2, 3), keepdim=True), min=1e-9)
    A = torch.tensor(np.array(angles_found), dtype=torch.float32, device=dev)[None]
    print(f"adapter-style input: projections {tuple(P.shape)} "
          f"range[{P.min():.4f},{P.max():.4f}] mean={P.mean():.4f}")
    print(f"  angles [{A.min():.1f},{A.max():.1f}]")

    # ---- 3. model on both -------------------------------------------------
    model = create_recon_model(opt)
    model.device = torch.device(dev)
    model.setup(opt)
    for nn in model.model_names:
        setattr(model, 'net' + nn, getattr(model, 'net' + nn).to(dev))
    model.eval()

    gt = np.load(aug / 'ct_resampled.npz')['data'].astype(np.float32)
    gt = np.clip((gt - (-1000.0)) / 2000.0, 0, 1)

    def score(v):
        e = ((v - gt) ** 2).mean()
        return 10 * np.log10(1.0 / max(float(e), 1e-12))

    for label, batch in (('test.py path', {k: (v.to(dev) if torch.is_tensor(v) else v)
                                           for k, v in ref.items()}),
                         ('adapter path', {'projections': P, 'angles': A,
                                           'N': int(A.shape[1])})):
        with torch.no_grad():
            out = model.forward(batch, None, device=dev).clamp(0, 1)[0].cpu().numpy()
        print(f'{label:14s} -> PSNR {score(out):.2f} dB')
    print('\n(test.py reference for this case @10v: see results_per_sample.csv)')


if __name__ == '__main__':
    main()
