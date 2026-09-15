#!/usr/bin/env python
"""
Standalone smoke test: build the model and run ONE forward + backward +
optimizer step on synthetic data at the real geometry.  Validates the compute
graph (encoder -> bin aggregation -> FDK injection -> refinement -> decoder ->
loss -> backward -> step) without the data pipeline, so it surfaces shape /
dtype / device / OOM errors fast before launching a full job.

Run on a single GPU:
    python smoke_test.py --encoder_backbone convnextv2_nano.fcmae_ft_in22k_in1k
    python smoke_test.py --n_views 100        # check peak memory at max N
    python smoke_test.py --encoder_backbone ''  # from-scratch encoder path
"""
import argparse, traceback
import numpy as np
import torch
from types import SimpleNamespace
from models import create_recon_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoder_backbone', default='convnextv2_nano.fcmae_ft_in22k_in1k')
    ap.add_argument('--n_views', type=int, default=8)
    ap.add_argument('--Nx', type=int, default=256)
    ap.add_argument('--Ny', type=int, default=256)
    ap.add_argument('--Nz', type=int, default=128)
    ap.add_argument('--H', type=int, default=384)
    ap.add_argument('--W', type=int, default=768)
    ap.add_argument('--total_arc', type=float, default=200.0, help='deg spanned by the views')
    ap.add_argument('--bf16', action='store_true', default=True)
    args = ap.parse_args()

    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    opt = SimpleNamespace(
        gpu_ids=[0] if dev != 'cpu' else [], isTrain=True,
        checkpoints_dir='./checkpoints', name='smoke',
        Nx=args.Nx, Ny=args.Ny, Nz=args.Nz, vol_size=2.0,
        H=args.H, W=args.W, psize=0.7273,
        use_binned_aggr=True, aggr_bins=36, aggr_layers=8, fuse_depth=0,
        refine_depth=12, use_rope=True,
        encoder_backbone=args.encoder_backbone, lr=2e-4, backbone_lr_mult=0.1,
        lambda_L1=-1.0, lambda_ssim=-1.0, lambda_ref=0.5, lambda_perc=1.0,
        use_fdk=True, use_ref_inject=0,
        proj_coverage_halflife=10.0, aggr_chunk_size=1024, aggr_vox_chunk=4096)

    m = create_recon_model(opt)
    for n in m.model_names:
        if isinstance(n, str):
            getattr(m, 'net' + n).to(dev)
    ng = [(g['lr'], len(g['params'])) for g in m.optimizer_base.param_groups]
    print(f"built {type(m).__name__} on {dev} | optim groups (lr,n): {ng}")

    B, N = 1, args.n_views
    img = {
        'projections': torch.rand(B, N, args.H, args.W, device=dev),
        'angles': torch.tensor(np.linspace(0, args.total_arc, N)[None], dtype=torch.float32),
    }
    target = torch.rand(B, args.Nx, args.Ny, args.Nz, device=dev)

    try:
        m.optimizer_base.zero_grad()
        ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if (args.bf16 and dev != 'cpu') \
            else torch.autocast(device_type='cpu', enabled=False)
        with ctx:
            vols = m.forward(img, None, device=dev)
            loss = m.compute_volume_loss(vols, target)
            loss = loss + m.compute_multi_scale_loss(vols, target)
        assert torch.isfinite(loss), "non-finite loss"
        loss.backward()
        m.optimizer_base.step()
        if dev != 'cpu':
            print(f"peak GPU mem: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
        print(f"vols {tuple(vols.shape)} | loss {loss.item():.4f} | "
              f"FDK gate c/m/f {m.loss_M_fdk_gate_coarse:.3f}/"
              f"{m.loss_M_fdk_gate_mid:.3f}/{m.loss_M_fdk_gate_fine:.3f}")
        print("=== SMOKE TEST PASSED ===")
    except Exception:
        traceback.print_exc()
        print("=== SMOKE TEST FAILED ===")
        raise SystemExit(1)


if __name__ == '__main__':
    main()
