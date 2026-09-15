import time
from options.test_options import TestOptions
import os
import json
from ct_dataset import (
    scan_manifest, ScanLoader, CTAugmentor, SingleCTDataset,
    filter_manifest, sample_manifest, split_cases,
    ImageDataset, single_batch_collate,
)
import torch
torch.cuda.empty_cache()
from functools import partial
from accelerate import Accelerator
from pathlib import Path
from models import create_recon_model
import numpy as np
import wandb
from wandb import Settings
from ct_generate_drr import generate_drr
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from utils import get_logger_images
from pytorch_msssim import ssim as compute_ssim
import pandas as pd


def make_fixed_test_dataloader(aug_paths, opt, num_images, angle_mode):
    """Single image dataloader with a fixed projection count and a single angle mode."""
    dataset = ImageDataset(aug_paths, img_type=getattr(opt, 'img_type', 'npy'))
    collate = partial(
        single_batch_collate,
        select_mode=angle_mode,
        H=opt.H, W=opt.W,
        resw=opt.resw, resh=opt.resh, psize=opt.psize,
        min_num_images=num_images,
        max_num_images=num_images,
        add_noise=False,
        extra_angles=getattr(opt, 'extra_angles', ''),
    )
    return DataLoader(dataset, batch_size=len(aug_paths), shuffle=False,
                      collate_fn=collate, num_workers=0)


def _save_eval_nifti(vol_np, path, spacing):
    """Write an (Nx,Ny,Nz) float32 volume as .nii.gz with isotropic `spacing` mm."""
    import nibabel as nib
    affine = np.diag([spacing, spacing, spacing, 1.0]).astype(np.float64)
    img = nib.Nifti1Image(vol_np.astype(np.float32), affine)
    img.header.set_xyzt_units('mm')
    nib.save(img, str(path))


def _soft_tissue_mae_hu(pred, gt, hu_lo=-160.0, hu_hi=240.0):
    """Per-volume MAE in HU restricted to the soft-tissue window.

    pred/gt: (B, Nx, Ny, Nz) in [0,1] (== HU [-1000, 1000]). The mask is
    defined on GT only, so prediction errors cannot move voxels in/out of the
    window. Returns a list of B floats (NaN if a volume has no soft tissue).
    """
    lo = (hu_lo + 1000.0) / 2000.0
    hi = (hu_hi + 1000.0) / 2000.0
    out = []
    for b in range(pred.shape[0]):
        m = (gt[b] >= lo) & (gt[b] <= hi)
        if m.any():
            out.append(float((pred[b] - gt[b]).abs()[m].mean() * 2000.0))
        else:
            out.append(float('nan'))
    return out


def _gradient_error_hu(pred, gt):
    """Per-volume mean |∇pred − ∇gt| in HU/voxel (edge-fidelity metric).

    Forward differences along the three spatial axes, averaged. Sensitive to
    edge sharpness/placement in a way plain PSNR/SSIM is not (targets the
    gradient-loss ablation claim). pred/gt: (B, Nx, Ny, Nz) in [0,1].
    """
    err = None
    for axis in (1, 2, 3):
        e = (pred.diff(dim=axis) - gt.diff(dim=axis)).abs().flatten(1).mean(dim=1)
        err = e if err is None else err + e
    return (err / 3.0 * 2000.0).cpu().tolist()


def run_test_epoch(model, test_loader, device, opt, num_images, angle_mode,
                   log_metrics_fn=None,
                   log_visuals_fn=None, log_visual_freq=0,
                   save_nifti_dir=None):
    """One full test pass for a given (num_images, angle_mode) config.

    log_metrics_fn(batch_metrics_dict, step): called every batch for intermediate logging.
    log_visual_freq: call log_visuals_fn every N batches (0 = disabled).
    step = batch_idx is consistent across configs because the loader uses a
    pre-shuffled fixed ordering (indices permuted once with a fixed seed, shuffle=False).
    save_nifti_dir: if set, write float32 clip(0,1) pred/<case>.nii.gz and
    gt/<case>.nii.gz there (same layout the cbct_recon_compare test adapters
    produce, so eval/unified_eval.py scores all methods identically).
    """
    psnr_list, ssim_list, loss_list = [], [], []
    st_mae_list, grad_err_list = [], []
    if save_nifti_dir is not None:
        (Path(save_nifti_dir) / 'pred').mkdir(parents=True, exist_ok=True)
        (Path(save_nifti_dir) / 'gt').mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch_idx, test_data in enumerate(test_loader):
            batch_cts = test_data['ct_a'].squeeze(1)

            image_loader = make_fixed_test_dataloader(
                test_data['aug_img_path_a'], opt, num_images, angle_mode
            )

            if len(image_loader.dataset) == 0:
                print(f"[skip] batch {batch_idx}: no valid projection folders.")
                continue

            image_data = next(iter(image_loader))

            pred_vols = model.forward(image_data, None, device=device)
            gt = batch_cts.to(device)

            loss = model.compute_volume_loss(pred_vols, gt, model.vols_ref, model.ref_vol)
            loss_list.append(loss.item())

            pred_c = pred_vols.clamp(0, 1)
            gt_c = gt.clamp(0, 1)

            mse_per = ((pred_c - gt_c) ** 2).flatten(1).mean(dim=1)
            psnr_per = -10.0 * torch.log10(mse_per + 1e-8)
            batch_psnr = psnr_per.cpu().tolist()
            psnr_list.extend(batch_psnr)

            pred_5d, gt_5d = pred_c.unsqueeze(1), gt_c.unsqueeze(1)
            batch_ssim = []
            for b in range(pred_5d.shape[0]):
                s = compute_ssim(pred_5d[b:b+1], gt_5d[b:b+1], data_range=1.0, size_average=True)
                batch_ssim.append(s.item())
            ssim_list.extend(batch_ssim)

            batch_st_mae = _soft_tissue_mae_hu(pred_c, gt_c)
            batch_grad_err = _gradient_error_hu(pred_c, gt_c)
            st_mae_list.extend(batch_st_mae)
            grad_err_list.extend(batch_grad_err)

            if save_nifti_dir is not None:
                # scan_uid (case_id__scan_id) — a case can contain many scans;
                # keying dumps by case_id alone overwrites them (1226 files for
                # 1328 scans). Falls back to case_id for older datasets.
                case_ids = test_data.get('scan_uid', test_data['case_id'])
                for b in range(pred_c.shape[0]):
                    case = str(case_ids[b])
                    _save_eval_nifti(pred_c[b].cpu().numpy(),
                                     Path(save_nifti_dir) / 'pred' / f'{case}.nii.gz',
                                     opt.vol_size)
                    _save_eval_nifti(gt_c[b].cpu().numpy(),
                                     Path(save_nifti_dir) / 'gt' / f'{case}.nii.gz',
                                     opt.vol_size)

            if log_metrics_fn is not None:
                log_metrics_fn({
                    "loss": loss.item(),
                    "psnr": float(np.mean(batch_psnr)),
                    "ssim": float(np.mean(batch_ssim)),
                    "st_mae": float(np.nanmean(batch_st_mae)),
                    "grad_err": float(np.mean(batch_grad_err)),
                }, step=batch_idx)

            if (log_visual_freq > 0 and log_visuals_fn is not None
                    and batch_idx % log_visual_freq == 0):
                log_visuals_fn(pred_vols, gt, batch_idx)

    return {
        "loss": float(np.mean(loss_list)),
        "psnr": float(np.mean(psnr_list)),
        "ssim": float(np.mean(ssim_list)),
        "st_mae": float(np.nanmean(st_mae_list)) if st_mae_list else float('nan'),
        "grad_err": float(np.mean(grad_err_list)) if grad_err_list else float('nan'),
        "n_samples": len(psnr_list),
    }


def test(opt):
    accelerator = Accelerator(mixed_precision="bf16", log_with="wandb")
    accelerator.init_trackers(
        project_name="CBCT-reconstruction-test",
        config=vars(opt),
        init_kwargs={"wandb": {"settings": Settings(init_timeout=300)}},
    )
    accelerator.print(f"Running on {accelerator.num_processes} GPU(s), "
                      f"mixed precision: {accelerator.mixed_precision}")

    # Use batch_idx as a custom x-axis so all configs can log steps 0..N
    # independently without conflicting with wandb's monotonic global step.
    if accelerator.is_main_process:
        wandb.define_metric("batch_idx")
        wandb.define_metric("test/*", step_metric="batch_idx")
        wandb.define_metric("test_slices/*", step_metric="batch_idx")

    model = create_recon_model(opt)
    device = accelerator.device
    model.device = device
    model.setup(opt)

    # ── build manifest & test split ──────────────────────────────────────────
    all_datasets = opt.datasets.split(",")
    manifests = []
    for i, dataset in enumerate(all_datasets):
        manifest = scan_manifest(dataset, verbose=1)
        manifest['case_id'] = manifest['case_id'].apply(lambda x: f"D{i}_" + x)
        manifests.append(manifest)

    manifest_all = pd.concat(manifests, axis=0, ignore_index=True)
    avail = set(manifest_all.case_id.unique())

    # Prefer the exact RESERVED test ids the run recorded at train time. This is
    # robust to any later filter/universe drift and guarantees the train/val
    # cases are never evaluated here. Fall back to recomputing for older runs.
    # Which recorded split to evaluate on: 'test' (reserved, default) or 'val'
    # (for convergence-matched best-val checkpoint selection — NEVER touches the
    # reserved test set).
    _eval_split = getattr(opt, 'eval_split', 'test')
    split_path = Path(opt.checkpoints_dir) / opt.name / 'split_cases.json'
    if split_path.is_file():
        saved = json.loads(split_path.read_text())
        test_cases = np.array([c for c in saved[_eval_split] if c in avail])
        missing = len(saved[_eval_split]) - len(test_cases)
        print(f"[split] '{_eval_split}' set from {split_path}: {len(test_cases)} cases"
              + (f" ({missing} not in manifest)" if missing else ""))
    else:
        manifest_all = filter_manifest(
            manifest_all, max_res=opt.filter_max_res, min_phys_z=opt.filter_min_phys_z)
        manifest_all = sample_manifest(manifest_all, opt.max_dataset_size)
        # Sort so the recomputed split matches training's (which now sorts too);
        # manifest row order is not order-stable across runs.
        _, _, test_cases = split_cases(
            np.array(sorted(map(str, manifest_all.case_id.unique()))),
            test_size=0.2, val_frac=opt.val_frac, seed=42)
        print(f"[split] split_cases.json not found; recomputed reserved test set: "
              f"{len(test_cases)} cases (pass the training --filter_max_res/"
              f"--filter_min_phys_z to match the run's universe exactly)")

    # Cap test set so all configs run over the exact same fixed subset,
    # making step numbers comparable across configs in wandb.
    if opt.max_test_size < len(test_cases):
        test_cases = test_cases[:opt.max_test_size]
    print(f"Test cases: {len(test_cases)}")

    loader = ScanLoader(
        max_n_aug=opt.aug_max_aug, mode='continue',
        aug_generator=CTAugmentor(opt, generate_drr),
        aug_prefix=opt.aug_prefix,
        fixed_aug_index=(opt.eval_aug_index if getattr(opt, 'eval_aug_index', -1) >= 0 else None),
    )
    # Pre-shuffle once with a fixed seed so every config iterates the same order.
    # DataLoader shuffle=True re-permutes on each iter(), so different configs
    # would see different orderings. Instead we permute indices once, wrap with
    # Subset, and set shuffle=False.
    def _make_loader(dataset, seed=42):
        g = torch.Generator()
        g.manual_seed(seed)
        idx = torch.randperm(len(dataset), generator=g).tolist()
        return DataLoader(Subset(dataset, idx), batch_size=opt.batch_size, shuffle=False)

    single_test_dataset = SingleCTDataset(manifest_all, test_cases, loader)
    single_test_loader = _make_loader(single_test_dataset)
    print(f'Test set: {len(single_test_dataset)} cases, '
          f'{len(single_test_loader)} batches')

    for name in model.model_names:
        net = getattr(model, 'net' + name)
        setattr(model, 'net' + name, accelerator.prepare(net))
    model.eval()

    # ── build the matrix of configs to evaluate ──────────────────────────────
    sub_modes = [m.strip() for m in opt.angle_select_mode.split(",") if m.strip()]
    angle_modes = sub_modes if len(sub_modes) == 1 else sub_modes + [opt.angle_select_mode]

    # ── sweep ─────────────────────────────────────────────────────────────────
    scenarios = [("unpaired", single_test_loader)]

    eval_nifti_root = None
    if opt.save_eval_nifti and accelerator.is_main_process:
        eval_nifti_root = (Path(opt.eval_nifti_dir) if opt.eval_nifti_dir
                           else Path(opt.checkpoints_dir) / opt.name / 'eval_nifti')
        print(f"[eval_nifti] saving float32 pred/gt NIfTI under {eval_nifti_root}")

    results = []
    for scenario_name, test_loader in scenarios:
        for num_images in opt.num_images_list:
            for angle_mode in angle_modes:
                tag = f"{scenario_name}_n{num_images}_{angle_mode}"
                print(f"\n[Test] {tag}")

                # Default-arg capture (_tag=tag) avoids the Python loop-closure bug
                # where all closures would otherwise share the last loop value.
                def log_metrics(batch_metrics, step, _tag=tag):
                    if not accelerator.is_main_process:
                        return
                    wandb.log({
                        f"test/{_tag}/loss": batch_metrics["loss"],
                        f"test/{_tag}/psnr": batch_metrics["psnr"],
                        f"test/{_tag}/ssim": batch_metrics["ssim"],
                        f"test/{_tag}/st_mae": batch_metrics["st_mae"],
                        f"test/{_tag}/grad_err": batch_metrics["grad_err"],
                        "batch_idx": step,
                    })

                def log_visuals(pred, gt, batch_idx, _tag=tag):
                    if not accelerator.is_main_process:
                        return
                    visuals = model.generate_visualization_slices(pred, gt)
                    wandb.log({
                        f"test_slices/{_tag}": get_logger_images(visuals),
                        "batch_idx": batch_idx,
                    })

                metrics = run_test_epoch(
                    model, test_loader, device, opt,
                    num_images, angle_mode,
                    log_metrics_fn=log_metrics,
                    log_visuals_fn=log_visuals,
                    log_visual_freq=opt.log_test_visual_freq,
                    save_nifti_dir=(eval_nifti_root / f'n{num_images}_{angle_mode}'
                                    if eval_nifti_root is not None else None),
                )
                print(f"  loss={metrics['loss']:.4f}  PSNR={metrics['psnr']:.2f} dB  "
                      f"SSIM={metrics['ssim']:.4f}  ST-MAE={metrics['st_mae']:.1f} HU  "
                      f"gradErr={metrics['grad_err']:.2f} HU/vox  (n={metrics['n_samples']})")

                results.append({"scenario": scenario_name, "num_images": num_images,
                                 "angle_mode": angle_mode, **metrics})

    # ── summary ───────────────────────────────────────────────────────────────
    if accelerator.is_main_process:
        df = pd.DataFrame(results)
        print("\n===== Test Results =====")
        print(df.to_string(index=False))
        out_path = Path(opt.checkpoints_dir) / opt.name / "test_results.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Saved to {out_path}")
        if eval_nifti_root is not None:
            print("Score with the unified pipeline, e.g.:\n"
                  f"  python eval/unified_eval.py --method main --n_views <V> \\\n"
                  f"    --pred_dir {eval_nifti_root}/n<V>_<mode>/pred \\\n"
                  f"    --gt_dir {eval_nifti_root}/n<V>_<mode>/gt \\\n"
                  f"    --out_dir {eval_nifti_root}/n<V>_<mode>")

    accelerator.end_training()


if __name__ == '__main__':
    opt = TestOptions().parse()
    test(opt)
