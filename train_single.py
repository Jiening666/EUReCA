import time
from options.train_options import TrainOptions
import os
import shutil
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"
from ct_dataset import create_single_img_dataloader

import torch
import torch.nn.functional as F
torch.cuda.empty_cache()
#torch.autograd.set_detect_anomaly(True)
from tqdm import tqdm

import random
import math
from math import sqrt
from datetime import timedelta
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs
from pathlib import Path
from models import create_recon_model
import numpy as np

import wandb
from wandb import Settings

from ct_dataset import scan_manifest, ScanLoader, CTAugmentor, SingleCTDataset, filter_manifest, sample_manifest, split_cases
from ct_dataset import plan_case_draws, load_case_views, build_draw, build_angle_map
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from ct_generate_drr import generate_drr
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split
import json


from utils import get_logger_images
from pytorch_msssim import ssim as compute_ssim

import pandas as pd

#import torch.multiprocessing as mp

def cuda_check(name, tensor):
    torch.cuda.synchronize()
    print(f"[OK] {name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, contiguous={tensor.is_contiguous()}")


def iter_lr_multiplier(t, warmup_iters, total_iters, eta_min_frac=0.0):
    """LR multiplier in [eta_min_frac, 1] as a function of the global step `t`.

    Iteration-space analogue of the epoch-based cosine_warmup: linear ramp over
    [0, warmup_iters), then cosine decay to eta_min_frac over the remainder.
    A pure function of t, so it is identical across dataset sizes (matched
    schedule) and needs no scheduler state for --resume (t == total_iters,
    which is already checkpointed).
    """
    if warmup_iters > 0 and t < warmup_iters:
        return float(t + 1) / float(warmup_iters)
    denom = max(1, total_iters - warmup_iters)
    progress = min(1.0, float(t - warmup_iters) / float(denom))
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return eta_min_frac + (1.0 - eta_min_frac) * cos


class _TrainState:
    """Counters carried through accelerator.save_state/load_state (full resume)."""
    def __init__(self):
        self.total_iters = 0
        self.epoch = 0
    def state_dict(self):
        return {'total_iters': self.total_iters, 'epoch': self.epoch}
    def load_state_dict(self, sd):
        self.total_iters = int(sd.get('total_iters', 0))
        self.epoch = int(sd.get('epoch', 0))


def run_validation(model, val_loader, device, opt, max_batches=0, want_loss=False):
    """Validation pass over `val_loader` (the reserved-test-free VAL split).

    max_batches <= 0 runs the entire val set; otherwise stops after that many
    batches (the lightweight "quick" signal). Runs on every rank (collective-safe
    for DDP); metrics are this rank's shard mean. Returns a dict:
        psnr, ssim, n, vis_vols, vis_gt, vis_nviews[, loss]
    vis_* are the LAST batch's volumes so an optional slice panel stays consistent
    with model.fdk_vol (which reflects the most recent forward)."""
    psnr_list, ssim_list, loss_list = [], [], []
    vis_vols, vis_gt, vis_nviews = None, None, None
    model.eval()
    with torch.no_grad():
        for n, val_data in enumerate(val_loader):
            if max_batches and n >= max_batches:
                break
            gt = val_data['ct_a'].squeeze(1).to(device)
            val_image_loader = create_single_img_dataloader(val_data['aug_img_path_a'], opt)
            val_image_data = next(iter(val_image_loader))
            vols = model.forward(val_image_data, None, device=device)
            if want_loss:
                loss = model.compute_volume_loss(vols, gt, model.vols_ref, model.ref_vol)
                loss_list.append(loss.item())
            pred_c, gt_c = vols.clamp(0, 1), gt.clamp(0, 1)
            mse = ((pred_c - gt_c) ** 2).flatten(1).mean(dim=1)
            psnr_list.extend((-10.0 * torch.log10(mse + 1e-8)).cpu().tolist())
            p5, g5 = pred_c.unsqueeze(1), gt_c.unsqueeze(1)
            for b in range(p5.shape[0]):
                ssim_list.append(compute_ssim(p5[b:b+1], g5[b:b+1],
                                              data_range=1.0, size_average=True).item())
            vis_vols, vis_gt = vols, gt
            vis_nviews = int(val_image_data['projections'].shape[1])
    model.train()
    # Filter non-finite (a degenerate sample, or NaN weights) so the logged
    # metric isn't silently poisoned; an all-NaN result still surfaces as nan.
    psnr_f = [x for x in psnr_list if np.isfinite(x)]
    ssim_f = [x for x in ssim_list if np.isfinite(x)]
    out = {
        'psnr': float(np.mean(psnr_f)) if psnr_f else float('nan'),
        'ssim': float(np.mean(ssim_f)) if ssim_f else float('nan'),
        'n': len(psnr_f),
        'vis_vols': vis_vols, 'vis_gt': vis_gt, 'vis_nviews': vis_nviews,
    }
    if want_loss:
        out['loss'] = float(np.mean(loss_list)) if loss_list else float('nan')
    return out


def train(opt):
    #mp.set_start_method("spawn", force=True)
    # broadcast_buffers=False: the model has no per-rank-updated buffers (bin_dir,
    # FDK ramp/Parker buffers are constant), so DDP's per-forward buffer-sync
    # broadcast is pure overhead — and it is the exact collective that hung at an
    # epoch boundary. Disabling it removes that sync (and one deadlock surface).
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=True, broadcast_buffers=False)
    # Raise the NCCL collective timeout (default 600s): with variable N and
    # iterative warmups a single step can legitimately run long, and the
    # default would kill the job at a per-step all-reduce.
    pg_kwargs = InitProcessGroupKwargs(
        timeout=timedelta(seconds=getattr(opt, 'ddp_timeout', 3600)))
    # use_seedable_sampler: make distributed shuffling reproducible via a
    # deterministic per-epoch seed instead of broadcasting the RNG state every
    # __iter__. The RNG-state broadcast can fail ("Invalid mt19937 state") and
    # desync the ranks -> NCCL collective timeout. Version-robust (the kwarg
    # moved into DataLoaderConfiguration in newer accelerate).
    try:
        from accelerate import DataLoaderConfiguration
        accelerator = Accelerator(
            mixed_precision="bf16", log_with="wandb",
            dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
            kwargs_handlers=[ddp_kwargs, pg_kwargs])
    except (ImportError, TypeError):
        accelerator = Accelerator(
            mixed_precision="bf16", log_with="wandb",
            use_seedable_sampler=True,
            kwargs_handlers=[ddp_kwargs, pg_kwargs])
    # Seed EVERYTHING identically across arms before building loaders/model so
    # that (a) the seedable sampler yields the SAME case order every run, (b) the
    # per-epoch data pipeline (aug pick, which is python-random driven) matches,
    # and (c) weight init is identical. Combined with the per-step reseeding in
    # plan_case_draws (views/blade), this makes two arms process the identical
    # case+views+aug at each step -> comparable wandb slices, tighter ablation.
    # Placed BEFORE accelerator.prepare() so the SeedableRandomSampler picks it up.
    from accelerate.utils import set_seed as _set_seed
    _set_seed(getattr(opt, 'seed', 42))

    # Full-state resume bookkeeping. accel_state/ holds model+optimizer+scheduler
    # +RNG+counters; wandb_id.txt lets us continue the SAME wandb run.
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    state_dir = os.path.join(save_dir, 'accel_state')
    wandb_id_file = os.path.join(save_dir, 'wandb_id.txt')
    _resume = getattr(opt, 'resume', False)
    _wb = {"settings": Settings(init_timeout=300)}
    if _resume and os.path.exists(wandb_id_file):
        _wb.update({"id": open(wandb_id_file).read().strip(), "resume": "allow"})
    accelerator.init_trackers(
        project_name="CBCT-reconstruction",
        config=vars(opt), # It's good practice to log your hyperparameters here
        init_kwargs={"wandb": _wb},
    )
    if accelerator.is_main_process and not (_resume and os.path.exists(wandb_id_file)):
        try:
            import wandb as _wandb
            if _wandb.run is not None:
                os.makedirs(save_dir, exist_ok=True)
                open(wandb_id_file, 'w').write(_wandb.run.id)
        except Exception:
            pass
    #accelerator = Accelerator(mixed_precision="bf16")
    # --- Verification  ---
    accelerator.print(f"Distributed Status: Running on {accelerator.num_processes} GPUs/processes.")
    print(f"Process {accelerator.process_index} is using device: {accelerator.device}")
    accelerator.print(f"Mixed Precision: {accelerator.mixed_precision}")
    # --------------------------
    
    model = create_recon_model(opt)      # create a model given opt.model and other options
    device = accelerator.device
    model.device = device
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # only_processed reprocess continue
    
    all_datasets = opt.datasets.split(",")
    # Build + cache the per-dataset manifests on RANK 0 ONLY, then every rank
    # reads the cache. Previously every rank independently walked the filesystem
    # and read tens of thousands of NIfTI headers over NFS (~30 min startup,
    # with N-rank contention). scan_manifest now caches to disk and reads headers
    # in parallel; pass --rescan_manifest to force a rebuild after adding data.
    _mcache = os.path.join(opt.checkpoints_dir, 'manifest_cache')
    _rescan = getattr(opt, 'rescan_manifest', False)
    if accelerator.is_main_process:
        for dataset in all_datasets:
            scan_manifest(dataset, verbose=1, cache_dir=_mcache, rescan=_rescan)
    accelerator.wait_for_everyone()
    manifests = []
    for i, dataset in enumerate(all_datasets):
        manifest = scan_manifest(dataset, verbose=1, cache_dir=_mcache)
        manifest['case_id'] = manifest['case_id'].apply(lambda x: f"D{i}_" + x)
        manifests.append(manifest)
    
    manifest_all = pd.concat(manifests, axis=0, ignore_index=True)
    
    manifest_all = filter_manifest(manifest_all, max_res=opt.filter_max_res, min_phys_z=opt.filter_min_phys_z)
    manifest_all = sample_manifest(manifest_all, opt.max_dataset_size)
    
    # SORT before splitting: split_cases is seeded, but train_test_split only
    # reproduces if the INPUT ORDER is canonical. manifest_all's row order is NOT
    # order-stable across runs (scan_manifest reads headers in parallel ->
    # completion-order rows; concurrently-launched jobs also race the manifest
    # cache). Unsorted, concurrently-launched ablation arms get DIFFERENT
    # train/val/test splits from the SAME case pool -> unmatched data + different
    # reserved test set per arm. Sorting makes the split depend only on the pool.
    all_cases = np.array(sorted(map(str, manifest_all.case_id.unique())))
    # 3-way split: train (80%, unchanged) / val (monitoring) / test (RESERVED —
    # only test.py touches it). Persist the actual ids so test.py evaluates the
    # exact reserved cases regardless of any later filter/universe drift.
    train_cases, val_cases, test_cases = split_cases(
        all_cases, test_size=0.2, val_frac=opt.val_frac, seed=42)
    # Ablation subset: deterministic stratified draw from the train split ONLY
    # (val/test untouched). Per-dataset counts use largest-remainder rounding so
    # the total is exactly train_subset_n and proportions match the full split.
    train_cases_full = train_cases
    if getattr(opt, 'train_subset_n', 0) > 0 and opt.train_subset_n < len(train_cases):
        rng = np.random.RandomState(opt.train_subset_seed)
        by_ds = {}
        for c in sorted(map(str, train_cases)):
            by_ds.setdefault(c.split('_', 1)[0], []).append(c)
        keys = sorted(by_ds)
        if getattr(opt, 'train_subset_nested', False):
            # Nested stratified draw (data-size ablation). Build ONE deterministic
            # global order of all train cases that is independent of the target
            # size, so the first-N prefix is nested across sizes AND stratified.
            # Each dataset's cases get a shuffled rank; the fractional position
            # (rank+0.5)/len_ds interleaves datasets proportionally, so every
            # prefix length holds ~constant per-dataset proportions.
            ranked = []
            for k in keys:
                lst = list(by_ds[k])
                rng.shuffle(lst)                       # seeded, fixed regardless of N
                L = len(lst)
                for rank, c in enumerate(lst):
                    ranked.append(((rank + 0.5) / L, k, c))
            ranked.sort(key=lambda t: (t[0], t[1], t[2]))
            global_order = [c for _, _, c in ranked]
            train_cases = np.array(sorted(global_order[:opt.train_subset_n]))
            from collections import Counter as _Counter
            per_ds = dict(_Counter(c.split('_', 1)[0] for c in train_cases))
            accelerator.print(f"[subset] NESTED train subset: {len(train_cases)} cases "
                              f"(seed {opt.train_subset_seed}), per-dataset "
                              f"{dict(sorted(per_ds.items()))}")
        else:
            quota = [len(by_ds[k]) * opt.train_subset_n / len(train_cases) for k in keys]
            counts = [int(q) for q in quota]
            for _ in range(opt.train_subset_n - sum(counts)):
                i = max(range(len(keys)), key=lambda j: quota[j] - counts[j])
                counts[i] += 1
            train_subset = []
            for k, n_k in zip(keys, counts):
                train_subset.extend(rng.choice(by_ds[k], size=min(n_k, len(by_ds[k])),
                                               replace=False))
            train_cases = np.array(sorted(train_subset))
            accelerator.print(f"[subset] train subset: {len(train_cases)} cases "
                              f"(seed {opt.train_subset_seed}), per-dataset "
                              f"{dict(zip(keys, counts))}")
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'split_cases.json'), 'w') as _f:
            json.dump({
                'seed': 42, 'test_size': 0.2, 'val_frac': opt.val_frac,
                'datasets': opt.datasets,
                'filter_max_res': opt.filter_max_res,
                'filter_min_phys_z': opt.filter_min_phys_z,
                'max_dataset_size': (None if opt.max_dataset_size == float('inf')
                                     else opt.max_dataset_size),
                'train': list(map(str, train_cases_full)),
                'val':   list(map(str, val_cases)),
                'test':  list(map(str, test_cases)),
                'train_subset_n': (opt.train_subset_n
                                   if getattr(opt, 'train_subset_n', 0) > 0 else None),
                'train_subset_seed': getattr(opt, 'train_subset_seed', 0),
                'train_subset_nested': bool(getattr(opt, 'train_subset_nested', False)),
                'train_subset': (list(map(str, train_cases))
                                 if train_cases is not train_cases_full else None),
            }, _f, indent=1)
    accelerator.print(f"[split] train={len(train_cases)}  val={len(val_cases)}  "
                      f"test(reserved)={len(test_cases)}")

    loader = ScanLoader(max_n_aug=opt.aug_max_aug, mode='continue', aug_generator=CTAugmentor(opt, generate_drr), aug_prefix=opt.aug_prefix)


    train_dataset = SingleCTDataset(manifest_all, train_cases, loader)
    val_dataset = SingleCTDataset(manifest_all, val_cases, loader)


    ct_loader = DataLoader(
        dataset=train_dataset,
        batch_size=opt.batch_size,      # Number of samples per pass
        shuffle=True,       # Mixes data every epoch (crucial for training)
        num_workers=0       # Number of subprocesses for data loading
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=opt.batch_size,     # choose based on your GPU memory
        shuffle=True      # quick-val samples vary; full-val covers all anyway
    )
    
    dataset_size = len(ct_loader)    # number of batches per epoch
    print('The number of batches = %d' % dataset_size)
    
    # Prepare every sub-network generically from model_names so model
    # variants with extra nets (binned aggregation, ref alignment, skip
    # fusion) are always covered without editing this list.
    for _name in model.model_names:
        setattr(model, 'net' + _name,
                accelerator.prepare(getattr(model, 'net' + _name)))

    (model.optimizer_base,
     ct_loader,
     val_loader
    ) = accelerator.prepare(
        model.optimizer_base,
        ct_loader,
        val_loader
    )
    
    
    
    # visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    visual_count = 0
    start_epoch = opt.epoch_count

    model.accelerator = accelerator

    # Full-state resume: register the step/epoch counters + LR schedulers so
    # accelerator.save_state/load_state captures them alongside the (already
    # prepared) models and optimizers.
    train_state = _TrainState()
    accelerator.register_for_checkpointing(train_state, *getattr(model, 'schedulers', []))
    if getattr(opt, 'resume', False):
        if getattr(opt, 'continue_train', False):
            accelerator.print("[resume] NOTE: --resume supersedes --continue_train "
                              "(load_state overwrites the weights-only load).")
        # Try the live checkpoint first, then the one-generation backup. A save
        # killed mid-write can leave a truncated optimizer_*.bin that makes
        # load_state raise PytorchStreamReader ("failed finding central
        # directory"); rather than crash, fall back to .bak, then to weights.
        loaded = False
        for _cand in (state_dir, state_dir + '.bak'):
            if not os.path.isdir(_cand):
                continue
            try:
                accelerator.load_state(_cand)
                total_iters = train_state.total_iters
                start_epoch = train_state.epoch
                accelerator.print(f"[resume] restored full state from {_cand}: "
                                  f"epoch={start_epoch}, total_iters={total_iters}")
                loaded = True
                break
            except Exception as _e:
                accelerator.print(f"[resume] WARNING: could not load state from "
                                  f"{_cand} ({type(_e).__name__}: {_e}); "
                                  f"trying next fallback.")
        if not loaded:
            accelerator.print(f"[resume] no usable saved state at {state_dir}; "
                              f"starting fresh (weights only if --continue_train).")
        # The resumed wandb run keeps its internal step from the previous
        # session, which is usually ahead of the checkpoint's total_iters
        # (wandb logs every print_freq, state saves every save_latest_freq).
        # wandb drops logs at steps behind its counter, so fast-forward
        # total_iters past it. Rounded up to a batch_size multiple to keep the
        # `% print_freq` / `% save_latest_freq` triggers firing, and broadcast
        # so all DDP ranks stay in lock-step.
        _wb_step = 0
        if accelerator.is_main_process and wandb.run is not None:
            _wb_step = int(wandb.run.step)
        from accelerate.utils import broadcast_object_list
        _sync = [_wb_step]
        broadcast_object_list(_sync, from_process=0)
        _wb_step = _sync[0]
        if _wb_step > total_iters:
            _bs = max(int(opt.batch_size), 1)
            _ff = ((_wb_step + _bs - 1) // _bs) * _bs
            accelerator.print(f"[resume] wandb run step ({_wb_step}) is ahead of "
                              f"checkpoint total_iters ({total_iters}); "
                              f"fast-forwarding total_iters to {_ff} so logs "
                              f"are not dropped.")
            total_iters = _ff

    def _save_full_state(_epoch):
        """Snapshot model+optimizers+schedulers+RNG+counters for exact resume.

        Atomic: write to a fresh temp dir, then on the main process rotate the
        previous good checkpoint to `<state_dir>.bak` and swap the new one in.
        A job killed mid-save can only corrupt the temp dir, never the live
        checkpoint, and `.bak` gives a one-generation fallback.
        """
        train_state.total_iters = total_iters
        train_state.epoch = _epoch
        tmp_dir = state_dir + '.tmp'
        bak_dir = state_dir + '.bak'
        if accelerator.is_main_process and os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)          # drop any debris from a killed save
        accelerator.wait_for_everyone()
        accelerator.save_state(tmp_dir)     # collective: all ranks must call
        accelerator.wait_for_everyone()     # ensure all ranks finished writing
        if accelerator.is_main_process:
            if os.path.isdir(state_dir):
                if os.path.isdir(bak_dir):
                    shutil.rmtree(bak_dir)
                os.replace(state_dir, bak_dir)   # demote current good -> backup
            os.replace(tmp_dir, state_dir)       # promote new -> live (atomic)
        accelerator.wait_for_everyone()

    # Parameters for gradient clipping (prevents the gradient explosions that
    # cause NaN loss). Gathered from the prepared sub-networks.
    _clip_params = [p for _n in model.model_names
                    for p in getattr(model, 'net' + _n).parameters()]


    # DDP lock-step guard: every rank must run the SAME number of training
    # batches per epoch, or a rank with fewer batches finishes early, races
    # through the collective-free (no_grad) test loop, and parks at the
    # epoch-end save barrier while the other rank is still issuing gradient
    # all-reduces -> NCCL collective-ordering mismatch -> 1h watchdog timeout.
    # accelerate's even_batches only pads the LAST batch; any larger imbalance
    # (or a variable per-step collective count) still desyncs. Capping the
    # outer loop to the global-minimum batch count makes the per-epoch
    # collective sequence identical on all ranks.
    _local_batches = torch.tensor([len(ct_loader)], device=device)
    _all_batches = accelerator.gather(_local_batches)
    steps_per_epoch = int(_all_batches.min().item())
    accelerator.print(f"[ddp] capping training to {steps_per_epoch} batches/epoch "
                      f"(per-rank counts: {_all_batches.tolist()})")

    # Background pool that prefetches upcoming cases' projection views while the
    # GPU trains the current case (0 workers = synchronous load). Reads are I/O-
    # bound (np.load releases the GIL), so threads overlap with GPU compute and
    # keep the cold-NFS read latency off the critical path. prefetch_depth cases
    # are kept in flight, so a read longer than one case's compute still overlaps
    # across several cases; the pool needs >= depth workers to load them at once.
    _pf_workers = getattr(opt, 'prefetch_workers', 2)
    prefetch_depth = max(1, int(getattr(opt, 'prefetch_depth', 2)))
    proj_pool = (ThreadPoolExecutor(max_workers=max(_pf_workers, prefetch_depth))
                 if _pf_workers > 0 else None)

    # In --total_iters mode the run is bounded by iterations, not epochs, so the
    # epoch loop is effectively unbounded and we break when total_iters is hit.
    iter_mode = getattr(opt, 'total_iters', 0) > 0
    epoch_upper = (10 ** 9 if iter_mode
                   else opt.n_epochs + opt.n_epochs_decay + 1)
    stop_training = False
    for epoch in range(start_epoch, epoch_upper):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        if stop_training:
            break
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        # Reproducible per-epoch reshuffle: the seedable sampler permutes by
        # (seed + epoch), so the order is identical across arms yet varies each
        # epoch. Without this the sampler stays at epoch 0 (same order forever).
        if hasattr(ct_loader, 'set_epoch'):
            ct_loader.set_epoch(epoch)
        data_iter = iter(ct_loader)

        def _prep_case(_data, _base_iter):
            """Plan a case's inner draws (main thread; seeds N per rank-synced
            step) and kick off its projection-view load (background thread)."""
            _folder = _data['aug_img_path_a'][0]
            _a2f = build_angle_map(_folder, opt)
            _plans, _needed = plan_case_draws(_a2f, opt, _base_iter)
            if proj_pool is not None:
                _fut = proj_pool.submit(load_case_views, _folder, _a2f, _needed, opt)
                return {'data': _data, 'plans': _plans, 'future': _fut, 'cache': None}
            return {'data': _data, 'plans': _plans, 'future': None,
                    'cache': load_case_views(_folder, _a2f, _needed, opt)}

        # Keep `prefetch_depth` cases in flight. plan_base predicts each case's
        # base total_iters (advances by num_batch_img*batch_size per case, which
        # every rank does deterministically), so N stays synced across ranks even
        # when we prefetch several cases ahead.
        _G = opt.num_batch_img * opt.batch_size
        _plan_base = [total_iters]
        pending = deque()

        def _submit_next():
            try:
                _t = time.perf_counter()
                _d = next(data_iter)   # loads next case's CT (ct_resampled.npz)
                print(f"CT Data loading time: {time.perf_counter() - _t:.2f} seconds")
            except StopIteration:
                return False
            pending.append(_prep_case(_d, _plan_base[0]))
            _plan_base[0] += _G
            return True

        iter_data_time = time.time()
        for _ in range(prefetch_depth):   # prime the pipeline `depth` cases deep
            if not _submit_next():
                break
        pbar = tqdm(range(steps_per_epoch), leave=True,
                    disable=not accelerator.is_main_process)
        for i in pbar:  # inner loop within one epoch (bounded by steps_per_epoch)
            if not pending:   # data exhausted (won't happen before the cap)
                break
            entry = pending.popleft()
            _submit_next()    # refill so `depth` cases stay prefetching ahead
            data = entry['data']
            plans = entry['plans']
            batch_cases = data['case_id']
            print(f"Processing cases: {batch_cases}")
            # ct_a: source vol to reconstruct; ct_b: reference vol
            batch_cts_a = data['ct_a'].squeeze(1)
            print(f"CT source shape: {batch_cts_a.shape}")
            aug_path_a = data['aug_path_a']
            print(f"Aug path: {aug_path_a}")

            # Collect THIS case's projections (already loading in the background).
            _pw0 = time.perf_counter()
            case_cache = (entry['cache'] if entry['future'] is None
                          else entry['future'].result())
            print(f"Proj load wait: {time.perf_counter() - _pw0:.2f} seconds")

            for j in range(opt.num_batch_img):
                model.optimizer_base.zero_grad()
                iter_start_time = time.time()  # timer for computation per iteration
                dtime = iter_start_time - iter_data_time
                if total_iters % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time
                total_iters += opt.batch_size
                epoch_iter += opt.batch_size

                # Iteration-based LR schedule (data-size ablation): set the LR
                # every step from total_iters so the schedule is identical across
                # dataset sizes. Scales each param group off its initial_lr (set
                # by the scheduler at construction, preserving backbone_lr_mult).
                # --finetune_lr > 0 overrides the cosine with a CONSTANT multiplier
                # (× initial_lr) — for a low-LR continuation past the original
                # --total_iters without the cosine warm-restart LR jump.
                if iter_mode:
                    _ft = getattr(opt, 'finetune_lr', 0.0)
                    _mult = (_ft if _ft > 0
                             else iter_lr_multiplier(total_iters, opt.warmup_iters,
                                                     opt.total_iters))
                    for _g in model.optimizer_base.param_groups:
                        _g['lr'] = _g.get('initial_lr', _g['lr']) * _mult

                # N / angle-selection / blade were planned per-step in
                # plan_case_draws (seeded by base_iter + j -> identical N across
                # DDP ranks, preserving the old lock-step contract). Here we only
                # slice the prefetched views from RAM — no disk, no per-step RNG.
                image_data = build_draw(case_cache, plans[j])
                # image_data keys: projections (1 N H W), angles (1 N), N

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                _fwd0 = time.perf_counter()
                vols = model.forward(image_data, None, device=device)

                loss = model.compute_volume_loss(vols, batch_cts_a.to(device),
                                                 model.vols_ref, model.ref_vol)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t5 = time.perf_counter()
                print(f"gpu_forward time: {t5 - _fwd0:.2f} seconds")
                if (model.lambda_l1 > 0) or (model.lambda_ssim > 0):
                    pred_drrs = model.get_pred_drrs(vols, image_data['angles'].to(device))
                    loss = loss + model.compute_render_loss(pred_drrs, image_data['projections'].to(device), angles=image_data['angles'].to(device))
                else:
                    pred_drrs = None
                    model.loss_L_render_L1 = 0.0
                    model.loss_L_render_SSIM = 0.0
                    model.loss_L_render = 0.0

                
                if getattr(opt, 'lambda_maga', 1.0) > 0:
                    model.loss_L_MAGA = opt.lambda_maga * model.compute_multi_scale_loss(vols, batch_cts_a.to(device))
                    loss = loss + model.loss_L_MAGA
                else:
                    model.loss_L_MAGA = 0.0

                # Perceptual loss removed (it used the now-deleted reference
                # 3D encoder; FDK-only model).

                #model.loss_L_MAGA_ref = model.compute_multi_scale_loss(model.vols_ref, model.ref_vol)
                #loss = loss + 0.5 * model.loss_L_MAGA_ref


                # NaN/Inf guard: skip the step so one bad step can't poison
                # every weight (the optimizer would otherwise step on NaN grads
                # and the model stays NaN forever). DDP-safe: all ranks decide
                # together via an all-reduce, so none deadlock at backward.
                nan_flag = accelerator.reduce(
                    torch.tensor(float(not torch.isfinite(loss)), device=device),
                    reduction="sum")
                if nan_flag.item() > 0:
                    if not torch.isfinite(loss):
                        _fin = lambda t: (bool(torch.isfinite(t).all())
                                          if torch.is_tensor(t) else None)
                        print(f"[NaN] iter={total_iters} rank={accelerator.process_index} "
                              f"N={image_data['projections'].shape[1]} loss={float(loss)} | "
                              f"vols_finite={_fin(vols)} "
                              f"fdk_finite={_fin(getattr(model, 'fdk_vol', None))} "
                              f"proj_finite={_fin(image_data['projections'])} "
                              f"L_vol={float(model.loss_L_vol):.4f} "
                              f"L_MAGA={float(model.loss_L_MAGA):.4f}")
                    model.optimizer_base.zero_grad(set_to_none=True)
                    continue

                accelerator.backward(loss)
                if getattr(opt, 'grad_clip', 0) > 0:
                    accelerator.clip_grad_norm_(_clip_params, opt.grad_clip)
                model.optimizer_base.step()
                t6 = time.perf_counter()
                print(f"Backward and optimization time: {t6 - t5:.2f} seconds")

                if accelerator.is_main_process:
                    if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                        visual_count += 1
                        losses = model.get_current_losses()
                        losses["N_views"] = float(image_data['projections'].shape[1])
                        accelerator.log(
                            losses,
                            step=total_iters
                        )

                    if visual_count % opt.display_freq == 0:
                        n_caption = f"| N={int(image_data['projections'].shape[1])} views"
                        if (model.lambda_l1 > 0) or (model.lambda_ssim > 0):
                            visuals_drr = model.generate_visualization_drrs(pred_drrs, image_data['projections'].to(device))
                            train_drr_images = get_logger_images(visuals_drr, caption_suffix=n_caption)
                            accelerator.log({"train_rendered_images": train_drr_images}, step=total_iters)

                        visuals_slice = model.generate_visualization_slices(vols, batch_cts_a.to(device))
                        train_slice_images = get_logger_images(visuals_slice, caption_suffix=n_caption)
                        accelerator.log({"train_slices": train_slice_images}, step=total_iters)
                
                    if total_iters % opt.save_latest_freq == 0:
                        print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                        model.save_networks('latest')
                        # Periodic NUMBERED net snapshot (weights + split/opt only,
                        # ~780MB) so we can pick each arm's BEST-VAL checkpoint
                        # post-hoc (convergence-matched analysis). Main process only.
                        _sf = getattr(opt, 'snapshot_freq', 0)
                        if _sf > 0 and total_iters % _sf == 0 and accelerator.is_main_process:
                            _snapd = os.path.join(save_dir, 'snap', f'iter_{total_iters}')
                            os.makedirs(_snapd, exist_ok=True)
                            for _f in os.listdir(save_dir):
                                if _f.startswith('latest_net_') and _f.endswith('.pth'):
                                    shutil.copy2(os.path.join(save_dir, _f), _snapd)
                            for _m in ('split_cases.json', 'train_opt.ini'):
                                _s = os.path.join(save_dir, _m)
                                if os.path.isfile(_s):
                                    shutil.copy2(_s, _snapd)
                            print(f'[snapshot] iter {total_iters} -> {_snapd}')

                if total_iters % opt.save_latest_freq == 0:
                    _save_full_state(epoch)   # collective: every rank must call
                    # DDP-wrapped modules must be driven by every rank. Running
                    # validation only on rank 0 can leave ranks entering
                    # different collectives on the next training step.
                    qtb = getattr(opt, 'quick_test_batches', 3)
                    if qtb > 0:
                        r = run_validation(model, val_loader, device, opt,
                                           max_batches=qtb, want_loss=False)
                        if accelerator.is_main_process:
                            accelerator.log({"val_quick_psnr": r['psnr'],
                                             "val_quick_ssim": r['ssim']}, step=total_iters)
                            if r['vis_vols'] is not None:
                                vis = model.generate_visualization_slices(r['vis_vols'], r['vis_gt'])
                                accelerator.log(
                                    {"val_quick_slices": get_logger_images(vis)},
                                    step=total_iters)
                            print(f"[quick val @ iter {total_iters}] "
                                  f"PSNR={r['psnr']:.2f} dB  SSIM={r['ssim']:.4f}")

                    # Larger validation every val_full_every quick-vals. Stateless
                    # trigger (total_iters is a multiple of save_latest_freq here),
                    # so the cadence survives resume. Replaces the old per-epoch
                    # full test, which was too slow.
                    vfe = getattr(opt, 'val_full_every', 10)
                    if vfe > 0 and total_iters > 0 and \
                            (total_iters // opt.save_latest_freq) % vfe == 0:
                        rf = run_validation(model, val_loader, device, opt,
                                            max_batches=getattr(opt, 'val_full_batches', 30),
                                            want_loss=True)
                        if accelerator.is_main_process:
                            accelerator.log({"val_full_loss": rf['loss'],
                                             "val_full_psnr": rf['psnr'],
                                             "val_full_ssim": rf['ssim']}, step=total_iters)
                            if rf['vis_vols'] is not None:
                                cap = f"| N={rf['vis_nviews']} views"
                                visf = model.generate_visualization_slices(rf['vis_vols'], rf['vis_gt'])
                                accelerator.log(
                                    {"val_full_slices": get_logger_images(visf, caption_suffix=cap)},
                                    step=total_iters)
                            print(f"[FULL val @ iter {total_iters}] loss={rf['loss']:.4f}  "
                                  f"PSNR={rf['psnr']:.2f} dB  SSIM={rf['ssim']:.4f}  (n={rf['n']})")
                
                t3 = time.perf_counter()

                # Iteration-based stop (data-size ablation). All ranks reach this
                # at the same total_iters (deterministic per-step planning), so the
                # collective save is safe. Force a final save since the stop point
                # need not land on a save_latest_freq boundary.
                if iter_mode and total_iters >= opt.total_iters:
                    if accelerator.is_main_process:
                        print(f'[iter-stop] total_iters={total_iters} >= target '
                              f'{opt.total_iters}; saving latest + full state, stopping')
                    model.save_networks('latest')
                    _save_full_state(epoch + 1)
                    stop_training = True
                    break
            if stop_training:
                break
            t0 = time.perf_counter()

        # NOTE: the old per-epoch full test on the (then-shared) test set is gone.
        # Validation now runs on the VAL split on an iteration cadence: a quick
        # pass every save_latest_freq iters + a larger pass every val_full_every
        # quick-vals (see the block above). The reserved test set is evaluated
        # only by test.py.

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')   # overwrite single 'latest' snapshot
            _save_full_state(epoch + 1)     # resume restarts at the NEXT epoch

            iter_data_time = time.time()
        
        if not iter_mode:
            model.update_learning_rate()    # epoch-based schedule; iter_mode sets LR per-step
        if accelerator.is_main_process:
            print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
    
    if proj_pool is not None:
        proj_pool.shutdown(wait=False)
    accelerator.end_training()
    


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    train(opt)
    
    # test_opt = TestOptions().parse()  # get test options
    # # hard-code some parameters for test
    # test_opt.batch_size = 1  # test code only supports batch_size = 1
    # #test_opt.trace_all_model = True # make pt files for all saved intermedia models
    # trace(test_opt)
    
    
        
