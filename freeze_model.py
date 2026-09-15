#!/usr/bin/env python
"""
Freeze a trained model into a self-contained bundle UNDER its checkpoint dir,
so it can be tested regardless of later code-base changes.

A faithful reproduction needs the matching *code*, not just the weights (the
forward has numpy geometry / FFT / checkpointing / dynamic control flow that
can't be serialized to TorchScript/ONNX). So this snapshots the inference
source + weights + config together.

Bundle layout:  checkpoints/<name>/frozen/<name>_<gitsha>_<date>/
    code/                 inference source snapshot (models/, options/, *.py)
    ckpt/<name>/          the {epoch}_net_*.pth weights + the *_opt.ini config
    inference_opt.ini     geometry+architecture+FDK flags pulled from train_opt
    run_test.sh           ready-to-run: cd code && python <entry> @../inference_opt.ini ...
    MANIFEST.txt          git sha, date, weights, full train config reference

Usage:
    python freeze_model.py --name hal_iter
    python freeze_model.py --name hal_iter --epoch latest
"""
import argparse
import datetime
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Source needed to instantiate + run the model (relative to repo root).
CODE_ITEMS = [
    "models", "options",
    "threeD_geometry.py", "ct_generate_drr.py", "ct_dataset.py", "utils.py",
    "test.py",
]

# Flags from train_opt.ini that define the model/geometry/data location (kept
# for the test command). Everything else (lr, schedule, save freq, DDP, paired,
# iterative-training, etc.) is training-only and dropped.
KEEP = {
    # geometry
    'H', 'W', 'psize', 'resw', 'resh', 'Nx', 'Ny', 'Nz', 'Nz_temp', 'vol_size',
    # binned aggregation / transformer
    'use_binned_aggr', 'aggr_bins', 'aggr_layers', 'aggr_chunk_size',
    'aggr_vox_chunk', 'fuse_depth', 'refine_depth', 'fuse_view_chunk',
    'use_rope', 'proj_coverage_halflife',
    # FDK
    'use_fdk', 'fdk_filter', 'fdk_sad', 'fdk_vox_chunk', 'fdk_pad', 'fdk_cutoff',
    'fdk_parker', 'fdk_parker_gamma_sign',
    # reference / iterative
    'use_ref_inject', 'iter_max', 'lambda_align', 'align_ridge',
    # data location + filtering (so the bundle runs out-of-the-box; override as needed)
    'datasets', 'data_root', 'filter_max_res', 'filter_min_phys_z',
    'angle_select_mode', 'aug_prefix', 'img_type',
}


def git_sha(root):
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"],
            text=True, stderr=subprocess.DEVNULL).strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "nogit"


def parse_opt_ini(path):
    """train_opt.ini -> list of (key, value-or-None) preserving only KEEP keys.

    Format is one token per line: '--key' then (for valued flags) its value on
    the next line; store_true flags are a lone '--key'.
    """
    toks = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if s and not s.startswith('#'):
            toks.append(s)
    kept = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if not t.startswith('--'):
            i += 1
            continue
        key = t[2:]
        has_val = (i + 1 < len(toks)) and not toks[i + 1].startswith('--')
        val = toks[i + 1] if has_val else None
        if key in KEEP:
            kept.append((key, val))
        i += 2 if has_val else 1
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--epoch", default="latest")
    ap.add_argument("--checkpoints_dir", default="./checkpoints")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    ckpt_src = Path(args.checkpoints_dir) / args.name
    if not ckpt_src.is_dir():
        sys.exit(f"[freeze] checkpoint dir not found: {ckpt_src}")
    weights = sorted(ckpt_src.glob(f"{args.epoch}_net_*.pth"))
    if not weights:
        sys.exit(f"[freeze] no {args.epoch}_net_*.pth in {ckpt_src}")

    sha = git_sha(root)
    date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle = ckpt_src / "frozen" / f"{args.name}_{sha}_{date}"
    (bundle / "code").mkdir(parents=True, exist_ok=True)
    (bundle / "ckpt" / args.name).mkdir(parents=True, exist_ok=True)

    # 1. snapshot inference code (skip caches and any nested frozen bundles)
    copied = []
    for item in CODE_ITEMS:
        s = root / item
        if not s.exists():
            continue
        d = bundle / "code" / item
        if s.is_dir():
            shutil.copytree(s, d, ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", "frozen"))
        else:
            shutil.copy2(s, d)
        copied.append(item)

    # 2. weights + the exact opt configs
    for w in weights:
        shutil.copy2(w, bundle / "ckpt" / args.name / w.name)
    for cfg in list(ckpt_src.glob("*_opt.ini")) + list(ckpt_src.glob("*_opt.txt")):
        shutil.copy2(cfg, bundle / "ckpt" / args.name / cfg.name)

    # 3. distilled inference_opt.ini (model/geometry/data flags only)
    train_ini = ckpt_src / "train_opt.ini"
    inf_lines = []
    if train_ini.exists():
        for k, v in parse_opt_ini(train_ini):
            inf_lines.append(f"--{k}")
            if v is not None:
                inf_lines.append(v)
    (bundle / "inference_opt.ini").write_text("\n".join(inf_lines) + "\n")

    # 4. run_test.sh — runnable from the frozen code with the bundled weights
    run = bundle / "run_test.sh"
    run.write_text(
        "#!/usr/bin/env bash\n"
        "# Test this FROZEN model from its snapshotted code (immune to later\n"
        "# code-base changes). Append/override test flags as needed; data path\n"
        "# comes from inference_opt.ini (override with --datasets / --data_root).\n"
        "set -e\n"
        'cd "$(dirname "$0")/code"\n'
        f"python test.py @../inference_opt.ini \\\n"
        f"    --checkpoints_dir ../ckpt --name {args.name} --epoch {args.epoch} \\\n"
        f"    --num_images_list 10 25 50 100 --max_test_size 30 \\\n"
        f"    \"$@\"\n"
    )
    os.chmod(run, 0o755)

    # 5. manifest
    full_cfg = train_ini.read_text() if train_ini.exists() else "(no train_opt.ini)"
    (bundle / "MANIFEST.txt").write_text(
        f"frozen model : {args.name}\n"
        f"date         : {date}\n"
        f"git          : {sha}\n"
        f"epoch        : {args.epoch}\n"
        f"weights      : {[w.name for w in weights]}\n"
        f"code         : {copied}\n"
        f"entry        : test.py\n\n"
        f"TEST:  bash run_test.sh [extra/override flags]\n"
        f"REAL DATA: run root-level test_real.py/test_npy.py with "
        f"--use_frozen --frozen_tag {bundle.name}\n"
        f"(reproduction is faithful only with this frozen code/ — new working-tree\n"
        f" code may load these weights but behave differently.)\n\n"
        f"===== full train_opt.ini =====\n{full_cfg}\n"
    )

    print(f"[freeze] '{args.name}' ({args.epoch}, git {sha}) -> {bundle}")
    print(f"  weights: {len(weights)} files | code: {', '.join(copied)}")
    print(f"  inference flags kept: {len(inf_lines)} tokens")
    print(f"  test with:  bash {run} <extra flags>")


if __name__ == "__main__":
    main()
