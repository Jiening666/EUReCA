#!/usr/bin/env python
"""
Split genuine 4D CT NIfTIs (multiple stacked volumes, e.g. (X,Y,Z,3)) into
separate single-volume scan folders so each volume becomes its own 3D scan
and is usable as training data (instead of the pipeline silently keeping only
volume 0).

Layout assumed:  <root>/<case>/<scan>/ct.nii.gz
For a 4D file at <case>/<scan>/ct.nii.gz with K volumes and a NUMERIC scan
folder (e.g. "000"), the volumes are written to sibling folders:
    <case>/000/ct.nii.gz, <case>/001/ct.nii.gz, ..., <case>/00{K-1}/ct.nii.gz
(volume 0 overwrites the original folder; the original 4D file is backed up
to  <scan>/ct_4d_original.nii.gz  first).  Each output keeps the source affine.

Safety:
  * DRY-RUN by default — prints the planned actions; pass --apply to execute.
  * Never clobbers a pre-existing sibling scan folder that already has a
    ct.nii.gz (skips that source file with a warning).
  * Backs up the original 4D file (reversible) rather than deleting it.

Usage:
    python scripts/split_4d_ct.py /path/ds1,/path/ds2            # dry run
    python scripts/split_4d_ct.py /path/ds1 --apply              # execute
    python scripts/split_4d_ct.py --files a.nii.gz,b.nii.gz --apply
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import nibabel as nib


def find_4d(roots, glob_pat):
    out = []
    for r in roots:
        root = Path(r)
        files = sorted(root.glob(glob_pat)) or sorted(root.rglob("*.nii.gz"))
        for p in files:
            try:
                shape = nib.load(str(p)).shape
            except Exception as e:                       # noqa: BLE001
                print(f"[warn] unreadable: {p} ({e})")
                continue
            if len([d for d in shape if d != 1]) > 3:    # genuine 4D
                out.append(p)
    return out


def plan_split(p: Path):
    """Return (K, [(target_path, vol_index), ...], backup_path) or None."""
    scan_dir = p.parent
    case_dir = scan_dir.parent
    base = scan_dir.name
    if not base.isdigit():
        print(f"[skip] non-numeric scan folder '{base}' for {p} "
              f"(unsure how to name siblings)")
        return None
    base_idx = int(base)
    shape = nib.load(str(p)).shape
    sq = [d for d in shape if d != 1]
    K = sq[3]
    targets = [(case_dir / f"{base_idx + i:03d}" / "ct.nii.gz", i) for i in range(K)]
    # Backup must NOT end in .nii.gz, or scan_manifest/_find_original (which
    # glob "*.nii.gz" and take the first match) could load the 4D backup
    # instead of the new 3D ct.nii.gz. Rename to ct.nii.gz -> ct.nii.gz to
    # restore.
    backup = scan_dir / "ct.nii.gz.4dbak"
    return K, targets, backup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="*", help="dataset root(s), space/comma-sep")
    ap.add_argument("--files", default="", help="explicit comma-sep 4D file paths")
    ap.add_argument("--glob", default="*/*/*.nii.gz")
    ap.add_argument("--apply", action="store_true", help="execute (default: dry run)")
    args = ap.parse_args()

    roots = []
    for r in args.roots:
        roots.extend(s for s in r.split(",") if s.strip())

    if args.files:
        files = [Path(s) for s in args.files.split(",") if s.strip()]
    else:
        files = find_4d(roots, args.glob)

    print(f"{'APPLY' if args.apply else 'DRY-RUN'}: {len(files)} genuine-4D CT file(s)\n")
    n_done, n_skipped, n_new = 0, 0, 0
    for p in files:
        planned = plan_split(p)
        if planned is None:
            n_skipped += 1
            continue
        K, targets, backup = planned

        # collision check on the NEW siblings (i>=1)
        collide = [t for t, i in targets if i > 0 and t.exists()]
        if collide:
            print(f"[skip] {p}\n    would clobber existing: {collide[0]} ...")
            n_skipped += 1
            continue

        print(f"{p}  ({K} volumes)")
        print(f"    backup original 4D -> {backup}")
        for t, i in targets:
            tag = "(overwrite orig folder)" if i == 0 else "(new scan)"
            if i > 0:
                n_new += 1
            print(f"    vol {i} -> {t}  {tag}")

        if args.apply:
            img = nib.load(str(p))
            data = np.squeeze(img.get_fdata()).astype(np.float32)   # (X,Y,Z,K)
            # back up the 4D first, then write volumes (vol0 reuses the folder)
            if not backup.exists():
                p.rename(backup)
            for t, i in targets:
                t.parent.mkdir(parents=True, exist_ok=True)
                nib.save(nib.Nifti1Image(data[..., i], img.affine), str(t))
            n_done += 1
        print()

    print("================ SUMMARY ================")
    verb = "split" if args.apply else "would split"
    print(f"{verb}: {n_done if args.apply else len(files) - n_skipped} file(s); "
          f"new scan folders: {n_new}; skipped: {n_skipped}")
    if not args.apply:
        print("\n(dry run — re-run with --apply to execute)")


if __name__ == "__main__":
    main()
