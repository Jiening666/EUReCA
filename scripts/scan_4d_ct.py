#!/usr/bin/env python
"""
Scan dataset roots for CT NIfTIs that are not 3D (e.g. (X,Y,Z,1) or genuine
4D), which crash the DRR / resample pipeline (permute / ndimage.zoom rank
errors).  Reads HEADERS ONLY (nib.load(...).shape is lazy), so it is fast over
thousands of files and never loads voxel data.

Usage:
    python scripts/scan_4d_ct.py /path/ds1,/path/ds2,...
    python scripts/scan_4d_ct.py /path/ds1 /path/ds2        # space- or comma-sep
    python scripts/scan_4d_ct.py --glob "*/*/*.nii.gz" /path/ds

Originals are expected at  <root>/<case>/<scan>/*.nii.gz  (the layout
scan_manifest assumes); falls back to a recursive search if none match.
Exits non-zero if any non-3D CT is found.
"""
import argparse
import sys
from pathlib import Path

import nibabel as nib


def iter_ct_files(root: Path, pattern: str):
    files = sorted(root.glob(pattern))
    if not files:                      # layout differs — search recursively
        files = sorted(root.rglob("*.nii.gz"))
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+",
                    help="dataset root(s), space- and/or comma-separated")
    ap.add_argument("--glob", default="*/*/*.nii.gz",
                    help="glob (relative to each root) for original CTs "
                         "(default: <case>/<scan>/*.nii.gz)")
    args = ap.parse_args()

    roots = []
    for r in args.roots:
        roots.extend(s for s in r.split(",") if s.strip())

    total, bad, unreadable = 0, [], []
    for r in roots:
        root = Path(r)
        if not root.exists():
            print(f"[warn] root does not exist: {root}")
            continue
        files = iter_ct_files(root, args.glob)
        print(f"[scan] {root}: {len(files)} NIfTI file(s)")
        for p in files:
            total += 1
            try:
                shape = nib.load(str(p)).shape          # header only, lazy
            except Exception as e:                      # noqa: BLE001
                unreadable.append((p, repr(e)))
                continue
            ndim = len(shape)
            # 3D is fine; >3 with all extra dims == 1 is harmless but flagged.
            if ndim != 3:
                squeezed = tuple(d for d in shape if d != 1)
                bad.append((p, shape, len(squeezed) == 3))
            if total % 200 == 0:
                print(f"  ...checked {total}")

    print("\n================ RESULT ================")
    print(f"checked: {total} CT file(s)")
    if bad:
        print(f"\nNON-3D CT files: {len(bad)}")
        for p, shape, fixable in bad:
            tag = "(singleton -> auto-coerces to 3D)" if fixable \
                  else "(GENUINE 4D -> pipeline uses first volume)"
            print(f"  {shape}  {tag}\n    {p}")
    else:
        print("\nNo non-3D CT files found.")
    if unreadable:
        print(f"\nUNREADABLE: {len(unreadable)}")
        for p, err in unreadable:
            print(f"  {p}\n    {err}")

    sys.exit(1 if (bad or unreadable) else 0)


if __name__ == "__main__":
    main()
