"""
deidentify_prostate.py — replace MRN-based folder names with anonymized IDs.

The initial prep named each scan folder  <MRN>_<site>  (MRN = the scan id from
the screenshot table, which is PHI).  This step:

  1. Assigns a stable anonymized id (anonNNN) to each scan, ordered by
     acquisition datetime so the numbering is deterministic across re-runs.
  2. Renames  <MRN>_<site>  ->  <anon_id>_<site>.
  3. Rewrites each folder's scan_meta.json into a de-identified form
     (drops MRN, zip name, and exact dates; adds anon_id).
  4. Writes TWO correspondence files:
       correspondence_deid.{csv,json}  — anon_id -> site + scan info, NO MRN.
                                          Safe to share alongside the images.
       correspondence_key.{csv,json}   — anon_id -> MRN + zip + datetime + all
                                          info.  The re-identification key;
                                          keep this private / access-controlled.

Idempotent: reads correspondence.json if present, else correspondence_key.json,
so re-running after a rename is a no-op.

NOTE: this only de-identifies folder names + scan_meta.json + correspondence.
The extracted scan folders still contain ClinicalInfo.wox and XIM header
properties that may hold PHI — scrubbing those is a separate step (not done
here).
"""

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path


def _rename_with_retry(old_path, new_path, attempts=5, delay=1.5):
    """Windows holds transient dir handles (Explorer thumbnailing); retry."""
    for k in range(attempts):
        try:
            old_path.rename(new_path)
            return True
        except PermissionError:
            if k == attempts - 1:
                return False
            time.sleep(delay)
    return False

# De-identified view: what is safe to ship next to the pixel data.
DEID_FIELDS = ["anon_id", "folder", "site", "n_proj", "angle_start", "angle_stop"]
# Re-identification key: everything, including the MRN linkage.
KEY_FIELDS = ["anon_id", "folder", "scan_id", "zip", "scan_datetime",
              "jpg_time", "site", "n_proj", "angle_start", "angle_stop"]


def sanitize(site):
    return re.sub(r"[^a-z0-9]+", "_", site.strip().lower()).strip("_")


def load_source(out_root):
    for name in ("correspondence.json", "correspondence_key.json"):
        p = out_root / name
        if p.is_file():
            return json.loads(p.read_text()), name
    raise SystemExit(f"No correspondence.json / correspondence_key.json in {out_root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="D:/Jiening/prostate/processed")
    ap.add_argument("--prefix", default="anon",
                    help="anonymized id prefix (folder = <prefix>NNN_<site>)")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    records, src_name = load_source(out_root)
    print(f"Loaded {len(records)} records from {src_name}")

    # Stable order → deterministic anon ids.
    records = sorted(records, key=lambda r: (r["scan_datetime"], str(r["scan_id"])))

    deid_rows, key_rows, unresolved = [], [], []
    for i, r in enumerate(records, 1):
        anon = f"{args.prefix}{i:03d}"
        site_slug = sanitize(r["site"])
        new_folder = f"{anon}_{site_slug}"

        # Current folder may still be MRN-based (first run) or already anon (rerun).
        old_folder = r["folder"]
        old_path = out_root / old_folder
        new_path = out_root / new_folder
        actual_folder = new_folder   # what the correspondence will record

        already_renamed = new_path.is_dir() and not old_path.is_dir()
        if already_renamed or args.dry_run:
            if args.dry_run:
                print(f"  {old_folder}  ->  {new_folder}")
        elif old_path.resolve() == new_path.resolve():
            pass
        elif not old_path.is_dir():
            print(f"  [warn] missing folder {old_folder}; skipping")
        elif new_path.exists():
            print(f"  [warn] target {new_folder} exists; skipping")
        else:
            print(f"  {old_folder}  ->  {new_folder}")
            if not _rename_with_retry(old_path, new_path):
                print(f"  [LOCKED] could not rename {old_folder} "
                      f"(close any Explorer/viewer window on it, then re-run)")
                actual_folder = old_folder        # keep correspondence truthful
                unresolved.append(old_folder)

        # De-identified scan_meta.json inside the (resulting) folder.
        cur_path = out_root / actual_folder
        if not args.dry_run and cur_path.is_dir():
            meta_path = cur_path / "scan_meta.json"
            meta = json.loads(meta_path.read_text()) if meta_path.is_file() else dict(r)
            for k in ("scan_id", "zip", "scan_datetime", "jpg_time"):
                meta.pop(k, None)
            meta = {"anon_id": anon, "folder": actual_folder,
                    **{k: v for k, v in meta.items() if k != "folder"}}
            meta_path.write_text(json.dumps(meta, indent=2))

        base = {**r, "anon_id": anon, "folder": actual_folder}
        deid_rows.append({k: base.get(k) for k in DEID_FIELDS})
        key_rows.append({k: base.get(k) for k in KEY_FIELDS})

    if args.dry_run:
        print("\n[dry run] no files changed.")
        return

    _write(out_root / "correspondence_deid.json", out_root / "correspondence_deid.csv",
           deid_rows, DEID_FIELDS)
    _write(out_root / "correspondence_key.json", out_root / "correspondence_key.csv",
           key_rows, KEY_FIELDS)

    # Remove the original MRN-in-folder-name correspondence ONLY when every
    # folder was renamed (otherwise a re-run still needs it as the source).
    if not unresolved:
        for name in ("correspondence.json", "correspondence.csv"):
            p = out_root / name
            if p.is_file():
                p.unlink()

    print(f"\nDone. {len(records) - len(unresolved)}/{len(records)} scans renamed.")
    print(f"  shareable : {out_root/'correspondence_deid.csv'}  (no MRN)")
    print(f"  key (PHI) : {out_root/'correspondence_key.csv'}   (MRN linkage)")
    if unresolved:
        print(f"\n  [!] {len(unresolved)} folder(s) still locked and NOT renamed:")
        for f in unresolved:
            print(f"        {f}")
        print("      Close any Explorer/image-viewer window on them and re-run "
              "this script to finish.")


def _write(json_path, csv_path, rows, fields):
    json_path.write_text(json.dumps(rows, indent=2))
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
