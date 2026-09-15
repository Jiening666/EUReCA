"""
prepare_prostate_xim.py — End-to-end prep of the `prostate/` XIM zip dumps.

What it does
------------
1. UNZIP every `*.zip` in the prostate folder into  <out>/<scanid>_<site>/ ,
   where the (scanid, site) name is looked up from the screenshot table
   (`*.jpg`) by matching the zip's timestamp to the table's date/time.
   The full zip↔folder↔patient correspondence is written to
   `correspondence.csv` and `correspondence.json`.

2. PROCESS each projection (`Proj_*.xim`), reusing the validated physics in
   `xim_dataset.py`:
       a. bowtie flat-field normalisation  (raw / FilterBowtie flood)
       b. blade band → 0  (outside the collimated Y-band, set to 0)
       c. -log  → line integrals
       d. zoom (anisotropic detector px → isotropic `psize` mm) + centre-crop
          to the cbct_recon model input size (H × W).

3. SAVE every projection as BOTH .png and .npy, each tagged with its gantry
   angle (in the filename), plus a stacked `projections.npy` (N,H,W) and the
   matching `angles.npy` (N,) for direct model ingestion.

Geometry / preprocessing notes are derived from Scan.xml and Reconstruction.xml
(see module docstring `RECON_PIPELINE_NOTES` at the bottom and the printed
summary).

Usage
-----
    python prepare_prostate_xim.py \
        --prostate_dir D:/Jiening/prostate \
        --out_dir      D:/Jiening/prostate/processed \
        [--H 384 --W 768 --psize 0.7273] \
        [--no_bowtie] [--max_projections 60] \
        [--no_individual]        # skip per-projection png/npy, keep stacks only
"""

import argparse
import csv
import json
import re
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np

# Vectorised XIM decoder (~170x faster than the per-pixel reference loop).
# Patch it into xim_dataset so _process_xim / load_bowtie use the fast path.
import xim_dataset as _xd
from xim_fast import read_xim as _fast_read_xim
_xd.read_xim = _fast_read_xim

# Reuse the validated XIM physics already in the repo.
from xim_dataset import (
    parse_scan_xml_xim,
    find_acquisition_dir,
    find_xim_files,
    find_bowtie_xim,
    load_bowtie,
    blade_pixel_mask,
    _process_xim,
)

# ---------------------------------------------------------------------------
# 0.  Patient table transcribed from the screenshot (*.jpg)
#     Each entry: scan id, anatomical site, and the acquisition time shown.
#     Times are matched to zip timestamps by (month, day, hour, minute).
# ---------------------------------------------------------------------------
# (scan_id, site, month, day, "h:mmAM/PM")
SITE_TABLE = [
    (35515840, "pubic",            6, 3, "10:48am"),
    (38290908, "prostate bed",     6, 5, "10:47am"),
    (38394829, "esophagus",        6, 5,  "9:28am"),
    (39150400, "HN",               6, 5, "10:03am"),
    (38086598, "prostate bed",     6, 5,  "8:03am"),
    (39150791, "prone rectum",     6, 5,  "8:24am"),
    (39148974, "utuerus",          6, 5,  "9:18am"),
    (39142635, "prostate bed lns", 6, 5, "10:39am"),
    (39148502, "lt pelvic",        6, 5, "11:51am"),
    (39049948, "prostatebed",      6, 5, "11:32am"),
    (  494659, "bladder",          6, 5, "11:22am"),
    (38066598, "thyroid",          6, 5,  "7:42am"),
    (  438512, "prostatebed",      6, 5, "11:40am"),
    (39150550, "tonsil",           6, 5, "12:16pm"),
    (39145842, "tongue",           6, 5, "12:06pm"),
]


def _table_minutes(month, day, timestr):
    """Return absolute minute-of-month key for a table row."""
    t = datetime.strptime(timestr, "%I:%M%p")
    return ((month * 31 + day) * 24 + t.hour) * 60 + t.minute


def _zip_datetime(zip_path):
    """Parse 'YYYY-MM-DD_HHMMSS.zip' → datetime."""
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})_(\d{2})(\d{2})(\d{2})", zip_path.stem)
    if not m:
        return None
    y, mo, d, h, mi, s = map(int, m.groups())
    return datetime(y, mo, d, h, mi, s)


def _zip_minutes(dt):
    return ((dt.month * 31 + dt.day) * 24 + dt.hour) * 60 + dt.minute


def match_zip_to_patient(zip_path, tolerance_min=5):
    """Find the SITE_TABLE row closest in time to this zip (same day)."""
    dt = _zip_datetime(zip_path)
    if dt is None:
        return None, None
    zmin = _zip_minutes(dt)
    best, best_d = None, 1 << 30
    for sid, site, mo, day, tstr in SITE_TABLE:
        if mo != dt.month or day != dt.day:
            continue
        d = abs(_table_minutes(mo, day, tstr) - zmin)
        if d < best_d:
            best, best_d = (sid, site, tstr), d
    if best is None or best_d > tolerance_min:
        return None, dt
    return best, dt


def sanitize(site):
    s = site.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


# ---------------------------------------------------------------------------
# 1.  Extraction
# ---------------------------------------------------------------------------

def extract_zip(zip_path, dest_dir):
    """
    Extract `zip_path` into `dest_dir`, stripping the leading GUID folder so
    that `dest_dir` itself becomes the scan_root (contains Scan.xml,
    Acquisitions/, Calibrations/...).
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        # Detect a single common top-level folder (the GUID) to strip.
        tops = {n.split("/", 1)[0] for n in names if "/" in n}
        strip = tops.pop() + "/" if len(tops) == 1 else ""
        for n in names:
            if n.endswith("/"):
                continue
            rel = n[len(strip):] if strip and n.startswith(strip) else n
            if not rel:
                continue
            target = dest_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(n) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    return dest_dir


# ---------------------------------------------------------------------------
# 2 + 3.  Process one scan_root → projections + angles → png/npy
# ---------------------------------------------------------------------------

def _save_png(img, path, vmax=None):
    """Save one projection as a grayscale PNG (angle is carried in the filename)."""
    from PIL import Image
    hi = vmax if vmax else float(img.max())
    a = np.clip(img / (hi + 1e-8), 0.0, 1.0)
    Image.fromarray((a * 255).astype(np.uint8)).save(path)


def process_scan(scan_root, out_dir, H, W, psize, bowtie_correction,
                 max_projections=None, save_individual=True):
    """Run the full XIM → line-integral pipeline and dump png/npy + stacks."""
    scan_root = Path(scan_root)
    out_dir = Path(out_dir)
    proj_png = out_dir / "projections_png"
    proj_npy = out_dir / "projections_npy"
    if save_individual:
        proj_png.mkdir(parents=True, exist_ok=True)
        proj_npy.mkdir(parents=True, exist_ok=True)

    geom = parse_scan_xml_xim(scan_root / "Scan.xml")
    acq_dir = find_acquisition_dir(scan_root, geom["acq_id"])
    xims = find_xim_files(acq_dir)

    if max_projections and len(xims) > max_projections:
        idx = np.linspace(0, len(xims) - 1, max_projections).round().astype(int)
        xims = [xims[i] for i in idx]

    mask = blade_pixel_mask(geom)

    bowtie = None
    if bowtie_correction:
        bpath = find_bowtie_xim(scan_root, geom.get("mode_id"))
        if bpath is not None:
            bowtie = load_bowtie(bpath)
            print(f"    bowtie flood : {bpath.relative_to(scan_root)}")
        else:
            print("    [warn] no bowtie flood found → 99th-pct fallback norm")

    imgs, angles, skipped = [], [], []
    for i, xim in enumerate(xims):
        try:
            # normalise=False → keep physical line-integral values in the npy.
            img, angle = _process_xim(
                xim, bowtie=bowtie, mask=mask, geom=geom,
                psize_mm=psize, out_H=H, out_W=W, normalise=False,
            )
        except (ZeroDivisionError, ValueError) as e:
            # Halcyon scans often end with a degenerate (0,0) trailing frame.
            skipped.append(xim.name)
            continue
        if img.shape != (H, W) or not np.isfinite(img).all():
            skipped.append(xim.name)
            continue
        imgs.append(img)
        angles.append(angle)

    if skipped:
        print(f"    [skip] {len(skipped)} unreadable/empty projection(s): "
              f"{skipped[0]}{' …' if len(skipped) > 1 else ''}")
    if not imgs:
        raise ValueError("no valid projections after filtering")

    projs = np.stack(imgs).astype(np.float32)        # (N, H, W)
    angs = np.asarray(angles, dtype=np.float32)      # (N,)

    # Stacked, model-ready arrays (angles carried alongside).
    np.save(out_dir / "projections.npy", projs)
    np.save(out_dir / "angles.npy", angs)
    # Model-ready copy: per-VOLUME global-max normalisation, matching the
    # training collate (single_batch_collate divides the whole stack by one max
    # over all views; _load_proj does not per-image normalise). This preserves
    # relative inter-view intensity. NOTE: the true model input max depends on
    # which views are selected; test_npy.py re-normalises the chosen subset.
    gmax = float(projs.max())
    np.save(out_dir / "projections_modelnorm.npy",
            (projs / max(gmax, 1e-9)).astype(np.float32))

    if save_individual:
        vmax = float(np.percentile(projs, 99.5))
        for i, (img, ang) in enumerate(zip(imgs, angles)):
            tag = f"Proj_{i:05d}_ang{ang:07.2f}"
            np.save(proj_npy / f"{tag}.npy", img)
            _save_png(img, proj_png / f"{tag}.png", vmax=vmax)

    return {
        "n_proj": int(projs.shape[0]),
        "shape": list(projs.shape),
        "angle_start": float(angs[0]),
        "angle_stop": float(angs[-1]),
        "geom": {k: geom[k] for k in (
            "SID", "SAD", "full_H", "full_W", "res_h", "res_w",
            "blade_y1", "blade_y2", "blade_x1", "blade_x2", "mode_id")},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--prostate_dir", default="D:/Jiening/prostate")
    ap.add_argument("--out_dir", default="D:/Jiening/prostate/processed")
    ap.add_argument("--H", type=int, default=384)
    ap.add_argument("--W", type=int, default=768)
    ap.add_argument("--psize", type=float, default=0.7273)
    ap.add_argument("--no_bowtie", action="store_true",
                    help="skip bowtie flat-field (use 99th-pct fallback)")
    ap.add_argument("--max_projections", type=int, default=None,
                    help="uniformly subsample to this many projections")
    ap.add_argument("--no_individual", action="store_true",
                    help="skip per-projection png/npy; keep stacked arrays only")
    ap.add_argument("--skip_extract", action="store_true",
                    help="reuse already-extracted folders")
    args = ap.parse_args()

    prostate = Path(args.prostate_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    zips = sorted(prostate.glob("*.zip"))
    if not zips:
        raise SystemExit(f"No zips found in {prostate}")

    correspondence = []
    for zp in zips:
        match, dt = match_zip_to_patient(zp)
        if match is None:
            print(f"[skip] {zp.name}: no patient-table match "
                  f"(time {dt})")
            continue
        sid, site, jpg_time = match
        folder = f"{sid}_{sanitize(site)}"
        scan_root = out_root / folder
        print(f"\n=== {zp.name}  →  {folder}  ({site}, {jpg_time}) ===")

        if not args.skip_extract or not (scan_root / "Scan.xml").is_file():
            extract_zip(zp, scan_root)

        try:
            info = process_scan(
                scan_root, scan_root, H=args.H, W=args.W, psize=args.psize,
                bowtie_correction=not args.no_bowtie,
                max_projections=args.max_projections,
                save_individual=not args.no_individual,
            )
        except Exception as e:  # keep going across scans
            print(f"    [error] {e}")
            continue

        row = {
            "zip": zp.name,
            "folder": folder,
            "scan_id": sid,
            "site": site,
            "scan_datetime": dt.isoformat(),
            "jpg_time": jpg_time,
            **info,
        }
        correspondence.append(row)
        (scan_root / "scan_meta.json").write_text(json.dumps(row, indent=2))
        print(f"    saved {info['n_proj']} projections "
              f"{tuple(info['shape'])}, angles "
              f"{info['angle_start']:.1f}→{info['angle_stop']:.1f}°")

    # Global correspondence files
    (out_root / "correspondence.json").write_text(
        json.dumps(correspondence, indent=2))
    if correspondence:
        keys = ["zip", "folder", "scan_id", "site", "scan_datetime",
                "jpg_time", "n_proj", "angle_start", "angle_stop"]
        with open(out_root / "correspondence.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in correspondence:
                w.writerow(r)
    print(f"\nDone. {len(correspondence)}/{len(zips)} scans → {out_root}")
    print("Correspondence: correspondence.csv / correspondence.json")


# ---------------------------------------------------------------------------
# Notes distilled from Scan.xml + Reconstruction.xml (Varian Halcyon iCBCT).
# The steps marked [done] are applied above; [todo] are recommended extras.
# ---------------------------------------------------------------------------
RECON_PIPELINE_NOTES = """
Vendor recon plugin chain (Reconstruction.xml), mapped to our pipeline:
  SaturationCorrection            [todo] clip/repair saturated detector pixels
  DetectorCrosstalkCompensation   [todo] (hardware-specific; minor)
  ScintillatorGapCorrection       [partial] handled implicitly by flood/bowtie
  NonPulsedBeamCompensation       [todo] this mode is Non-Pulsed (continuous kV)
  ApplyBeamShape / PhysicsCorr.   [done]  bowtie flood normalisation
  PenumbraCompensation            [done]  blade band masking → 0
  LogNormalizationPlugIn          [done]  -log(raw/flood) line integrals
  GridGapCorrection               [todo] anti-scatter GRID_15_1 gaps
  DefectCorrectionPlugIn          [todo] DefectMap.xim bad-pixel inpainting
  RingSuppressionPlugin (MEDIUM)  [todo] post-recon / detector ring removal
Geometry: SID 1540 / SAD 1000 mm, HALF trajectory, StartAngle -126 / Stop 85,
  SourceAngleOffset -90; detector 384x3072 @ (1.12 x 0.28) mm/px (Bin2x8).
"""

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(RECON_PIPELINE_NOTES)
    main()
