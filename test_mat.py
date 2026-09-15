"""
test_mat.py — Reconstruct with the cbct_recon model from iTools .mat projections.

This is the .mat analogue of test_npy.py.  Instead of the XIM-derived
projections.npy, it ingests the projections that Varian iTools / DataInspector
exports as one `Proj_XXXXX.mat` per view (key `Proj`, shape (det_H, det_W)).
Those projections are ALREADY the vendor-corrected line integrals (scatter /
beam-hardening / -log all applied inside iTools), so the ONLY thing this script
does to each frame is the geometry resample the model was trained on:

    detector pixels (anisotropic, at the imager plane)
        --zoom-->  isotropic `psize` mm/px at ISOCENTRE
        --centre-crop-->  (H, W)

i.e. exactly step 5 of realdata/xim_dataset._process_xim, with no bowtie / -log /
blade masking (iTools already did all of that).  The resampled stack matches
`itools_projs_iso.npy` to corr > 0.999.

Angles
------
.mat files carry no angle metadata, so the per-view angle is resolved in order:
    1. --angles_file <path.npy>                 (explicit, must match #views)
    2. angles.npy in the scan folder            (XIM KVSourceRtn; the validated
                                                 model path — used only if its
                                                 length matches the #.mat views)
    3. Scan.xml in the scan folder              linspace(StartAngle, StopAngle)
    4. --start_angle / --stop_angle             linspace fallback (Halcyon arc)
The resolved angle is then mapped to the training convention exactly like
test_npy.py:  angle = (angle_sign * a + angle_offset) mod 360   (default +1/-90).

Examples
--------
# one scan folder holding Proj_*.mat (+ angles.npy), 30 views, model + FDK
python test_mat.py --name my_ckpt --checkpoints_dir ./checkpoints \
    --mat_root D:/Jiening/prostate/processed --scan test --num_views 30 --fdk

# every subfolder under the root that contains Proj_*.mat
python test_mat.py --name my_ckpt --mat_root D:/Jiening/prostate/processed \
    --all --num_views 60 --output_dir ./outputs/nii_mat
"""

import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path

# Anaconda on Windows can load Intel OpenMP twice (NumPy/MKL + PyTorch).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import scipy.io as sio
import torch
import nibabel as nib
from scipy import ndimage
from accelerate import Accelerator

from frozen_model_utils import (
    add_frozen_model_args,
    maybe_inject_frozen_args,
    maybe_inject_train_opt_args,
    maybe_prepend_frozen_code_path,
)
maybe_prepend_frozen_code_path()
from options.test_options import TestOptions
from models import create_recon_model
from models.fdk import DifferentiableFDK


# Halcyon detector geometry (imager plane). Overridable on the CLI or read from
# Scan.xml when present. Matches realdata/xim_dataset + reconstruct_itools_*.py.
DET_RES_H = 1.12     # mm/px, detector row pitch  (ImagerResY)
DET_RES_W = 0.28     # mm/px, detector col pitch  (ImagerResX, Bin2x8)
SID = 1540.0         # mm, source-to-imager distance


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_opt():
    base   = TestOptions()
    parser = base.initialize(
        argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            fromfile_prefix_chars='@',
        )
    )
    parser.add_argument('--mat_root', type=str, required=True,
                        help='Folder containing scan subfolders with Proj_*.mat '
                             '(or a single scan folder used with --scan .)')
    parser.add_argument('--scan', type=str, default=None,
                        help='Scan subfolder name under --mat_root (e.g. test). '
                             'Use "." for --mat_root itself. Omit with --all.')
    parser.add_argument('--all', action='store_true',
                        help='Process every subfolder that contains Proj_*.mat')
    parser.add_argument('--mat_glob', type=str, default='Proj_*.mat',
                        help='Glob for the per-view .mat files inside a scan folder')
    parser.add_argument('--mat_subdir', type=str, default='projections_mat',
                        help='Subfolder under each scan folder that holds the '
                             '.mat (iTools dumper writes here). Falls back to the '
                             'scan folder itself if no .mat are found in it.')
    parser.add_argument('--mat_key', type=str, default='Proj',
                        help='Variable name of the projection array inside each .mat')
    parser.add_argument('--num_views', type=int, default=None,
                        help='Uniformly sub-sample to this many views (None = all)')
    parser.add_argument('--view_indices', type=int, nargs='*', default=None,
                        help='Explicit zero-based view indices (overrides --num_views)')
    parser.add_argument('--fdk', action='store_true',
                        help='Also reconstruct an analytic FDK volume from ALL '
                             'projections (ignores --num_views) as '
                             '<scan>_fdk_<N>views.nii.gz.')
    parser.add_argument('--fdk_save_filter', type=str, default=None,
                        choices=['ramp', 'hann'],
                        help='Override the filter for the standalone --fdk volume '
                             '(defaults to --fdk_filter).')
    # Angle resolution + training-convention mapping (mirrors test_npy.py).
    parser.add_argument('--angles_file', type=str, default='',
                        help='Explicit .npy of per-view angles (must match #views). '
                             'Overrides angles.npy / Scan.xml / --start_angle.')
    parser.add_argument('--start_angle', type=float, default=-126.0,
                        help='linspace start when no angles.npy/Scan.xml is found')
    parser.add_argument('--stop_angle', type=float, default=85.0,
                        help='linspace stop when no angles.npy/Scan.xml is found')
    parser.add_argument('--angle_sign', type=float, default=1.0,
                        help='angle = (angle_sign*a + angle_offset) mod 360. Use -1 '
                             'to reverse rotation direction (mirror / chirality).')
    parser.add_argument('--angle_offset', type=float, default=-90.0,
                        help='Added after --angle_sign to match the training '
                             'convention. Default -90 matches test_npy.py for '
                             'angles.npy (XIM KVSourceRtn). For a raw Halcyon arc '
                             '(--start/--stop) with FDK, 0 matches '
                             'reconstruct_itools_fdk.py --angle_mode raw.')
    # Detector geometry for the .mat -> iso resample (Scan.xml overrides if found).
    parser.add_argument('--det_res_h', type=float, default=DET_RES_H,
                        help='Detector row pitch [mm/px] at the imager plane')
    parser.add_argument('--det_res_w', type=float, default=DET_RES_W,
                        help='Detector col pitch [mm/px] at the imager plane')
    parser.add_argument('--sid', type=float, default=SID,
                        help='Source-to-imager distance [mm] (SAD comes from --fdk_sad)')
    parser.add_argument('--no_input_norm', action='store_true',
                        help='Skip per-volume global-max normalisation of the input '
                             'stack (use for raw-scale FDK diagnostics).')
    parser.add_argument('--output_dir', type=str, default='',
                        help='Where to write <scan>_<K>views.nii.gz '
                             '(default: checkpoints_dir/name/recon_nii_mat)')
    add_frozen_model_args(parser)
    parser.set_defaults(phase='test')

    argv        = maybe_inject_frozen_args(parser)
    if argv is None:                       # not frozen: replay arch from train_opt.ini
        argv    = maybe_inject_train_opt_args(parser)
    opt         = parser.parse_args(argv)
    opt.isTrain = False
    base.opt    = opt
    if not opt.output_dir:
        opt.output_dir = str(Path(opt.checkpoints_dir) / opt.name / "recon_nii_mat")
    return opt


# ---------------------------------------------------------------------------
# .mat -> isocentre-grid resample (mirrors _process_xim step 5)
# ---------------------------------------------------------------------------

def resample_to_iso(proj, res_h, res_w, mag, psize, out_H, out_W):
    """Anisotropic detector px -> isotropic psize mm/px at iso, centre-cropped.

    `proj` is an iTools-corrected line-integral frame (det_H, det_W). res_h/res_w
    are the imager-plane pitches; dividing by the magnification M = SID/SAD moves
    them to the isocentre plane before matching `psize`. This is exactly the zoom
    + centre-crop of realdata/xim_dataset._process_xim, minus bowtie/-log/mask
    (iTools already applied those).
    """
    zoom_h = (res_h / mag) / psize
    zoom_w = (res_w / mag) / psize
    zoomed = ndimage.zoom(proj, (zoom_h, zoom_w), order=1)
    zh, zw = zoomed.shape

    top  = max(0, (zh - out_H) // 2)
    left = max(0, (zw - out_W) // 2)
    img  = zoomed[top:top + out_H, left:left + out_W]

    if img.shape != (out_H, out_W):          # zero-pad if smaller than target
        pad = np.zeros((out_H, out_W), dtype=np.float32)
        ch, cw = img.shape
        pad[(out_H - ch) // 2:(out_H - ch) // 2 + ch,
            (out_W - cw) // 2:(out_W - cw) // 2 + cw] = img
        img = pad
    return img.astype(np.float32)


def _load_mat_frame(path, key):
    """Return the (det_H, det_W) float32 array from one Proj_*.mat."""
    d = sio.loadmat(str(path))
    if key in d:
        arr = d[key]
    else:                                    # fall back to the first 2-D array
        arrays = [v for k, v in d.items()
                  if not k.startswith("__") and getattr(v, "ndim", 0) == 2]
        if not arrays:
            raise KeyError(f"{path.name}: no 2-D array (key {key!r} absent)")
        arr = arrays[0]
    return np.asarray(arr, dtype=np.float32)


# ---------------------------------------------------------------------------
# Scan.xml geometry / arc (self-contained; no realdata import)
# ---------------------------------------------------------------------------

def _read_scan_xml(xml_path):
    """Minimal Scan.xml reader: SID, res_h/res_w, StartAngle/StopAngle. None-safe."""
    try:
        raw = Path(xml_path).read_bytes()
        try:
            text = raw.decode("utf-16")
        except UnicodeDecodeError:
            text = raw.decode("utf-8", errors="replace")
        root = ET.fromstring(text)
    except Exception:
        return {}

    def _num(tag):
        for el in root.iter():
            if el.tag.rsplit('}', 1)[-1] == tag and el.text:
                try:
                    return float(el.text.strip())
                except ValueError:
                    return None
        return None

    return {
        "SID":         _num("SID"),
        "res_h":       _num("ImagerResY"),
        "res_w":       _num("ImagerResX"),
        "start_angle": _num("StartAngle"),
        "stop_angle":  _num("StopAngle"),
    }


def resolve_angles(scan_dir, n_views, opt):
    """Return per-view angles (n_views,) BEFORE the sign/offset mapping, plus a
    short string describing the source."""
    if opt.angles_file:
        a = np.load(opt.angles_file).astype(np.float32)
        if a.shape[0] != n_views:
            raise ValueError(f"--angles_file has {a.shape[0]} angles, {n_views} views")
        return a, f"file:{Path(opt.angles_file).name}"

    npy = scan_dir / "angles.npy"
    if npy.is_file():
        a = np.load(npy).astype(np.float32)
        if a.shape[0] == n_views:
            return a, "angles.npy"
        print(f"    [angles] angles.npy has {a.shape[0]} != {n_views} views; "
              f"falling back to arc")

    xml = _read_scan_xml(scan_dir / "Scan.xml") if (scan_dir / "Scan.xml").is_file() else {}
    if xml.get("start_angle") is not None and xml.get("stop_angle") is not None:
        a = np.linspace(xml["start_angle"], xml["stop_angle"], n_views).astype(np.float32)
        return a, f"Scan.xml arc {xml['start_angle']:.1f}->{xml['stop_angle']:.1f}"

    a = np.linspace(opt.start_angle, opt.stop_angle, n_views).astype(np.float32)
    return a, f"cli arc {opt.start_angle:.1f}->{opt.stop_angle:.1f}"


# ---------------------------------------------------------------------------
# Load one scan folder of .mat -> model-ready tensors
# ---------------------------------------------------------------------------

def select_views(N, num_views=None, view_indices=None):
    if view_indices is not None:
        return np.asarray(sorted(view_indices), dtype=int)
    if num_views is not None and num_views < N:
        return np.linspace(0, N - 1, num_views).round().astype(int)
    return np.arange(N, dtype=int)


def _mat_dir(scan_dir, opt):
    """Directory that actually holds the Proj_*.mat: prefer <scan>/<mat_subdir>
    (where the iTools dumper writes), else the scan folder itself (flat layout).
    Scan.xml / angles.npy still resolve against the scan folder, not this one."""
    scan_dir = Path(scan_dir)
    sub = scan_dir / opt.mat_subdir
    if opt.mat_subdir and sub.is_dir() and any(sub.glob(opt.mat_glob)):
        return sub
    return scan_dir


def load_scan_mat(scan_dir, opt, num_views=None, view_indices=None):
    """Load + resample + normalise all Proj_*.mat in a scan folder.

    Returns image_data dict (projections (1,K,H,W), angles (1,K)), the selected
    view indices, and the mapped angles.
    """
    scan_dir = Path(scan_dir)
    mats = sorted(_mat_dir(scan_dir, opt).glob(opt.mat_glob))
    if not mats:
        raise FileNotFoundError(f"no {opt.mat_glob} in {scan_dir}")

    # Geometry: Scan.xml overrides the CLI/Halcyon defaults when present.
    xml = _read_scan_xml(scan_dir / "Scan.xml") if (scan_dir / "Scan.xml").is_file() else {}
    res_h = xml.get("res_h") or opt.det_res_h
    res_w = xml.get("res_w") or opt.det_res_w
    sid   = xml.get("SID") or opt.sid
    mag   = sid / opt.fdk_sad                      # SAD = --fdk_sad (default 1000)

    N = len(mats)
    angles_raw, ang_src = resolve_angles(scan_dir, N, opt)

    idx = select_views(N, num_views, view_indices)
    mats_sel = [mats[i] for i in idx]
    angles_raw = angles_raw[idx]

    imgs = [resample_to_iso(_load_mat_frame(p, opt.mat_key),
                            res_h, res_w, mag, opt.psize, opt.H, opt.W)
            for p in mats_sel]
    projs = np.stack(imgs).astype(np.float32)      # (K, H, W)

    # Map to the training angle convention (same as test_npy.py).
    angs = np.mod(opt.angle_sign * angles_raw + opt.angle_offset, 360.0).astype(np.float32)

    # Per-VOLUME global-max normalisation (matches single_batch_collate / the
    # per-volume input norm the model was trained on; see test_npy.load_scan).
    if not opt.no_input_norm:
        projs = projs / max(float(projs.max()), 1e-9)

    image_data = {
        'projections': torch.from_numpy(projs).unsqueeze(0),   # (1, K, H, W)
        'angles':      torch.from_numpy(angs).unsqueeze(0),     # (1, K)
        'N':           int(projs.shape[0]),
        '_angle_src':  ang_src,
        '_geom':       (res_h, res_w, sid, mag),
    }
    return image_data, idx, angs


# ---------------------------------------------------------------------------
# NIfTI output
# ---------------------------------------------------------------------------

def save_nii(vol_np, path, spacing):
    affine = np.diag([spacing, spacing, spacing, 1.0]).astype(np.float64)
    img = nib.Nifti1Image(vol_np.astype(np.float32), affine)
    img.header.set_xyzt_units('mm')
    nib.save(img, str(path))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    opt = build_opt()

    mat_root = Path(opt.mat_root)
    if opt.all:
        scans = sorted(d.name for d in mat_root.iterdir()
                       if d.is_dir() and any(_mat_dir(d, opt).glob(opt.mat_glob)))
    elif opt.scan:
        scans = [opt.scan]
    else:
        raise SystemExit("Specify --scan <folder> (use '.' for --mat_root) or --all")
    if not scans:
        raise SystemExit(f"No scans with {opt.mat_glob} under {mat_root}")

    # Model
    accelerator = Accelerator(mixed_precision="bf16")
    device      = accelerator.device
    model = create_recon_model(opt)
    model.device = device
    model.setup(opt)
    for name in model.model_names:
        net = getattr(model, 'net' + name)
        setattr(model, 'net' + name, accelerator.prepare(net))
    model.eval()

    # Analytic FDK (optional) — same iso-plane geometry as the model.
    fdk = None
    if opt.fdk:
        fdk = DifferentiableFDK(
            opt.Nx, opt.Ny, opt.Nz, opt.vol_size, opt.H, opt.W, opt.psize,
            SAD=getattr(opt, 'fdk_sad', 1000.0),
            filter=(opt.fdk_save_filter or opt.fdk_filter),
            vox_chunk=getattr(opt, 'fdk_vox_chunk', 131072),
            parker=opt.fdk_parker, pad=opt.fdk_pad, cutoff=opt.fdk_cutoff,
        ).to(device)

    out_dir = Path(opt.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}\nVolume grid: {opt.Nx}x{opt.Ny}x{opt.Nz} "
          f"@ {opt.vol_size} mm\nAngle map: angle = ({opt.angle_sign:+.0f}*a "
          f"{opt.angle_offset:+.1f}) mod 360"
          f"{'  | FDK: on (all views%s)' % (', Parker' if opt.fdk_parker else '') if fdk is not None else ''}\n")

    for scan in scans:
        scan_dir = mat_root if scan == "." else mat_root / scan
        tag_name = mat_root.name if scan == "." else scan
        try:
            image_data, idx, angs = load_scan_mat(
                scan_dir, opt, opt.num_views, opt.view_indices)
        except FileNotFoundError as e:
            print(f"[skip] {tag_name}: {e}")
            continue

        K = image_data['N']
        res_h, res_w, sid, mag = image_data['_geom']
        print(f"=== {tag_name}: {K} views  (angles {angs.min():.1f}..{angs.max():.1f}°, "
              f"src={image_data['_angle_src']}, M={mag:.3f}, "
              f"det {res_h:.3f}x{res_w:.3f} mm) ===")

        image_data['projections'] = image_data['projections'].to(device)
        with torch.no_grad():
            vol = model.forward(image_data, None, device=device)   # (1,Nx,Ny,Nz)
        vol_np = vol.squeeze(0).cpu().float().numpy()

        out_path = out_dir / f"{tag_name}_{K}views.nii.gz"
        save_nii(vol_np, out_path, opt.vol_size)
        np.save(out_dir / f"{tag_name}_{K}views_idx.npy", idx)
        print(f"    model  saved {vol_np.shape}  ->  {out_path}")

        # ── FDK over ALL projections ────────────────────────────────────────
        if fdk is not None:
            fdk_data, _, _ = load_scan_mat(
                scan_dir, opt, num_views=None, view_indices=None)
            N_all = fdk_data['N']
            with torch.no_grad():
                fvol = fdk(fdk_data['projections'].to(device),
                           fdk_data['angles'].to(device))   # (1,1,Nx,Ny,Nz)
            fvol_np = fvol.squeeze(0).squeeze(0).cpu().float().numpy()
            fdk_path = out_dir / f"{tag_name}_fdk_{N_all}views.nii.gz"
            save_nii(fvol_np, fdk_path, opt.vol_size)
            print(f"    fdk    saved {fvol_np.shape}  ({N_all} views)  ->  {fdk_path}")
        print()

    print("Done.")


if __name__ == '__main__':
    main()
