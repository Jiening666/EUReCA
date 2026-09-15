"""
xim_dataset.py — Load Halcyon CBCT projections directly from XIM files.

Expected scan folder structure:
    <scan_root>/
        Scan.xml
        Acquisitions/
            <acq_id>/
                Proj_00000.xim
                Proj_00001.xim
                ...
        Calibrations/
            AIR-Bowtie-<mode>/
                Current/
                    FilterBowtie.xim

Processing pipeline per projection:
  1. Parse Scan.xml  → geometry, blade positions, acquisition ID
  2. Discover Acquisitions/<acq_id>/Proj_*.xim  (sorted)
  3. Load bowtie flat-field from Calibrations/AIR-Bowtie-*/Current/FilterBowtie.xim
  4. Per XIM:
       a. Read raw detector counts via ReadXim.read_xim()
       b. Resize bowtie to match if needed, divide raw by bowtie
       c. Outside-blade pixels → set corrected = 1  (so -log → 0)
       d. Negative log → line integrals (attenuation image)
       e. Explicitly zero outside-blade region
       f. Zoom from native (anisotropic) detector pixels to isotropic psize mm/px
       g. Centre-crop to (H, W)
  5. Read gantry angle from XIM properties (KVSourceRtn); linear fallback from
     Scan.xml StartAngle/StopAngle if the property is absent.

Subset selection (at most one; priority: xim_json > xim_indices > xim_mask):
  xim_indices : list of int indices into the sorted Proj_*.xim list
  xim_mask    : 0/1 sequence, or path to .txt / .npy file, one entry per XIM
  xim_json    : path to JSON with int indices or filename strings
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset

from ReadXim import read_xim   # already in cbct_recon/

_SCAN_XML_NS = "http://baden.varian.com/cr.xsd"


# ---------------------------------------------------------------------------
# Scan.xml parsing
# ---------------------------------------------------------------------------

def _xml_text(root, tag, dtype=str, ns=_SCAN_XML_NS):
    el = root.find(f".//{{{ns}}}{tag}")
    if el is None:
        el = root.find(f".//{tag}")
    if el is None or not el.text:
        return None
    try:
        return dtype(el.text.strip())
    except (ValueError, TypeError):
        return None


def parse_scan_xml_xim(xml_path):
    """
    Parse a Halcyon Scan.xml (UTF-16 LE with BOM) and return a geometry dict.

    Returns
    -------
    dict with keys:
        SID, SAD, full_H, full_W, res_h, res_w,
        blade_y1, blade_y2, blade_x1, blade_x2,
        imager_lat, imager_lng, source_angle_offset,
        acq_id          – acquisition folder name (str)
        start_angle     – StartAngle [IEC continuous, deg]
        stop_angle      – StopAngle  [IEC continuous, deg]
        n_projections   – Count from first <Acquisition> block
        mode_id         – e.g. "125kV_FullFan_RTI8643iL"
    """
    raw = Path(xml_path).read_bytes()
    try:
        text = raw.decode("utf-16")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")
    root = ET.fromstring(text)

    acq_el = root.find(f".//{{{_SCAN_XML_NS}}}Acquisition")
    if acq_el is None:
        acq_el = root.find(".//Acquisition")
    acq_id      = acq_el.get("Id")             if acq_el is not None else None
    n_proj_text = acq_el.findtext(f"{{{_SCAN_XML_NS}}}Count") or (
                  acq_el.findtext("Count"))     if acq_el is not None else None

    return {
        "SID":                  _xml_text(root, "SID",                float),
        "SAD":                  _xml_text(root, "SAD",                float),
        "full_H":               _xml_text(root, "ImagerSizeY",        int),
        "full_W":               _xml_text(root, "ImagerSizeX",        int),
        "res_h":                _xml_text(root, "ImagerResY",         float),
        "res_w":                _xml_text(root, "ImagerResX",         float),
        "blade_y1":             _xml_text(root, "BladeY1",            float),
        "blade_y2":             _xml_text(root, "BladeY2",            float),
        "blade_x1":             _xml_text(root, "BladeX1",            float),
        "blade_x2":             _xml_text(root, "BladeX2",            float),
        "imager_lat":           _xml_text(root, "ImagerLat",          float) or 0.0,
        "imager_lng":           _xml_text(root, "ImagerLng",          float) or 0.0,
        "source_angle_offset":  _xml_text(root, "SourceAngleOffset",  float) or 0.0,
        "acq_id":               acq_id,
        "start_angle":          _xml_text(root, "StartAngle",         float),
        "stop_angle":           _xml_text(root, "StopAngle",          float),
        "n_projections":        int(n_proj_text) if n_proj_text else None,
        "mode_id":              _xml_text(root, "ModeId"),
    }


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_acquisition_dir(scan_root, acq_id=None):
    """Return Path to Acquisitions/<acq_id>/; auto-detects if acq_id is None."""
    acq_base = Path(scan_root) / "Acquisitions"
    if not acq_base.is_dir():
        raise FileNotFoundError(f"Acquisitions/ not found under {scan_root}")
    if acq_id is not None:
        d = acq_base / str(acq_id)
        if not d.is_dir():
            raise FileNotFoundError(f"Acquisition folder not found: {d}")
        return d
    for d in sorted(acq_base.iterdir()):
        if d.is_dir():
            return d
    raise FileNotFoundError(f"No acquisition subfolders under {acq_base}")


def find_xim_files(acq_dir):
    """Return sorted list of Proj_*.xim Paths."""
    xims = sorted(Path(acq_dir).glob("Proj_*.xim"))
    if not xims:
        raise FileNotFoundError(f"No Proj_*.xim files found in {acq_dir}")
    return xims


def find_bowtie_xim(scan_root, mode_id=None):
    """
    Search Calibrations/AIR-Bowtie-*/Current/FilterBowtie.xim.

    Prefers a folder whose name contains `mode_id`; falls back to first match.
    Returns None if nothing is found.
    """
    cal_base = Path(scan_root) / "Calibrations"
    if not cal_base.is_dir():
        return None

    candidates = []
    for d in sorted(cal_base.glob("AIR-Bowtie-*")):
        p = d / "Current" / "FilterBowtie.xim"
        if p.is_file():
            candidates.append((d.name, p))

    if not candidates:
        return None

    if mode_id:
        for name, p in candidates:
            if mode_id in name or name in mode_id:
                return p

    return candidates[0][1]


# ---------------------------------------------------------------------------
# Bowtie / flat-field
# ---------------------------------------------------------------------------

def load_bowtie(bowtie_path):
    """
    Load FilterBowtie.xim as float32.
    Non-positive values are replaced by 1 to avoid division by zero.
    """
    d   = read_xim(str(bowtie_path))
    arr = d["pixel_data"].astype(np.float32)
    arr = np.where(arr > 0, arr, 1.0)
    return arr


# ---------------------------------------------------------------------------
# Blade mask
# ---------------------------------------------------------------------------

def blade_pixel_mask(geom):
    """
    Return a bool ndarray (full_H, full_W) where True = inside collimated field.

    Physics: blade positions are given at source plane (mm, IEC, relative to
    isocenter).  They project to the detector by multiplying by SID/SAD.
    Detector row 0 = top (+Y); increasing row → decreasing physical Y.
    Detector col 0 = left (−X); increasing col → increasing physical X.
    """
    mag   = geom["SID"] / geom["SAD"]
    H, W  = geom["full_H"], geom["full_W"]
    res_h = geom["res_h"]
    res_w = geom["res_w"]

    # Project blade edges to detector plane [mm]
    y2_det = geom["blade_y2"] * mag   # upper blade → smaller row index
    y1_det = geom["blade_y1"] * mag   # lower blade → larger  row index
    x1_det = geom["blade_x1"] * mag   # left  blade
    x2_det = geom["blade_x2"] * mag   # right blade

    # Physical extent of the full detector from its centre [mm]
    det_top_mm  = H * res_h / 2
    det_left_mm = W * res_w / 2

    # Lateral detector offset shifts the isocenter column position [pixels]
    iso_col_shift = geom["imager_lat"] / res_w

    # Convert blade projections to pixel indices
    row_top   = int(round((det_top_mm  - y2_det) / res_h))
    row_bot   = int(round((det_top_mm  - y1_det) / res_h))
    col_left  = int(round((det_left_mm + x1_det) / res_w - iso_col_shift))
    col_right = int(round((det_left_mm + x2_det) / res_w - iso_col_shift))

    row_top   = max(0, min(row_top,  H))
    row_bot   = max(0, min(row_bot,  H))
    col_left  = max(0, min(col_left, W))
    col_right = max(0, min(col_right, W))

    mask = np.zeros((H, W), dtype=bool)
    mask[row_top:row_bot, col_left:col_right] = True
    return mask


# ---------------------------------------------------------------------------
# Subset selection
# ---------------------------------------------------------------------------

def resolve_xim_subset(all_xims, xim_indices=None, xim_mask=None, xim_json=None):
    """
    Select a subset of XIM paths from `all_xims`.

    Priority: xim_json > xim_indices > xim_mask.
    If all three are None, returns all_xims unchanged.

    Parameters
    ----------
    all_xims    : list of Path, sorted Proj_*.xim files
    xim_indices : list/tuple of int – zero-based indices into all_xims
    xim_mask    : 0/1 sequence (list/ndarray), or path str/Path to a
                  .npy or .txt file with one value per XIM (1 = include)
    xim_json    : str/Path to JSON file with a list of int indices
                  or filename strings (e.g. ["Proj_00010.xim", ...])

    Returns
    -------
    list of Path
    """
    if xim_json is not None:
        with open(xim_json) as f:
            entries = json.load(f)
        name_map = {p.name: p for p in all_xims}
        selected = []
        for e in entries:
            if isinstance(e, int):
                selected.append(all_xims[e])
            else:
                selected.append(name_map[str(e)])
        return selected

    if xim_indices is not None:
        return [all_xims[i] for i in xim_indices]

    if xim_mask is not None:
        if isinstance(xim_mask, (str, Path)):
            p = Path(xim_mask)
            if p.suffix == ".npy":
                mask_arr = np.load(str(p)).astype(bool)
            else:
                mask_arr = np.array(
                    [int(line.strip()) for line in p.read_text().splitlines()
                     if line.strip()],
                    dtype=bool,
                )
        else:
            mask_arr = np.asarray(xim_mask, dtype=bool)
        return [p for p, keep in zip(all_xims, mask_arr) if keep]

    return all_xims


# ---------------------------------------------------------------------------
# Angle extraction from XIM properties
# ---------------------------------------------------------------------------

_ANGLE_PROP_KEYS = ("KVSourceRtn", "GantryRtn", "GantryAngle")


def _xim_gantry_angle(xim_dict):
    """Extract gantry angle [degrees, 0-360) from XIM properties dict."""
    props = xim_dict.get("properties", {})
    for key in _ANGLE_PROP_KEYS:
        if key in props:
            return float(props[key]) % 360.0
    return None


def _fallback_angle(xim_path, geom):
    """Linear interpolation of gantry angle from filename index + Scan.xml arc."""
    try:
        idx = int(Path(xim_path).stem.split("_")[-1])
    except ValueError:
        return 0.0
    n   = geom["n_projections"] or 1
    a0  = geom["start_angle"] or 0.0
    a1  = geom["stop_angle"]  or 0.0
    return (a0 + idx * (a1 - a0) / max(n - 1, 1)) % 360.0


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------

def _process_xim(xim_path, bowtie, mask, geom, psize_mm, out_H, out_W,
                 normalise=True):
    """
    Load one XIM projection, apply corrections, resample, and return
    (img: float32 ndarray (out_H, out_W), angle_deg: float).

    Steps
    -----
    1. Read raw detector counts.
    2. Flat-field correct with bowtie (or 99th-pct of valid region).
    3. Set outside-blade pixels to corrected=1  →  -log = 0 there.
    4. Negative log  →  line integrals (attenuation image).
    5. Enforce zero outside blade after log.
    6. Zoom to isotropic psize; centre-crop to (out_H, out_W).
    7. Optional per-image max normalisation.
    """
    d   = read_xim(str(xim_path))
    raw = d["pixel_data"].astype(np.float32)

    # Guard against XIM dimensions differing from full detector in Scan.xml
    if raw.shape != (geom["full_H"], geom["full_W"]):
        # Resize mask to match actual XIM dimensions
        scale_h = raw.shape[0] / geom["full_H"]
        scale_w = raw.shape[1] / geom["full_W"]
        mask_r  = ndimage.zoom(mask.astype(np.float32),
                               (scale_h, scale_w), order=0).astype(bool)
        eff_res_h = geom["res_h"] / scale_h
        eff_res_w = geom["res_w"] / scale_w
    else:
        mask_r    = mask
        eff_res_h = geom["res_h"]
        eff_res_w = geom["res_w"]

    # 1. Flat-field correction
    if bowtie is not None:
        if bowtie.shape != raw.shape:
            bt = ndimage.zoom(bowtie,
                              (raw.shape[0] / bowtie.shape[0],
                               raw.shape[1] / bowtie.shape[1]),
                              order=1)
        else:
            bt = bowtie
        corrected = raw / np.where(bt > 0, bt, 1.0)
    else:
        valid_vals = raw[mask_r]
        p99 = float(np.percentile(valid_vals, 99)) if valid_vals.size else 1.0
        corrected  = raw / max(p99, 1.0)

    # 2. Outside-blade → 1.0 so that -log gives exactly 0
    corrected = np.where(mask_r, corrected.clip(min=1e-9), 1.0)

    # 3. Negative log (line integrals)
    proj = -np.log(corrected)

    # 4. Enforce zero outside blade and clip
    proj = np.where(mask_r, proj, 0.0).clip(min=0.0)

    # 5. Zoom to isotropic psize + centre-crop.
    #    psize is the pixel pitch at the ISOCENTRE plane (the model places its
    #    virtual detector at X=0; DRRs are likewise defined at iso).  The XIM
    #    res_h/res_w are at the IMAGER plane (SID).  Convert imager->iso by the
    #    magnification M = SID/SAD before matching psize, otherwise projections
    #    are over-magnified by M (~1.54x for Halcyon) and wrongly cropped.
    mag    = geom["SID"] / geom["SAD"]
    zoom_h = (eff_res_h / mag) / psize_mm
    zoom_w = (eff_res_w / mag) / psize_mm
    zoomed = ndimage.zoom(proj, (zoom_h, zoom_w), order=1)
    zh, zw = zoomed.shape

    top  = max(0, (zh - out_H) // 2)
    left = max(0, (zw - out_W) // 2)
    img  = zoomed[top:top + out_H, left:left + out_W]

    # Zero-pad if smaller than target (edge case)
    if img.shape != (out_H, out_W):
        pad = np.zeros((out_H, out_W), dtype=np.float32)
        ch, cw = img.shape
        pad[(out_H - ch) // 2:(out_H - ch) // 2 + ch,
            (out_W - cw) // 2:(out_W - cw) // 2 + cw] = img
        img = pad

    # 6. Normalise
    if normalise:
        mx = img.max()
        if mx > 1e-9:
            img /= mx

    # 7. Gantry angle
    angle = _xim_gantry_angle(d)
    if angle is None:
        angle = _fallback_angle(xim_path, geom)

    return img.astype(np.float32), float(angle)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class XimProjectionDataset(Dataset):
    """
    Torch Dataset of Halcyon CBCT projections loaded directly from XIM files.

    Parameters
    ----------
    scan_root         : folder containing Scan.xml, Acquisitions/, Calibrations/
    H, W              : model input height / width [pixels]  (default 384 × 768)
    psize             : target isotropic pixel size [mm]     (default 0.7273)
    normalise         : divide each image by its max         (default True)
    bowtie_correction : apply flat-field / bowtie correction (default True)
    xim_indices       : list of int indices to select; None = all
    xim_mask          : 0/1 sequence or path, one per XIM;  None = all
    xim_json          : path to JSON with int indices or filename strings
    """

    def __init__(
        self,
        scan_root: str,
        H: int = 384,
        W: int = 768,
        psize: float = 0.7273,
        normalise: bool = False,
        bowtie_correction: bool = True,
        xim_indices=None,
        xim_mask=None,
        xim_json=None,
    ):
        self.H, self.W = H, W
        self.psize     = psize
        self.normalise = normalise
        self.scan_root = Path(scan_root)

        # 1. Geometry from Scan.xml
        xml_path = self.scan_root / "Scan.xml"
        if not xml_path.is_file():
            raise FileNotFoundError(f"Scan.xml not found at {xml_path}")
        self.geom = parse_scan_xml_xim(xml_path)

        # 2. XIM files
        acq_dir  = find_acquisition_dir(self.scan_root, self.geom["acq_id"])
        all_xims = find_xim_files(acq_dir)
        self.xims = resolve_xim_subset(all_xims, xim_indices, xim_mask, xim_json)
        if not self.xims:
            raise ValueError("No XIM files selected after subset filter.")

        # 3. Blade mask
        self.mask = blade_pixel_mask(self.geom)

        # 4. Bowtie calibration
        self.bowtie = None
        if bowtie_correction:
            bpath = find_bowtie_xim(self.scan_root, self.geom.get("mode_id"))
            if bpath is not None:
                self.bowtie = load_bowtie(bpath)
                print(f"[XimDataset] bowtie  : {bpath}")
            else:
                print("[XimDataset] WARNING: bowtie not found; "
                      "falling back to per-image 99th-percentile normalisation.")

        # Summary
        valid_rows = np.flatnonzero(self.mask.any(axis=1))
        valid_cols = np.flatnonzero(self.mask.any(axis=0))
        row_range  = f"{valid_rows[0]}–{valid_rows[-1]}" if valid_rows.size else "none"
        col_range  = f"{valid_cols[0]}–{valid_cols[-1]}" if valid_cols.size else "none"
        print(
            f"[XimDataset] scan_root   : {self.scan_root}\n"
            f"             projections : {len(self.xims)} / {len(all_xims)}\n"
            f"             detector    : {self.geom['full_H']} × {self.geom['full_W']} px"
            f"  ({self.geom['res_h']} × {self.geom['res_w']} mm/px)\n"
            f"             blade mask  : rows {row_range}  cols {col_range}\n"
            f"             output      : {H} × {W} px @ {psize} mm/px"
        )

    def __len__(self):
        return len(self.xims)

    def __getitem__(self, idx):
        img, angle = _process_xim(
            self.xims[idx],
            bowtie=self.bowtie,
            mask=self.mask,
            geom=self.geom,
            psize_mm=self.psize,
            out_H=self.H,
            out_W=self.W,
            normalise=self.normalise,
        )
        return img, angle    # (H, W) float32,  scalar float


# ---------------------------------------------------------------------------
# Convenience loader (returns model-ready tensors in one call)
# ---------------------------------------------------------------------------

def load_xim_projections(
    scan_root: str,
    H: int = 384,
    W: int = 768,
    psize: float = 0.7273,
    bowtie_correction: bool = True,
    xim_indices=None,
    xim_mask=None,
    xim_json=None,
    normalise: bool = False,
) -> dict:
    """
    Load all selected XIM projections and return model-ready tensors.

    Returns
    -------
    dict:
        "projections" : (1, N, H, W) float32 tensor
        "angles"      : (1, N)       float32 tensor  [degrees, 0-360)
        "N"           : int
    """
    ds = XimProjectionDataset(
        scan_root=scan_root, H=H, W=W, psize=psize,
        normalise=normalise, bowtie_correction=bowtie_correction,
        xim_indices=xim_indices, xim_mask=xim_mask, xim_json=xim_json,
    )

    imgs, angles = [], []
    for img, angle in ds:
        imgs.append(img)
        angles.append(angle)

    projs = torch.tensor(np.stack(imgs)).float()   # (N, H, W)
    angs  = torch.tensor(angles).float()           # (N,)

    # Batch-level per-image max normalisation (matches single_batch_collate)
    mx    = projs.amax(dim=(1, 2), keepdim=True)
    projs = projs / torch.clamp(mx, min=1e-9)

    return {
        "projections": projs.unsqueeze(0),   # (1, N, H, W)
        "angles":      angs.unsqueeze(0),    # (1, N)
        "N":           len(imgs),
    }
