"""
Real Halcyon CBCT projection loader for cbct_recon model testing.

Processing pipeline per projection:
  1. Parse Scan.xml  → BladeX1/X2/Y1/Y2, SID, SAD, full detector geometry
  2. Project blade edges to the detector plane (multiply by SID/SAD)
  3. Convert blade-plane mm positions to pixel offsets in the full detector
  4. Embed blade-cropped .npy  →  zero-padded full detector  (384 × 3072)
  5. Zoom to isotropic psize  (1.12 mm → 0.7273 mm, 0.28 mm → 0.7273 mm)
  6. Centre-crop to model input  (384 × 768)
  7. Convert angle: meta_data radians → degrees in [0, 360)
"""

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Scan.xml parsing
# ---------------------------------------------------------------------------

_SCAN_XML_NS = "http://baden.varian.com/cr.xsd"


def _find_xml_value(root, tag, dtype=float, ns=_SCAN_XML_NS):
    """Return the text of the first matching element, or None if absent."""
    el = root.find(f".//{{{ns}}}{tag}")
    if el is None:                          # try without namespace
        el = root.find(f".//{tag}")
    return dtype(el.text.strip()) if el is not None else None


def parse_scan_xml(xml_path):
    """
    Parse a Halcyon Scan.xml (UTF-16 LE, with BOM) and extract the blade
    positions and full detector geometry needed for padding computation.

    Returns
    -------
    dict with keys:
        SID        – source-to-imager distance [mm]
        SAD        – source-to-isocenter distance [mm]
        full_H     – full detector rows     (ImagerSizeY)
        full_W     – full detector columns  (ImagerSizeX)
        res_h      – row spacing   [mm/pixel]  (ImagerResY)
        res_w      – column spacing [mm/pixel]  (ImagerResX)
        blade_y1   – lower Y blade at source plane [mm]  (BladeY1)
        blade_y2   – upper Y blade at source plane [mm]  (BladeY2)
        blade_x1   – left  X blade at source plane [mm]  (BladeX1)
        blade_x2   – right X blade at source plane [mm]  (BladeX2)
        imager_lat – lateral  detector offset [mm]  (ImagerLat, usually 0)
        imager_lng – longitudinal detector offset [mm]  (ImagerLng, usually 0)
    """
    raw = Path(xml_path).read_bytes()
    text = raw.decode("utf-16")            # handles BOM automatically
    root = ET.fromstring(text)

    return {
        "SID":        _find_xml_value(root, "SID"),
        "SAD":        _find_xml_value(root, "SAD"),
        "full_H":     _find_xml_value(root, "ImagerSizeY", int),
        "full_W":     _find_xml_value(root, "ImagerSizeX", int),
        "res_h":      _find_xml_value(root, "ImagerResY"),
        "res_w":      _find_xml_value(root, "ImagerResX"),
        "blade_y1":   _find_xml_value(root, "BladeY1"),
        "blade_y2":   _find_xml_value(root, "BladeY2"),
        "blade_x1":   _find_xml_value(root, "BladeX1"),
        "blade_x2":   _find_xml_value(root, "BladeX2"),
        "imager_lat": _find_xml_value(root, "ImagerLat") or 0.0,
        "imager_lng": _find_xml_value(root, "ImagerLng") or 0.0,
    }


# ---------------------------------------------------------------------------
# Blade-based padding offset computation
# ---------------------------------------------------------------------------

def compute_blade_offsets(scan_xml_params, crop_h, crop_w):
    """
    Compute where the blade-cropped projection sits inside the full detector,
    using the actual blade jaw positions from Scan.xml.

    Physics
    -------
    The Halcyon X-ray source is at distance SAD from the isocenter.
    The detector is at distance SID from the source.
    Magnification  m = SID / SAD.

    A blade edge at  B_mm  (in mm at the source plane, relative to isocenter)
    projects to the detector at  B_mm * m.

    In image coordinates:
      row 0  = top of detector  = physical Y = +(full_H * res_h / 2) mm
      col 0  = left of detector = physical X = -(full_W * res_w / 2) mm

    The lateral detector offset (ImagerLat, positive → detector shifted right)
    shifts the isocenter projection:
      col_iso = full_W / 2 - imager_lat / res_w

    Parameters
    ----------
    scan_xml_params : dict from parse_scan_xml()
    crop_h, crop_w  : shape of the blade-cropped .npy data (from meta_data.json
                      scanner.nDetector), used only for a sanity clamp.

    Returns
    -------
    y_start : int  – first row  of the crop within the full detector
    x_start : int  – first col  of the crop within the full detector
    full_H  : int  – full detector row count
    full_W  : int  – full detector col count
    """
    p = scan_xml_params
    SID, SAD         = p["SID"],      p["SAD"]
    full_H, full_W   = p["full_H"],   p["full_W"]
    res_h, res_w     = p["res_h"],    p["res_w"]
    imager_lat       = p["imager_lat"]

    mag = SID / SAD   # magnification from source plane to detector plane

    # --- project blade edges to detector plane ---
    blade_y2_det = p["blade_y2"] * mag   # upper blade edge in detector Y [mm]
    blade_x1_det = p["blade_x1"] * mag   # left  blade edge in detector X [mm]

    # --- full detector half-extents from centre [mm] ---
    det_top_mm   = full_H * res_h / 2    # physical Y at row 0
    det_left_mm  = full_W * res_w / 2    # physical |X| at col 0

    # --- isocenter column offset due to lateral detector shift ---
    # (positive imager_lat shifts the detector to the right, moving the
    #  isocenter projection to the left by imager_lat / res_w columns)
    iso_col_shift = imager_lat / res_w   # in pixels

    # --- convert blade-plane positions to pixel indices ---
    # row 0 = top = +det_top_mm   → row_start is where upper blade projects
    y_start_f = (det_top_mm   - blade_y2_det) / res_h
    # col 0 = left = -det_left_mm → col_start is where left blade projects
    #   (+iso_col_shift moves the crop right when detector is shifted right)
    x_start_f = (det_left_mm  + blade_x1_det) / res_w - iso_col_shift

    y_start = int(round(y_start_f))
    x_start = int(round(x_start_f))

    # --- clamp so the crop stays inside the canvas ---
    y_start = max(0, min(y_start, full_H - crop_h))
    x_start = max(0, min(x_start, full_W - crop_w))

    return y_start, x_start, full_H, full_W


def _fallback_offsets(crop_h, crop_w, full_h, full_w):
    """
    Fallback when Scan.xml is unavailable.

    Halcyon's Y blade is asymmetric (upper blade crops more), so the crop is
    flush with the bottom.  X blade is symmetric.
    """
    y_start = full_h - crop_h          # crop flush with bottom
    x_start = (full_w - crop_w) // 2   # symmetric
    return y_start, x_start


# ---------------------------------------------------------------------------
# Per-image processing helpers
# ---------------------------------------------------------------------------

def _embed_in_full_detector(img, y_start, x_start, full_H, full_W):
    """
    Place blade-cropped projection into a zero-padded full-detector canvas.

    Pixels outside the beam (unexposed detector area) are zero, representing
    no attenuation outside the collimated field.
    """
    canvas = np.zeros((full_H, full_W), dtype=np.float32)
    h, w   = img.shape
    canvas[y_start:y_start + h, x_start:x_start + w] = img
    return canvas


def _resample_and_crop(img_full, pixel_h_mm, pixel_w_mm,
                       target_psize_mm, out_h, out_w):
    """
    Zoom from native (anisotropic) detector pixel sizes to isotropic
    `target_psize_mm`, then centre-crop to (out_h, out_w).

    The centre of the zoomed image corresponds to the detector centre
    (isocenter projection), so the crop is always anchored to anatomy.
    """
    zoom_h = pixel_h_mm / target_psize_mm   # e.g. 1.12 / 0.7273 ≈ 1.540
    zoom_w = pixel_w_mm / target_psize_mm   # e.g. 0.28  / 0.7273 ≈ 0.385

    zoomed = ndimage.zoom(img_full, (zoom_h, zoom_w), order=1)
    zh, zw = zoomed.shape

    top  = max(0, (zh - out_h) // 2)
    left = max(0, (zw - out_w) // 2)
    cropped = zoomed[top:top + out_h, left:left + out_w]

    # zero-pad if zoomed image is smaller than the target (edge case)
    if cropped.shape != (out_h, out_w):
        pad = np.zeros((out_h, out_w), dtype=np.float32)
        ch, cw = cropped.shape
        dh = max(0, (out_h - ch) // 2)
        dw = max(0, (out_w - cw) // 2)
        pad[dh:dh + ch, dw:dw + cw] = cropped
        cropped = pad

    return cropped.astype(np.float32)


def _angle_rad_to_deg(angle_rad):
    """Convert meta_data radian angle to degrees in [0, 360)."""
    return float(np.degrees(angle_rad) % 360.0)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RealCBCTProjectionDataset(Dataset):
    """
    Dataset of real Halcyon CBCT projections loaded from a meta_data.json folder.

    Selects projections whose "time" (respiratory phase 0–1) falls within
    [target_time - time_tol, target_time + time_tol].

    Blade positions are read from ``Scan.xml`` in the same directory to
    compute the exact detector-plane padding offsets.  If Scan.xml is absent,
    a geometric fallback (Y flush-bottom, X symmetric) is used instead.

    Each item is processed to match the cbct_recon model input:
      - Embedded in the full detector (384 × 3072) using blade-derived offsets
      - Zoomed to isotropic ``psize`` mm/pixel
      - Centre-cropped to (H, W)
      - Angle converted from radians → degrees in [0, 360)

    Parameters
    ----------
    data_root    : folder containing meta_data.json (and optionally Scan.xml)
    split        : "train" or "test"
    target_time  : respiratory phase to select  (0 = exhale, 0.5 = inhale …)
    time_tol     : half-window tolerance around target_time  (default 0.05)
    H, W         : model input height / width in pixels     (default 384 × 768)
    psize        : model isotropic pixel size in mm         (default 0.7273)
    normalise    : divide each image by its per-image max   (matches DRR loader)
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        target_time: float = 0.0,
        time_tol: float = 0.05,
        H: int = 384,
        W: int = 768,
        psize: float = 0.7273,
        normalise: bool = True,
    ):
        self.data_root = Path(data_root)
        self.H, self.W = H, W
        self.psize     = psize
        self.normalise = normalise

        # --- meta_data.json ---
        with open(self.data_root / "meta_data.json") as f:
            self.meta = json.load(f)

        scanner       = self.meta["scanner"]
        n_det         = scanner["nDetector"]      # [crop_h, crop_w]
        s_det         = scanner["sDetector"]      # [phys_h_mm, phys_w_mm]
        self.crop_h   = n_det[0]
        self.crop_w   = n_det[1]
        # Native pitches AT THE ISO PLANE (detector-plane pitch / magnification):
        # e.g. rows 1.12 / 1.54 ≈ 0.7273 mm, cols 0.28 / 1.54 ≈ 0.1818 mm.
        self.pixel_h  = s_det[0] / n_det[0] / scanner["DSD"] * scanner["DSO"]
        self.pixel_w  = s_det[1] / n_det[1] / scanner["DSD"] * scanner["DSO"]
        self._mag     = scanner["DSD"] / scanner["DSO"]   # iso → detector plane

        # --- Scan.xml: blade positions → padding offsets ---
        xml_path = self.data_root / "Scan.xml"
        if xml_path.exists():
            xml_params = parse_scan_xml(xml_path)
            self.y_start, self.x_start, self.full_H, self.full_W = \
                compute_blade_offsets(xml_params, self.crop_h, self.crop_w)
            self._offset_source = "Scan.xml"
        else:
            # fallback: derive full detector dimensions from meta_data pixel sizes.
            # 430.08 × 860.16 mm is the full Halcyon panel at the DETECTOR plane
            # (384 × 3072 at 1.12 × 0.28 mm/pixel); pixel_h/pixel_w are iso-plane
            # pitches, so scale the panel extent to the iso plane before dividing.
            self.full_H = round(430.08 / self._mag / self.pixel_h)   # ≈ 384
            self.full_W = round(860.16 / self._mag / self.pixel_w)   # ≈ 3072
            self.y_start, self.x_start = _fallback_offsets(
                self.crop_h, self.crop_w, self.full_H, self.full_W
            )
            self._offset_source = "fallback"

        # --- filter projections by respiratory phase ---
        all_projs = self.meta.get(f"proj_{split}", [])
        lo, hi    = target_time - time_tol, target_time + time_tol
        self.projs = [p for p in all_projs if lo <= p["time"] <= hi]

        if not self.projs:
            available = sorted({p["time"] for p in all_projs})
            raise ValueError(
                f"No projections found for split='{split}', "
                f"target_time={target_time} ± {time_tol}.\n"
                f"Available time values: {available}"
            )

        self._print_geometry()

    # ------------------------------------------------------------------

    def _print_geometry(self):
        print(
            f"[RealCBCTProjectionDataset] offset source: {self._offset_source}\n"
            f"  full detector : {self.full_H} × {self.full_W} px"
            f"  ({self.full_H * self.pixel_h:.1f} × {self.full_W * self.pixel_w:.1f} mm)\n"
            f"  blade crop    : {self.crop_h} × {self.crop_w} px  "
            f"  placed at y_start={self.y_start}, x_start={self.x_start}\n"
            f"  after zoom+crop → {self.H} × {self.W} px "
            f"  @ {self.psize} mm/px\n"
            f"  projections selected: {len(self.projs)}"
        )

    def __len__(self):
        return len(self.projs)

    def __getitem__(self, idx):
        entry   = self.projs[idx]
        rel     = entry["file_path"].replace("\\", os.sep)
        img     = np.load(self.data_root / rel).astype(np.float32)

        # 1. embed in full detector using blade offsets
        full    = _embed_in_full_detector(
            img, self.y_start, self.x_start, self.full_H, self.full_W
        )

        # 2. zoom to isotropic psize + centre-crop
        proc    = _resample_and_crop(
            full, self.pixel_h, self.pixel_w, self.psize, self.H, self.W
        )

        # 3. normalise (matches DRR training path)
        if self.normalise:
            mx = proc.max()
            if mx > 1e-9:
                proc /= mx

        # 4. angle in degrees [0, 360)
        angle_deg = _angle_rad_to_deg(entry["angle"])

        return proc, angle_deg     # (H, W) float32,  scalar float

    # ------------------------------------------------------------------

    def load_all(self):
        """
        Load every selected projection and return stacked tensors.

        Returns
        -------
        projections : torch.Tensor  (N, H, W)
        angles      : torch.Tensor  (N,)  degrees in [0, 360)
        """
        imgs, angles = [], []
        for img, angle in self:
            imgs.append(img)
            angles.append(angle)
        return (
            torch.tensor(np.stack(imgs)).float(),
            torch.tensor(angles).float(),
        )


# ---------------------------------------------------------------------------
# Collate / DataLoader helpers
# ---------------------------------------------------------------------------

def real_cbct_collate(batch):
    """
    Collate a list of (img, angle_deg) tuples into model-ready tensors.

    Returns
    -------
    dict:
      "projections" : (1, N, H, W)
      "angles"      : (1, N)  degrees in [0, 360)
      "N"           : int
    """
    imgs   = torch.tensor(np.stack([b[0] for b in batch])).float()  # (N, H, W)
    angles = torch.tensor([b[1] for b in batch]).float()            # (N,)

    # per-image normalise (matches single_batch_collate)
    mx   = imgs.amax(dim=(1, 2), keepdim=True)
    imgs = imgs / torch.clamp(mx, min=1e-9)

    return {
        "projections": imgs.unsqueeze(0),    # (1, N, H, W)
        "angles":      angles.unsqueeze(0),  # (1, N)
        "N":           len(batch),
    }


def create_real_cbct_dataloader(
    data_root: str,
    split: str = "train",
    target_time: float = 0.0,
    time_tol: float = 0.05,
    H: int = 384,
    W: int = 768,
    psize: float = 0.7273,
    num_workers: int = 0,
) -> DataLoader:
    """
    Build a DataLoader yielding one batch of all projections at the
    requested respiratory phase.

    Typical usage::

        loader = create_real_cbct_dataloader(
            data_root="/path/to/20260113",
            target_time=0.0,   # exhale
            time_tol=0.05,
        )
        batch = next(iter(loader))
        projections = batch["projections"]  # (1, N, 384, 768)
        angles      = batch["angles"]       # (1, N)  degrees

    Parameters
    ----------
    data_root   : folder with meta_data.json  (+Scan.xml if available)
    split       : "train" or "test"
    target_time : respiratory phase (0 = exhale, 0.5 = inhale)
    time_tol    : half-window tolerance
    H, W        : model input size  (must match training: default 384 × 768)
    psize       : model pixel size in mm  (must match training: default 0.7273)
    num_workers : DataLoader workers
    """
    ds = RealCBCTProjectionDataset(
        data_root=data_root, split=split,
        target_time=target_time, time_tol=time_tol,
        H=H, W=W, psize=psize,
    )
    return DataLoader(
        ds,
        batch_size=len(ds),
        shuffle=False,
        collate_fn=real_cbct_collate,
        num_workers=num_workers,
    )


# ---------------------------------------------------------------------------
# Multi-phase convenience loader
# ---------------------------------------------------------------------------

def load_all_phases(
    data_root: str,
    split: str = "train",
    time_values=None,
    time_tol: float = 0.05,
    H: int = 384,
    W: int = 768,
    psize: float = 0.7273,
):
    """
    Load projections for each respiratory phase in ``time_values``.

    Parameters
    ----------
    time_values : iterable of floats; if None, uses all unique time values
                  present in the metadata

    Returns
    -------
    dict  time_value → {"projections": (1, N, H, W), "angles": (1, N), "N": int}
    """
    meta_path = Path(data_root) / "meta_data.json"
    with open(meta_path) as f:
        meta = json.load(f)

    if time_values is None:
        time_values = sorted({p["time"] for p in meta.get(f"proj_{split}", [])})

    return {
        t: next(iter(create_real_cbct_dataloader(
            data_root=data_root, split=split,
            target_time=t, time_tol=time_tol,
            H=H, W=W, psize=psize,
        )))
        for t in time_values
    }
