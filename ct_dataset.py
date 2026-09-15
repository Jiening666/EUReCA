# Defer annotation evaluation so PEP 604 unions (e.g. `str | Path`) don't break
# imports on Python < 3.10 -- the comparator envs (DIF-Gaussian py3.9,
# GeoAware py3.8) import this module for the shared manifest/split.
from __future__ import annotations

import os
from pathlib import Path
import re
import math
import time
import ast
import pandas as pd
import nibabel as nib
import numpy as np
from scipy import ndimage
import random
from torch.utils.data import Dataset
import torch

from ct_generate_drr import generate_drr
from threeD_geometry import map_2d_plucker_ray_directions
from torch.utils.data import DataLoader
from functools import partial
from PIL import Image

from itertools import combinations
import configparser
from glob import glob
import shutil


def _read_nifti_geometry(nifti_path: Path):
    """
    Read just the header of a NIfTI file and return voxel grid + spacing.
    Returns None if the file can't be read or isn't 3D.
    """
    try:
        img = nib.load(str(nifti_path))
        shape = img.shape
        zooms = img.header.get_zooms()
    except Exception as e:
        print(f"[warn] failed to read header of {nifti_path}: {e}")
        return None
 
    if len(shape) < 3 or len(zooms) < 3:
        print(f"[warn] {nifti_path} is not 3D (shape={shape}), skipping geometry")
        return None
 
    dx, dy, dz = int(shape[0]), int(shape[1]), int(shape[2])
    rx, ry, rz = float(zooms[0]), float(zooms[1]), float(zooms[2])
    return {
        "dim_x": dx, "dim_y": dy, "dim_z": dz,
        "res_x": round(rx, 6), "res_y": round(ry, 6), "res_z": round(rz, 6),
        "phys_x": round(dx * rx, 4),
        "phys_y": round(dy * ry, 4),
        "phys_z": round(dz * rz, 4),
    }

def scan_manifest(
    root: str | Path,
    case_pattern: str = "*",
    scan_pattern: str = "*",
    original_glob: str = "*.nii.gz",
    aug_prefix: str = "aug",
    strict: bool = False,
    verbose: int = 0,
    progress_every: int = 25,
    cache_dir=None,
    rescan: bool = False,
) -> pd.DataFrame:
    """
    Walk `root` and build a flat manifest of scans on disk.
 
    Expected layout:
        root/
          <case_id>/
            <scan_id>/
              *.nii.gz          # original (required)
              aug0/             # optional, no transform
              aug1/             # optional, augmented
              ...
 
    Returns a DataFrame whose columns depend on `verbose`:
        verbose=0 (default, fast):
            case_id, scan_id, scan_root, original_path, n_aug
        verbose>=1 (reads each NIfTI header, slower):
            adds dim_x, dim_y, dim_z, res_x, res_y, res_z,
                 phys_x, phys_y, phys_z
 
    Parameters
    ----------
    root : path to the dataset root (e.g. ".../MELA")
    case_pattern, scan_pattern : glob filters if you want to restrict which
        case/scan folders are picked up (default: everything).
    original_glob : pattern that identifies the "original" CT file inside a
        scan folder. A scan folder must contain exactly one match.
    aug_prefix : folders starting with this string under a scan folder are
        counted as augmentations.
    strict : if True, raise on malformed folders instead of skipping them.
    verbose : 0 = basic info only (no nifti read). >=1 = also read each
        original nifti header to add voxel grid size, spacing, and
        physical extent. Use this when you intend to filter by geometry
        (e.g. via filter_manifest).
    progress_every : when verbose>=1, print progress every N scans.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Manifest root does not exist: {root}")

    # --- manifest cache: skip re-reading every NIfTI header on each launch. ---
    # Keyed by (root, aug_prefix, verbose); pass rescan=True to force a rebuild.
    _cache_file = None
    if cache_dir is not None:
        import hashlib
        _key = hashlib.md5(f"{root}|{aug_prefix}|{verbose}".encode()).hexdigest()[:10]
        _cache_file = Path(cache_dir) / f"manifest_{root.name}_{_key}.pkl"
        if not rescan and _cache_file.exists():
            try:
                df = pd.read_pickle(_cache_file)
                print(f"[scan_manifest] loaded cached manifest for {root.name} "
                      f"({len(df)} scans) from {_cache_file}")
                return df
            except Exception as _e:
                print(f"[scan_manifest] cache read failed ({_e}); rescanning {root}")
 
    base_cols = ["case_id", "scan_id", "scan_root", "original_path", "n_aug"]
    extra_cols = ["dim_x", "dim_y", "dim_z",
                  "res_x", "res_y", "res_z",
                  "phys_x", "phys_y", "phys_z"]
 
    rows = []
    for case_dir in sorted(root.glob(case_pattern)):
        if not case_dir.is_dir():
            continue
        case_id = case_dir.name
 
        for scan_dir in sorted(case_dir.glob(scan_pattern)):
            if not scan_dir.is_dir():
                continue
            scan_id = scan_dir.name
 
            # 1. Find the original *.nii.gz at the scan-folder level.
            #    glob, not rglob — augs are not originals.
            originals = [
                p for p in scan_dir.glob(original_glob)
                if p.is_file() and not p.name.startswith(aug_prefix)
            ]
            if len(originals) == 0:
                msg = f"No original {original_glob} in {scan_dir}"
                if strict: raise ValueError(msg)
                #log.warning("%s — skipping", msg)
                continue
            if len(originals) > 1:
                msg = f"Multiple originals in {scan_dir}: {[p.name for p in originals]}"
                if strict: raise ValueError(msg)
                #log.warning("%s — using first", msg)
            original_path = originals[0]
 
            # 2. Count aug* subfolders (non-recursive).
            aug_dirs = [
                d for d in scan_dir.iterdir()
                if d.is_dir() and d.name.startswith(aug_prefix)
            ]
            n_aug = len(aug_dirs)
 
            row = {
                "case_id": case_id,
                "scan_id": scan_id,
                "scan_root": scan_dir,
                "original_path": original_path,
                "n_aug": n_aug,
            }
 
            # Geometry (verbose>=1) is read AFTER the walk, in parallel (below).
 
            rows.append(row)

    # Parallel geometry read (verbose>=1): reading tens of thousands of gzip
    # NIfTI headers serially over NFS dominated startup. Threads overlap the
    # per-file open/read latency (I/O-bound), cutting a ~30-min scan to minutes.
    if verbose >= 1 and rows:
        from concurrent.futures import ThreadPoolExecutor

        def _geom(p):
            g = _read_nifti_geometry(p)
            if g is None:
                if strict:
                    raise ValueError(f"Could not read geometry of {p}")
                g = {k: float("nan") for k in extra_cols}
            return g

        n_workers = min(32, (os.cpu_count() or 4) * 4)
        done = 0
        with ThreadPoolExecutor(max_workers=n_workers) as _ex:
            for row, geom in zip(
                    rows, _ex.map(_geom, [r["original_path"] for r in rows])):
                row.update(geom)
                done += 1
                if progress_every and done % progress_every == 0:
                    print(f"  scanned {done} scans...")
 
    if not rows:
        #log.warning("scan_manifest found no scans under %s", root)
        pass
 
    cols = base_cols + (extra_cols if verbose >= 1 else [])
    df = pd.DataFrame(rows, columns=cols)
    if _cache_file is not None:
        try:
            _cache_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_pickle(_cache_file)
            print(f"[scan_manifest] cached manifest for {root.name} "
                  f"({len(df)} scans) -> {_cache_file}")
        except Exception as _e:
            print(f"[scan_manifest] cache write failed ({_e}); continuing")
    return df
 
 
def filter_manifest(
    df: pd.DataFrame,
    max_res: float = 2.0,
    min_phys_z: float = 380.0,
    drop_nan: bool = True,
) -> pd.DataFrame:
    """
    Filter a manifest produced by scan_manifest(..., verbose>=1).
 
    Keeps only rows where:
        res_x < max_res  AND  res_y < max_res  AND  res_z < max_res
        AND  phys_z > min_phys_z
 
    Parameters
    ----------
    df : DataFrame from scan_manifest with the extended geometry columns.
    max_res : upper bound (exclusive) on voxel spacing in any axis, in mm.
    min_phys_z : lower bound (exclusive) on z-axis physical extent, in mm.
    drop_nan : if True, rows with NaN in any of the relevant columns are
        treated as failing the filter (i.e. dropped). If False, they are
        kept (pandas comparisons against NaN are False, so this only
        matters when you want to inspect them separately).
 
    Returns
    -------
    A new DataFrame containing the surviving rows, index reset.
    """
    required = ["res_x", "res_y", "res_z", "phys_z"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"filter_manifest needs geometry columns {missing}. "
            f"Did you call scan_manifest(..., verbose=1)?"
        )
 
    if drop_nan:
        mask_valid = df[required].notna().all(axis=1)
    else:
        mask_valid = pd.Series(True, index=df.index)
 
    mask = (
        mask_valid
        & (df["res_x"] < max_res)
        & (df["res_y"] < max_res)
        & (df["res_z"] < max_res)
        & (df["phys_z"] > min_phys_z)
    )
    return df.loc[mask].reset_index(drop=True)

def sample_manifest(
    df: pd.DataFrame,
    n: int,
    seed: int | None = None,
    replace: bool = False,
) -> pd.DataFrame:
    """
    Randomly sample `n` rows from a manifest DataFrame.

    Parameters
    ----------
    df : manifest DataFrame (e.g. output of scan_manifest / filter_manifest).
    n : number of rows to sample.
    seed : optional RNG seed for reproducible sampling.
    replace : if False (default) and n > len(df), sample is capped at len(df).
              if True, sampling is with replacement and can exceed len(df).

    Returns
    -------
    A new DataFrame with `n` rows (or fewer if df is smaller and replace=False),
    with the index reset.
    """
    if n == float('inf'):
        if replace:
            raise ValueError("n=inf is not meaningful with replace=True")
        n = len(df)
    if not replace:
        n = int(min(n, len(df)))
    return df.sample(n=n, replace=replace, random_state=seed).reset_index(drop=True)


def split_cases(all_cases, test_size=0.2, val_frac=0.5, seed=42):
    """Deterministic train / val / test split over unique case ids.

    Two-stage, so the *train* set is byte-identical to the historical
    ``train_test_split(all_cases, test_size=0.2, random_state=42)`` split:

      1. carve off a held-out fraction ``test_size`` (default 0.2) exactly as
         before -> (train, held-out);
      2. split the held-out set into val : test by ``val_frac`` (default 0.5,
         i.e. 10% / 10% of the whole).

    ``val`` is used only for training-time monitoring; ``test`` is reserved and
    should only ever be touched by test.py. Same ``seed`` everywhere => the
    reserved test set is identical and reproducible across train and test.

    Returns (train_cases, val_cases, test_cases) as numpy arrays.
    """
    from sklearn.model_selection import train_test_split
    all_cases = np.asarray(all_cases)
    train_cases, heldout = train_test_split(
        all_cases, test_size=test_size, random_state=seed)
    # test_size here = the TEST share of the held-out set (1 - val_frac).
    val_cases, test_cases = train_test_split(
        heldout, test_size=(1.0 - val_frac), random_state=seed)
    return train_cases, val_cases, test_cases


# def scan_manifest(
#     root: str | Path,
#     case_pattern: str = "*",
#     scan_pattern: str = "*",
#     original_glob: str = "*.nii.gz",
#     aug_prefix: str = "aug",
#     strict: bool = False,
# ) -> pd.DataFrame:
#     """
#     Walk `root` and build a flat manifest of scans on disk.

#     Expected layout:
#         root/
#           <case_id>/
#             <scan_id>/
#               *.nii.gz          # original (required)
#               aug0/             # optional, no transform
#               aug1/             # optional, augmented
#               ...

#     Returns a DataFrame with columns:
#         case_id, scan_id, scan_root (absolute Path), original_path, n_aug

#     Parameters
#     ----------
#     root : path to the dataset root (e.g. ".../MELA")
#     case_pattern, scan_pattern : glob filters if you want to restrict which
#         case/scan folders are picked up (default: everything).
#     original_glob : pattern that identifies the "original" CT file inside a
#         scan folder. A scan folder must contain exactly one match.
#     aug_prefix : folders starting with this string under a scan folder are
#         counted as augmentations.
#     strict : if True, raise on malformed folders instead of skipping them.
#     """
#     root = Path(root).resolve()
#     if not root.is_dir():
#         raise FileNotFoundError(f"Manifest root does not exist: {root}")

#     rows = []
#     for case_dir in sorted(root.glob(case_pattern)):
#         if not case_dir.is_dir():
#             continue
#         case_id = case_dir.name

#         for scan_dir in sorted(case_dir.glob(scan_pattern)):
#             if not scan_dir.is_dir():
#                 continue
#             scan_id = scan_dir.name

#             # 1. Find the original *.nii.gz at the scan-folder level.
#             #    glob, not rglob — augs are not originals.
#             originals = [
#                 p for p in scan_dir.glob(original_glob)
#                 if p.is_file() and not p.name.startswith(aug_prefix)
#             ]
#             if len(originals) == 0:
#                 msg = f"No original {original_glob} in {scan_dir}"
#                 if strict: raise ValueError(msg)
#                 #log.warning("%s — skipping", msg)
#                 continue
#             if len(originals) > 1:
#                 msg = f"Multiple originals in {scan_dir}: {[p.name for p in originals]}"
#                 if strict: raise ValueError(msg)
#                 #log.warning("%s — using first", msg)
#             original_path = originals[0]

#             # 2. Count aug* subfolders (non-recursive).
#             aug_dirs = [
#                 d for d in scan_dir.iterdir()
#                 if d.is_dir() and d.name.startswith(aug_prefix)
#             ]
#             n_aug = len(aug_dirs)

#             rows.append({
#                 "case_id": case_id,
#                 "scan_id": scan_id,
#                 "scan_root": scan_dir,
#                 "original_path": original_path,
#                 "n_aug": n_aug,
#             })

#     if not rows:
#         #log.warning("scan_manifest found no scans under %s", root)
#         pass

#     df = pd.DataFrame(rows, columns=["case_id", "scan_id", "scan_root",
#                                      "original_path", "n_aug"])
#     return df


class ScanLoader:
    def __init__(self, max_n_aug=1, aug_generator=None, aug_prefix="aug", mode='continue',
                 fixed_aug_index=None):
        """
        max_n_aug: how many aug folders to maintain per scan (e.g. aug_0, aug_1, ..., aug_{max_n_aug-1})
        aug_generator: a CTAugmentor instance or similar with a .generate(scan_root, aug_index, is_identity_aug) method that creates the aug folder when called
        aug_prefix: prefix for augmentation folders (default "aug", so folders are aug_0, aug_1, etc.)
        mode: loading mode ('necessary' or 'continue' or 'force_new') - if 'necessary', only generate new augmentations if there is no processed aug existing; if 'continue', keep existing augmentations and only generate one new one and use it if there are less than max_n_aug existing, otherwise reuse existing ones; if 'force_new', always generate a new augmentation (overwriting existing ones)
        """
        self.aug_generator = aug_generator  # callable: (scan_root) -> new aug folder
        self.max_n_aug = max_n_aug
        self.aug_prefix = aug_prefix
        self.mode = mode
        # pin every load to one aug folder (e.g. 0 = the identity aug) for
        # deterministic, comparator-matched evaluation dumps; None = legacy
        # random selection
        self.fixed_aug_index = fixed_aug_index

    def prepare(self, scan_root, mode=None, aug_prefix=None):
        """
        mode options:
            - 'force_new': always generate a new augmentation (overwriting existing aug_0, aug_1, etc.)
            - 'continue': keep existing augmentations and only generate new ones if needed to reach max_n_aug
        """
        if mode is None:
            mode = self.mode
        if aug_prefix is None:
            aug_prefix = self.aug_prefix
        if mode == 'force_new':
            for i in range(self.max_n_aug):
                self._cleanup_aug(scan_root, i)
                self.aug_generator.generate(scan_root, aug_index=i, is_identity_aug=(i==0), overwrite=True, aug_prefix=aug_prefix)
        elif mode == 'continue':
            #existing_aug_count = len(aug_dirs)
            #for i in range(existing_aug_count, self.max_n_aug):
            for i in range(self.max_n_aug):
                if not self._check_aug_integrity(scan_root, aug_index=i):
                    self._cleanup_aug(scan_root, i, aug_prefix)
                    self.aug_generator.generate(scan_root, aug_index=i, is_identity_aug=(i==0), overwrite=False, aug_prefix=aug_prefix)
        elif mode == 'necessary':
            if not any(self._check_aug_integrity(scan_root, aug_index=i) for i in range(self.max_n_aug)):
                self._cleanup_aug(scan_root, 0, aug_prefix)
                self.aug_generator.generate(scan_root, aug_index=0, is_identity_aug=True, overwrite=True, aug_prefix=aug_prefix)
    
    def _check_aug_integrity(self,scan_root, aug_index, aug_prefix=None):
        scan_root = Path(scan_root)
        if aug_prefix is None:
            aug_prefix = self.aug_prefix
        aug_path  = scan_root / f"{aug_prefix}_{aug_index}"
        ct_path = aug_path / 'ct_resampled.npz'
        drr_path = aug_path / 'img' / 'proj_358.00.png'
        return ct_path.is_file() and drr_path.is_file()
    
    def _cleanup_aug(self, scan_root, aug_index, aug_prefix=None):
        if aug_prefix is None:
            aug_prefix = self.aug_prefix
        aug_path = Path(scan_root) / f"{aug_prefix}_{aug_index}"
        if aug_path.is_dir():
            try:
                shutil.rmtree(aug_path)
            except OSError as e:
                if e.errno == 39:  # Directory not empty
                    time.sleep(0.5)  # Wait for the FS to catch up
                    shutil.rmtree(aug_path)
                else:
                    raise e      
    
    def load(self, scan_root, mode=None, aug_prefix=None):
        if mode is None:
            mode = self.mode
        if aug_prefix is None:
            aug_prefix = self.aug_prefix
        if self.fixed_aug_index is not None:
            fixed = Path(scan_root) / f"{aug_prefix}_{self.fixed_aug_index}"
            if self._check_aug_integrity(scan_root, self.fixed_aug_index, aug_prefix):
                return self._read_aug(str(fixed))
            print(f"[ScanLoader] warning: fixed aug {fixed} missing/incomplete; "
                  f"falling back to legacy selection")
        aug_dirs = sorted(glob(f"{scan_root}/{aug_prefix}_[0-9]"))
        aug_dirs = [d for d in aug_dirs if self._check_aug_integrity(scan_root, int(d.split('_')[-1]), aug_prefix)]
        if mode == 'force_new' or not aug_dirs:
            aug_id = len(aug_dirs) if len(aug_dirs) < self.max_n_aug else random.randint(0, self.max_n_aug-1)
            aug_path = self.aug_generator.generate(scan_root, aug_index=aug_id, is_identity_aug=(aug_id==0), overwrite=True, aug_prefix=aug_prefix)
        elif mode == 'continue':
            if len(aug_dirs) < self.max_n_aug:
                aug_id = len(aug_dirs)
                aug_path = self.aug_generator.generate(scan_root, aug_index=aug_id, is_identity_aug=(aug_id==0), overwrite=False, aug_prefix=aug_prefix)
            else:
                aug_path = random.choice(aug_dirs)
                if not self._check_aug_integrity(scan_root, int(aug_path.split('_')[-1]), aug_prefix):
                    aug_path = self.aug_generator.generate(scan_root, aug_index=int(aug_path.split('_')[-1]), is_identity_aug=(aug_id==0), overwrite=True, aug_prefix=aug_prefix)
        else:
            aug_path = random.choice(aug_dirs)   # or deterministic policy
        return self._read_aug(aug_path)

    def _load_scan(self, aug_path, ct_min=-1000, ct_max=1000):
        ct_path = os.path.join(aug_path, 'ct_resampled.npz')
        print(f'Loading CT from {ct_path}')
        ct_data = _safe_npz_load(ct_path)
        
        ct_data = (ct_data - ct_min) / (ct_max - ct_min)  # Normalize to [0, 1]
        ct_data = np.clip(ct_data, 0.0, 1.0)  # Clip to [0, 1] range after normalization

        return torch.from_numpy(ct_data).unsqueeze(0)

    
    def _read_aug(self, aug_path):
        ct   = self._load_scan(aug_path)
        aug_img_path = f"{aug_path}/img/"
        return {"ct": ct, "aug_path": str(aug_path), "aug_img_path": aug_img_path}


class SingleCTDataset(Dataset):
    def __init__(self, manifest, cases, loader):
        self.items = manifest[manifest.case_id.isin(cases)].reset_index(drop=True)
        self.loader = loader

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        row = self.items.iloc[i]
        case_id = row.case_id

        a_data = self.loader.load(row.scan_root)
        out = {
            "case_id": case_id,
            # scan-level unique id: a case_id can hold MANY scans (e.g.
            # msk_prostate 328 has 8 CBCTs), so dumps keyed by case_id alone
            # silently overwrite each other.
            "scan_uid": f"{case_id}__{row.scan_id}",
            **{f"{k}_a": v for k, v in a_data.items()}
        }
        return out


class CTAugmentor:
    """
    Pipeline for resampling a CT and producing augmented `aug_<k>/` folders.

    One instance is configured from `opt` and an RNG seed; call `.generate(...)`
    once per (scan_root, aug_index) you want to write.

    Reads from `opt`:
        Nx, Ny, Nz                # target voxel grid
        vol_size                  # target voxel resolution (mm), isotropic
        aug_max_rot_deg           # max rotation per axis, degrees   (default 5.0)
        aug_max_shift_mm          # max translation per axis, mm     (default 5.0)
        aug_rot_axes              # subset of "xyz"                  (default "xyz")
        aug_seed                  # int or None                      (default None)
        d_angle                   # passed through to generate_drr   (default 2)
        proj_H, proj_W, proj_res  # recorded in aug.ini if present
    """

    # ----------------------------- construction -----------------------------

    def __init__(self, opt, drr_fn, rng=None):
        self.opt = opt
        self.drr_fn = drr_fn                      # injected `generate_drr`
        self.target_shape      = (opt.Nx, opt.Ny, opt.Nz)
        #self.target_shape_temp      = (opt.Nx, opt.Ny, opt.Nz_temp)
        self.target_resolution = (opt.vol_size, opt.vol_size, opt.vol_size)

        self.max_rot_deg  = float(getattr(opt, "aug_max_rot_deg",  5.0))
        self.max_shift_mm = float(getattr(opt, "aug_max_shift_mm", 5.0))
        self.rot_axes     = getattr(opt, "aug_rot_axes", "xyz")
        self.d_angle      = getattr(opt, "d_angle", 2)
        self.fill_value   = getattr(opt, "aug_fill_value", -1000)
        self.interp_order = getattr(opt, "aug_interp_order", 1)

        self.rng = rng if rng is not None else np.random.default_rng(
            getattr(opt, "aug_seed", None)
        )

    # ---------------------------- public API --------------------------------

    def generate(self, scan_root, aug_index=0, is_identity_aug=True, overwrite=False, aug_prefix="aug"):
        """
        Produce `<scan_root>/aug_<aug_index>/` containing ct, DRRs, aug.ini.
        Returns the aug folder Path.
        """
        scan_root = Path(scan_root)
        aug_path  = scan_root / f"{aug_prefix}_{aug_index}"
        if aug_path.exists() and not overwrite:
            return aug_path
        aug_path.mkdir(parents=True, exist_ok=True)
        (aug_path / "img").mkdir(exist_ok=True)

        start_time = time.perf_counter()
        raw_ct_path = self._find_original(scan_root)
        end_time = time.perf_counter()
        print(f"Found original CT at {raw_ct_path} in {end_time - start_time:.2f} seconds")
        
        transform = None if is_identity_aug else self.sample_transform(raw_ct_path)
        
        start_time = time.perf_counter()
        ct = self.refine_resample(raw_ct_path, transform=self.invert_transform(transform))
        end_time = time.perf_counter()
        print(f"Resampled CT in {end_time - start_time:.2f} seconds")
        # ct_temp = self.refine_resample(raw_ct_path, transform=transform)
        # ct = self._center_crop(ct_temp, self.target_shape)
        np.savez_compressed(aug_path / "ct_resampled.npz", data=ct)

        self.drr_fn(
            self.opt,
            #ct=ct_temp,
            ct_file=raw_ct_path,
            output_folder=str(aug_path / "img"),
            output_format="both",
            d_angle=self.d_angle,
            transform=transform,
        )

        self._write_aug_ini(aug_path, transform, is_identity_aug)
        return aug_path

    def sample_transform(self, raw_ct_path=None):
        """4x4 rigid transform (rotation + translation) in physical mm space."""
        rng = self.rng
        angles_rad = {
            ax: np.deg2rad(rng.uniform(-self.max_rot_deg, self.max_rot_deg))
                if ax in self.rot_axes else 0.0
            for ax in "xyz"
        }
        
        shift = rng.uniform(-self.max_shift_mm, self.max_shift_mm, size=3)
        if raw_ct_path is not None:
            ct_nifti = nib.load(raw_ct_path)
            shape = np.array(ct_nifti.shape[:3])
            zooms = np.array(ct_nifti.header.get_zooms()[:3])
            
            ct_extent = shape * zooms
            target_extent = np.array([
                self.opt.vol_size * self.opt.Nx,
                self.opt.vol_size * self.opt.Ny,
                self.opt.vol_size * self.opt.Nz,
            ])
            
            physical_margin = np.maximum((ct_extent - target_extent) / 2.0, 0.0)
            additional_shift = rng.uniform(-physical_margin, physical_margin)
            shift += additional_shift

        cx, sx = np.cos(angles_rad["x"]), np.sin(angles_rad["x"])
        cy, sy = np.cos(angles_rad["y"]), np.sin(angles_rad["y"])
        cz, sz = np.cos(angles_rad["z"]), np.sin(angles_rad["z"])

        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx

        T = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = shift
        return T

    @staticmethod
    def invert_transform(T):
        """Inverse of a 4x4 rigid transform (rotation + translation)."""
        if T is None:
            return None
        R = T[:3, :3]
        t = T[:3, 3]
        T_inv = np.eye(4)
        T_inv[:3, :3] = R.T
        T_inv[:3, 3] = -R.T @ t
        return T_inv
    
    def refine_resample(self, raw_ct_path, transform=None):
        """
        Resample CT to (target_shape, target_resolution); optionally apply a
        4x4 transform in physical (mm) space, pivoting at the volume center.
        """
        ct_nifti  = nib.load(str(raw_ct_path))
        ct_data   = ct_nifti.get_fdata().astype(np.float32)
        # Coerce to 3D: some NIfTIs carry a trailing singleton (e.g.
        # (X,Y,Z,1)) or a genuine 4th dim; the resample logic below assumes 3D
        # (src_res uses get_zooms()[:3]).  Drop singleton dims, take the first
        # volume of any real 4th dim, and fail loudly (naming the file) only if
        # it still isn't 3D.
        if ct_data.ndim != 3:
            ct_data = np.squeeze(ct_data)
            while ct_data.ndim > 3:
                ct_data = ct_data[..., 0]
            if ct_data.ndim != 3:
                raise ValueError(f"Expected a 3D CT but got shape "
                                 f"{ct_nifti.shape} for {raw_ct_path}")
        src_res   = np.asarray(ct_nifti.header.get_zooms()[:3], dtype=np.float64)
        src_shape = np.asarray(ct_data.shape, dtype=np.float64)
        tgt_res   = np.asarray(self.target_resolution, dtype=np.float64)
        tgt_shape = np.asarray(self.target_shape,      dtype=np.float64)

        if transform is None:
            return self._resample_and_center(ct_data, src_res, tgt_shape, tgt_res)

        transform = np.asarray(transform, dtype=np.float64)
        assert transform.shape == (4, 4)
        R     = transform[:3, :3]
        t     = transform[:3,  3]
        R_inv = np.linalg.inv(R)
        t_inv = -R_inv @ t

        S_src_inv = np.diag(1.0 / src_res)
        S_tgt     = np.diag(tgt_res)
        c_src     = (src_shape - 1) / 2.0
        c_tgt     = (tgt_shape - 1) / 2.0

        matrix = S_src_inv @ R_inv @ S_tgt
        offset = S_src_inv @ (R_inv @ (-S_tgt @ c_tgt) + t_inv) + c_src

        out = ndimage.affine_transform(
            ct_data,
            matrix=matrix,
            offset=offset,
            output_shape=tuple(int(s) for s in tgt_shape),
            order=self.interp_order,
            mode="constant",
            cval=self.fill_value,
            prefilter=False,
        )
        return out.astype(np.float32)

    # ---------------------------- internals ---------------------------------

    def _resample_and_center(self, ct_data, src_res, tgt_shape, tgt_res):
        zoom_factors = src_res / tgt_res
        resampled = ndimage.zoom(ct_data, zoom_factors, order=self.interp_order)

        out = np.full(tuple(int(s) for s in tgt_shape),
                      self.fill_value, dtype=np.float32)
        rs = np.asarray(resampled.shape)
        ts = np.asarray(tgt_shape, dtype=int)
        src_start  = np.maximum(0, (rs - ts) // 2)
        dest_start = np.maximum(0, (ts - rs) // 2)
        size       = np.minimum(rs, ts)

        out[
            dest_start[0]:dest_start[0]+size[0],
            dest_start[1]:dest_start[1]+size[1],
            dest_start[2]:dest_start[2]+size[2],
        ] = resampled[
            src_start[0]:src_start[0]+size[0],
            src_start[1]:src_start[1]+size[1],
            src_start[2]:src_start[2]+size[2],
        ]
        return out
    
    def _center_crop(self, ct_data, tgt_shape):
        out = np.full(tuple(int(s) for s in tgt_shape),
                      self.fill_value, dtype=np.float32)
        rs = np.asarray(ct_data.shape)
        ts = np.asarray(tgt_shape, dtype=int)
        src_start  = np.maximum(0, (rs - ts) // 2)
        dest_start = np.maximum(0, (ts - rs) // 2)
        size       = np.minimum(rs, ts)

        out[
            dest_start[0]:dest_start[0]+size[0],
            dest_start[1]:dest_start[1]+size[1],
            dest_start[2]:dest_start[2]+size[2],
        ] = ct_data[
            src_start[0]:src_start[0]+size[0],
            src_start[1]:src_start[1]+size[1],
            src_start[2]:src_start[2]+size[2],
        ]
        return out
    

    @staticmethod
    def _find_original(scan_root):
        originals = [p for p in Path(scan_root).glob("*.nii.gz") if p.is_file()]
        if not originals:
            raise FileNotFoundError(f"No *.nii.gz original under {scan_root}")
        return originals[0]

    def _write_aug_ini(self, aug_path, transform, is_identity_aug):
        T_record = np.eye(4) if transform is None else transform
        cfg = configparser.ConfigParser()
        cfg["ct"] = {
            "dimension":  ",".join(map(str, self.target_shape)),
            "resolution": ",".join(map(str, self.target_resolution)),
        }
        cfg["transform"] = {
            "is_identity": str(is_identity_aug),
            "matrix": ";".join(
                ",".join(f"{v:.10g}" for v in row) for row in T_record
            ),
        }
        cfg["projection"] = {
            "dimension":  f"{getattr(self.opt, 'proj_H', '')},"
                          f"{getattr(self.opt, 'proj_W', '')}",
            "resolution": f"{getattr(self.opt, 'proj_res', '')}",
            "d_angle":    f"{self.d_angle}",
        }
        with open(aug_path / "aug.ini", "w") as f:
            cfg.write(f)


def preprocess_dataset(manifest, opt, aug_force_new=False):
    """
    Complete for each scan fill required augmentation folders, resampled CT, DRRs, and aug.ini.
     - If augmentation folders already exist, they will be overwritten if opt.aug_force_new is True.
     - If augmentation folders already exist and opt.aug_force_new is False, they will be left as-is.
    process and update the manifest and save it on the run in opt.checkpoints_dir/opt.name/manifest.csv
    """
    loader = ScanLoader(max_n_aug=opt.aug_max_aug, aug_generator=CTAugmentor(opt, generate_drr), aug_prefix=opt.aug_prefix)
    manifest_path = os.path.join(opt.checkpoints_dir, opt.name, 'manifest.csv')
    for _, row in manifest.iterrows():
        scan_root = row.scan_root
        print("Starting preprocessing for scan:", scan_root)
        for aug_index in range(opt.aug_max_aug):
            if aug_force_new:
                loader.prepare(scan_root, mode='force_new')
            else:
                loader.prepare(scan_root, mode='continue')
        aug_dirs = sorted(glob(f"{scan_root}/{opt.aug_prefix}_[0-9]"))
        print(f"Finished preprocessing for {scan_root}. Augmentations available: {len(aug_dirs)}")        
        


def preprocess_ct_volumes(opt, force_reprocess=False, reprocess_num=100):
    """
    Scans root_dir, resamples CTs, saves as .npz, and updates manifest.csv.
    """
    root_dir = opt.data_root
    target_size=(opt.Nx, opt.Ny, opt.Nz) 
    target_resolution=(opt.vol_size, opt.vol_size, opt.vol_size)
    
    manifest_path = os.path.join(root_dir, 'manifest.csv')
    #case_ids = [d for d in os.listdir(root_dir) if d.startswith('BDMAP')]
    case_ids = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')]
    if reprocess_num=="all":
        reprocess_num = len(case_ids)
        
    count = 0
    data = []
    for case_id in case_ids:
        case_path = os.path.join(root_dir, case_id)
        
        cbct_list = [d for d in os.listdir(case_path) if d.endswith('.nii.gz')]
        num_cbct = len(cbct_list)
        if num_cbct < 2:
            print(f"Skipping {case_id} because it has only {num_cbct} CBCT(s)")
            continue
        
        processing_wrong = False
        scan_ids = []
        for i, cbct_filename in enumerate(cbct_list):
            scan_id = cbct_filename.replace('.nii.gz', '')
            scan_ids.append(scan_id)
            raw_ct_path = os.path.join(case_path, cbct_filename)
            
            scan_output_dir = os.path.join(case_path, scan_id)
            os.makedirs(scan_output_dir, exist_ok=True)
            
            processed_npz_path = os.path.join(scan_output_dir, 'ct_resampled.npz')
            drr_check_path = os.path.join(scan_output_dir, 'img', 'aug_0','proj_358.00.png')
            
        
            if force_reprocess:
                is_processed = False
            else:
                is_processed = os.path.exists(processed_npz_path) and os.path.exists(drr_check_path)
        
            if (not is_processed) and count < reprocess_num:
                # 1. Generate Resampled CT
                try:
                    # Load and Resample
                    ct_resampled = refine_resample(raw_ct_path, target_shape=target_size, target_resolution=target_resolution)
                    
                    # Save as compressed numpy
                    np.savez_compressed(processed_npz_path, data=ct_resampled)
                    is_processed1 = True
                    print(f"Processed {case_id}: saved resampled CT")
                    
                except Exception as e:
                    print(f"Error processing {case_id}: {e}")
                    is_processed1 = False
                
                # Generate DRRs    
                try:
                    generate_drr(raw_ct_path, opt, output_folder= scan_id + '/img/', output_format='both')
                    print(f"Processed {case_id}: saved DRRs")
                    is_processed2 = True
                except Exception as e:
                    print(f"Error generating DRRs for {case_id}: {e}")
                    is_processed2 = False
                
                is_processed = is_processed1 and is_processed2   
                if is_processed:
                    print(f"Finished processing {case_id} - {(i+1)}/{num_cbct} CBCTs")
                    count += 1
                else:
                    processing_wrong = True
                    print(f"Processing failed for {case_id}. Skipping.")
            elif is_processed:
                print(f"{case_id} already processed. Skipping.")
                
        if processing_wrong is False:
            print(f"Finished processing {case_id} ({count}/{reprocess_num})")
        
                
        
        data.append({
            'case_id': case_id,
            'scan_ids': scan_ids,
            "num_cbct": num_cbct,
            'is_processed': is_processed
        })

    df = pd.DataFrame(data)
    df.to_csv(manifest_path, index=False)
    print(f"Manifest saved to {manifest_path}")
    return df

def refine_resample(raw_ct_path, target_shape=(256, 256, 256), target_resolution=(500/256, 500/256, 500/256)):
    # 1. Load data and extract physical metadata
    ct_nifti = nib.load(raw_ct_path)
    ct_data = ct_nifti.get_fdata().astype(np.float32)
    current_resolution = ct_nifti.header.get_zooms()[:3]
    
    # 2. Calculate zoom factors based on PHYSICAL resolution
    # Formula: zoom = current_spacing / target_spacing
    zoom_factors = [c / t for c, t in zip(current_resolution, target_resolution)]
    
    # 3. Resample the entire volume first
    # This keeps the voxel proportions correct relative to real-world MM
    ct_resampled = ndimage.zoom(ct_data, zoom_factors, order=1)
    
    # 4. Center-Crop or Center-Pad to reach the exact target_shape
    # This ensures the "Isocenter" (volume center) remains the anchor
    final_data = np.zeros(target_shape, dtype=np.float32)-1000
    
    resampled_shape = ct_resampled.shape
    
    # Calculate start/end indices for the source (resampled) and destination (final)
    # This logic handles both shrinking and growing
    src_start = [max(0, (resampled_shape[i] - target_shape[i]) // 2) for i in range(3)]
    dest_start = [max(0, (target_shape[i] - resampled_shape[i]) // 2) for i in range(3)]
    
    size = [min(resampled_shape[i], target_shape[i]) for i in range(3)]
    
    final_data[
        dest_start[0]:dest_start[0]+size[0],
        dest_start[1]:dest_start[1]+size[1],
        dest_start[2]:dest_start[2]+size[2]
    ] = ct_resampled[
        src_start[0]:src_start[0]+size[0],
        src_start[1]:src_start[1]+size[1],
        src_start[2]:src_start[2]+size[2]
    ]
    
    return final_data


class CTProjectionDataset(Dataset):
    def __init__(self, opt, mode='only_processed', reprocess_num=5, max_samples=None):
        """
        Args:
            mode: 'only_processed' (skip missing), 
                  or 'reprocess' 
                  or 'continue' continue to process missing ones without deleting existing processed files
        """
        
        self.root_dir = opt.data_root
        self.manifest_path = os.path.join(self.root_dir, 'manifest.csv')

        # 1. Check/Generate Manifest
        if not os.path.exists(self.manifest_path):
            print("Manifest not found. Running preprocessing...")
            self.df = preprocess_ct_volumes(opt, reprocess_num=1)
        
        if mode=='only_processed':
            self.df = pd.read_csv(self.manifest_path)
            #self.df['is_processed'] = self.df['processed_path'].apply(os.path.exists)
        elif mode == 'reprocess':
            self.df = preprocess_ct_volumes(opt, force_reprocess=True, reprocess_num=reprocess_num)
        elif mode == 'continue':
            self.df = preprocess_ct_volumes(opt, reprocess_num=reprocess_num)
                

        # Only keep processed samples
        self.df = self.df[self.df['is_processed'] == True]
        print(f"Total processed samples available: {len(self.df)}")
        
        if max_samples:
            max_samples = min(max_samples, len(self.df))
            self.df = self.df.head(max_samples)

        # 3. Flatten samples (Case + Augmentation)
        self.samples = []
        for _, row in self.df.iterrows():
            img_root = os.path.join(self.root_dir, row['case_id'], 'img')
            if os.path.exists(img_root):
                aug_list = os.listdir(img_root)
                aug = aug_list[0]
                self.samples.append({
                    'case_id': row['case_id'],
                    'aug': aug,
                    'ct_path': row['processed_path'] #npz file path
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        
        # Load CT efficiently
        print(f"Loading {s['ct_path']}")
        ct_data = np.load(s['ct_path'])['data']
        
        ct_data = (ct_data - ct_data.min())/1000
        # Load Projections (logic remains same as previous step)
        aug_path = os.path.join(self.root_dir, s['case_id'], 'img', s['aug'])
        return {"ct": torch.from_numpy(ct_data).unsqueeze(0), "aug_path":[aug_path], "case_id": s['case_id']}


class ImageDataset(Dataset):
    """
    Each item is a pair of aug folders (scan_a, scan_b) from the same case.
    Resolves the intersection of available angles at init so the collate fn
    can always load angle-corresponded images from both scans.
    """
    def __init__(self, aug_paths, img_type="png"):
        """
        Args:
            paired_aug_paths: list of (folder_a, folder_b) tuples,
                              e.g. from PairedCTProjectionDataset aug_path fields.
        """
        self.type = img_type
        self.items = []
        for folder_a in aug_paths:
            angles_a = self._angle_map(folder_a, img_type)
            if len(angles_a) == 0:
                print(f"Warning: no angles in {folder_a}, skipping.")
                continue
            self.items.append({
                "folder_a": folder_a,
                "angle_to_file_a": angles_a,
                "angles": sorted(angles_a.keys()),
                "type": img_type,
            })

    @staticmethod
    def _angle_map(folder, img_type):
        """Returns {angle_float: filename} for all matching files in folder."""
        mapping = {}
        for f in os.listdir(folder):
            if f.endswith(f".{img_type}"):
                try:
                    angle = float(f.split('_')[1].split(f'.{img_type}')[0])
                    mapping[angle] = f
                except (IndexError, ValueError):
                    pass
        return mapping

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

def parse_fixed_angles(spec):
    """Parse an --extra_angles spec ("45,135,225,315", list, or '') -> [float]."""
    if not spec:
        return []
    if isinstance(spec, (list, tuple, np.ndarray)):
        vals = [float(a) for a in spec]
    else:
        vals = [float(a) for a in str(spec).split(',') if a.strip()]
    # de-dup targets while preserving order; snapping keeps this count constant
    out = []
    for v in vals:
        v = v % 360.0
        if v not in out:
            out.append(v)
    return out


def snap_fixed_angles(angles_avail, targets):
    """Snap each target angle to the closest available projection angle.

    Projections live on a discrete grid (e.g. every 2 deg: 0,2,...,358), so a
    requested 45 deg snaps to its nearest on-grid neighbour (circular distance,
    ties -> smaller angle). Returns one angle per target (no de-dup) so the
    view count stays constant across items in a batch.
    """
    return [
        min(angles_avail,
            key=lambda a: min(abs(a - t), 360.0 - abs(a - t)))
        for t in targets
    ]


def _arc_binning_select(angles, N, min_num, arc_deg):
    """Binning within an arc of arc_deg degrees (use 360 for full circle)."""
    num = len(angles)
    arc_len = int(num * arc_deg / 360) if arc_deg < 360 else num
    start_idx = np.random.randint(0, num)
    base_count = min(min_num, N)
    boundaries = np.linspace(0, arc_len, 2 * min_num + 1).astype(int)

    # Work in arc-relative offsets so sorting gives arc order even when wrapping past 0
    chosen_offsets = set()
    offset = random.randint(0, 1)
    for i in range(base_count):
        bin_idx = 2 * i + offset
        lo, hi = int(boundaries[bin_idx]), int(boundaries[bin_idx + 1])
        if lo < hi:
            chosen_offsets.add(random.randrange(lo, hi))

    remaining = N - len(chosen_offsets)
    if remaining > 0:
        pool = [i for i in range(arc_len) if i not in chosen_offsets]
        chosen_offsets.update(random.sample(pool, min(remaining, len(pool))))

    # Sort offsets in arc order, then map to global angle indices (handles wrap-around)
    return [angles[(start_idx + off) % num] for off in sorted(chosen_offsets)]


def choose_angles(select_mode, angles, N, min_num):
    # --- angle selection (identical logic to batch_collate) ---
    num = len(angles)
    # A few aug folders render marginally fewer views than the largest test
    # config (e.g. 99 available vs N=100 requested on the 211deg/2deg grid).
    # Cap instead of asserting so those cases stay in the test set; the
    # warning makes the (~1% fewer views) substitution auditable.
    if N > num or num < min_num:
        print(f"[choose_angles] warning: requested N={N} (min_num={min_num}) "
              f"but only {num} angles available; using {min(N, num)}")
        N = min(N, num)
    assert num >= 1, "No angles available to select from"

    # Support comma-separated modes with equal probability
    modes = [m.strip() for m in select_mode.split(",") if m.strip()]
    mode = random.choice(modes) # equal probability if multiple modes specified

    if mode == "random":
        chosen = sorted(random.sample(angles, min(N, num)))
    elif mode == "uniform":
        start_idx = np.random.randint(0, num)
        indices = [(start_idx + int(i)) % num
                    for i in np.linspace(0, num - 1, num=N, endpoint=False)]
        chosen = [angles[i] for i in indices]
    elif mode == "shortarc":
        chosen = _arc_binning_select(angles, N, min_num, arc_deg=90)
    elif mode == "shortarc120":
        chosen = _arc_binning_select(angles, N, min_num, arc_deg=120)
    elif mode == "halcyon":
        chosen = _arc_binning_select(angles, N, min_num, arc_deg=211)
    elif mode == "truebeam":
        chosen = _arc_binning_select(angles, N, min_num, arc_deg=200)
    elif mode == "binning":
        chosen = _arc_binning_select(angles, N, min_num, arc_deg=360)
    elif mode == "2orthogonal":
        # Always exactly 2 views, 90 deg apart (N is ignored). The first view
        # starts anywhere on the circle; the second snaps to the on-grid angle
        # closest to start+90 (circular distance, excluding the start itself).
        assert num >= 2, "2orthogonal needs at least 2 available angles"
        start = angles[np.random.randint(0, num)]
        target = (start + 90.0) % 360.0
        partner = min(
            (a for a in angles if a != start),
            key=lambda a: min(abs(a % 360.0 - target),
                              360.0 - abs(a % 360.0 - target)))
        chosen = sorted([start, partner])
    else:  # sequential
        start_idx = np.random.randint(0, num)
        indices = [(start_idx + i) % num for i in range(N)]
        chosen = [angles[i] for i in indices]

    return chosen


def sample_num_images(min_num, max_num, sampling="uniform"):
    """Sample the number of views N for one batch.

    "uniform"    : random.randint(min_num, max_num) — original behavior.
    "loguniform" : N = floor(exp(U(log min, log(max+1)))) — equal probability
                   mass per octave instead of per count.  With a wide range
                   (e.g. 10–500) plain uniform sampling almost never draws
                   sparse counts (P(N<=50) ~ 8%), so the model forgets the
                   sparse-view regime; log-uniform keeps every scale of N
                   equally represented (10–20 is as likely as 250–500).
    """
    if max_num <= min_num:
        return min_num
    if sampling == "loguniform":
        lo = math.log(min_num)
        hi = math.log(max_num + 1)
        return min(int(math.exp(random.uniform(lo, hi))), max_num)
    return random.randint(min_num, max_num)


def single_batch_collate(
    batch,
    select_mode="uniform",
    H=224,
    W=224,
    resw=0.5,
    resh=0.5,
    psize=1.0,
    min_num_images=5,
    max_num_images=10,
    add_noise=True,
    blade_blocking=False,
    blade_max_frac=0.15,
    blade_prob=0.5,
    num_images_sampling="uniform",
    extra_angles="",
):
    N = sample_num_images(min_num_images, max_num_images, num_images_sampling)
    # Fixed angles (e.g. "45,135,225,315") are ALWAYS loaded on top of the N
    # sampled views, snapped per item to the closest on-grid projection.
    extra_targets = parse_fixed_angles(extra_angles)
    extras_per_item = [
        snap_fixed_angles(item["angles"], extra_targets) if extra_targets else []
        for item in batch
    ]
    # Never request more views than the sparsest item in the batch provides
    # (matters when max_num_images exceeds an aug folder's projection count).
    # Fixed angles are removed from the sampling pool, so cap N accordingly.
    N = min(N, min(len(item["angles"]) - len(ex)
                   for item, ex in zip(batch, extras_per_item)))
    zoom = [resh / psize, resw / psize]

    # Sample blade widths once for the whole batch so all images share the same blocking.
    top_w = bot_w = 0
    if blade_blocking and random.random() < blade_prob:
        top_w, bot_w = _sample_blade_widths(H, blade_max_frac)

    proj_a_batch = []
    angles_batch, plucker_rays_batch = [], []

    for item, extras in zip(batch, extras_per_item):
        folder_a = item["folder_a"]
        angles = item["angles"]
        angle_to_file_a = item["angle_to_file_a"]
        img_type = item["type"]

        if extras:
            _ex = set(extras)
            pool = [a for a in angles if a not in _ex]
        else:
            pool = angles
        chosen = choose_angles(select_mode, pool, N, min_num_images) + extras

        projs_a, angles_a = [], []
        for angle in chosen:
            img_a = _load_proj(folder_a, angle_to_file_a[angle], img_type, zoom, H, W, add_noise)
            if top_w or bot_w:
                img_a = _apply_blade_widths(img_a, top_w, bot_w)

            projs_a.append(img_a)
            angles_a.append(angle)

        proj_a_batch.append(np.stack(projs_a))
        angles_batch.append(np.array(angles_a))

    projections = torch.tensor(np.stack(proj_a_batch)).float()
    max_vals = torch.amax(projections, dim=(1, 2, 3), keepdim=True)
    projections /= torch.clamp(max_vals, min=1e-9)
    
    return {
        "projections": projections,   # B (N + n_extra) H W
        "angles": torch.tensor(np.stack(angles_batch)).float(),          # B (N + n_extra)
        # actual per-item view count (modes like 2orthogonal ignore the sampled N)
        "N": len(angles_batch[0]),
    }



def create_single_img_dataloader(aug_paths, opt, num_workers=0):
    dataset = ImageDataset(aug_paths, img_type=getattr(opt, 'img_type', 'npy'))

    custom_collate = partial(
        single_batch_collate,
        select_mode=opt.angle_select_mode,
        H=opt.H,
        W=opt.W,
        resw=opt.resw,
        resh=opt.resh,
        psize=opt.psize,
        min_num_images=opt.min_num_images,
        max_num_images=opt.max_num_images,
        num_images_sampling=getattr(opt, 'num_images_sampling', 'uniform'),
        #add_noise=not opt.no_add_noise,
        add_noise=False,
        blade_blocking=getattr(opt, 'blade_blocking', False),
        blade_max_frac=getattr(opt, 'blade_max_frac', 0.15),
        blade_prob=getattr(opt, 'blade_prob', 0.5),
        extra_angles=getattr(opt, 'extra_angles', ''),
    )

    return DataLoader(
        dataset,
        batch_size=len(aug_paths),
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=num_workers,
    )



def _sample_blade_widths(H: int, max_frac: float) -> tuple[int, int]:
    """Sample top and bottom band heights for one blade-blocking event."""
    max_w = max(1, int(H * max_frac))
    return random.randint(0, max_w), random.randint(0, max_w)


def _apply_blade_widths(img: np.ndarray, top_w: int, bot_w: int) -> np.ndarray:
    """Zero out pre-sampled top/bottom bands in-place."""
    if top_w:
        img[:top_w, :] = 0.0
    if bot_w:
        img[img.shape[0] - bot_w:, :] = 0.0
    return img


def _load_proj(folder, filename, img_type, zoom, H, W, add_noise):
    if img_type == "npy":
        img = np.load(os.path.join(folder, filename))
    else:
        img = np.array(Image.open(os.path.join(folder, filename)).convert("L"))

    #img = img / np.max(img)

    img = ndimage.zoom(img, zoom, order=1)
    h, w = img.shape
    img = img[(h - H) // 2:(h - H) // 2 + H, (w - W) // 2:(w - W) // 2 + W]
    if add_noise:
        img = add_speckle_noise(img)
    return img


# ─── Per-case projection prefetch (load once per case, slice per inner draw) ───
#
# Replaces the old per-inner-step `next(iter(create_single_img_dataloader(...)))`,
# which re-opened N projection files from disk on EVERY draw (num_batch_img times
# per case). Cold NFS opens dominated the measured "forward time". These helpers:
#   1. plan_case_draws  — MAIN thread: draw N / blade / chosen-angles for all the
#      inner steps of a case, seeded per-step so N is identical across DDP ranks
#      (the collate's old `random.seed(_step_seed)` contract, made explicit).
#   2. load_case_views  — WORKER thread (pure I/O, no RNG): read the UNION of the
#      chosen views once. One open per view per case instead of per draw.
#   3. build_draw       — MAIN thread: slice one draw's views from the in-RAM
#      cache and reproduce the collate's blade + per-volume max normalisation.

def plan_case_draws(angle_to_file, opt, base_iter):
    """Plan the opt.num_batch_img inner draws for one case (B=1).

    base_iter is the training total_iters BEFORE this case's inner loop (identical
    on every rank), so N is DDP-synced. Reproduces single_batch_collate's RNG
    order (N -> blade -> choose_angles) by seeding the global RNGs exactly as the
    old inner loop did. Returns (plans, needed_angles).
    """
    angles_avail = sorted(angle_to_file.keys())
    sampling = getattr(opt, 'num_images_sampling', 'uniform')
    blade_on = getattr(opt, 'blade_blocking', False)
    blade_prob = getattr(opt, 'blade_prob', 0.5)
    blade_max_frac = getattr(opt, 'blade_max_frac', 0.15)
    # Fixed angles are always loaded on top of the N sampled views. They are
    # snapped deterministically (no RNG) so the DDP seeding contract holds.
    extra_targets = parse_fixed_angles(getattr(opt, 'extra_angles', ''))
    extras = snap_fixed_angles(angles_avail, extra_targets) if extra_targets else []
    if extras:
        _ex = set(extras)
        pool = [a for a in angles_avail if a not in _ex]
    else:
        pool = angles_avail
    plans, needed = [], set()
    for j in range(opt.num_batch_img):
        step_seed = 1234567 + base_iter + (j + 1) * opt.batch_size
        random.seed(step_seed)
        np.random.seed(step_seed % (2 ** 31 - 1))
        N = sample_num_images(opt.min_num_images, opt.max_num_images, sampling)
        N = min(N, len(pool))
        top_w = bot_w = 0
        if blade_on and random.random() < blade_prob:
            top_w, bot_w = _sample_blade_widths(opt.H, blade_max_frac)
        chosen = choose_angles(opt.angle_select_mode, pool, N,
                               opt.min_num_images) + extras
        plans.append({'N': len(chosen), 'chosen': chosen,
                      'top_w': top_w, 'bot_w': bot_w})
        needed.update(chosen)
    return plans, needed


def load_case_views(folder, angle_to_file, needed_angles, opt):
    """Read the needed projection views for one case into a {angle: ndarray} dict.
    Pure I/O + resize (no RNG) so it is safe to run in a prefetch thread.

    The per-view files are read in PARALLEL: each is a separate cold NFS open
    (latency-bound), so N serial opens dominated the prefetch wait. np.load
    releases the GIL during I/O, so a small thread pool overlaps the round-trips
    (N opens W-wide ~= N/W x latency). --prefetch_view_workers sets W."""
    zoom = [opt.resh / opt.psize, opt.resw / opt.psize]
    img_type = getattr(opt, 'img_type', 'npy')
    needed = list(needed_angles)

    def _one(a):
        return a, _load_proj(folder, angle_to_file[a], img_type, zoom,
                             opt.H, opt.W, add_noise=False)

    workers = int(getattr(opt, 'prefetch_view_workers', 8))
    if workers > 1 and len(needed) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(workers, len(needed))) as _ex:
            return dict(_ex.map(_one, needed))
    return dict(_one(a) for a in needed)


def build_draw(cache, plan, device=None):
    """Assemble one draw's image_data (B=1) from the in-RAM view cache, matching
    single_batch_collate's output (blade bands + per-volume max normalisation)."""
    projs = []
    for a in plan['chosen']:
        img = cache[a]
        if plan['top_w'] or plan['bot_w']:
            img = _apply_blade_widths(img.copy(), plan['top_w'], plan['bot_w'])
        projs.append(img)
    projections = torch.tensor(np.stack(projs)).float().unsqueeze(0)   # (1, N, H, W)
    max_vals = torch.amax(projections, dim=(1, 2, 3), keepdim=True)
    projections = projections / torch.clamp(max_vals, min=1e-9)
    angles = torch.tensor(np.array(plan['chosen'], dtype=np.float32)).unsqueeze(0)
    if device is not None:
        projections = projections.to(device, non_blocking=True)
    return {"projections": projections, "angles": angles, "N": plan['N']}


def build_angle_map(folder, opt):
    """{angle: filename} for a case's projection folder (one listdir per case)."""
    return ImageDataset._angle_map(folder, getattr(opt, 'img_type', 'npy'))


def add_projection_noise(img, poisson_scale=30, gaussian_std=0.01):
    """
    Add Poisson (quantum noise) + Gaussian (electronic/scatter noise)
    """
    # Poisson noise (simulate photon counting)
    scaled = img * poisson_scale
    poisson = np.random.poisson(scaled) / poisson_scale

    # Gaussian noise (electronics + scatter)
    gaussian = poisson + np.random.normal(
        loc=0.0,
        scale=gaussian_std,
        size=poisson.shape
    )

    return np.clip(gaussian, 0.0, 1.0)

def add_speckle_noise(img, var= 0.001):
    """
    Adds speckle noise to an image.
    img: Input image (any scale, but usually float for math)
    sigma: Variance of the noise. Higher = more grain.
    """
    sigma = var ** 0.5
    # Generate noise with mean 0 and standard deviation sigma
    # The shape must match the input image
    noise = np.random.normal(loc=0.0, scale=sigma, size=img.shape)
    
    # Multiplicative step: Noise is scaled by the pixel intensity
    noisy_img = img + (img * noise)
    
    # Ensure values remain within physical limits (e.g., 0 to max)
    # If using 0-255, change 1.0 to 255
    return np.clip(noisy_img, 0, np.max(img))

def normalize_min_max(
        array: np.ndarray,
        min_val: float = 0.0,
        max_val: float = 1.0,
        normalize_dims: int = 2,
        eps: float = 1e-8) -> np.ndarray:
        
    if normalize_dims < 1 or normalize_dims > array.ndim:
        raise ValueError(f"normalize_dims must be between 1 and {array.ndim}, got {normalize_dims}")
    
    # Determine which dimensions to reduce over (the last normalize_dims dimensions)
    reduce_dims = tuple(range(array.ndim - normalize_dims, array.ndim))
    
    # Compute min and max. keepdims=True is essential for broadcasting back to the original shape
    array_min = np.amin(array, axis=reduce_dims, keepdims=True)
    array_max = np.amax(array, axis=reduce_dims, keepdims=True)
    
    # Normalize to [0, 1]
    array_range = array_max - array_min
    normalized = (array - array_min) / (array_range + eps)
    
    # Scale to [min_val, max_val]
    scaled = normalized * (max_val - min_val) + min_val
    
    return scaled


def batch_collate(
    batch,
    select_mode="uniform", # "random", "uniform", "sequential"
    H=224,
    W=224,
    resw=0.5,
    resh=0.5,
    psize=1.0,
    min_num_images=5,
    max_num_images=10,
    add_noise=True,
    blade_blocking=False,
    blade_max_frac=0.15,
    blade_prob=0.5,
    num_images_sampling="uniform",
):

    B = len(batch)
    # sample N once for whole batch
    N = sample_num_images(min_num_images, max_num_images, num_images_sampling)
    N = min(N, min(len(item["files"]) for item in batch))
    projections_batch = []
    angles_batch = []
    plucker_rays_batch = []

    zoom = [resh/psize, resw/psize]

    # Sample blade widths once for the whole batch so all images share the same blocking.
    top_w = bot_w = 0
    if blade_blocking and random.random() < blade_prob:
        top_w, bot_w = _sample_blade_widths(H, blade_max_frac)

    for item in batch:
        folder = item["folder"]
        files = item["files"]
        img_type = item["type"]

        mode = select_mode
        

        # Pre-extract angles to help with sorting and uniform selection
        # This creates a list of (filename, angle) tuples
        file_angle_pairs = []
        for f in files:
            angle = float(f.split('_')[1].split(f'.{img_type}')[0])
            file_angle_pairs.append((f, angle))
        num_files = len(file_angle_pairs)
        
        # 1. Random Selection (Your current logic)
        if mode == "random":
            selected = np.random.choice(
                files, 
                size=min(N, len(files)), 
                replace=False
            )
        # 2. Uniform Selection (with random starting offset)
        if mode == "uniform":
            # Pick a random starting index
            start_idx = np.random.randint(0, num_files)
            # Generate evenly spaced offsets and apply modulo to wrap around the list
            indices = [(start_idx + int(i)) % num_files for i in np.linspace(0, num_files - 1, num=N, endpoint=False)]
            selected = [file_angle_pairs[i][0] for i in indices]

        # 3. Sequential Selection (Circular with Delta)
        elif mode == "sequential":
            # Start anywhere in the circle
            start_idx = np.random.randint(0, num_files)
            delta = 1  
            # Take N steps of size 'delta', wrapping around the end to the head
            indices = [(start_idx + i * delta) % num_files for i in range(N)]
            selected = [file_angle_pairs[i][0] for i in indices]
        
        
        projections = []
        angles = []
        plucker_rays = []
        for f in selected:
            
            angle = float(f.split('_')[1].split(f'.{img_type}')[0])
            if img_type == "npy":
                img = np.load(os.path.join(folder, f))
            elif img_type == "png":
                img = Image.open(os.path.join(folder, f)).convert("L")
                img = np.array(img)
            
            img = img / np.max(img)
            
            img = ndimage.zoom(img, zoom, order=1)

            h, w = img.shape
            top = (h - H)//2
            left = (w - W)//2
            img = img[top:top+H, left:left+W]

            #img = normalize_min_max(img, min_val=0.0, max_val=1.0, normalize_dims=2)
            if top_w or bot_w:
                img = _apply_blade_widths(img, top_w, bot_w)
            if add_noise:
                img = add_speckle_noise(img)

            # prepare plucker rays for this projections (patch level, so that it can be concatenated with ViT patch features)
            plucker_ray = map_2d_plucker_ray_directions(H/16, W/16, psize*16, angle) # plucker_ray is at the patch level, so we multiply psize by 16 to get the effective pixel size at the patch level
            
            plucker_rays.append(plucker_ray)
            projections.append(img)
            angles.append(angle)

        projections_batch.append(np.stack(projections))
        angles_batch.append(np.array(angles))
        plucker_rays_batch.append(np.stack(plucker_rays))

    return {
        "projections": torch.tensor(np.stack(projections_batch)).float(),  # B N H W
        "angles": torch.tensor(np.stack(angles_batch)).float(),
        "plucker_rays": torch.tensor(np.stack(plucker_rays_batch)).float(),  # B N 6 H/16 W/16
        "N": N
    }

def create_img_dataloader(img_roots, opt, num_workers=8):
    dataset = ImageDataset(img_roots)

    custom_collate = partial(
        batch_collate,
        select_mode=opt.angle_select_mode,
        H=opt.H,
        W=opt.W,
        resw=opt.resw,
        resh=opt.resh,
        psize=opt.psize,
        min_num_images=opt.min_num_images,
        max_num_images=opt.max_num_images,
        num_images_sampling=getattr(opt, 'num_images_sampling', 'uniform'),
        add_noise=not opt.no_add_noise,
        blade_blocking=getattr(opt, 'blade_blocking', False),
        blade_max_frac=getattr(opt, 'blade_max_frac', 0.15),
        blade_prob=getattr(opt, 'blade_prob', 0.5),
    )
    
    loader = DataLoader(
        dataset,
        batch_size=len(img_roots),   # B
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=num_workers
        )
    return loader


def _safe_npz_load(path, key='data'):
    """Load npz with copy-on-read to avoid SIGBUS from mmap on truncated files."""
    try:
        with np.load(path, mmap_mode=None) as f:  # mmap_mode=None forces full read
            return f[key].copy()
    except Exception as e:
        raise RuntimeError(f"Failed to load {path}: {e}") from e