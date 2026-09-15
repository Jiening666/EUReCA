"""
Convert RSNA 2023 Abdominal Trauma DICOM scans to compressed NIfTI.

Input layout:
    .../RSNA2023/train_images/
        <patient_id>/<series_id>/*.dcm
        e.g. 205/62534/24.dcm, 25.dcm, ...

Output layout:
    <OUTPUT_ROOT>/<patient_id>/<series_id>/ct.nii.gz

Parallelism:
    Patients are converted in parallel across processes (ProcessPoolExecutor).
    Every patient is independent and the script is resumable (existing
    ct.nii.gz are skipped), so this is a safe fan-out. Each worker is pinned to
    1 SimpleITK thread so NUM_WORKERS processes don't oversubscribe the cores.

    (GPU is deliberately NOT used: the cost here is reading many small .dcm
    files and gzip-compressing the output -- I/O + CPU bound, nothing a GPU
    accelerates.)
"""

import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import SimpleITK as sitk


# -------------------------------------------------------------------
# Configuration  --  EDIT THESE PATHS
# -------------------------------------------------------------------
INPUT_ROOT = r"./RSNA2023_raw/train_images"
OUTPUT_ROOT = r"./RSNA2023"   # <-- change to whatever you want
MIN_SLICES = 10                     # only save series with strictly more than this many files

# Number of patients converted in parallel. Each worker is pinned to 1 SimpleITK
# thread (see _init_worker) so NUM_WORKERS processes don't oversubscribe cores.
# None -> use all available CPUs.
NUM_WORKERS = None


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def find_all_series_in_patient(patient_dir):
    """
    For RSNA, each patient folder contains one or more series sub-folders,
    each of which holds the .dcm files directly.

    Returns a list of (series_folder, series_uid, file_count, file_names).
    """
    reader = sitk.ImageSeriesReader()
    found = []

    # Each direct child of the patient directory is a series folder
    for series_folder in sorted(p for p in patient_dir.iterdir() if p.is_dir()):
        try:
            series_ids = reader.GetGDCMSeriesIDs(str(series_folder))
        except Exception:
            continue

        if not series_ids:
            # Fall back: some RSNA series have inconsistent UIDs across files.
            # In that case, just grab every .dcm in the folder as one series,
            # sorted numerically by filename stem (1.dcm, 2.dcm, ...).
            dcm_files = sorted(
                series_folder.glob("*.dcm"),
                key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
            )
            if dcm_files:
                # Use the folder name as a pseudo series UID
                pseudo_uid = series_folder.name
                found.append((str(series_folder), pseudo_uid, len(dcm_files),
                              [str(f) for f in dcm_files]))
            continue

        for sid in series_ids:
            try:
                fnames = reader.GetGDCMSeriesFileNames(str(series_folder), sid)
            except Exception:
                continue
            if fnames:
                found.append((str(series_folder), sid, len(fnames), list(fnames)))

    return found


def load_and_save_series(file_names, out_path):
    """Load a single DICOM series (given explicit file list) and save it as .nii.gz."""
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(file_names)
    image = reader.Execute()

    arr = sitk.GetArrayFromImage(image)   # int16, in HU after rescale
    arr[arr < -1000] = -1000              # clamp FOV padding and any sub-air values
    clean = sitk.GetImageFromArray(arr)
    clean.CopyInformation(image)
    image = clean

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(out_path)
    writer.SetUseCompression(True)
    writer.Execute(image)

    return {
        "n_files": len(file_names),
        "size": image.GetSize(),
        "spacing": tuple(round(s, 3) for s in image.GetSpacing()),
    }


def process_patient(patient_dir, out_root, min_slices=MIN_SLICES):
    """Find all series under patient_dir and save those with > min_slices files."""
    series_list = find_all_series_in_patient(patient_dir)
    if not series_list:
        raise RuntimeError(f"No DICOM series found anywhere under {patient_dir}")

    patient_id = patient_dir.name
    saved = []
    skipped_small = []

    for series_folder, series_uid, n_files, file_names in series_list:
        if n_files <= min_slices:
            skipped_small.append((series_uid, n_files))
            continue

        # Use the series folder name (RSNA series ID) for the output path,
        # since RSNA's own series IDs are numeric and human-friendly.
        series_id = Path(series_folder).name

        out_path = out_root / patient_id / series_id / "ct.nii.gz"
        if out_path.exists():
            # skip ones already converted, makes the script resumable
            saved.append({
                "series_uid": series_uid,
                "n_files": n_files,
                "size": "(exists)",
                "spacing": "(exists)",
                "out_path": out_path,
            })
            continue

        info = load_and_save_series(file_names, str(out_path))
        saved.append({
            "series_uid": series_uid,
            "n_files": info["n_files"],
            "size": info["size"],
            "spacing": info["spacing"],
            "out_path": out_path,
        })

    return saved, skipped_small, len(series_list)


# -------------------------------------------------------------------
# Parallel worker
# -------------------------------------------------------------------
def _init_worker():
    """Runs once per worker process: pin SimpleITK to a single thread so that
    NUM_WORKERS processes don't each spin up ITK threads and thrash the CPU."""
    try:
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    except Exception:
        pass


def _process_patient_job(patient_dir_str):
    """Worker entry point: process one patient, return a picklable report."""
    patient_dir = Path(patient_dir_str)
    patient_id = patient_dir.name
    try:
        saved, skipped_small, _n_total = process_patient(patient_dir, Path(OUTPUT_ROOT))
        lines = []
        for s in saved:
            uid_disp = str(s["series_uid"])[:24]
            lines.append(
                f"[ok]    {patient_id}  series={uid_disp}  "
                f"files={s['n_files']}  size={s['size']}  spacing={s['spacing']}"
            )
        return {
            "patient_id": patient_id,
            "ok": True,
            "saved": lines,
            "skipped_small": [c for _, c in skipped_small],
            "n_saved": len(saved),
            "error": None,
        }
    except Exception as e:
        return {
            "patient_id": patient_id,
            "ok": False,
            "saved": [],
            "skipped_small": [],
            "n_saved": 0,
            "error": f"{e}\n{traceback.format_exc()}",
        }


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    in_root = Path(INPUT_ROOT)
    out_root = Path(OUTPUT_ROOT)

    if not in_root.is_dir():
        raise RuntimeError(f"INPUT_ROOT does not exist: {in_root}")

    workers = NUM_WORKERS or os.cpu_count() or 1

    print(f"Scanning: {in_root}")
    print(f"Output:   {out_root}")
    print(f"Min slices (exclusive): {MIN_SLICES}")
    print(f"Parallel workers: {workers}\n")

    # RSNA patient folders are numeric (e.g. "205"), not prefixed
    patient_dirs = sorted(
        (p for p in in_root.iterdir() if p.is_dir()),
        key=lambda p: int(p.name) if p.name.isdigit() else p.name,
    )
    total = len(patient_dirs)
    print(f"Found {total} patient folders.\n")

    n_done = n_failed = 0
    n_series_saved = 0

    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
        futures = {
            ex.submit(_process_patient_job, str(p)): p.name for p in patient_dirs
        }
        for i, fut in enumerate(as_completed(futures), start=1):
            r = fut.result()
            patient_id = r["patient_id"]

            if not r["ok"]:
                n_failed += 1
                print(f"[error] {patient_id}: {r['error']}")
            else:
                if not r["saved"]:
                    small_str = ", ".join(str(c) for c in r["skipped_small"])
                    print(f"[skip]  {patient_id}  no series > {MIN_SLICES} slices  "
                          f"(found counts: {small_str})")
                else:
                    for line in r["saved"]:
                        print(line)
                    if r["skipped_small"]:
                        small_str = ", ".join(str(c) for c in r["skipped_small"])
                        print(f"        (skipped small series in {patient_id}: {small_str})")
                n_done += 1
                n_series_saved += r["n_saved"]

            if i % 50 == 0 or i == total:
                print(f"        ...progress {i}/{total} patients "
                      f"(saved={n_series_saved}, failed={n_failed})")

    print(f"\nFinished. patients_processed={n_done}, "
          f"patients_failed={n_failed}, series_saved={n_series_saved}")


if __name__ == "__main__":
    main()
