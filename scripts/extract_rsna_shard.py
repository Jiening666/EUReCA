#!/usr/bin/env python3
# Extract one shard (chunk file of member names) from the RSNA2023 zip, using
# multiple processes. Called by scripts/extract_rsna_array.slurm, one instance
# per SLURM array task.
#
# All paths are absolute -- do NOT rely on the process's current working
# directory (SLURM array tasks / manual test runs can start from anywhere,
# e.g. $HOME instead of the dataset dir, which is what caused the earlier
# "FileNotFoundError: chunk_000" -- the file lookup was relative to the wrong cwd).
#
# Each worker process opens the zip ONCE (via Pool initializer) and reuses that
# handle for every file it extracts. An earlier version opened a fresh
# zipfile.ZipFile(ZIP_PATH) inside extract_one() -- i.e. once per FILE, not per
# worker. With ~1.5M entries, parsing the central directory isn't free, so
# paying that cost on every single extraction (93538 times for a ~93k shard)
# swamped the actual (fast) per-file extraction and made the job look hung.
#
# Resumable: skips a file if it's already been extracted on disk, so a
# restarted shard doesn't redo completed work.
#
# Usage: python3 extract_rsna_shard.py <chunk_file_abs_path> <nproc>
import os
import sys
import zipfile
from multiprocessing import Pool

ZIP_PATH = "/path/to/RSNA2023/rsna-2023-abdominal-trauma-detection.zip"
OUT_DIR = "/path/to/RSNA2023"

_zf = None  # persistent per-worker-process zip handle


def _init_worker():
    global _zf
    _zf = zipfile.ZipFile(ZIP_PATH)  # parsed once per worker, not once per file


def extract_one(name):
    dest = os.path.join(OUT_DIR, name)
    if os.path.exists(dest):
        return
    _zf.extract(name, OUT_DIR)


def main():
    chunk_file, nproc = sys.argv[1], int(sys.argv[2])
    with open(chunk_file) as f:
        names = [line.rstrip("\n") for line in f if line.strip()]
    print(f"{chunk_file}: {len(names)} files, {nproc} procs", flush=True)
    with Pool(nproc, initializer=_init_worker) as p:
        for i, _ in enumerate(p.imap_unordered(extract_one, names, chunksize=200)):
            if i % 20000 == 0:
                print(f"{chunk_file}: {i}/{len(names)}", flush=True)
    print(f"{chunk_file}: done ({len(names)}/{len(names)})", flush=True)


if __name__ == "__main__":
    main()
