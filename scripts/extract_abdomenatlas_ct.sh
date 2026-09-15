#!/usr/bin/env bash
#
# Extract AbdomenAtlas3.0Mini CT scans and arrange them as
#   <out>/<BDMAP id>/<fraction id>/ct.nii.gz     (same layout as AMOS/LUNA16 sets)
#
# AbdomenAtlas ships CT-only (no MRI to filter). Each archive is named
#   AbdomenAtlas3_images_BDMAP_<first_id>_BDMAP_<last_id>.tar.gz
# and (per the standard AbdomenAtlas layout) contains one folder per case:
#   BDMAP_XXXXXXXX/ct.nii.gz
# We don't hard-assume that nesting though: we recursively find every *.nii.gz
# inside the extracted archive and pull the BDMAP id out of whichever path
# component matches BDMAP_[0-9]+, so odd/nested layouts still work.
#
# Resume support: before extracting an archive, its contents are listed with
# `tar -tzf` (streamed, no disk writes) and the archive is skipped entirely if
# every BDMAP id in it already has a converted ct.nii.gz.
#
# MULTI-CORE: archives are independent (disjoint BDMAP id ranges, separate temp
# dirs), so the WHOLE per-archive job -- resume check AND extraction -- runs
# inside one xargs -P worker per archive. (An earlier version ran the resume
# check for all archives serially before any parallel extraction started;
# `tar -tzf` must decompress the entire gzip stream to list contents since gzip
# isn't seekable, so a large archive could sit silently for minutes during that
# serial pre-scan, looking like a hang. Folding the check into each parallel
# worker fixes that.)
# Default parallelism is min(nproc, 8) -- capped because each parallel worker
# extracts a full archive to disk at once, so raw nproc on a big node can spike
# disk usage. Override with a 3rd argument or JOBS=N env var.
#
# Usage:
#   bash scripts/extract_abdomenatlas_ct.sh [SRC_DIR] [OUT_DIR] [JOBS]
#   SRC_DIR  folder with AbdomenAtlas3_images_*.tar.gz   (default: AbdomenAtlas3.0Mini/image_only)
#   OUT_DIR  output folder                               (default: AbdomenAtlas3.0Mini_ct)
#   JOBS     archives to process in parallel             (default: min(nproc,8))

set -euo pipefail

SRC="${1:-AbdomenAtlas3.0Mini/image_only}"
OUT="${2:-AbdomenAtlas3.0Mini_ct}"
OUT="${OUT%/}"

def_jobs() { local n; n="$(nproc 2>/dev/null || echo 4)"; [ "$n" -gt 8 ] && n=8; echo "$n"; }
JOBS="${3:-${JOBS:-$(def_jobs)}}"

mkdir -p "$OUT" "$OUT/.extract_tmp"
trap 'rm -rf "$OUT/.extract_tmp"' EXIT

shopt -s nullglob
tars=("$SRC"/*.tar.gz)
if [ ${#tars[@]} -eq 0 ]; then
  echo "No .tar.gz files found in '$SRC'." >&2
  exit 1
fi
echo ">> ${#tars[@]} archive(s) found, using up to $JOBS parallel job(s)"

bdmap_id_from_path() {  # arg: path to a .nii.gz -> echoes BDMAP_XXXXXXXX or ""
  # Prefer the file's immediate parent directory (the real per-case folder,
  # e.g. .../BDMAP_00001654/ct.nii.gz). Matching anywhere in the full path
  # instead (previously: first BDMAP_NNN found via `head -n1`) is unsafe: our
  # temp extraction dir is named after the archive, which itself embeds the
  # archive's FIRST case id (e.g. .extract_tmp/AbdomenAtlas3_images_BDMAP_
  # BDMAP_00001625_BDMAP_00001856/...), so `head -n1` grabbed that instead of
  # the real per-file id -- collapsing every case in an archive onto one id.
  local f="$1" parent base
  parent="$(basename "$(dirname "$f")")"
  if [[ "$parent" =~ ^BDMAP_[0-9]+$ ]]; then
    echo "$parent"; return
  fi
  base="$(basename "$f" .nii.gz)"
  if [[ "$base" =~ ^BDMAP_[0-9]+$ ]]; then
    echo "$base"; return
  fi
  # last resort: rightmost BDMAP_NNN token (closest to the file, not an ancestor)
  echo "$f" | grep -oE 'BDMAP_[0-9]+' | tail -n1
}
export -f bdmap_id_from_path
export OUT

# ---- per-archive worker: resume check + extraction, all in one job --------
process_one() {
  local z="$1" name entries id all_done e f dest work
  name="$(basename "$z")"
  echo ">> [$name] checking..."

  mapfile -t entries < <(tar -tzf "$z" 2>/dev/null | grep -E '\.nii\.gz$' || true)
  if [ ${#entries[@]} -eq 0 ]; then
    echo "WARN: [$name] no .nii.gz found listing archive (corrupt?)" >&2
    return
  fi
  all_done=1
  for e in "${entries[@]}"; do
    id="$(bdmap_id_from_path "$e")"
    if [ -z "$id" ] || [ ! -f "$OUT/$id/000/ct.nii.gz" ]; then
      all_done=0; break
    fi
  done
  if [ "$all_done" = "1" ]; then
    echo ">> [$name] skip (all ${#entries[@]} cases already converted)"
    return
  fi

  echo ">> [$name] extracting (${#entries[@]} cases)"
  work="$OUT/.extract_tmp/$(basename "$z" .tar.gz)"
  rm -rf "$work"; mkdir -p "$work"
  tar -xzf "$z" -C "$work"

  shopt -s globstar
  for f in "$work"/**/*.nii.gz; do
    id="$(bdmap_id_from_path "$f")"
    if [ -z "$id" ]; then
      echo "WARN: [$name] can't derive BDMAP id from $f, skipping" >&2
      continue
    fi
    dest="$OUT/$id/000"
    [ -f "$dest/ct.nii.gz" ] && continue
    mkdir -p "$dest"
    mv -f "$f" "$dest/ct.nii.gz"
    echo "keep : [$name] $id"
  done

  rm -rf "$work"
}
export -f process_one

printf '%s\0' "${tars[@]}" | xargs -0 -P "$JOBS" -I{} bash -c 'process_one "$@"' _ {}

converted="$(find "$OUT" -type f -name ct.nii.gz 2>/dev/null | wc -l)"
echo "------------------------------------------"
echo "Done. total ct.nii.gz present: $converted"
echo "Output: $OUT/<BDMAP id>/000/ct.nii.gz"
