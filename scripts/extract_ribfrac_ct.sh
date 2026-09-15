#!/usr/bin/env bash
#
# Extract RibFrac CT scans and arrange them as
#   <out>/<case id>/<fraction id>/ct.nii.gz     (same layout as AMOS/LUNA16/AbdomenAtlas/TotalSegmentator)
#
# RibFrac images are already NIfTI (RibFracNNN-image.nii.gz), so this is
# extract-and-regroup, no format conversion.
#
# The dataset ships as separate image and label zips:
#   ribfrac-train-images-1.zip / -2.zip, ribfrac-val-images.zip, ribfrac-test-images.zip  <- CT, processed
#   ribfrac-*-labels*.zip                                                                 <- masks, SKIPPED
# We only glob *-images*.zip so the label archives are never touched. As a
# second guard, any path containing "label" is skipped too.
#
# Case id (RibFracNNN) comes from the FILENAME (e.g. RibFrac421-image.nii.gz),
# not the parent dir (train images sit in Part1/ Part2/ subfolders). RibFrac
# numbers are globally unique across splits (train 1-420, val 421-500,
# test 501-660), so train/val/test flatten into one pool without collision.
#
# MULTI-CORE + resumable: each image zip is processed in its own xargs -P
# worker with a private temp dir; a case already converted is skipped without
# re-extracting. Default parallelism min(nproc,8); override via 3rd arg or JOBS=N.
#
# Usage:
#   bash scripts/extract_ribfrac_ct.sh [SRC_DIR] [OUT_DIR] [JOBS]
#   SRC_DIR  folder containing the RibFrac zips   (default: RibFrac)
#   OUT_DIR  output folder                        (default: RibFrac_ct)
#   JOBS     zips processed in parallel           (default: min(nproc,8))

set -euo pipefail

SRC="${1:-RibFrac}"
OUT="${2:-RibFrac_ct}"
OUT="${OUT%/}"

def_jobs() { local n; n="$(nproc 2>/dev/null || echo 4)"; [ "$n" -gt 8 ] && n=8; echo "$n"; }
JOBS="${3:-${JOBS:-$(def_jobs)}}"

mkdir -p "$OUT" "$OUT/.extract_tmp"
trap 'rm -rf "$OUT/.extract_tmp"' EXIT

shopt -s nullglob
zips=("$SRC"/*images*.zip)   # CT image archives only; label zips excluded
if [ ${#zips[@]} -eq 0 ]; then
  echo "No *images*.zip found in '$SRC'." >&2
  exit 1
fi
echo ">> ${#zips[@]} image archive(s) found, using up to $JOBS parallel job(s)"

case_id_from_path() {  # arg: path -> echoes RibFracNNN or ""
  basename "$1" | grep -oE 'RibFrac[0-9]+' | head -n1
}
export -f case_id_from_path
export OUT

process_one() {
  local z="$1" name entries id all_done e f dest work base
  name="$(basename "$z")"
  echo ">> [$name] checking..."

  mapfile -t entries < <(unzip -Z1 "$z" 2>/dev/null | grep -E '\.nii\.gz$' | grep -viE 'label' || true)
  if [ ${#entries[@]} -eq 0 ]; then
    echo "WARN: [$name] no image .nii.gz found listing archive (corrupt/unexpected layout?)" >&2
    return
  fi
  all_done=1
  for e in "${entries[@]}"; do
    id="$(case_id_from_path "$e")"
    if [ -z "$id" ] || [ ! -f "$OUT/$id/000/ct.nii.gz" ]; then
      all_done=0; break
    fi
  done
  if [ "$all_done" = "1" ]; then
    echo ">> [$name] skip (all ${#entries[@]} cases already converted)"
    return
  fi

  echo ">> [$name] extracting (${#entries[@]} cases)"
  work="$OUT/.extract_tmp/$(basename "$z" .zip)"
  rm -rf "$work"; mkdir -p "$work"
  unzip -q -o "$z" -d "$work"

  shopt -s globstar
  for f in "$work"/**/*.nii.gz; do
    case "$f" in *[Ll]abel*) continue ;; esac      # belt-and-suspenders
    id="$(case_id_from_path "$f")"
    if [ -z "$id" ]; then
      echo "WARN: [$name] can't derive case id from $f, skipping" >&2
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

printf '%s\0' "${zips[@]}" | xargs -0 -P "$JOBS" -I{} bash -c 'process_one "$@"' _ {}

converted="$(find "$OUT" -type f -name ct.nii.gz 2>/dev/null | wc -l)"
echo "------------------------------------------"
echo "Done. total ct.nii.gz present: $converted"
echo "Output: $OUT/<case id>/000/ct.nii.gz"
