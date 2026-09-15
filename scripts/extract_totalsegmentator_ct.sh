#!/usr/bin/env bash
#
# Extract TotalSegmentator CT scans and arrange them as
#   <out>/<case id>/<fraction id>/ct.nii.gz     (same layout as AMOS/LUNA16/AbdomenAtlas)
#
# TotalSegmentator ships one folder per case:
#   sXXXX/ct.nii.gz
#   sXXXX/segmentations/<organ>.nii.gz   <- organ masks, NOT ct: always dropped
# We match the exact filename "ct.nii.gz" (not a *.nii.gz glob) so the many
# per-organ mask files under segmentations/ are never picked up. The case id
# is taken from the ct.nii.gz file's immediate parent directory (sXXXX),
# matching the fix applied to the AbdomenAtlas script (matching anywhere in
# the full path is unsafe once a temp extraction dir embeds similar-looking
# tokens).
#
# MULTI-CORE: if SRC contains more than one .zip (e.g. dataset shipped in
# parts), each is processed in parallel via xargs -P, one archive per worker,
# with its own temp extraction dir. Default parallelism is min(nproc, 8);
# override with a 3rd argument or JOBS=N env var. Resumable: a case already
# converted (ct.nii.gz present) is skipped without re-extracting.
#
# Usage:
#   bash scripts/extract_totalsegmentator_ct.sh [SRC_DIR] [OUT_DIR] [JOBS]
#   SRC_DIR  folder containing the TotalSegmentator .zip(s)   (default: TotalSegmentator)
#   OUT_DIR  output folder                                    (default: TotalSegmentator_ct)
#   JOBS     archives to process in parallel                  (default: min(nproc,8))

set -euo pipefail

SRC="${1:-TotalSegmentator}"
OUT="${2:-TotalSegmentator_ct}"
OUT="${OUT%/}"

def_jobs() { local n; n="$(nproc 2>/dev/null || echo 4)"; [ "$n" -gt 8 ] && n=8; echo "$n"; }
JOBS="${3:-${JOBS:-$(def_jobs)}}"

mkdir -p "$OUT" "$OUT/.extract_tmp"
trap 'rm -rf "$OUT/.extract_tmp"' EXIT

shopt -s nullglob
zips=("$SRC"/*.zip)
if [ ${#zips[@]} -eq 0 ]; then
  echo "No .zip files found in '$SRC'." >&2
  exit 1
fi
echo ">> ${#zips[@]} archive(s) found, using up to $JOBS parallel job(s)"

case_id_from_path() {  # arg: path to a ct.nii.gz -> echoes sXXXX or ""
  local f="$1" parent
  parent="$(basename "$(dirname "$f")")"
  if [[ "$parent" =~ ^s[0-9]+$ ]]; then
    echo "$parent"; return
  fi
  # fallback: rightmost sNNN token closest to the file, not an ancestor dir
  echo "$f" | grep -oE 's[0-9]{3,}' | tail -n1
}
export -f case_id_from_path
export OUT

process_one() {
  local z="$1" name entries id all_done e f dest work
  name="$(basename "$z")"
  echo ">> [$name] checking..."

  mapfile -t entries < <(unzip -Z1 "$z" 2>/dev/null | grep -E '(^|/)ct\.nii\.gz$' || true)
  if [ ${#entries[@]} -eq 0 ]; then
    echo "WARN: [$name] no ct.nii.gz found listing archive (corrupt, or unexpected layout?)" >&2
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
  for f in "$work"/**/ct.nii.gz; do
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
