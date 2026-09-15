#!/usr/bin/env bash
#
# Convert LUNA16 CT scans (.mhd/.raw MetaImage pairs) to nii.gz, arranged as
#   <out>/<series uid>/<fraction id>/ct.nii.gz      (same layout as the AMOS set)
#
# LUNA16 ships 888 CTs in subset0.zip..subset9.zip; each subsetN/ holds pairs
#   <uid>.mhd (header) + <uid>.raw (voxels), named by the DICOM Series UID.
# We use that UID as the patient id, fraction 000, so the output joins directly
# with annotations.csv (its "seriesuid" column == this UID).
#
# Only subset*.zip is processed. seg-lungs-LUNA16.zip ALSO contains .mhd/.raw but
# those are lung MASKS, not CT -- so we never touch it here.
#
# Conversion needs an imaging library (pure bash can't re-encode MetaImage):
#   uses python3 + SimpleITK   ->   pip install SimpleITK
#   (alternatives: `c3d in.mhd -o out.nii.gz` or `plastimatch convert ...`)
#
# To bound disk use, each subset is extracted to a temp dir ON THE OUTPUT
# FILESYSTEM, converted, then deleted before the next subset.
#
# Usage:
#   bash scripts/extract_luna16_ct.sh [SRC_DIR] [OUT_DIR]
#   SRC_DIR  folder with subset0.zip..subset9.zip   (default: LUNA16)
#   OUT_DIR  output folder                          (default: LUNA16_ct)

set -euo pipefail

SRC="${1:-LUNA16}"
OUT="${2:-LUNA16_ct}"
OUT="${OUT%/}"

# ---- check converter ------------------------------------------------------
if ! command -v python3 >/dev/null 2>&1 || ! python3 -c "import SimpleITK" 2>/dev/null; then
  echo "Need python3 + SimpleITK (pip install SimpleITK)." >&2
  echo "Or adapt this script to use 'c3d' / 'plastimatch convert'." >&2
  exit 1
fi

mkdir -p "$OUT"
WORK="$OUT/.extract_tmp"
trap 'rm -rf "$WORK"' EXIT

shopt -s nullglob
zips=("$SRC"/subset*.zip)
if [ ${#zips[@]} -eq 0 ]; then
  echo "No subset*.zip found in '$SRC'." >&2
  exit 1
fi

total=0; skipped=0; failed=0
for z in "${zips[@]}"; do
  name="$(basename "$z")"

  # resume fast: read the zip's index only (no decompression) and skip the whole
  # subset if every .mhd in it already has a converted ct.nii.gz.
  mhd_count=0; need=0
  while IFS= read -r m; do
    mhd_count=$((mhd_count+1))
    uid="$(basename "$m" .mhd)"
    [ -f "$OUT/$uid/000/ct.nii.gz" ] || { need=1; break; }
  done < <(unzip -Z1 "$z" 2>/dev/null | grep -E '\.mhd$' || true)
  if [ "$mhd_count" -gt 0 ] && [ "$need" -eq 0 ]; then
    echo ">> skip $name (all $mhd_count scans already converted)"
    skipped=$((skipped+mhd_count)); continue
  fi

  echo ">> unzip $name"
  rm -rf "$WORK"; mkdir -p "$WORK"
  # don't let a corrupt archive abort the run: extract intact entries, warn, go on
  unzip -q -o "$z" -d "$WORK" \
    || echo "WARN: $name had extraction errors (corrupt? re-download it); converting intact scans only" >&2

  shopt -s globstar
  for mhd in "$WORK"/**/*.mhd; do
    uid="$(basename "$mhd" .mhd)"
    dest="$OUT/$uid/000"
    if [ -f "$dest/ct.nii.gz" ]; then            # resume: already done
      skipped=$((skipped+1)); continue
    fi
    mkdir -p "$dest"
    if python3 - "$mhd" "$dest/ct.nii.gz" <<'PY'
import sys, SimpleITK as sitk
img = sitk.ReadImage(sys.argv[1])          # reads .mhd, pulls in the .raw
sitk.WriteImage(img, sys.argv[2], useCompression=True)
PY
    then
      echo "convert : $uid"
      total=$((total+1))
    else
      echo "FAILED  : $uid (bad/missing .raw?)" >&2
      rm -f "$dest/ct.nii.gz"; failed=$((failed+1))
    fi
  done

  rm -rf "$WORK"          # free space before the next subset
done

echo "------------------------------------------"
echo "Done. converted: $total   already-present: $skipped   failed: $failed"
echo "Output: $OUT/<series uid>/000/ct.nii.gz"
[ "$failed" -gt 0 ] && echo "Some scans failed -- re-download the affected subset(s) and re-run; finished ones are skipped."
