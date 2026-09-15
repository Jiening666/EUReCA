#!/usr/bin/env bash
#
# Unzip the AMOS22 downloads, keep only CT scans (drop MRI), and arrange them as
#   <out>/<patient id>/<fraction id>/ct.nii.gz
#
# AMOS files are named amos_XXXX.nii.gz. Patient id = XXXX, fraction id = 000
# (AMOS has one scan per patient, no real fractions).
#
# MODALITY comes from the meta CSV that sits NEXT TO the zips
# (labeled_data_meta_0000_0599.csv). It has no modality column, so we read the
# "Manufacturer's Model Name" column: the MRI scanners are Prisma / SIGNA HDe /
# Ingenia / Achieva; everything else in the labeled set is CT. Any case NOT in
# the CSV (the unlabeled CT-only parts, e.g. ids 5900-6199) is kept as CT.
#
# Note: amos_id 500 appears as both CT and MRI in the CSV (ambiguous); it lands
# in the MRI list and is therefore dropped -- which is fine per the request.
#
# Pure bash (unzip + awk + grep); no Python.
#
# Folder layouts differ between archives (amos22.zip -> imagesTr/imagesVa/imagesTs,
# unlabeled zips -> flat). We just search recursively for *.nii.gz, so it doesn't
# matter. Segmentation/label volumes are skipped by path.
#
# Usage:
#   bash scripts/extract_amos_ct.sh [SRC_DIR] [OUT_DIR]
#   SRC_DIR  folder with the .zip files AND the meta CSV   (default: AMOS22)
#   OUT_DIR  output folder                                 (default: AMOS22_ct)

set -euo pipefail

SRC="${1:-AMOS22}"
OUT="${2:-AMOS22_ct}"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$OUT"

# ---- 0. locate meta CSV (next to the zips) --------------------------------
shopt -s nullglob
CSV="$SRC/labeled_data_meta_0000_0599.csv"
if [ ! -f "$CSV" ]; then
  cands=("$SRC"/labeled_data_meta*.csv)
  [ ${#cands[@]} -gt 0 ] && CSV="${cands[0]}"
fi
if [ ! -f "$CSV" ]; then
  echo "Meta CSV not found in '$SRC' (labeled_data_meta_0000_0599.csv)." >&2
  exit 1
fi
echo ">> meta CSV: $CSV"

# ---- 1. build MRI id list from CSV (by scanner model) ---------------------
# Columns: amos_id, BirthDate, Sex, Age, ModelName, Manufacturer, AcqDate, Site
MRI_LIST="$TMP/mri_ids.txt"
awk -F, 'NR>1 && $5 ~ /Prisma|SIGNA|Ingenia|Achieva/ { printf "%04d\n", $1+0 }' \
  "$CSV" | sort -u > "$MRI_LIST"
echo ">> MRI cases listed in CSV: $(wc -l < "$MRI_LIST")"

# ---- 2. unzip everything --------------------------------------------------
zips=("$SRC"/*.zip)
if [ ${#zips[@]} -eq 0 ]; then
  echo "No .zip files found in '$SRC'." >&2
  exit 1
fi
for z in "${zips[@]}"; do
  echo ">> unzip $z"
  unzip -q -o "$z" -d "$TMP"
done

# ---- 3. collect CTs -------------------------------------------------------
shopt -s globstar
kept=0; skipped=0
for f in "$TMP"/**/*.nii.gz; do
  # skip AMOS label dirs (labelsTr/labelsVa/labelsTs). Match the "labels/" dir,
  # NOT the bare substring "label" -- "amos_unlabeled_ct_*" contains "label"!
  case "$f" in */labels*/*) continue ;; esac

  base="$(basename "$f")"
  pid="$(echo "$base" | grep -oE '[0-9]+' | head -n1)"
  [ -n "$pid" ] || { echo "no id, skip: $base" >&2; continue; }
  pid="$(printf '%04d' "$((10#$pid))")"

  if grep -qxF "$pid" "$MRI_LIST"; then
    echo "skip MRI : $base"
    skipped=$((skipped+1))
    continue
  fi

  dest="$OUT/$pid/000"
  mkdir -p "$dest"
  cp -f "$f" "$dest/ct.nii.gz"
  echo "keep CT  : $base -> $dest/ct.nii.gz"
  kept=$((kept+1))
done

echo "------------------------------------------"
echo "Done. CT kept: $kept   MRI/other skipped: $skipped"
echo "Output: $OUT/<patient id>/000/ct.nii.gz"
