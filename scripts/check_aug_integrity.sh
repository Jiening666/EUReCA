#!/usr/bin/env bash
# Pre-flight dataset check — run BEFORE launching parallel test/eval jobs.
#
# test.py/train_single.py regenerate a case lazily when `_check_aug_integrity`
# fails, and that check only probes ct_resampled.npz + img/proj_358.00.png. A
# dir left half-rendered by a killed job keeps both markers while holding too
# few angles, so nothing repairs it and every job that needs more views than it
# has dies on `Not enough angles to select from` — hours in. Worse, several
# jobs hitting one broken case at once race each other's regeneration.
#
# A complete hal_* dir holds 360 files (180 angles x {png,npy}).
#
# Usage: bash scripts/check_aug_integrity.sh [datasets_root] [expected_count]
set -uo pipefail
ROOT="${1:-./data}"
WANT="${2:-360}"

bad=0
while read -r d; do
  n=$(ls "$d/img" 2>/dev/null | wc -l)
  if [[ "$n" -ne "$WANT" ]]; then
    npy=$(ls "$d"/img/*.npy 2>/dev/null | wc -l)
    echo "BAD  files=$n (npy=$npy, max usable views=$npy)  $d"
    bad=$((bad + 1))
  fi
done < <(find "$ROOT" -mindepth 4 -maxdepth 4 -type d -name 'hal_*' 2>/dev/null)

if [[ "$bad" -eq 0 ]]; then
  echo "OK: every hal_* dir under $ROOT has $WANT files"
else
  echo "$bad damaged dir(s). Repair with a scratch-render + atomic swap (see the"
  echo "2026-08-01 entry in the Obsidian ablation note) — never rmtree in place"
  echo "while other jobs are reading, and re-check before launching tests."
fi
exit $((bad > 0))
