#!/usr/bin/env bash
#SBATCH --job-name=kaggle_dl
#SBATCH --output=kaggle_dl_%j.log
#SBATCH --error=kaggle_dl_%j.log
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=cpu          # <-- needs outbound internet; use an xfer node if compute is firewalled

set -euo pipefail

# ---- edit / override via --export ------------------------------------------
DEST="${DEST:-$SCRATCH/data/RSNA2023}"
COMP="${COMP:-rsna-2023-abdominal-trauma-detection}"
export UNZIP="${UNZIP:-1}"
export KEEP_ZIP="${KEEP_ZIP:-0}"            # delete the 400 GB zip after unzip
# Credentials: easiest is to point at a kaggle.json you've uploaded:
export KAGGLE_CONFIG_DIR="${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}"
# ---------------------------------------------------------------------------

echo "Job $SLURM_JOB_ID on $(hostname) -> $DEST  ($COMP)"
module load python 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/download_kaggle.sh" "$DEST" "$COMP"

echo "Done. Data in $DEST"
