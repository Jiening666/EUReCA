#!/usr/bin/env bash
#SBATCH --job-name=mela_dl
#SBATCH --output=mela_dl_%j.log
#SBATCH --error=mela_dl_%j.log
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=cpu          # <-- change to a partition that has internet access
# Some clusters only allow outbound traffic from login/data-transfer nodes.
# If compute nodes are firewalled, run download_mela.sh on the login node instead,
# or submit to a dedicated data-transfer / xfer partition.

set -euo pipefail

# ---- edit these two paths --------------------------------------------------
DEST="${DEST:-$SCRATCH/data/MELA}"          # where the data should land
URLS="${URLS:-$SLURM_SUBMIT_DIR/mela_urls.txt}"
# ---------------------------------------------------------------------------

echo "Job $SLURM_JOB_ID on $(hostname)  ->  $DEST"

# Make a fast resumable HTTP tool available. Adjust to your cluster's modules.
# aria2c is preferred; the script falls back to wget or curl if it's absent.
module load aria2 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/download_mela.sh" "$DEST" "$URLS"

echo "Done. Data in $DEST"
