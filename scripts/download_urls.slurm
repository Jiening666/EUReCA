#!/usr/bin/env bash
#SBATCH --job-name=dl
#SBATCH --output=dl_%j.log
#SBATCH --error=dl_%j.log
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=cpu          # <-- use a partition/node that has internet access
# Many HPC clusters firewall compute nodes. If so, submit to a data-transfer
# partition, or just run download_urls.sh on the login node instead.

set -euo pipefail

# ---- set these (or pass on the command line via --export) ------------------
DEST="${DEST:-$SCRATCH/data/download}"           # where files land
URLS="${URLS:-$SLURM_SUBMIT_DIR/urls.txt}"       # url list file
export JOBS="${JOBS:-3}"                          # files downloaded concurrently
export CONN="${CONN:-4}"                          # connections per file (aria2c)
# ---------------------------------------------------------------------------

echo "Job $SLURM_JOB_ID on $(hostname)  ->  $DEST  (JOBS=$JOBS CONN=$CONN)"
module load aria2 2>/dev/null || true            # falls back to wget/curl if absent

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/download_urls.sh" "$DEST" "$URLS"

echo "Done. Files in $DEST"
