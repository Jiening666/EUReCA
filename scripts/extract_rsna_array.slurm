#!/bin/bash
#SBATCH --job-name=rsna_unzip
#SBATCH --array=0-15
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH -p cpu
#SBATCH -o /path/to/RSNA2023/unzip_%A_%a.out
#SBATCH -e /path/to/RSNA2023/unzip_%A_%a.err

# All paths absolute -- SLURM array tasks don't inherit your shell's cwd from
# where you ran `sbatch`; relying on a relative "cd" + relative chunk filename
# is what caused the earlier FileNotFoundError.
DATA_DIR="/path/to/RSNA2023"
SCRIPT="scripts/extract_rsna_shard.py"   # adjust if you keep it elsewhere

chunk_file="$DATA_DIR/$(printf 'chunk_%03d' "$SLURM_ARRAY_TASK_ID")"
echo "host: $(hostname)  chunk: $chunk_file  cpus: $SLURM_CPUS_PER_TASK"

if [ ! -f "$chunk_file" ]; then
  echo "ERROR: chunk file not found: $chunk_file (re-run split to regenerate it, then resubmit this array index)" >&2
  exit 1
fi

python3 "$SCRIPT" "$chunk_file" "$SLURM_CPUS_PER_TASK"
