#!/usr/bin/env bash
#SBATCH --job-name=hal_s90
#SBATCH --partition=gpu
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --constraint=h200|h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --output=log/s90_%j.log
#SBATCH --error=log/s90_err_%j.log

# 90 deg short-arc arm, RETRAIN of hal_s2 on corrected FDK weighting.
#
# The original hal_s2 trained while Parker short-scan weighting was active on a
# 90 deg arc, which is invalid (Parker compensates redundancy over a
# pi + 2*gamma_max ~ 211 deg span; a 90 deg arc has no redundant overlap). It
# zeroed view 0 outright and attenuated ~44% of the arc. Commit e2e3381 added a
# 180 deg floor to the guard, so a 90 deg arc now uses angular-interval
# weighting. hal_s2's own numbers are self-consistent (same weighting at train
# and test) but are NOT comparable to anything trained after that fix.
#
# Config MIRRORS hal_s2 exactly -- same recipe, 15..50 loguniform views, 50
# epochs, full dataset, split seed 42 -- so the ONLY difference from hal_s2 is
# the corrected FDK prior. That makes s90 vs s2 a clean read on the fix, and
# s90 vs hal_s120 vs hal_n2 the 90/120/211 arc-width sweep.
#
# Note --max_num_images 50 is kept for exact parity with hal_s2, but a 90 deg
# arc on a 180-view 2 deg grid only HAS 45 distinct projections
# (arc_len = num*arc_deg/360), so _arc_binning_select caps draws above 45. That
# is a physical limit of the geometry, not a config error; ~10% of loguniform
# draws land there. hal_s120 (arc_len = 60) realizes its full range.
#
# Resume after a crash / timeout:
#   RESUME=1 sbatch --export=ALL,RESUME=1 scripts/shortarc90_hal.slurm

set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
ENV_PATH="${ENV_PATH:-$CONDA_ROOT/envs/cbct_recon}"
source "$(conda info --base 2>/dev/null || echo "$CONDA_ROOT")/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"

REPO="${REPO:-$HOME/cbct_recon}"
cd "$REPO"
mkdir -p log

CKPT_DIR="${CKPT_DIR:-./checkpoints}"
export WANDB_DIR="$CKPT_DIR/wandb"
mkdir -p "$CKPT_DIR" "$WANDB_DIR"

NAME="${NAME:-hal_s90}"

EXTRA=()
if [[ "${RESUME:-0}" == "1" ]]; then EXTRA+=(--resume); fi

echo "Job $SLURM_JOB_ID $NAME (90 deg short arc, corrected FDK) on $(hostname)"
nvidia-smi || true

accelerate launch \
    --mixed_precision=bf16 \
    --multi_gpu \
    --num_processes=2 \
    --main_process_port 29670 \
    train_single.py \
    @./options/datasets/all_iris_ext.ini \
    @./options/geometry/hal.ini \
    --name "$NAME" \
    --checkpoints_dir "$CKPT_DIR" \
    --use_binned_aggr \
    --use_rope \
    --use_fdk \
    --use_ref_inject 1 \
    --fdk_filter hann \
    --fdk_film 1 \
    --fdk_parker_gamma_sign -1.0 \
    --encoder_backbone convnextv2_nano.fcmae_ft_in22k_in1k \
    --refine_depth 12 \
    --aggr_bins 36 \
    --aggr_layers 8 \
    --bin_ray_pe 1 \
    --lambda_L1 -1.0 \
    --lambda_ssim -1.0 \
    --lambda_perc 1.0 \
    --lambda_align 0.1 \
    --align_ridge 1.0 \
    --blade_blocking \
    --blade_max_frac 0.07 \
    --blade_prob 0.2 \
    --self_ref_aug 0.1 \
    --self_ref_prob 0.0 \
    --lr 0.0002 \
    --lr_policy cosine \
    --backbone_lr_mult 0.1 \
    --beta1 0.5 \
    --grad_clip 1.0 \
    --batch_size 1 \
    --num_batch_img 5 \
    --angle_select_mode shortarc \
    --min_num_images 15 \
    --max_num_images 50 \
    --num_images_sampling loguniform \
    --filter_max_res 5.0 \
    --filter_min_phys_z 300.0 \
    --aug_max_aug 1 \
    --n_epochs 50 \
    --n_epochs_decay 0 \
    --save_epoch_freq 1 \
    "${EXTRA[@]}"

echo "Done: $NAME"
