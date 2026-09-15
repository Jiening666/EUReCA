#!/usr/bin/env bash
# Train the main EUReCA model (8-100 views, 211-degree Halcyon arc).
# This is the exact configuration of the released `eureca_main` weights;
# 2x A100 80GB, ~50 epochs. Adjust partition / account / paths for your cluster.
#SBATCH --job-name=eureca_main
#SBATCH --partition=gpu                 # <-- your GPU partition
##SBATCH --account=YOUR_ACCOUNT
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --output=log/%x_%j.log
#SBATCH --error=log/%x_%j.log

set -euo pipefail
mkdir -p log

# ---- environment (edit) ---------------------------------------------------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-eureca}"

NAME="${NAME:-eureca_main}"
CKPT_DIR="${CKPT_DIR:-./checkpoints}"       # each run writes ~3 GB (weights + optimizer state)
DATASETS_INI="${DATASETS_INI:-./options/datasets/example.ini}"

# ---- training -------------------------------------------------------------
# Add --resume to the same command to continue after a crash.
accelerate launch --mixed_precision=bf16 --multi_gpu --num_processes=2 \
    train_single.py \
    @"$DATASETS_INI" \
    @./options/geometry/hal.ini \
    --name "$NAME" \
    --checkpoints_dir "$CKPT_DIR" \
    --use_binned_aggr --use_rope --use_fdk \
    --encoder_backbone convnextv2_nano.fcmae_ft_in22k_in1k \
    --encoder_view_chunk 16 --backbone_lr_mult 0.1 \
    --bin_ray_pe 1 --fdk_film 1 \
    --angle_select_mode halcyon \
    --min_num_images 8 --max_num_images 100 --num_images_sampling loguniform \
    --lambda_L1 -1.0 --lambda_ssim -1.0 \
    --blade_blocking --filter_max_res 5.0 --filter_min_phys_z 300.0

echo "Done."
