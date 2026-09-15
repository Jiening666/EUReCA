#!/usr/bin/env bash
#SBATCH --job-name=hal_4_135
#SBATCH --partition=gpu
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --constraint=h200|h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --output=log/4v_%j.log
#SBATCH --error=log/4v_err_%j.log

# Fixed 4-view specialist. Every draw loads EXACTLY the --extra_angles set and
# nothing else (--min_num_images 0 --max_num_images 0 means zero sampled views
# on top), so --angle_select_mode is inert here and only kept for parity with
# the hal_4 config.
#
# ANGLES defaults to 0,45,90,135 -- four views on a 45 deg pitch spanning a
# 135 deg arc. This SUPERSEDES the original hal_4 run, which used
# 45,135,225,315 (a 90 deg pitch around the full circle).
#
# Note the projections live on a 2 deg grid, so 45 and 135 are not on-grid and
# snap to the nearest available angle (ties -> smaller): the run actually sees
# 0, 44, 90, 134. That is inherent to the DRR sampling and matches how hal_4's
# own targets snapped to 44,134,224,314.
#
# FDK weighting: the new set spans 135 deg -> coverage 134 deg, below the 180 deg
# Parker floor from e2e3381, so it uses angular-interval weighting and all four
# views contribute (shares .333/.167/.167/.333).
#
# !! The OLD hal_4 set did NOT get that treatment: 45,135,225,315 has a max gap
# of 90 deg, so coverage reads 270 deg and Parker stays ACTIVE -- which zeroed
# the first and last views (shares .001/.533/.467/.000). The old hal_4 therefore
# trained on a ~2-view FDK prior. Its coverage metric (360 - largest gap)
# under-reports a sparse FULL-circle set as a 270 deg arc; that limitation is
# unrelated to this run and is left as-is deliberately.
#
# Override to reproduce the old geometry:
#   NAME=hal_4_orig ANGLES=45,135,225,315 sbatch --export=ALL,NAME,ANGLES \
#       scripts/fourview_hal.slurm
#
# Resume after a crash / timeout:
#   RESUME=1 sbatch --export=ALL,RESUME=1 scripts/fourview_hal.slurm

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

NAME="${NAME:-hal_4_135}"
ANGLES="${ANGLES:-0,45,90,135}"

EXTRA=()
if [[ "${RESUME:-0}" == "1" ]]; then EXTRA+=(--resume); fi

echo "Job $SLURM_JOB_ID $NAME (fixed 4 views: $ANGLES) on $(hostname)"
nvidia-smi || true

accelerate launch \
    --mixed_precision=bf16 \
    --multi_gpu \
    --num_processes=2 \
    --main_process_port 29680 \
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
    --angle_select_mode halcyon \
    --extra_angles "$ANGLES" \
    --min_num_images 0 \
    --max_num_images 0 \
    --num_images_sampling loguniform \
    --filter_max_res 5.0 \
    --filter_min_phys_z 300.0 \
    --aug_max_aug 1 \
    --n_epochs 50 \
    --n_epochs_decay 0 \
    --save_epoch_freq 1 \
    "${EXTRA[@]}"

echo "Done: $NAME"
