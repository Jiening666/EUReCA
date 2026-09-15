#!/usr/bin/env bash
#SBATCH --job-name=hal_s120
#SBATCH --partition=gpu
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --constraint=h200|h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --output=log/s120_%j.log
#SBATCH --error=log/s120_err_%j.log

# 120 deg short-arc arm (--angle_select_mode shortarc120), the wider companion to
# the 90 deg hal_s2 run. Config MIRRORS hal_s2 exactly -- same production recipe,
# same view-count sampling (15..50 loguniform), same 50-epoch budget, full
# dataset, split seed 42 -- so the ONLY difference between the two arms is the
# arc width. That makes s2 (90) vs s120 (120) vs hal_n2 (211) a clean
# arc-coverage sweep.
#
# NOTE on the FDK prior: a 120 deg arc has coverage 120 deg, which is below the
# 180 deg floor added to the Parker guard in models/fdk.py, so this run uses
# angular-interval weighting throughout. That is the correct behaviour (there is
# no redundant overlap to compensate on a sub-half scan) and it also means this
# arm is NOT comparable to the pre-fix hal_s2 checkpoint, which trained under the
# old invalid Parker weighting. Retrain s2 on current code before comparing.
#
# Resume after a crash / timeout:
#   RESUME=1 sbatch --export=ALL,RESUME=1 scripts/shortarc120_hal.slurm

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

NAME="${NAME:-hal_s120}"

EXTRA=()
if [[ "${RESUME:-0}" == "1" ]]; then EXTRA+=(--resume); fi

echo "Job $SLURM_JOB_ID $NAME (120 deg short arc) on $(hostname)"
nvidia-smi || true

accelerate launch \
    --mixed_precision=bf16 \
    --multi_gpu \
    --num_processes=2 \
    --main_process_port 29660 \
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
    --angle_select_mode shortarc120 \
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
