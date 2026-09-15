#!/usr/bin/env bash
#SBATCH --job-name=hal_2ortho
#SBATCH --partition=gpu
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --constraint=h200|h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --output=log/2ortho_%j.log
#SBATCH --error=log/2ortho_err_%j.log

# Two-orthogonal-view specialist of the MAIN model: every draw loads exactly 2
# projections 90 deg apart, with the pair starting anywhere on the 360 deg
# circle (--angle_select_mode 2orthogonal). This is the sparsest clinically
# meaningful geometry (AP + lateral) and anchors the view-count sweep below the
# 4-fixed-view study.
#
# Config mirrors the PRODUCTION hal_n2 run and the hal_fix{10,20,50,100}v
# fixed-view specialists exactly, so the 2-view point is comparable with that
# curve. min/max_num_images are pinned to 2 for clarity — the 2orthogonal mode
# overrides the sampled view count anyway. Same 50-epoch budget, full dataset,
# same split (seed 42 -> identical split_cases.json contents as hal_n2).
#
# Resume after a crash / timeout:
#   RESUME=1 sbatch --export=ALL,RESUME=1 scripts/twoortho_hal.slurm

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

NAME="${NAME:-hal_2ortho}"

EXTRA=()
if [[ "${RESUME:-0}" == "1" ]]; then EXTRA+=(--resume); fi

echo "Job $SLURM_JOB_ID $NAME (2 orthogonal views) on $(hostname)"
nvidia-smi || true

accelerate launch \
    --mixed_precision=bf16 \
    --multi_gpu \
    --num_processes=2 \
    --main_process_port 29650 \
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
    --angle_select_mode 2orthogonal \
    --min_num_images 2 \
    --max_num_images 2 \
    --num_images_sampling uniform \
    --filter_max_res 5.0 \
    --filter_min_phys_z 300.0 \
    --aug_max_aug 1 \
    --n_epochs 50 \
    --n_epochs_decay 0 \
    --save_epoch_freq 1 \
    "${EXTRA[@]}"

echo "Done: $NAME"
