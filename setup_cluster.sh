#!/usr/bin/env bash
# One-shot cluster setup for the restructured model.
# Run from the repo root on a node WITH internet (login node) so the pretrained
# encoder weights get cached; the smoke test runs only if a GPU is visible.
#
#   git pull && bash setup_cluster.sh
#
set -e

echo "[1/3] installing timm (pretrained encoder) ..."
# --no-deps so pip does NOT upgrade torch/torchvision (which causes a
# 'torchvision::nms does not exist' mismatch); install timm's small,
# torch-independent deps explicitly. If torchvision is already broken from an
# earlier 'pip install timm', run: bash fix_torchvision.sh
pip install --no-deps "timm>=0.9.12"
pip install huggingface_hub safetensors pyyaml

echo "[2/3] pre-caching pretrained encoder weights into ${HF_HOME:-$HOME/.cache/huggingface} ..."
python - <<'PY'
import timm
timm.create_model('convnextv2_nano.fcmae_ft_in22k_in1k', pretrained=True,
                  features_only=True, out_indices=(1, 2, 3), in_chans=1)
print("  weights cached OK")
PY

echo "[3/3] smoke test (model graph + memory) ..."
if python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    python smoke_test.py --n_views 100 \
        --encoder_backbone convnextv2_nano.fcmae_ft_in22k_in1k
else
    echo "  no GPU on this node — run on a GPU node:"
    echo "    python smoke_test.py --n_views 100 --encoder_backbone convnextv2_nano.fcmae_ft_in22k_in1k"
fi

echo
echo "DONE. If you saw '=== SMOKE TEST PASSED ===', you are ready to train."
echo "Next: a 1-GPU pipeline check, then the full multi-GPU launch:"
echo "  CUDA_VISIBLE_DEVICES=0 python train_single.py @./options/datasets/all_iris.ini \\"
echo "    @./options/geometry/hal.ini --name hal_smoke --use_binned_aggr --use_rope --use_fdk \\"
echo "    --encoder_backbone convnextv2_nano.fcmae_ft_in22k_in1k --encoder_view_chunk 16 \\"
echo "    --min_num_images 8 --max_num_images 24 --num_images_sampling loguniform \\"
echo "    --lambda_L1 -1.0 --lambda_ssim -1.0 --num_batch_img 1"
