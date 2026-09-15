#!/usr/bin/env bash
# Download AbdomenAtlas 3.0 Mini (https://huggingface.co/datasets/AbdomenAtlas/AbdomenAtlas3.0Mini)
# to a cluster. Resumable: safe to re-run after interruption or job timeout.
#
# Repo contents (~586 GB total):
#   image_only/   41 tar.gz archives, ~571 GB  (CT volumes)
#   mask_only/    tar.gz archives              (per-voxel annotations)
#   TrainTestIDS/ train/test split CSVs
#   AbdomenAtlas3.0MiniWithMeta.csv  (16 MB metadata)
#   pdf_reports.zip  (2 GB radiology reports)
#
# Usage:
#   bash download_abdomenatlas3.0mini.sh /path/to/data/AbdomenAtlas3.0Mini [images|masks|meta|all]
#   default component: all
#
# Requirements: pip install -U "huggingface_hub[cli]" hf_transfer
# Optional: export HF_TOKEN=hf_xxx   (not required — dataset is public — but avoids rate limits)

set -euo pipefail

REPO="AbdomenAtlas/AbdomenAtlas3.0Mini"
DEST="${1:?Usage: $0 <destination_dir> [images|masks|meta|all]}"
COMPONENT="${2:-all}"

mkdir -p "$DEST"

# Fast parallel transfer backend (big win on cluster links)
export HF_HUB_ENABLE_HF_TRANSFER=1
# Keep the HF cache next to the data instead of filling $HOME quota
export HF_HOME="${HF_HOME:-$DEST/.hf_cache}"

# huggingface_hub >= 0.34 ships `hf`; older versions only have `huggingface-cli`
if command -v hf >/dev/null 2>&1; then
    DL() { hf download "$REPO" --repo-type dataset --local-dir "$DEST" "$@"; }
else
    DL() { huggingface-cli download "$REPO" --repo-type dataset --local-dir "$DEST" "$@"; }
fi

case "$COMPONENT" in
    meta)
        DL --include "TrainTestIDS/*" "AbdomenAtlas3.0MiniWithMeta.csv" "README.md"
        ;;
    images)
        DL --include "image_only/*"
        ;;
    masks)
        DL --include "mask_only/*"
        ;;
    reports)
        DL --include "pdf_reports.zip"
        ;;
    all)
        DL  # everything (~586 GB)
        ;;
    *)
        echo "Unknown component '$COMPONENT' (use images|masks|meta|reports|all)" >&2
        exit 1
        ;;
esac

echo "Download of '$COMPONENT' complete -> $DEST"
echo
echo "To extract the archives afterwards (parallel, 4 at a time):"
echo "  ls $DEST/image_only/*.tar.gz | xargs -P 4 -I{} tar -xzf {} -C $DEST/image_only/"
echo "  ls $DEST/mask_only/*.tar.gz  | xargs -P 4 -I{} tar -xzf {} -C $DEST/mask_only/"
