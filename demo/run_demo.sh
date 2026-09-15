#!/usr/bin/env bash
# Reconstruct the bundled LUNA16 demo case with a released EUReCA model.
#
#   bash demo/run_demo.sh                    # eureca_main, 20 views
#   bash demo/run_demo.sh eureca_main 50     # eureca_main, 50 views
#   bash demo/run_demo.sh eureca_4view       # 4 fixed views (0/45/90/135)
#   bash demo/run_demo.sh eureca_2ortho      # 2 orthogonal views (0/90)
#   bash demo/run_demo.sh eureca_arc90 30    # 30 views inside a 90-degree arc
#
# Requires: python download_weights.py --model <name> --demo   (weights/ + demo/data/)
# The demo case ships 180 projections at 2-degree steps over 360 degrees; each
# model is evaluated on the acquisition geometry it was trained for (a 211-degree
# arc for the main model), so views are sub-selected from that arc, starting at 0.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-eureca_main}"
N="${2:-}"
WEIGHTS="${WEIGHTS:-weights}"
DATA="${DATA:-demo/data}"
SCAN=luna16_27966
OUT="${OUT:-outputs/demo/$MODEL}"

# Angles available: 0,2,...,358  ->  index = angle / 2
case "$MODEL" in
  eureca_main)   ARC=210; N="${N:-20}" ;;
  eureca_arc90)  ARC=90;  N="${N:-30}" ;;
  eureca_arc120) ARC=120; N="${N:-30}" ;;
  eureca_4view)  IDX="0 22 45 67" ;;     # 0 / 44 / 90 / 134 deg (nearest to 0/45/90/135)
  eureca_2ortho) IDX="0 45" ;;           # 0 / 90 deg
  *) echo "unknown model $MODEL"; exit 1 ;;
esac
if [ -z "${IDX:-}" ]; then
  IDX=$(python -c "import numpy as np; print(' '.join(str(i) for i in np.linspace(0, $ARC//2, $N).round().astype(int)))")
fi
echo "model=$MODEL  views: $IDX"

python test_npy.py @./options/geometry/hal.ini \
    --name "$MODEL" --checkpoints_dir "$WEIGHTS" \
    --npy_root "$DATA" --scan "$SCAN" --view_indices $IDX \
    --angle_sign 1 --angle_offset 0 \
    --fdk --output_dir "$OUT"

# (the all-view FDK volume is written for visual comparison; its intensity scale
#  follows the max-normalised projections, so it is not scored in HU here)
python demo/evaluate.py "$OUT"/${SCAN}_[0-9]*views.nii.gz --gt "$DATA/$SCAN/gt_hu.nii.gz"
