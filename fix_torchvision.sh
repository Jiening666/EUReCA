#!/usr/bin/env bash
# Repair a torch/torchvision mismatch (e.g. after `pip install timm` upgraded
# torchvision past your torch's compatibility -> "operator torchvision::nms
# does not exist").  Keeps your torch fixed; reinstalls the torchvision that
# MATCHES the already-installed torch + its CUDA build.
set -e

read TV IDX <<EOF
$(python - <<'PY'
import torch
# torchvision minor tracks torch minor with a fixed +15 offset
# (torch 2.0 <-> tv 0.15, 2.1 <-> 0.16, ... 2.11 <-> 0.26).
major, minor = (int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
tv_minor = minor + 15 if major == 2 else minor  # extend if torch hits 3.x
cu = torch.version.cuda
idx = "cu" + cu.replace(".", "") if cu else "cpu"
print(f"0.{tv_minor}", idx)
PY
)
EOF

echo "torch $(python -c 'import torch;print(torch.__version__)') -> torchvision==$TV.*  (index: $IDX)"
pip install --no-deps --force-reinstall "torchvision==$TV.*" \
    --index-url "https://download.pytorch.org/whl/$IDX"

echo "verifying ..."
python -c "import torchvision; print('torchvision', torchvision.__version__, 'OK'); import timm; print('timm', timm.__version__, 'imports OK')"
echo "FIXED. Re-run: bash setup_cluster.sh"
