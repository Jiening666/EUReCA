#!/usr/bin/env python
"""Download released EUReCA weights (and the demo case) from the Hugging Face Hub.

    python download_weights.py                    # main model -> ./weights/eureca_main
    python download_weights.py --model all        # every released model
    python download_weights.py --demo             # also fetch demo/data/luna16_27966

Each model folder holds `latest_net_<module>.safetensors` for every sub-network,
`train_opt.ini` (full training config, replayed automatically by test*.py) and
`manifest.json` (parameter counts + sha256). Point `--checkpoints_dir ./weights
--name eureca_main` at it.
"""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "jzhu35/EUReCA"
DEMO_REPO = "jzhu35/EUReCA-demo"
MODELS = {
    "eureca_main":   "main model: 8-100 views on a 211-degree Halcyon arc",
    "eureca_4view":  "4 fixed views at 0/45/90/135 degrees",
    "eureca_2ortho": "2 orthogonal views",
    "eureca_arc90":  "15-45 views inside a 90-degree arc",
    "eureca_arc120": "15-45 views inside a 120-degree arc",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="eureca_main", choices=list(MODELS) + ["all"])
    ap.add_argument("--out", default="weights", help="checkpoints_dir to populate")
    ap.add_argument("--demo", action="store_true",
                    help="also download the demo projections to demo/data")
    a = ap.parse_args()

    names = list(MODELS) if a.model == "all" else [a.model]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for n in names:
        print(f"[{n}] {MODELS[n]}")
        snapshot_download(REPO, allow_patterns=[f"{n}/*"], local_dir=str(out))
        print(f"    -> {out / n}")
    if a.demo:
        d = Path("demo/data")
        d.mkdir(parents=True, exist_ok=True)
        snapshot_download(DEMO_REPO, repo_type="dataset", local_dir=str(d))
        print(f"demo case -> {d}")


if __name__ == "__main__":
    main()
