#!/usr/bin/env python
"""Publish the staged EUReCA release to the Hugging Face Hub (maintainer script).

Run on the machine that holds the staged folders (see README of the release
plan), after `hf auth login`:

    python hf/upload.py --stage /path/to/staged/release --cards hf/

Creates (if needed) the model repo jzhu35/EUReCA and the dataset repo
jzhu35/EUReCA-demo, uploads one folder per model (safetensors + train_opt.ini +
manifest.json; the `pth/` backups are excluded) and the demo case, then writes
the cards. Re-running only uploads changed files.
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi

MODEL_REPO = "jzhu35/EUReCA"
DEMO_REPO = "jzhu35/EUReCA-demo"
MODELS = ["eureca_main", "eureca_4view", "eureca_2ortho", "eureca_arc90", "eureca_arc120"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, help="staged release root (holds eureca_*/ and demo/)")
    ap.add_argument("--cards", default="hf", help="folder with model_card.md / dataset_card.md")
    ap.add_argument("--private", action="store_true", help="create repos private (flip public later on the Hub)")
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--skip_demo", action="store_true")
    a = ap.parse_args()

    api = HfApi()
    me = api.whoami()["name"]
    print(f"logged in as {me}")
    stage, cards = Path(a.stage), Path(a.cards)

    api.create_repo(MODEL_REPO, repo_type="model", private=a.private, exist_ok=True)
    for m in a.models:
        src = stage / m
        assert (src / "manifest.json").exists(), src
        print(f"uploading {m} ...")
        api.upload_folder(
            repo_id=MODEL_REPO, repo_type="model", folder_path=str(src), path_in_repo=m,
            allow_patterns=["*.safetensors", "train_opt.ini", "manifest.json", "inference_opt.ini"],
            commit_message=f"add {m} weights",
        )
    api.upload_file(path_or_fileobj=str(cards / "model_card.md"), path_in_repo="README.md",
                    repo_id=MODEL_REPO, repo_type="model", commit_message="model card")
    print(f"model repo: https://huggingface.co/{MODEL_REPO}")

    if not a.skip_demo:
        api.create_repo(DEMO_REPO, repo_type="dataset", private=a.private, exist_ok=True)
        print("uploading demo case ...")
        api.upload_folder(repo_id=DEMO_REPO, repo_type="dataset", folder_path=str(stage / "demo"),
                          commit_message="add luna16_27966 demo case")
        api.upload_file(path_or_fileobj=str(cards / "dataset_card.md"), path_in_repo="README.md",
                        repo_id=DEMO_REPO, repo_type="dataset", commit_message="dataset card")
        print(f"demo repo:  https://huggingface.co/datasets/{DEMO_REPO}")


if __name__ == "__main__":
    main()
