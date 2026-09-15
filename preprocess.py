import time
from options.train_options import TrainOptions
import os
from ct_dataset import scan_manifest, preprocess_dataset, filter_manifest, sample_manifest
import pandas as pd


def preprocess(opt):
    manifest = scan_manifest(opt.data_root)
    preprocess_dataset(manifest, opt)
    
def preprocess_all(opt):
    all_datasets = opt.datasets.split(",")
    manifests = []
    for i, dataset in enumerate(all_datasets):
        manifest = scan_manifest(dataset, verbose=1)
        manifest['case_id'] = manifest['case_id'].apply(lambda x: f"D{i}_" + x)
        manifests.append(manifest)

    manifest_all = pd.concat(manifests, axis=0, ignore_index=True)
    # Use the SAME filter thresholds as training (train_single.py), otherwise
    # cases that pass training's looser filter but not filter_manifest's strict
    # defaults (res<2.0, phys_z>380) get no augs here and are re-generated on the
    # fly during training. opt.filter_max_res / opt.filter_min_phys_z come from
    # --filter_max_res / --filter_min_phys_z.
    manifest_all = filter_manifest(manifest_all,
                                   max_res=opt.filter_max_res,
                                   min_phys_z=opt.filter_min_phys_z)
    manifest_all = sample_manifest(manifest_all, opt.max_dataset_size)

    # Only job 0 writes the full manifest so jobs don't race on the same file
    if opt.job_index == 0:
        manifest_all.to_csv(os.path.join(opt.checkpoints_dir, opt.name, "manifest.csv"), index=False)

    # Each job processes every num_jobs-th row (round-robin split)
    if opt.num_jobs > 1:
        manifest_slice = manifest_all.iloc[opt.job_index::opt.num_jobs].reset_index(drop=True)
        print(f"[Job {opt.job_index}/{opt.num_jobs}] Processing {len(manifest_slice)}/{len(manifest_all)} scans")
    else:
        manifest_slice = manifest_all

    preprocess_dataset(manifest_slice, opt)
    

if __name__ == '__main__':
    opt = TrainOptions().parse()
    preprocess_all(opt)