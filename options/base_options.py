import argparse
from email import parser
import os
#from util import util
#import torch
#import models
#import data
from pathlib import Path

class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        #parser.add_argument('--dataroot', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        
        parser.add_argument('--data_root', type=str, default="./data", help='root path for training data volumes, most important entry point!')
        
        parser.add_argument('--datasets', type=str, default="./data/LUNA16,./data/AMOS22", help='root paths for training data volumes, most important entry point! Separate multiple paths with comma if needed. For example: --datasets /path1,/path2')
        
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more, only a subset is loaded.')
        
        parser.add_argument('--aug_prefix', type=str, default="hal", help='prefix for augmented CT paths')
        
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        
        parser.add_argument('--filter_max_res', type=float, default=2.0, help='maximum resolution in mm to include a scan for training (e.g. 2.0mm means only include scans with resolution finer than 2.0mm)')
        parser.add_argument('--filter_min_phys_z', type=float, default=380.0, help='minimum physical size in z direction in mm to include a scan for training (e.g. 300mm means only include scans with physical z size larger than 300mm) - this ensures the scan is enough to generate DRRs with the desired number of slices)')
        
        
        parser.add_argument('--aug_max_aug', type=int, default=1, help='maximum number of augmented CT to generate for each original CT')
        
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--model', type=str, default='ddd', help='chooses which model to use. [ddd], drr2dtitemplate netD must be DPT')
        parser.add_argument('--name', type=str, default='test0211', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--use_wandb', action='store_true', help='use wandb')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--H', type=int, default=384, help='Height of the output images (for Vit)')
        parser.add_argument('--W', type=int, default=768, help='Width of the output images (for Vit)')
        parser.add_argument('--psize', type=float, default=0.7273, help='physical size of pixel of the ourput image (for Vit) in mm | 1.12/1.5*256/512')
        parser.add_argument('--no_add_noise', action='store_true', help='do not add noise to the images')
        
        parser.add_argument('--resw', type=float, default=0.7273, help='pixel size of the original images in mm')
        parser.add_argument('--resh', type=float, default=0.7273, help='pixel size of the original images in mm')
        parser.add_argument('--min_num_images', type=int, default=10, help='minimum number of images to sample for each volume')
        parser.add_argument('--max_num_images', type=int, default=10, help='maximum number of images to sample for each volume')
        parser.add_argument('--num_images_sampling', type=str, default='uniform',
                            choices=['uniform', 'loguniform'],
                            help='How to sample the per-batch view count N in '
                                 '[min_num_images, max_num_images]. "uniform" = original '
                                 'random.randint. "loguniform" = equal probability per octave; '
                                 'use with a wide range (e.g. 10-500) so sparse counts stay '
                                 'well represented while training up to dense scans.')
        
        parser.add_argument('--num_batch_img', type=int, default=5, help='Load once ct, and corresponding batches images are loaded n times')
        
        parser.add_argument('--angle_select_mode', type=str, default='uniform', help='mode for selecting angles. Options: "random", "uniform", "shortarc" (90 deg arc), "shortarc120" (120 deg arc), "halcyon" (211), "binning" (360), "truebeam" (200), "2orthogonal" (always exactly 2 views 90 deg apart, random start; ignores min/max_num_images — do not combine with other modes unless N=2) or combined modes like "uniform,halcyon"')
        parser.add_argument('--extra_angles', type=str, default='',
                            help='comma-separated gantry angles ALWAYS loaded in addition to the '
                                 'N sampled views, each snapped to the closest available projection '
                                 '(e.g. "45,135,225,315" snaps to 44,134,224,314 on a 2-degree grid). '
                                 'Empty string disables.')

        parser.add_argument('--Nx', type=int, default=256, help='Number of voxels in x direction of the CT volume')
        parser.add_argument('--Ny', type=int, default=256, help='Number of voxels in y direction of the CT volume')
        parser.add_argument('--Nz', type=int, default=128, help='Number of voxels in z direction of the CT volume')
        parser.add_argument('--Nz_temp', type=int, default=190, help='Number of voxels in z direction of the CT volume for DRR generation')
        parser.add_argument('--vol_size', type=float, default=2.0, help='pixel size of the original images in mm')
        
        parser.add_argument('--lambda_L1', type=float, default=1.0, help='weight for L1 loss')
        parser.add_argument('--lambda_ssim', type=float, default=1.0, help='weight for SSIM loss')
        parser.add_argument('--use_perc_loss', action='store_true', help='use perceptual loss via the reference encoder feature maps')
        parser.add_argument('--lambda_perc', type=float, default=1.0, help='weight for perceptual loss')
        parser.add_argument('--lambda_maga', type=float, default=1.0,
                            help='weight for the multi-scale Scharr gradient-difference loss '
                                 '(L_MAGA). 0 disables it (ablation arm A2). Default 1.0 '
                                 'reproduces the previous unconditional behaviour.')
        
        parser.add_argument('--aug_max_shift_mm', type=float, default=5.0, help='maximum shift in mm for data augmentation')
        parser.add_argument('--aug_max_rot_deg', type=float, default=5.0, help='maximum rotation angle in degree for data augmentation')

        # Blade blocking augmentation
        parser.add_argument('--blade_blocking', action='store_true', help='simulate collimator blade blocking: zero out random top/bottom bands in each projection batch')
        parser.add_argument('--blade_max_frac', type=float, default=0.07, help='max fraction of image height blocked by each blade (top and bottom independently)')
        parser.add_argument('--blade_prob', type=float, default=0.2, help='probability that a given batch has blade blocking applied')

        # Parallel job splitting
        parser.add_argument('--job_index', type=int, default=0, help='0-based index of this job (for parallel preprocessing)')
        parser.add_argument('--num_jobs',  type=int, default=1, help='total number of parallel preprocessing jobs')

        # 2D encoder backbone
        parser.add_argument('--encoder_backbone', type=str, default='',
                            help="timm backbone for the 2D encoder, e.g. 'convnextv2_nano' "
                                 "(or a full tag like 'convnextv2_nano.fcmae_ft_in22k_in1k'). "
                                 "Empty = from-scratch ConvNeXt + 2D FPN. When set: pretrained "
                                 "features_only taps (strides 8/16/32, in_chans=1), 2D FPN disabled, "
                                 "per-image input z-scoring on.")
        parser.add_argument('--encoder_view_chunk', type=int, default=8,
                            help="Encode projections in chunks of this many views, each chunk "
                                 "gradient-checkpointed, to bound peak memory at large N. "
                                 "0 = disable (single call, no checkpoint).")
        parser.add_argument('--backbone_lr_mult', type=float, default=0.1,
                            help="LR multiplier for the pretrained encoder backbone param "
                                 "group (relative to --lr). Only applies when --encoder_backbone "
                                 "is set; the rest of the (cold) network trains at full --lr.")
        # Model depth controls — reduce to trade capacity for speed
        parser.add_argument('--fuse_depth',   type=int, default=0,  help='Per-view FusionViT depth. 0 = skip the transformer (keep ModLN ray conditioning); capacity moved to the 3D transformers')
        parser.add_argument('--aggr_layers',  type=int, default=8,  help='Binned cross-view aggregator (Stage-2) layers — O(K), constant in N')
        parser.add_argument('--bin_ray_pe', type=int, default=0,
                            help='Add a per-(voxel,bin) Plücker ray positional encoding to the '
                                 'coarse Stage-2 transformer tokens (geometry-aware cross-view '
                                 'attention). 0 = off (bin-centre [sinθ,cosθ] geom only).')
        parser.add_argument('--refine_depth', type=int, default=12, help='VolumeRefinementTransformer (3D self-refinement) depth')
        parser.add_argument('--use_rope', action='store_true', help='add 3D RoPE to VolumeRefinementTransformer')
        # Memory controls
        parser.add_argument('--aggr_chunk_size', type=int, default=1024,
                            help='Chunk size for the chunked 2D→3D aggregation. Lower values reduce '
                                 'peak memory at the cost of more loop iterations.')
        # Angular-binned aggregation (twod_to_threed_model_binning.py)
        parser.add_argument('--use_binned_aggr', action='store_true',
                            help='Use angular-binned two-stage view aggregation '
                                 '(twoD2threeDBinningModel). Scales to hundreds of '
                                 'projections; default off = original per-view aggregation.')
        parser.add_argument('--aggr_bins', type=int, default=36,
                            help='Number of angular bins K for binned aggregation, in the '
                                 'absolute gantry frame, shared by coarse/mid/fine scales. '
                                 'Must be even (conjugate pairing k <-> k+K/2).')
        parser.add_argument('--aggr_vox_chunk', type=int, default=8192,
                            help='Voxel chunk size for binned mid/fine aggregation. Peak '
                                 'memory is O(aggr_vox_chunk * K * C) regardless of N.')
        # Reference alignment + multi-scale prior injection (binned model only)
        parser.add_argument('--lambda_align', type=float, default=0.1,
                            help='Weight of the reference-alignment loss: NCC between the '
                                 'affine-warped reference volume and the GT source volume, '
                                 'plus the near-affine attention consistency term. 0 disables '
                                 'the loss (the transform is still fitted and used).')
        parser.add_argument('--align_ridge', type=float, default=1.0,
                            help='Ridge strength pulling the fitted affine toward identity. '
                                 'High = conservative (stays near identity until the '
                                 'correspondence attention becomes reliable).')
        parser.add_argument('--self_ref_prob', type=float, default=0.0,
                            help='Fraction of paired batches that use a self-reference (the '
                                 'source scan itself, lightly augmented) instead of the '
                                 'different-time scan. Supplies the missing "a matching prior '
                                 'is useful" training signal so the reference branch learns to '
                                 'exploit a good prior. 0 = original (different-time only).')
        parser.add_argument('--self_ref_aug', type=float, default=0.1,
                            help='Intensity/noise augmentation strength for the self-reference '
                                 'volume (0 = exact copy of the source GT; higher = more '
                                 'imperfect prior, discourages trivial identity copying).')
        parser.add_argument('--use_ref_inject', type=int, default=1,
                            help='1 = use the reference-injection path (alignment + skip '
                                 'fusion, incl. the null-ref skip on single batches); 0 = '
                                 'disable it entirely for a clean FDK-only / projection-only '
                                 'experiment (no align, no null-ref skip).')
        # FDK prior injection (binned model; additive, in-frame, no alignment)
        parser.add_argument('--use_fdk', action='store_true',
                            help='Inject an FDK backprojection of the source projections into '
                                 'the decoder skips (coverage-gated). In-frame, needs no '
                                 'alignment; streaky at few views, ~correct at many. Off = '
                                 'original path unchanged.')
        parser.add_argument('--fdk_zero_prior', action='store_true',
                            help='Ablation (arm A3): keep the whole FDK encoder+fusion branch '
                                 'but feed it an all-zero volume instead of the FDK '
                                 'reconstruction. Parameter-matched control that isolates the '
                                 'INFORMATION in the FDK prior from the branch capacity. '
                                 'FiLM conditioning (geometry descriptors) is kept.')
        parser.add_argument('--fdk_film', type=int, default=1,
                            help='FiLM-condition the FDK encoder on a global descriptor '
                                 '[log N, angular spread, mean coverage] so it adapts feature '
                                 'extraction to the FDK reliability (trust detail at high N, '
                                 'robust low-freq at low N). 0 = unconditioned encoder.')
        # FDK reconstruction settings (model's --use_fdk prior AND standalone).
        # Defaults match the validated real-data config: hann + padding + Parker.
        parser.add_argument('--fdk_filter', type=str, default='hann',
                            choices=['ramp', 'hann'],
                            help='FDK filter: hann (windowed, less noise; default) or ramp '
                                 '(Ram-Lak, sharp).')
        parser.add_argument('--fdk_sad', type=float, default=1000.0,
                            help='Source-to-axis distance (mm) for FDK; must match the DRR '
                                 'geometry (renderer default 1000).')
        parser.add_argument('--fdk_vox_chunk', type=int, default=131072,
                            help='Voxel chunk for the FDK backprojection (memory control).')
        parser.add_argument('--fdk_pad', action=argparse.BooleanOptionalAction, default=True,
                            help='Pad the fan axis to next-pow2 >= 2*W before the ramp so the '
                                 'FFT is a LINEAR convolution (no wrap-around). On by default; '
                                 'disable with --no-fdk_pad.')
        parser.add_argument('--fdk_parker', action=argparse.BooleanOptionalAction, default=True,
                            help='Parker short-scan redundancy weighting. On by default; auto '
                                 'falls back to interval weighting for near-full (>270deg) '
                                 'arcs (e.g. full-360 training DRRs). Disable with --no-fdk_parker.')
        parser.add_argument('--fdk_cutoff', type=float, default=1.0,
                            help='FDK ramp cutoff (fraction of Nyquist). 1.0 = full Ram-Lak; '
                                 'lower (e.g. 0.8) = smooth roll-off, less noise.')
        parser.add_argument('--fdk_parker_gamma_sign', type=float, default=-1.0,
                            help='Sign (+1/-1) of the fan angle gamma in the Parker weight, '
                                 'i.e. the detector u-axis orientation convention. Default -1 '
                                 'matches the validated test geometry; the WRONG sign puts the '
                                 'taper on the wrong fan side and leaves left-right shading. '
                                 'Verify on the FDK viz column (uniform background = correct).')
        parser.add_argument('--quick_test_batches', type=int, default=3,
                            help='Number of val batches for the lightweight validation run '
                                 'at each model save (every save_latest_freq iters); not the '
                                 'full val set. 0 disables it.')
        parser.add_argument('--val_frac', type=float, default=0.5,
                            help='Fraction of the held-out 20%% used for VALIDATION '
                                 '(training-time monitoring); the remainder is the RESERVED '
                                 'TEST set that only test.py touches. 0.5 = 10%%/10%%. The '
                                 'train set is unchanged regardless. Must match between train '
                                 'and test.py (persisted to checkpoints/<name>/split_cases.json).')
        parser.add_argument('--val_full_every', type=int, default=10,
                            help='Run the larger full-ish validation every N quick '
                                 'validations (i.e. every N*save_latest_freq iters). '
                                 '0 disables the larger validation.')
        parser.add_argument('--val_full_batches', type=int, default=30,
                            help='Number of val batches for the larger validation '
                                 '(--val_full_every). 0 = the entire validation set.')
        parser.add_argument('--rescan_manifest', action='store_true',
                            help='Force a rebuild of the per-dataset scan manifests '
                                 '(ignore the cache under checkpoints/manifest_cache). '
                                 'Use after adding/regenerating data; otherwise the '
                                 'cached manifest is reused for a fast startup.')
        parser.add_argument('--prefetch_workers', type=int, default=2,
                            help='Background threads that prefetch a case\'s projection '
                                 'views while the GPU trains the previous case. 0 = load '
                                 'synchronously (no prefetch).')
        parser.add_argument('--prefetch_depth', type=int, default=2,
                            help='How many upcoming cases to keep prefetching ahead. '
                                 '>1 lets a projection read longer than one case\'s '
                                 'compute overlap across several cases (hides cold-NFS '
                                 'read spikes). Pool uses max(prefetch_workers, depth).')
        parser.add_argument('--prefetch_view_workers', type=int, default=8,
                            help='Parallel per-view file reads within one case. Each '
                                 'proj_<angle>.npy is a separate cold NFS open '
                                 '(latency-bound); reading them W-wide cuts an N-file '
                                 'read from ~N to ~N/W round-trips. 1 = serial.')
        parser.add_argument('--grad_clip', type=float, default=1.0,
                            help='Max global grad norm for clipping (prevents the gradient '
                                 'explosions that produce NaN loss). 0 disables clipping.')
        parser.add_argument('--ddp_timeout', type=int, default=3600,
                            help='NCCL process-group collective timeout in seconds (default '
                                 '600). Raised so a legitimately long step (large N + '
                                 'iterative warmups) does not trip the watchdog.')
        parser.add_argument('--fuse_view_chunk', type=int, default=32,
                            help='View chunk size for gradient-checkpointed ModLN+FusionViT '
                                 'in the binned model. Bounds the O(N) 2D-fusion activation '
                                 'memory at large projection counts.')

        self.initialized = True
        return parser
    
    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        def convert_arg_line_to_args(arg_line):
            # This allows you to use '#' for comments in your txt file
            for arg in arg_line.split():
                if not arg.strip() or arg.startswith('#'):
                    continue
                yield arg
        
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, fromfile_prefix_chars='@')
            parser.convert_arg_line_to_args = convert_arg_line_to_args
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        #model_option_setter = models.get_option_setter(model_name)
        #parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        #dataset_name = opt.dataset_mode
        #dataset_option_setter = data.get_option_setter(dataset_name)
        #parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser

        # Inference (isTrain=False): replay the trained run's architecture flags
        # from checkpoints/<name>/train_opt.ini so --name alone rebuilds the right
        # network (the model is built from flags, not the checkpoint). CLI still
        # wins; no-op for training, under --use_frozen, or if the file is absent.
        argv = None
        if not getattr(self, 'isTrain', True):
            try:
                from frozen_model_utils import maybe_inject_train_opt_args
                argv = maybe_inject_train_opt_args(parser)
            except Exception:
                argv = None
        return parser.parse_args(argv)

    def print_options_(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def print_options(self, opt):
        """Modified to handle boolean flags without printing 'True' or 'False'"""
        message = '----------------- Options ---------------\n'
        clean_format = ''
        
        for k, v in sorted(vars(opt).items()):
            # 1. Terminal Message (Keep this as-is for your own reading)
            default = self.parser.get_default(k)
            comment = '\t[default: %s]' % str(default) if v != default else ''
            if 'host_name' in k:
                continue
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
            
            # 2. Clean Response File Logic
            # Check the type of the default value to identify booleans
            if isinstance(default, bool):
                # For store_true: Only add the flag if the current value is True
                if v is True and default is False:
                    clean_format += f'--{k}\n'
                # For store_false: Only add the flag if the current value is False
                elif v is False and default is True:
                    clean_format += f'--{k}\n'
            else:
                # For all other types (strings, numbers, paths), keep the two-line format
                if (v is not None) and (v != '') and (k != 'phase'):
                    clean_format += f'--{k}\n{v}\n'

        message += '----------------- End -------------------'
        print(message)

        # Save to disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.ini'.format(opt.phase))  
        
        with open(file_name, 'wt') as opt_file:
            opt_file.write(clean_format)
            
        if opt.phase == 'workflow':
            if opt.not_under_data_root == False:
                template_dir = Path(opt.data_root) / Path(opt.template_dir)
            else:
                template_dir = Path(opt.template_dir)
            template_dir.mkdir(parents=True, exist_ok=True)
            file_name = template_dir / f'{opt.phase}_opt.ini'
            with open(file_name, 'wt') as opt_file:
                opt_file.write(clean_format)

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        self.print_options(opt)

        self.opt = opt
        return self.opt


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
       
def int_or_inf(x):
    if x.lower() == "inf":
        return float("inf")
    return int(x)