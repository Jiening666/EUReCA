from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')

        parser.add_argument('--print_freq', type=int, default=1, help='frequency of showing training results on console')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--test_epoch_freq', type=int, default=1, help='frequency of testing the model at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--load_iter', type=int, default=-1, help='load the latest model iter')
        parser.add_argument('--epoch', type=str, default='latest', help='load the latest model epoch')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--n_epochs', type=int, default=50, help='total training epochs; for cosine this is the CosineAnnealingLR T_max. Matches ILV (n_epoch=50).')
        parser.add_argument('--n_epochs_decay', type=int, default=0, help='extra linear-decay epochs (used only by lr_policy=linear). Keep 0 for cosine so the loop runs exactly n_epochs == T_max (nonzero would step the cosine past T_max and raise the LR again).')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='cosine_warmup', help='learning rate policy. [linear | step | plateau | cosine | cosine_warmup]. cosine_warmup = 5-epoch linear ramp then cosine to 0 over n_epochs (ILV-style; stabilizes the pretrained-backbone + cold-head start).')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--resume', action='store_true',
                            help='Full crash-recovery resume of the SAME run: restore model + '
                                 'optimizers + schedulers + RNG + step/epoch counters (and the '
                                 'wandb run) via accelerator.save_state/load_state. Differs from '
                                 '--continue_train, which only loads network weights (tolerant, '
                                 'for cross-run / architecture changes).')
        # Ablation-study subset: subsample the TRAIN split (after the standard
        # 80/10/10 split; val/test untouched) to a fixed number of cases,
        # stratified by source dataset (the D{i}_ case-id prefix). Deterministic
        # given the seed, so every ablation arm run with the same flags trains
        # on the identical case list; the chosen ids are recorded in
        # split_cases.json under 'train_subset'.
        parser.add_argument('--train_subset_n', type=int, default=0,
                            help='if > 0, subsample the train split to this many cases '
                                 '(stratified by dataset). 0 = use the full train split.')
        parser.add_argument('--train_subset_seed', type=int, default=0,
                            help='RNG seed for the train-subset draw.')
        parser.add_argument('--seed', type=int, default=42,
                            help='global RNG seed (torch/numpy/random/cuda), set at '
                                 'startup before building loaders/model. Identical across '
                                 'ablation arms -> identical case order + aug picks + weight '
                                 'init, so same-step wandb slices are comparable.')
        parser.add_argument('--train_subset_nested', action='store_true',
                            help='Data-size ablation: build a deterministic '
                                 'fractional-rank stratified ORDER of the train cases '
                                 'and take the first --train_subset_n as the subset, so '
                                 'any two subset sizes are NESTED (smaller subset of '
                                 'larger) and each is stratified by source dataset. '
                                 'Without this, --train_subset_n draws an independent '
                                 'random subset per size (not nested).')
        # Iteration-based schedule (data-size ablation): matches the LR schedule
        # AND the stop point in ITERATION space across arms of different dataset
        # size. When --total_iters > 0 it overrides the epoch-based schedule/stop:
        # LR is set every optimizer step from total_iters (warmup then cosine to
        # eta_min), and training stops at total_iters. The default (0) leaves the
        # original epoch-based cosine_warmup untouched.
        parser.add_argument('--total_iters', type=int, default=0,
                            help='if > 0, train for exactly this many optimizer '
                                 'iterations (in total_iters units = batch_size per '
                                 'step) and drive the LR schedule off it, ignoring '
                                 '--n_epochs. Makes runs with different #train cases '
                                 'directly comparable. 0 = epoch-based (unchanged).')
        parser.add_argument('--warmup_iters', type=int, default=0,
                            help='linear LR warmup length (in total_iters units) for '
                                 '--total_iters mode. 0 = no warmup.')
        parser.add_argument('--snapshot_freq', type=int, default=0,
                            help='if > 0, save a NUMBERED net-weights snapshot every this '
                                 'many iters (under <ckpt>/snap/iter_N/) for post-hoc '
                                 'best-val checkpoint selection. Should be a multiple of '
                                 '--save_latest_freq. 0 = off.')
        parser.add_argument('--finetune_lr', type=float, default=0.0,
                            help='if > 0 (with --total_iters, on a --resume), use a '
                                 'CONSTANT LR multiplier of this value (× initial_lr) '
                                 'instead of the cosine schedule — a low-LR continuation '
                                 'past the original budget with no cosine warm-restart '
                                 'jump. E.g., 0.05 = 5%% of peak LR held flat. 0 = cosine.')
        # parser.add_argument('--ptv_diameter', type=int, required=True, help='ptv diameter in mm')
        self.isTrain = True
        return parser
