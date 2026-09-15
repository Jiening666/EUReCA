from .train_options import TrainOptions


class TestOptions(TrainOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = TrainOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        #parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=1000, help='how many test images to run')
        parser.add_argument('--not_trace_all_model', action='store_true', help='don\'t make pt file for all saved intermedia models')
        # rewrite devalue values
        parser.add_argument('--num_images_list', type=int, nargs='+', default=[10, 20, 30, 40, 50], help='list of ints')
        parser.add_argument('--log_test_visual_freq', type=int, default=10, help='log visualization slices every N batches during test (0 = disabled)')
        parser.add_argument('--max_test_size', type=int, default=500, help='maximum number of test cases to evaluate (caps the 20%% split)')
        parser.add_argument('--eval_split', type=str, default='test', choices=['test', 'val'],
                            help="which recorded split to evaluate: 'test' (reserved, default) "
                                 "or 'val' (for best-val checkpoint selection; never touches test).")
        parser.add_argument('--save_eval_nifti', action='store_true', help='save float32 clip(0,1) pred + gt NIfTI per case for each (num_images, angle_mode) config, so cbct_recon_compare/eval/unified_eval.py can score this model with the same metric code as the comparators')
        parser.add_argument('--eval_nifti_dir', type=str, default='', help='output root for --save_eval_nifti (default: <checkpoints_dir>/<name>/eval_nifti)')
        parser.add_argument('--eval_aug_index', type=int, default=-1, help='pin test-time aug-folder selection to this index (0 = identity aug hal_0, matching the comparators); -1 = legacy random selection')

        parser.set_defaults(model='test')
        parser.set_defaults(phase='test')
        # To avoid cropping, the load_size should be the same as crop_size
        self.isTrain = False
        return parser
