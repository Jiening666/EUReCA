import argparse
import sys
from pathlib import Path


_FROZEN_CONTEXT = None


def add_frozen_model_args(parser):
    parser.add_argument(
        '--use_frozen', action='store_true',
        help='Load model weights/options from checkpoints/<name>/frozen instead of the main checkpoint dir.',
    )
    parser.add_argument(
        '--frozen_tag', type=str, default='',
        help='Frozen bundle folder name under checkpoints/<name>/frozen. If omitted with --use_frozen, uses the newest bundle.',
    )
    return parser


def _requested_frozen_context():
    probe = argparse.ArgumentParser(add_help=False, fromfile_prefix_chars='@')
    probe.add_argument('--name', default='test0211')
    probe.add_argument('--checkpoints_dir', default='./checkpoints')
    probe.add_argument('--use_frozen', action='store_true')
    probe.add_argument('--frozen_tag', default='')
    known, _ = probe.parse_known_args()
    if not (known.use_frozen or known.frozen_tag):
        return None
    return known


def resolve_frozen_bundle(checkpoints_dir, name, frozen_tag=''):
    frozen_root = Path(checkpoints_dir) / name / 'frozen'
    if not frozen_root.is_dir():
        raise SystemExit(
            f"[frozen] no frozen bundles found for '{name}' at {frozen_root}. "
            "Run freeze_model.py first, or omit --use_frozen to load the main checkpoint."
        )

    if frozen_tag:
        bundle = frozen_root / frozen_tag
        if not bundle.is_dir():
            available = ', '.join(p.name for p in sorted(frozen_root.iterdir()) if p.is_dir())
            raise SystemExit(
                f"[frozen] requested --frozen_tag '{frozen_tag}' was not found under {frozen_root}."
                f"{' Available: ' + available if available else ''}"
            )
    else:
        bundles = [p for p in frozen_root.iterdir() if p.is_dir()]
        if not bundles:
            raise SystemExit(
                f"[frozen] no frozen bundles found under {frozen_root}. "
                "Run freeze_model.py first, or omit --use_frozen to load the main checkpoint."
            )
        bundle = max(bundles, key=lambda p: p.stat().st_mtime)

    inf = bundle / 'inference_opt.ini'
    ckpt = bundle / 'ckpt'
    if not inf.is_file():
        raise SystemExit(f"[frozen] missing inference options: {inf}")
    ckpt_name_dir = ckpt / name
    if not ckpt_name_dir.is_dir():
        raise SystemExit(f"[frozen] missing frozen checkpoint dir: {ckpt_name_dir}")

    epochs = sorted({
        p.name.split('_net_', 1)[0]
        for p in ckpt_name_dir.glob('*_net_*.pth')
        if '_net_' in p.name
    })
    if not epochs:
        raise SystemExit(f"[frozen] no *_net_*.pth weights found in {ckpt_name_dir}")
    if len(epochs) > 1:
        raise SystemExit(
            f"[frozen] multiple epochs found in {ckpt_name_dir}: {epochs}. "
            "Use a bundle with one frozen epoch."
        )

    return bundle, inf, ckpt, epochs[0]


def maybe_prepend_frozen_code_path():
    global _FROZEN_CONTEXT
    known = _requested_frozen_context()
    if known is None:
        return None

    bundle, inf, ckpt, epoch = resolve_frozen_bundle(
        known.checkpoints_dir, known.name, known.frozen_tag)
    code_dir = bundle / 'code'
    if not (code_dir / 'models').is_dir() or not (code_dir / 'options').is_dir():
        raise SystemExit(
            f"[frozen] frozen code is incomplete at {code_dir}; expected models/ and options/."
        )
    sys.path.insert(0, str(code_dir))
    _FROZEN_CONTEXT = (known.name, bundle, inf, ckpt, epoch)
    print(f"[frozen] importing model/options from: {code_dir}")
    return code_dir


def maybe_inject_frozen_args(parser):
    global _FROZEN_CONTEXT
    known = _requested_frozen_context()
    if known is None:
        return None

    if _FROZEN_CONTEXT is None:
        bundle, inf, ckpt, epoch = resolve_frozen_bundle(
            known.checkpoints_dir, known.name, known.frozen_tag)
        _FROZEN_CONTEXT = (known.name, bundle, inf, ckpt, epoch)
    name, bundle, inf, ckpt, epoch = _FROZEN_CONTEXT

    argv = (
        [f'@{inf}']
        + sys.argv[1:]
        + ['--checkpoints_dir', str(ckpt), '--name', name, '--epoch', epoch]
    )
    parser.set_defaults(frozen_bundle=str(bundle))
    print(f"[frozen] loading frozen bundle: {bundle}")
    return argv


# Architecture / model-structural options replayed from a run's saved
# checkpoints/<name>/train_opt.ini so that --name alone rebuilds the trained
# network. test*.py construct the model from CLI flags (NOT from the checkpoint),
# so these must match the trained run or the state_dict load fails / loads random
# weights. Runtime-only and training-only knobs (losses, LR, augmentation,
# dataset filters, view sampling, resume/continue) are deliberately excluded.
_ARCH_OPT_KEYS = (
    # 2D encoder
    'encoder_backbone',
    # view aggregation + transformer structure
    'use_binned_aggr', 'aggr_bins', 'aggr_layers', 'bin_ray_pe',
    'fuse_depth', 'refine_depth', 'use_rope',
    # reference / FDK injection paths (build modules + forward behaviour)
    'use_ref_inject', 'use_fdk', 'fdk_film', 'fdk_zero_prior',
    # image + volume grid (structural — normally from the geometry .ini)
    'H', 'W', 'psize', 'Nx', 'Ny', 'Nz', 'vol_size',
)


def _tokenise_arg_file(path):
    """One token per non-empty, non-comment line (matches BaseOptions arg files:
    '--key' then its value on the next line; a bare '--flag' for store_true)."""
    toks = []
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if s and not s.startswith('#'):
            toks.append(s)
    return toks


def maybe_inject_train_opt_args(parser, keys=_ARCH_OPT_KEYS):
    """Replay architecture flags from checkpoints/<name>/train_opt.ini.

    Returns a new argv with the saved arch flags prepended BEFORE the user's CLI
    args (argparse takes the last occurrence, so anything on the command line still
    wins), or None to leave sys.argv untouched. No-op under --use_frozen (a frozen
    bundle already carries a curated inference_opt.ini) or when the run has no saved
    train_opt.ini (older runs — the user must pass the flags explicitly)."""
    probe = argparse.ArgumentParser(add_help=False, fromfile_prefix_chars='@')
    probe.add_argument('--name', default='test0211')
    probe.add_argument('--checkpoints_dir', default='./checkpoints')
    probe.add_argument('--use_frozen', action='store_true')
    probe.add_argument('--frozen_tag', default='')
    known, _ = probe.parse_known_args()
    if known.use_frozen or known.frozen_tag:
        return None

    opt_ini = Path(known.checkpoints_dir) / known.name / 'train_opt.ini'
    if not opt_ini.is_file():
        return None

    toks = _tokenise_arg_file(opt_ini)
    keyset = set(keys)
    injected, i, n = [], 0, len(toks)
    while i < n:
        t = toks[i]
        if not t.startswith('--'):
            i += 1
            continue
        # value = the next token unless it's another flag (a bare store_true flag)
        has_val = (i + 1 < n) and not toks[i + 1].startswith('--')
        if t[2:] in keyset:
            injected.append(t)
            if has_val:
                injected.append(toks[i + 1])
        i += 2 if has_val else 1

    if not injected:
        return None
    print(f"[opt] replaying architecture from {opt_ini}:\n"
          f"      {' '.join(injected)}\n"
          f"      (command-line flags still override these)")
    return injected + sys.argv[1:]
