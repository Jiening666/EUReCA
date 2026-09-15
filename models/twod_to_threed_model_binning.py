"""
Angular-binned two-stage view aggregation variant of twoD2threeDModel,
plus attention-derived reference alignment and multi-scale prior injection.

──────────────────────────── Binned aggregation ─────────────────────────────
The original per-voxel aggregators operate directly on the N view axis
(CLS-transformer over N+1 tokens at coarse, global softmax-weighted sum at
mid/fine).  At N in the hundreds this is slow / OOM and inaccurate (softmax
weights dilute toward 1/N).  Here, K angular bins (--aggr_bins, default 36)
in the absolute gantry frame give a two-stage scheme:

  Stage 1 (within-bin, O(N) total): learned scalar score + masked softmax
    reduces each bin's views to one feature per voxel.  Out-of-FOV views are
    masked out instead of contributing silent zeros.  Processed bin-by-bin,
    so the N axis is never materialised at once.
  Stage 2 (across-bin, O(K), constant in N): coarse = transformer over CLS +
    K tokens with a learned empty-token for unobserved bins (missing wedge is
    encoded, not dropped); mid/fine = scored softmax over occupied bins.
    Every bin token carries [sin θk, cos θk, log1p(count), conj_covered];
    the conjugate flag (k ↔ k+K/2) distinguishes Parker short scans from
    genuine limited-angle scans.

The coverage mask's angular spread is computed on 2θ (directions are lines),
so a 211° short scan is correctly credited as well-covered.

────────────────────── Reference alignment + injection ──────────────────────
The parent model's reference conduit is the 16³ coarse grid only: ref_f_1_4
and ref_f_1_8 were computed and discarded, and the source decoder's mid/fine
skips contained no reference information — so high-frequency prior content
(the thing sparse views cannot recover) never reached the reconstruction.

This model:
  1. estimates a rough affine ref→source transform from a dedicated
     correspondence cross-attention at the coarse scale (soft-argmax of the
     attention map + confidence-weighted ridge least-squares fit, pulled
     toward identity by --align_ridge).  The fit is differentiable; a
     train-time NCC loss between the affine-warped reference volume and the
     ground-truth source volume (weight --lambda_align) plus a near-affine
     attention-consistency term make the attention geometric.
  2. warps ref_f_1_8 / ref_f_1_4 feature volumes by the fitted transform
     (features are encoded ONCE in the reference frame; warping the feature
     maps is exact for translation and accurate for the small rotations in
     scope) and gate-fuses them into the SOURCE decoder skips.  Fusion convs
     are zero-initialised so injection starts as an exact identity.
  3. injects the UNWARPED ref features into the reference decode path's
     skips (its target is the reference frame) — opening the conduit for the
     vols_ref sanity check: with the conduit open, vols_ref error should
     drop far below the source error.

DDP / memory notes:
  * Both aggregation stages run INSIDE the aggregator modules' forward(), so
    when accelerate wraps each net in DistributedDataParallel every parameter
    is used within the wrapped call.  netalign / netskipfuse_* are called
    exactly once per paired forward (source+ref batched).  The model only
    reads module attributes through getattr(net, 'module', net).
  * Each voxel chunk's aggregation forward is gradient-checkpointed: stored
    activations are O(total * C), independent of N.
  * The alignment fit detaches its feature inputs: alignment objectives
    cannot distort the backprojected features, and the reconstruction loss
    cannot game the geometry; gradients still reach the align parameters
    through the attention → fit → warp chain.

Usage: --use_binned_aggr (construction goes through models.create_recon_model).
Options: --aggr_bins, --aggr_vox_chunk, --lambda_align, --align_ridge.

Note: 'aggr', 'aggr_fine', 'aggr_mid' checkpoints are NOT weight-compatible
with the original model, and 'align', 'skipfuse_mid', 'skipfuse_fine' are new.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .twod_to_threed_model import twoD2threeDModel, RefVolumeEncoder
from .fdk import DifferentiableFDK, CoverageGatedFusion
from threeD_geometry import map_3d_grid_to_2d_batch


# ─────────────────────────── Aggregator modules ─────────────────────────────

class _BinnedAggregatorBase(nn.Module):
    """
    Shared Stage-1 machinery (within-bin masked-softmax reduction) and bin
    geometry.  Subclasses implement Stage 2 in forward().  Keeping Stage 1
    inside the module means all parameters (score_view, geom_proj, ...) are
    exercised within a single wrapped forward call under DDP.
    """
    def __init__(self, embed_dim, K, ray_pe=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.K = K
        # Stage-1 scorer: per-view importance logit inside each bin.
        self.score_view = nn.Linear(embed_dim, 1, bias=False)
        self.geom_proj  = nn.Linear(4, embed_dim)

        # Optional per-(voxel, bin) Plücker ray positional encoding for Stage 2.
        # Encodes the actual ray from bin-k's source to the voxel — per-voxel
        # (cone-beam divergence), unlike the bin-center [sinθ,cosθ] in geom.
        self.ray_pe = ray_pe
        if ray_pe:
            self.ray_proj = nn.Sequential(
                nn.Linear(6, embed_dim), nn.GELU(),
                nn.Linear(embed_dim, embed_dim))
            nn.init.zeros_(self.ray_proj[-1].weight)   # zero-init -> identity at start
            nn.init.zeros_(self.ray_proj[-1].bias)
            # unit source direction at each bin centre, volume frame: SAD·(cosθ,sinθ,0)
            c = (torch.arange(K) + 0.5) * (2.0 * math.pi / K)
            self.register_buffer('bin_dir',
                                 torch.stack([c.cos(), c.sin(), torch.zeros_like(c)], -1))  # (K,3)

    def _ray_pe(self, vox_xyz, rho):
        """vox_xyz: (B, V, 3) mm voxel coords; rho: source radius (mm).
        Returns (B, V, K, C) Plücker ray PE for every (voxel, bin)."""
        v = (vox_xyz / rho).unsqueeze(2)                        # (B,V,1,3) in units of rho
        d = v - self.bin_dir.view(1, 1, self.K, 3)             # source(unit)->voxel
        d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-6)   # (B,V,K,3) unit direction
        m = torch.cross(v.expand(-1, -1, self.K, -1), d, dim=-1)  # (B,V,K,3) Plücker moment
        return self.ray_proj(torch.cat([d, m], dim=-1))        # (B,V,K,C)

    def _bin_geom(self, bin_count):
        """
        bin_count: (B, V, K) per-voxel in-FOV view count per bin.
        Returns geom (B, V, K, 4) and occupied (B, V, K) bool.
        conj_covered: this bin OR its conjugate (k + K/2) is occupied — a
        Parker short scan covers nearly all bins through conjugate rays.
        """
        K = self.K
        occupied = bin_count > 0
        conj = occupied | occupied.roll(K // 2, dims=-1)
        centers = (torch.arange(K, device=bin_count.device,
                                dtype=bin_count.dtype) + 0.5) * (2.0 * math.pi / K)
        geom = torch.stack([
            centers.sin().expand_as(bin_count),
            centers.cos().expand_as(bin_count),
            torch.log1p(bin_count),
            conj.to(bin_count.dtype),
        ], dim=-1)
        return geom, occupied

    def _stage1(self, feat_2d, grid_chunk, kidx, B, N, need_mean_max=False):
        """
        Within-bin reduction for one voxel chunk, fully vectorised: ONE
        grid_sample over all views, then a segmented (scatter-based) softmax
        grouped by bin.  Mathematically identical to a bin-by-bin masked
        softmax but ~K× fewer kernel launches — an earlier bin-loop version
        spent most of its time launching tiny kernels, in forward AND in the
        checkpoint recompute during backward.

        The (N, C, V) transient is bounded by the caller's voxel chunk size
        (and freed per chunk under checkpointing); it is NOT retained for
        backward thanks to the checkpoint around the caller.

        feat_2d    : (B*N, C, H, W)
        grid_chunk : (B*N, V, 2) on device — this chunk's voxel grid coords
        kidx       : (B, N) long, bin index per view
        Returns bin_feats (B, V, K, C), bin_count (B, V, K)
        and, if need_mean_max, the original all-view mean/max (B, V, C)
        computed in the same pass (semantics identical to the parent's
        reduce='mean'/'max', including zero-padding outside the FOV).
        """
        C, K = self.embed_dim, self.K
        device = grid_chunk.device
        V = grid_chunk.shape[1]

        # One vectorised sample of every view at this chunk's voxels.
        s = F.grid_sample(feat_2d.float(), grid_chunk.unsqueeze(1).float(),
                          mode='bilinear', padding_mode='zeros',
                          align_corners=True).squeeze(2)           # (B*N, C, V)
        in_fov = (grid_chunk.abs() <= 1.0).all(dim=-1)             # (B*N, V)
        # .float(): under bf16 autocast the Linear returns BFloat16, which
        # would mismatch the fp32 scatter buffers below — and the softmax
        # accumulation is safer in fp32 anyway (s itself is already fp32).
        logits = self.score_view(s.transpose(1, 2)).squeeze(-1).float()
        logits = logits.masked_fill(~in_fov, float('-inf'))

        s      = s.view(B, N, C, V)
        logits = logits.view(B, N, V)
        in_fov = in_fov.view(B, N, V)

        bin_feats, bin_count = [], []
        if need_mean_max:
            sums, maxs = [], []
        for b in range(B):
            kb = kidx[b].to(device)                                # (N,)

            # Segmented softmax over views grouped by bin.  The per-bin max
            # is DETACHED: softmax is shift-invariant, so the subtracted max
            # is a constant offset and needs no gradient (index_reduce's
            # backward support varies across torch versions).
            m = torch.full((K, V), float('-inf'), device=device)
            m.index_reduce_(0, kb, logits[b].detach(), 'amax',
                            include_self=True)
            m = torch.nan_to_num(m, neginf=0.0)                    # empty bins
            e = torch.exp(logits[b] - m[kb])                       # (N, V); OOF → 0
            denom = torch.zeros(K, V, device=device).index_add(0, kb, e)
            numer = (torch.zeros(K, C, V, device=device)
                     .index_add(0, kb, s[b] * e.unsqueeze(1)))     # (K, C, V)
            bf = numer / denom.unsqueeze(1).clamp_min(1e-12)
            bin_feats.append(bf.permute(2, 0, 1))                  # (V, K, C)
            cnt = (torch.zeros(K, V, device=device)
                   .index_add(0, kb, in_fov[b].float()))
            bin_count.append(cnt.transpose(0, 1))                  # (V, K)

            if need_mean_max:
                sums.append(s[b].sum(dim=0).transpose(0, 1))       # (V, C)
                maxs.append(s[b].amax(dim=0).transpose(0, 1))      # (V, C)

        bin_feats = torch.stack(bin_feats)                         # (B, V, K, C)
        bin_count = torch.stack(bin_count)                         # (B, V, K)
        if need_mean_max:
            return (bin_feats, bin_count,
                    torch.stack(sums) / N, torch.stack(maxs))
        return bin_feats, bin_count


class BinnedTransformerAggregator(_BinnedAggregatorBase):
    """
    Coarse aggregator: Stage 1 + Stage-2 transformer over CLS + K bin tokens
    (constant length).  Empty bins keep their slot with a shared learned
    empty-token so the attention can encode *which* directions were never
    observed (missing wedge), instead of being silently dropped.

    forward() handles one voxel chunk end-to-end and also returns the
    all-view mean/max for the parent's cat([attn, mean, max]) fusion.
    """
    def __init__(self, embed_dim=768, K=36, num_heads=8, num_layers=2, ray_pe=False):
        super().__init__(embed_dim, K, ray_pe=ray_pe)
        self.empty_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )

    def forward(self, feat_2d, grid_chunk, kidx, B, N, vox_xyz=None, rho=None):
        """
        feat_2d    : (B*N, C, H, W) fused 2D features
        grid_chunk : (B*N, V, 2)
        kidx       : (B, N) long
        vox_xyz    : (B, V, 3) mm voxel coords for this chunk (ray PE; optional)
        rho        : source radius (mm) for the ray PE
        Returns agg (B, V, C), view_mean (B, V, C), view_max (B, V, C)
        """
        C, K = self.embed_dim, self.K
        bin_feats, bin_count, v_mean, v_max = self._stage1(
            feat_2d, grid_chunk, kidx, B, N, need_mean_max=True)
        geom, occupied = self._bin_geom(bin_count)
        V = grid_chunk.shape[1]

        bf  = bin_feats.reshape(B * V, K, C)
        occ = occupied.reshape(B * V, K)
        x = torch.where(occ.unsqueeze(-1), bf, self.empty_embed.expand_as(bf))
        x = x + self.geom_proj(geom.reshape(B * V, K, 4))
        if self.ray_pe and vox_xyz is not None:
            # per-(voxel,bin) ray PE; added to occupied AND empty bins so the
            # transformer also sees the geometry of the missing-wedge directions.
            x = x + self._ray_pe(vox_xyz, rho).reshape(B * V, K, C)
        cls = self.cls_token.expand(x.size(0), 1, -1)
        x = self.transformer(torch.cat([cls, x], dim=1))
        agg = self.fc(x[:, 0]).view(B, V, C)
        return agg, v_mean, v_max


class BinnedViewWeightAggregator(_BinnedAggregatorBase):
    """
    Mid/fine aggregator: Stage 1 + O(K) scored softmax sum over occupied bins
    only.  Geometry embedding is added before scoring so both the weights and
    the content are direction-aware.  All-empty voxels output a learned
    constant.
    """
    def __init__(self, embed_dim, K=36, **kwargs):
        super().__init__(embed_dim, K)
        self.score_bin = nn.Linear(embed_dim, 1, bias=False)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )

    def forward(self, feat_2d, grid_chunk, kidx, B, N):
        """Returns (attn, mean, max), each (B, V, C) — the same attn+mean+max
        triple the coarse aggregator provides, for one voxel chunk."""
        C, K = self.embed_dim, self.K
        bin_feats, bin_count, v_mean, v_max = self._stage1(
            feat_2d, grid_chunk, kidx, B, N, need_mean_max=True)
        geom, occupied = self._bin_geom(bin_count)
        V = grid_chunk.shape[1]

        x   = bin_feats.reshape(B * V, K, C) + self.geom_proj(geom.reshape(B * V, K, 4))
        occ = occupied.reshape(B * V, K)
        logits = self.score_bin(x).squeeze(-1)                  # (B*V, K)
        logits = logits.masked_fill(~occ, float('-inf'))
        w = torch.nan_to_num(logits.softmax(dim=1), nan=0.0)   # all-empty → 0
        attn = self.fc((x * w.unsqueeze(-1)).sum(dim=1)).view(B, V, C)
        return attn, v_mean, v_max


# ─────────────────────────────── Model ──────────────────────────────────────

def _unwrap(net):
    """Underlying module of a DDP/DataParallel wrapper (or the net itself)."""
    return getattr(net, 'module', net)


class twoD2threeDBinningModel(twoD2threeDModel):

    def __init__(self, opt):
        super().__init__(opt)

        K = getattr(opt, 'aggr_bins', 36)
        if K % 2 != 0:
            raise ValueError(f'--aggr_bins must be even for conjugate-bin '
                             f'pairing (k <-> k+K/2), got {K}')

        aggr_layers = getattr(opt, 'aggr_layers', 2)
        # Same K at all three scales for now; constructors take K so the fine
        # scale can be ablated independently (e.g. 72) without other changes.
        self.bin_ray_pe = bool(getattr(opt, 'bin_ray_pe', False))
        self.netaggr      = BinnedTransformerAggregator(
            embed_dim=self.C_f, K=K, num_layers=aggr_layers, ray_pe=self.bin_ray_pe)
        # mid/fine aggregators keep the bin-centre geom encoding only (no ray PE).
        self.netaggr_fine = BinnedViewWeightAggregator(embed_dim=self.C_a, K=K)
        self.netaggr_mid  = BinnedViewWeightAggregator(embed_dim=self.C_b, K=K)
        # Fuse attn+mean+max at mid/fine (mirrors netrefine_coarse at coarse).
        self.netrefine_mid  = nn.Conv3d(self.C_b * 3, self.C_b, 1)
        self.netrefine_fine = nn.Conv3d(self.C_a * 3, self.C_a, 1)
        self.model_names = self.model_names + ['refine_mid', 'refine_fine']

        # Reference alignment + skip-fusion branch removed: this is an
        # FDK-only model with no (unaligned) reference input. _monitor_names
        # starts empty; the FDK block below appends its own monitors.
        self._monitor_names = ()

        # ── FDK prior injection (opt-in via --use_fdk; additive, in-frame) ──
        # A Feldkamp backprojection of the SOURCE's own projections is
        # reconstructed (no params, no_grad), encoded by a dedicated 3D CNN,
        # and injected into the decoder skips with a coverage-conditioned
        # gate.  Needs NO alignment (already in the source frame) and carries
        # the current anatomy (no inter-scan mismatch); reliability grows with
        # N, so the gate leans on it more as coverage rises.  Runs on EVERY
        # batch (no reference needed) → base optimizer.
        # Whether the reference-injection path (alignment + skipfuse, incl. the
        # null-ref skip on single batches) is used at all.  Off = a clean
        # FDK-only / projection-only experiment: no align, no null-ref skip.
        self.use_ref_inject = bool(getattr(opt, 'use_ref_inject', True))

        self.use_fdk = bool(getattr(opt, 'use_fdk', False))
        self.fdk_vol = None      # latest injected prior volume, for visualization
        if self.use_fdk:
            self.fdk = DifferentiableFDK(
                opt.Nx, opt.Ny, opt.Nz, opt.vol_size,
                opt.H, opt.W, opt.psize,
                SAD=getattr(opt, 'fdk_sad', 1000.0),
                filter=getattr(opt, 'fdk_filter', 'ramp'),
                vox_chunk=getattr(opt, 'fdk_vox_chunk', 131072),
                pad=getattr(opt, 'fdk_pad', True),
                cutoff=getattr(opt, 'fdk_cutoff', 1.0),
                parker=getattr(opt, 'fdk_parker', True),
                parker_gamma_sign=getattr(opt, 'fdk_parker_gamma_sign', -1.0))
                # Parker auto-disables for near-full (>270deg) arcs, so it is a
                # no-op on full-360 training DRRs and active on short real scans.
            # Global FiLM conditioning of the FDK encoder on [log N, angular
            # spread, mean coverage] so it adapts feature extraction to the
            # prior's reliability (trust detail at high N, robust low-freq at
            # low N).  cond_dim=0 (--fdk_film 0) restores the unconditioned form.
            self.fdk_film = bool(getattr(opt, 'fdk_film', True))
            self.netfdk_encoder   = RefVolumeEncoder(
                C_a=self.C_a, C_b=self.C_b, C_f=self.C_f,
                cond_dim=3 if self.fdk_film else 0)
            self.netfdkfuse_coarse = CoverageGatedFusion(self.C_f)
            self.netfdkfuse_mid    = CoverageGatedFusion(self.C_b)
            self.netfdkfuse_fine   = CoverageGatedFusion(self.C_a)
            self.model_names = self.model_names + [
                'fdk_encoder', 'fdkfuse_coarse', 'fdkfuse_mid', 'fdkfuse_fine']
            fdk_monitors = ('M_fdk_gate_coarse', 'M_fdk_gate_mid', 'M_fdk_gate_fine',
                            'M_fdk_inj_coarse', 'M_fdk_inj_mid', 'M_fdk_inj_fine')
            self._monitor_names = self._monitor_names + fdk_monitors
            self.loss_names = self.loss_names + list(fdk_monitors)
            for _m in fdk_monitors:
                setattr(self, 'loss_' + _m, torch.tensor(0.0))

        # Rebuild the optimizer: the parent's param groups still reference the
        # replaced aggregator modules and lack the new FDK nets.  FDK-only model
        # → a single optimizer (no reference/align sub-nets that needed a
        # separate, paired-only one).  The pretrained 2D-encoder backbone gets
        # its own param group at a reduced LR (it is pretrained; the rest of the
        # network is cold); when no pretrained backbone is used the group is
        # simply absent.
        def _all_named_params(model):
            for mname in model.model_names:
                if isinstance(mname, str):
                    net = getattr(model, 'net' + mname)
                    for pname, p in net.named_parameters():
                        yield f'net{mname}.{pname}', p

        bb_mult = getattr(opt, 'backbone_lr_mult', 0.1)
        bb_params, head_params = [], []
        for n, p in _all_named_params(self):
            if not p.requires_grad:
                continue
            (bb_params if 'netencoder.backbone.' in n else head_params).append(p)
        groups = [{'params': head_params, 'lr': opt.lr}]
        if bb_params:
            groups.append({'params': bb_params, 'lr': opt.lr * bb_mult})
        self.optimizer_base = torch.optim.AdamW(groups, lr=opt.lr, betas=(0.9, 0.95))
        self.optimizers = [self.optimizer_base]

    # ── Binning helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _bin_assignment(angles, K):
        """angles: (B, N) degrees (CPU) → (B, N) long bin index in [0, K)."""
        ang = torch.remainder(angles.float(), 360.0)
        return torch.div(ang * K, 360.0, rounding_mode='floor').long().clamp_(0, K - 1)

    def _prior_volume(self, prior_vol, projections, angles, device):
        """Return the in-frame prior volume to inject (B,1,Nx,Ny,Nz, [0,1]).

        prior_vol given (the model's own previous output on iterations >=2) is
        used as-is; otherwise (iteration 1) the source FDK is reconstructed.
        self.fdk has no learnable params and is not in model_names, so move its
        buffers to the input device here (idempotent).
        """
        if prior_vol is not None:
            return prior_vol
        if getattr(self.opt, 'fdk_zero_prior', False):
            # Ablation arm A3: parameter-matched control — the encoder+fusion
            # branch runs unchanged but carries no FDK information. FiLM cond
            # (geometry, not FDK) is left to the caller as usual.
            fdk_vol = torch.zeros(
                (projections.shape[0], 1, self.opt.Nx, self.opt.Ny, self.opt.Nz),
                device=device, dtype=projections.dtype)
            self.fdk_vol = fdk_vol.squeeze(1)
            return fdk_vol
        self.fdk.to(device)
        with torch.no_grad():
            fdk_vol = self.fdk(projections, angles)         # (B,1,Nx,Ny,Nz) [0,1]
        # Keep the ACTUAL FDK for the visualization column — set only when FDK
        # is computed (iteration 1), so it is not overwritten by the
        # previous-output prior on later iterations.
        self.fdk_vol = fdk_vol.squeeze(1).detach()          # (B,Nx,Ny,Nz)
        return fdk_vol

    def _fdk_cond(self, angles, coverage_mask, N, device):
        """Global FiLM descriptor for the FDK encoder, (B, 3):
        [log N (scaled), angular spread on 2theta, mean coverage].  Tells the
        encoder how reliable this FDK is so it can adapt feature extraction."""
        B = angles.shape[0]
        thetas2 = angles.float() * (math.pi / 90.0)                 # 2*theta (lines, not rays)
        R = torch.sqrt(thetas2.cos().mean(1) ** 2 + thetas2.sin().mean(1) ** 2)
        spread = (1.0 - R).to(device)                               # (B,) 0=clustered .. 1=uniform
        logN = torch.full((B,), math.log(max(int(N), 1)) / 5.0, device=device)  # ~[0.4,0.8] for N 8..50
        mean_cov = coverage_mask.float().mean(dim=1)                # (B,)
        return torch.stack([logN, spread, mean_cov], dim=1)         # (B, 3)

    def _inject_prior(self, prior_vol, V_f_2D, V_f_b, V_f_a, coverage_mask,
                      iter_level, B, device, fdk_cond=None):
        """
        Encode an in-frame prior volume (FDK or the model's previous output)
        and inject its features into the coarse (16³) feature AND the mid/fine
        decoder skips, each with a coverage- and iteration-conditioned gate.
        In-frame → no warp.
        coverage_mask: (B, L_coarse) at the 16³ grid; native for coarse,
        upsampled for mid/fine.
        fdk_cond: (B, 3) global descriptor [log N, angular_spread, mean cov] that
        FiLM-conditions the FDK encoder (None = unconditioned).
        """
        f14, f18, f116 = self.netfdk_encoder(prior_vol, fdk_cond)   # 64³, 32³, 16³

        L_fx = self.opt.Nx // 16
        L_fy = self.opt.Ny // 16
        L_fz = self.opt.Nz // 16
        cov = coverage_mask.view(B, 1, L_fx, L_fy, L_fz).float()
        cov_coarse = F.interpolate(cov, size=V_f_2D.shape[2:],
                                   mode='trilinear', align_corners=False)
        cov_mid    = F.interpolate(cov, size=V_f_b.shape[2:],
                                   mode='trilinear', align_corners=False)
        cov_fine   = F.interpolate(cov, size=V_f_a.shape[2:],
                                   mode='trilinear', align_corners=False)

        V_f_2D = self.netfdkfuse_coarse(V_f_2D, f116, cov_coarse, iter_level)
        V_f_b  = self.netfdkfuse_mid(V_f_b, f18, cov_mid, iter_level)
        V_f_a  = self.netfdkfuse_fine(V_f_a, f14, cov_fine, iter_level)

        fc, fm, ff = (_unwrap(self.netfdkfuse_coarse),
                      _unwrap(self.netfdkfuse_mid), _unwrap(self.netfdkfuse_fine))
        self.loss_M_fdk_gate_coarse = fc.last_gate.mean()
        self.loss_M_fdk_gate_mid    = fm.last_gate.mean()
        self.loss_M_fdk_gate_fine   = ff.last_gate.mean()
        self.loss_M_fdk_inj_coarse  = fc.last_inj_ratio.mean()
        self.loss_M_fdk_inj_mid     = fm.last_inj_ratio.mean()
        self.loss_M_fdk_inj_fine    = ff.last_inj_ratio.mean()
        return V_f_2D, V_f_b, V_f_a

    def generate_visualization_slices(self, pred_vol, real_vol):
        """Parent GT/pred slice panel + an extra FDK column when FDK ran.

        Appends 'fdk_slices' (axial/coronal/sagittal, same orientation as the
        GT/pred columns) so the logged image shows GT | pred | FDK.  Added only
        when an FDK volume is available (use_fdk), so the original 2-column
        panel is unchanged otherwise.
        """
        vis = super().generate_visualization_slices(pred_vol, real_vol)
        fdk = getattr(self, 'fdk_vol', None)
        if fdk is not None:
            mx, my, mz = fdk.shape[1] // 2, fdk.shape[2] // 2, fdk.shape[3] // 2
            def _n(x):
                return x.clamp(0.0, 1.0)
            ax = torch.flip(torch.rot90(_n(fdk[0, :, :, mz]), k=3, dims=[0, 1]), dims=[-1])
            co = torch.rot90(_n(fdk[0, :, my, :]), k=1, dims=[0, 1])
            sa = torch.rot90(_n(fdk[0, mx, :, :]), k=1, dims=[0, 1])
            vis["fdk_slices"] = torch.cat([ax, co, sa], dim=0)
        return vis

    # ── Overridden aggregation paths ────────────────────────────────────────

    def _fast_aggregate_attn(self, feat_2d, angles, B, N, Lx, Ly, Lz, C,
                             scale_3d, scale_2d, aggr_net, device,
                             attn_chunk_size=None):
        """
        Binned mid/fine aggregation.  Replaces the parent's one-shot
        (B*total, N, C) materialisation with voxel-chunked two-stage binning.
        Each chunk is one gradient-checkpointed forward of the (possibly
        DDP-wrapped) aggregator, so stored activations are O(total * C)
        regardless of N.  attn_chunk_size is accepted for signature
        compatibility and ignored.
        """
        total    = Lx * Ly * Lz
        vox_size = self.opt.vol_size * scale_3d
        H_feat, W_feat = feat_2d.shape[-2:]

        ip, iq = map_3d_grid_to_2d_batch(
            Lx, Ly, Lz, vox_size, vox_size, vox_size,
            angles.cpu().numpy(), H_feat, W_feat,
            self.opt.psize * scale_2d,
        )
        ip_t = torch.as_tensor(ip, dtype=torch.float32).view(B * N, total)
        iq_t = torch.as_tensor(iq, dtype=torch.float32).view(B * N, total)
        grid_flat = torch.stack((iq_t, ip_t), dim=-1).to(device)   # (B*N, total, 2)

        K = _unwrap(aggr_net).K
        kidx = self._bin_assignment(angles, K)
        # Scale the chunk down with N so the vectorised stage-1 transient
        # (~N × C × chunk floats) stays roughly constant; --aggr_vox_chunk is
        # the size used up to N=64.
        vox_chunk = getattr(self.opt, 'aggr_vox_chunk', 8192)
        vox_chunk = max(1024, int(vox_chunk * 64 / max(N, 64)))

        a_outs, m_outs, x_outs = [], [], []
        for start in range(0, total, vox_chunk):
            end = min(start + vox_chunk, total)
            agg, v_mean, v_max = checkpoint(
                aggr_net, feat_2d, grid_flat[:, start:end],
                kidx, B, N, use_reentrant=False)                   # each (B, V, C)
            a_outs.append(agg); m_outs.append(v_mean); x_outs.append(v_max)

        def _vol(outs):
            return (torch.cat(outs, dim=1).view(B, Lx, Ly, Lz, C)
                    .permute(0, 4, 1, 2, 3).contiguous())          # (B, C, Lx, Ly, Lz)
        return _vol(a_outs), _vol(m_outs), _vol(x_outs)            # attn, mean, max

    def _encode_and_aggregate_coarse(self, projections, angles, B, N, H, W, device):
        """
        Same as the parent, with two changes:
          1. attn / mean / max coarse reductions all come from one binned pass
             (the parent ran three separate sweeps over all N views).
          2. coverage angular spread is computed on 2*theta.
        Returns (V_f_2D, encoded, coverage_mask) — identical contract.
        """
        encoded  = self._encode_views(projections)                 # chunked + checkpointed over views
        f_coarse = encoded['coarse']                               # (B, N, C_f, H_c, W_c)
        _, _, _, H_c, W_c = f_coarse.shape
        plucker_scale = H // H_c

        plucker = self._compute_plucker(angles, B, N, H_c, W_c, plucker_scale, device)
        tokens  = (f_coarse.reshape(B * N, self.C_f, H_c, W_c)
                   .permute(0, 2, 3, 1).reshape(B * N, H_c * W_c, self.C_f))
        p_tok   = plucker.permute(0, 2, 3, 1).reshape(B * N, H_c * W_c, 6)

        # ModLN ray conditioning (always).  Optional per-view FusionViT (skipped
        # when fuse_depth=0): its activations grow O(N), so when used it is
        # chunked under gradient checkpointing (only chunk inputs/outputs stored,
        # the rest recomputed in backward).
        if self.netfuse is None:
            f_fused_2d = (self.netmodln(tokens, p_tok)
                          .reshape(B * N, H_c, W_c, self.C_f).permute(0, 3, 1, 2))
        else:
            def _fuse_views(t, p):
                return self.netfuse(self.netmodln(t, p))
            fuse_chunk = getattr(self.opt, 'fuse_view_chunk', 32)
            fused = []
            for s0 in range(0, B * N, fuse_chunk):
                fused.append(checkpoint(_fuse_views,
                                        tokens[s0:s0 + fuse_chunk],
                                        p_tok[s0:s0 + fuse_chunk],
                                        use_reentrant=False))
            f_fused_2d = (torch.cat(fused, dim=0)
                          .reshape(B * N, H_c, W_c, self.C_f).permute(0, 3, 1, 2))

        L_fx = self.opt.Nx // 16
        L_fy = self.opt.Ny // 16
        L_fz = self.opt.Nz // 16
        L_total = L_fx * L_fy * L_fz

        _vox = self.opt.vol_size * 16
        _ip, _iq = map_3d_grid_to_2d_batch(
            L_fx, L_fy, L_fz, _vox, _vox, _vox,
            angles.cpu().numpy(), H_c, W_c,
            self.opt.psize * plucker_scale,
        )
        _ip_t = torch.as_tensor(_ip, dtype=torch.float32).view(B * N, L_total)
        _iq_t = torch.as_tensor(_iq, dtype=torch.float32).view(B * N, L_total)
        coarse_grid = torch.stack((_iq_t, _ip_t), dim=-1).to(device)

        in_fov = (coarse_grid.abs() <= 1.0).all(dim=-1)            # (B*N, L_total)
        count_per_voxel = (in_fov.float()
                           .view(B, N, L_total)
                           .sum(dim=1)
                           .to(device))                            # (B, L_total)

        # Angular spread on 2θ: tomographic directions are *lines* — the ray
        # at θ measures (nearly) the same line integrals as θ+180°.  Doubling
        # the angles maps conjugate views onto each other, so a 211° Parker
        # short scan (data-complete) gets spread ≈ 1 ("well covered") while a
        # genuine 90° limited-angle scan remains penalised.  The parent's
        # plain-θ statistic under-credited short scans (spread ≈ 0.7).
        thetas2 = angles.float() * (math.pi / 90.0)                # 2θ in radians
        R = torch.sqrt(thetas2.cos().mean(1) ** 2 +
                       thetas2.sin().mean(1) ** 2)                 # (B,) in [0,1]
        angular_spread = (1.0 - R).to(device)

        base_hl = getattr(self.opt, 'proj_coverage_halflife', 10.0)
        eff_hl = base_hl / (0.1 + angular_spread).unsqueeze(1)     # (B, 1)
        coverage_mask = 1.0 - torch.exp(-count_per_voxel / eff_hl)

        # ── Binned coarse aggregation: attn + mean + max in one pass ────────
        # Each chunk is one checkpointed forward of the (possibly wrapped)
        # netaggr — Stage 1, mean/max accumulation and the Stage-2 transformer
        # all run inside it (DDP-safe, O(V*C) stored activations).
        K = _unwrap(self.netaggr).K
        kidx = self._bin_assignment(angles, K)
        # Same N-adaptive scaling as mid/fine (C_f=768 is the widest scale).
        chunk = getattr(self.opt, 'aggr_chunk_size', 1024)
        chunk = max(256, int(chunk * 64 / max(N, 64)))

        # 3D voxel coords (mm) for the optional Plücker ray PE — built with the
        # SAME convention as map_3d_grid_to_2d_batch (centred at L/2, 'ij' order,
        # coarse spacing vol_size*16), so it indexes the same voxels as
        # coarse_grid.  SAD matches the back-projection default (1000 mm).
        vox = rho = None
        if self.bin_ray_pe:
            vs = self.opt.vol_size * 16
            ax = [(torch.arange(L, device=device, dtype=torch.float32) - L / 2.0) * vs
                  for L in (L_fx, L_fy, L_fz)]
            gx, gy, gz = torch.meshgrid(ax[0], ax[1], ax[2], indexing='ij')
            vox = torch.stack([gx, gy, gz], -1).reshape(-1, 3).unsqueeze(0).expand(B, -1, -1)
            rho = getattr(self.opt, 'fdk_sad', 1000.0)

        aggs, means, maxs = [], [], []
        for start in range(0, L_total, chunk):
            end = min(start + chunk, L_total)
            vox_chunk = vox[:, start:end] if vox is not None else None
            agg, v_mean, v_max = checkpoint(
                self.netaggr, f_fused_2d, coarse_grid[:, start:end],
                kidx, B, N, vox_chunk, rho, use_reentrant=False)
            aggs.append(agg)
            means.append(v_mean)
            maxs.append(v_max)

        def _to_vol(chunks):
            t = torch.cat(chunks, dim=1)                           # (B, L_total, C_f)
            return (t.view(B, L_fx, L_fy, L_fz, self.C_f)
                     .permute(0, 4, 1, 2, 3).contiguous())

        V_bar       = self.netfeature_spatial_refine(_to_vol(aggs))
        coarse_mean = _to_vol(means)
        coarse_max  = _to_vol(maxs)
        V_f_2D = self.netrefine_coarse(torch.cat([V_bar, coarse_mean, coarse_max], dim=1))
        return V_f_2D, encoded, coverage_mask

    # ── Forward (overridden: ref alignment + multi-scale injection) ─────────

    def forward(self, image_data, ref_ct, device='cuda:0',
                prior_vol=None, iter_level=0.0):
        """
        Same contract as the parent, plus iterative-refinement inputs:
          prior_vol  : in-frame prior volume (B,1,Nx,Ny,Nz, [0,1]) to inject.
                       None on the first iteration → the source FDK is used;
                       on later iterations pass the model's own previous output.
          iter_level : scalar in [0,1] (0 = first/FDK iteration) conditioning
                       the injection gates on prior quality.
        Both default so single-pass callers are unchanged.

        Paired-path additions:
          * netref_encoder's 1/4 and 1/8 outputs are KEPT (parent discarded them),
          * a rough affine ref→source transform is fitted from correspondence
            attention on the coarse feature volumes,
          * affine-warped ref features are gate-fused into the SOURCE decoder
            skips; unwarped ref features into the REFERENCE decode skips.
        """
        projections = image_data['projections'].to(device)   # (B, N, H, W)
        angles      = image_data['angles']                   # (B, N), CPU tensor
        B, N, H, W  = projections.shape

        V_f_2D, encoded, coverage_mask = self._encode_and_aggregate_coarse(
            projections, angles, B, N, H, W, device)
        f_fine = encoded['fine']    # (B, N, C_a, H/8, W/8)
        f_mid  = encoded['mid']     # (B, N, C_b, H/16, W/16)
        _, _, _, H_f, W_f = f_fine.shape
        _, _, _, H_m, W_m = f_mid.shape
        scale_fine = H // H_f
        scale_mid  = H // H_m

        L_bx = self.opt.Nx // 8
        L_by = self.opt.Ny // 8
        L_bz = self.opt.Nz // 8
        L_ax = self.opt.Nx // 4
        L_ay = self.opt.Ny // 4
        L_az = self.opt.Nz // 4
        # mid/fine: attn + mean + max (same triple as coarse). Spatial-refine the
        # attn term, then fuse with the mean/max pools via netrefine_mid/fine.
        b_attn, b_mean, b_max = self._fast_aggregate_attn(
            f_mid.reshape(B * N, self.C_b, H_m, W_m), angles, B, N,
            L_bx, L_by, L_bz, self.C_b,
            scale_3d=8, scale_2d=scale_mid, aggr_net=self.netaggr_mid, device=device)
        a_attn, a_mean, a_max = self._fast_aggregate_attn(
            f_fine.reshape(B * N, self.C_a, H_f, W_f), angles, B, N,
            L_ax, L_ay, L_az, self.C_a,
            scale_3d=4, scale_2d=scale_fine, aggr_net=self.netaggr_fine, device=device)

        V_f_b = self.netfeature_spatial_refine_mid(b_attn)
        V_f_b = self.netrefine_mid(torch.cat([V_f_b, b_mean, b_max], dim=1))
        V_f_a = self.netfeature_spatial_refine_fine(a_attn)
        V_f_a = self.netrefine_fine(torch.cat([V_f_a, a_mean, a_max], dim=1))

        # ── FDK / previous-output prior injection into coarse + skips ───────
        if self.use_fdk:
            pv = self._prior_volume(prior_vol, projections, angles, device)
            fdk_cond = (self._fdk_cond(angles, coverage_mask, N, device)
                        if self.fdk_film else None)
            V_f_2D, V_f_b, V_f_a = self._inject_prior(
                pv, V_f_2D, V_f_b, V_f_a, coverage_mask, iter_level, B, device,
                fdk_cond=fdk_cond)

        # Self-refinement transformer (no reference cross-attention).
        V_f_3D = self.netrefine_tf(V_f_2D)
        vols = checkpoint(self.netdecoder, V_f_3D, V_f_b, V_f_a,
                          use_reentrant=False).squeeze(1)
        self.vols_ref = None
        self.ref_vol  = None
        return vols

    # ── Checkpoint loading ──────────────────────────────────────────────────

    def load_networks(self, epoch):
        """
        Like BaseModel.load_networks but tolerant to architecture evolution:
        falls back to strict=False when a net's state_dict has missing /
        unexpected keys (e.g. netalign gained alpha_raw after a run started),
        printing what was skipped instead of crashing --continue_train.

        Missing checkpoint FILES are skipped with a warning rather than
        crashing, so --continue_train works when the new run has nets the old
        one didn't (e.g. phase 1 single-pass FDK -> phase 2 adds nets), and a
        wrong --name / empty save_dir is reported clearly instead of a bare
        FileNotFoundError.
        """
        print(f'[load_networks] loading "{epoch}" checkpoints from {self.save_dir}')
        n_loaded, n_missing = 0, 0
        for name in self.model_names:
            if not isinstance(name, str):
                continue
            load_filename = '%s_net_%s.pth' % (epoch, name)
            load_path = os.path.join(self.save_dir, load_filename)
            net = getattr(self, 'net' + name)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            # Released weights are safetensors (`<epoch>_net_<name>.safetensors`);
            # training checkpoints are torch `.pth` state dicts. Prefer safetensors.
            st_path = os.path.splitext(load_path)[0] + '.safetensors'
            if os.path.exists(st_path):
                from safetensors.torch import load_file
                state_dict = load_file(st_path, device=str(self.device))
            elif os.path.exists(load_path):
                state_dict = torch.load(load_path, map_location=str(self.device),
                                        weights_only=True)
            else:
                print(f'  [skip] no checkpoint for net_{name} at {load_path} '
                      f'(kept at init)')
                n_missing += 1
                continue
            n_loaded += 1
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
            try:
                net.load_state_dict(state_dict)
            except RuntimeError as e:
                print(f'[load_networks] strict load failed for net_{name} '
                      f'({e});\n  retrying, skipping missing/shape-mismatched keys')
                # Drop keys absent from, or shape-incompatible with, the current
                # net (architecture evolved, e.g. a gate gained an input
                # channel) so the rest still loads instead of crashing.
                net_sd = net.state_dict()
                compatible = {k: v for k, v in state_dict.items()
                              if k in net_sd and v.shape == net_sd[k].shape}
                dropped = [k for k in state_dict if k not in compatible]
                missing, unexpected = net.load_state_dict(compatible, strict=False)
                if dropped:
                    print(f'  dropped (missing in ckpt / shape mismatch, kept at '
                          f'init): {sorted(set(dropped) | set(missing))}')
                if unexpected:
                    print(f'  unexpected keys (ignored): {unexpected}')
        print(f'[load_networks] loaded {n_loaded} net(s); {n_missing} missing '
              f'(kept at init).')
        if n_loaded == 0:
            print('[load_networks] WARNING: nothing loaded — check --name / '
                  '--checkpoints_dir / --epoch match the run you meant to continue.')

    # Volume loss is inherited from the parent (no alignment term in the
    # FDK-only model).
