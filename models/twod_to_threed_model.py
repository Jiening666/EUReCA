import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from .multiscale_cnn_encoder import MultiScaleCNNEncoder
from .multi_image_aggr import MultiImageAggregator, ViewWeightAggregator
from .renderer import DifferentiableDRR

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from threeD_geometry import map_3d_grid_to_2d_batch
from threeD_geometry import map_2d_plucker_ray_directions

from torch.utils.checkpoint import checkpoint
from pytorch_msssim import ssim, ms_ssim 

# ─────────────────────────── ModLN ──────────────────────────────────────────

class ModLN(nn.Module):
    """FiLM-style modulated LayerNorm conditioned on Plücker rays."""
    def __init__(self, feature_dim=768, ray_input_dim=6):
        super().__init__()
        self.ln = nn.LayerNorm(feature_dim, elementwise_affine=False)
        self.ray_to_mod = nn.Sequential(
            nn.Linear(ray_input_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim * 2),
        )

    def forward(self, h, ray_6d):
        # h: (..., C), ray_6d: (..., 6)
        h_norm = self.ln(h)
        gamma, beta = self.ray_to_mod(ray_6d).chunk(2, dim=-1)
        return h_norm * (1 + gamma) + beta


# ─────────────────────────── FusionViT ──────────────────────────────────────

class FusionViT(nn.Module):
    """Per-view 2D spatial self-attention to enrich coarse features."""
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                activation='gelu',
                batch_first=True,
                norm_first=True,
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B*N, L, C) — processes each view independently
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


# ─────────────────────────── RefVolumeEncoder ────────────────────────────────

class ConvNeXt3DBlock(nn.Module):
    def __init__(self, dim, expansion=4, layer_scale=1e-6):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, 5, padding=2, groups=dim)
        self.norm   = nn.GroupNorm(min(32, dim), dim)
        self.pw1    = nn.Conv3d(dim, dim * expansion, 1)
        self.pw2    = nn.Conv3d(dim * expansion, dim, 1)
        self.gamma  = nn.Parameter(layer_scale * torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        r = x
        x = F.gelu(self.norm(self.dwconv(x)))
        x = self.pw2(F.gelu(self.pw1(x)))
        return r + self.gamma * x


class RefVolumeEncoder(nn.Module):
    """
    3D CNN extracting features from an in-frame prior (FDK) volume at three
    scales:
      1/4  → (B, C_a, Nx/4,  Ny/4,  Nz/4 )
      1/8  → (B, C_b, Nx/8,  Ny/8,  Nz/8 )
      1/16 → (B, C_f, Nx/16, Ny/16, Nz/16)

    Optionally FiLM-conditioned on a small GLOBAL descriptor (e.g.
    [log N, angular_spread, mean coverage]) so the encoder can adapt feature
    extraction to the prior's reliability — trusting high-frequency detail when
    N is large / coverage is high, and leaning on robust low-frequency content
    when N is small (FDK streaky). cond_dim=0 disables it (identical to before).
    """
    def __init__(self, C_a=128, C_b=256, C_f=768, in_channels=1, cond_dim=0):
        super().__init__()
        mid = 64
        self.cond_dim = cond_dim
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, mid, 3, stride=2, padding=1),
            nn.GroupNorm(min(32, mid), mid), nn.GELU(),
            nn.Conv3d(mid, mid, 3, stride=2, padding=1),
            nn.GroupNorm(min(32, mid), mid), nn.GELU(),
        )
        self.stage1 = nn.Sequential(nn.Conv3d(mid, C_a, 1),
                                    ConvNeXt3DBlock(C_a), ConvNeXt3DBlock(C_a))
        self.down1  = nn.Sequential(nn.GroupNorm(min(32, C_a), C_a),
                                    nn.Conv3d(C_a, C_b, 2, stride=2))
        self.stage2 = nn.Sequential(ConvNeXt3DBlock(C_b), ConvNeXt3DBlock(C_b))
        self.down2  = nn.Sequential(nn.GroupNorm(min(32, C_b), C_b),
                                    nn.Conv3d(C_b, C_f, 2, stride=2))
        self.stage3 = nn.Sequential(ConvNeXt3DBlock(C_f), ConvNeXt3DBlock(C_f))

        if cond_dim > 0:
            # FiLM heads: global cond -> per-stage (gamma, beta). Zero-init so
            # the encoder starts IDENTICAL to the unconditioned version and
            # learns the conditioning from there (no disruption at step 0).
            self.film_a = nn.Linear(cond_dim, C_a * 2)
            self.film_b = nn.Linear(cond_dim, C_b * 2)
            self.film_f = nn.Linear(cond_dim, C_f * 2)
            for lin in (self.film_a, self.film_b, self.film_f):
                nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)

    @staticmethod
    def _film(h, lin, cond):
        gamma, beta = lin(cond).chunk(2, dim=1)          # (B, C) each
        return h * (1 + gamma[:, :, None, None, None]) + beta[:, :, None, None, None]

    def forward(self, x, cond=None):
        # x: (B, 1, Nx, Ny, Nz); cond: (B, cond_dim) or None
        use = cond is not None and self.cond_dim > 0
        x       = self.stem(x)
        f_1_4   = self.stage1(x)
        if use: f_1_4 = self._film(f_1_4, self.film_a, cond)
        f_1_8   = self.stage2(self.down1(f_1_4))
        if use: f_1_8 = self._film(f_1_8, self.film_b, cond)
        f_1_16  = self.stage3(self.down2(f_1_8))
        if use: f_1_16 = self._film(f_1_16, self.film_f, cond)
        return f_1_4, f_1_8, f_1_16


# ──────────────────── VolumeRefinementTransformer ────────────────────────────

class RoPE3D(nn.Module):
    """3-D Rotary Position Embedding.

    Splits the channel dim into three equal groups and applies standard 1-D
    RoPE independently along each of the x, y, z spatial axes.  The remaining
    channels (if embed_dim is not divisible by 3) are left untouched.

    embed_dim must be divisible by 6 (2 per axis pair × 3 axes).
    """

    def __init__(self, embed_dim: int, base: float = 10000.0):
        super().__init__()
        assert embed_dim % 6 == 0, "embed_dim must be divisible by 6 for RoPE3D"
        d = embed_dim // 3                          # channels per axis
        inv_freq = 1.0 / (base ** (torch.arange(0, d, 2).float() / d))
        self.register_buffer('inv_freq', inv_freq)  # (d/2,)
        self.d = d

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def _cos_sin(self, n: int, dtype, device) -> tuple:
        pos   = torch.arange(n, device=device, dtype=dtype)
        freqs = torch.outer(pos, self.inv_freq.to(dtype))  # (n, d/2)
        emb   = torch.cat([freqs, freqs], dim=-1)          # (n, d)
        return emb.cos(), emb.sin()

    def forward(self, x: torch.Tensor, spatial_shape: tuple) -> torch.Tensor:
        """
        x             : (B, Lx*Ly*Lz, C)
        spatial_shape : (Lx, Ly, Lz)
        """
        Lx, Ly, Lz = spatial_shape
        d = self.d
        dtype, device = x.dtype, x.device

        cos_x, sin_x = self._cos_sin(Lx, dtype, device)   # (Lx, d)
        cos_y, sin_y = self._cos_sin(Ly, dtype, device)
        cos_z, sin_z = self._cos_sin(Lz, dtype, device)

        # Broadcast to full (Lx, Ly, Lz, d) grid then flatten spatial dims
        cos_x = cos_x[:, None, None, :].expand(Lx, Ly, Lz, d).reshape(-1, d)
        sin_x = sin_x[:, None, None, :].expand(Lx, Ly, Lz, d).reshape(-1, d)
        cos_y = cos_y[None, :, None, :].expand(Lx, Ly, Lz, d).reshape(-1, d)
        sin_y = sin_y[None, :, None, :].expand(Lx, Ly, Lz, d).reshape(-1, d)
        cos_z = cos_z[None, None, :, :].expand(Lx, Ly, Lz, d).reshape(-1, d)
        sin_z = sin_z[None, None, :, :].expand(Lx, Ly, Lz, d).reshape(-1, d)

        x_x = x[..., :d]
        x_y = x[..., d:2 * d]
        x_z = x[..., 2 * d:3 * d]

        x_x = x_x * cos_x + self._rotate_half(x_x) * sin_x
        x_y = x_y * cos_y + self._rotate_half(x_y) * sin_y
        x_z = x_z * cos_z + self._rotate_half(x_z) * sin_z

        return torch.cat([x_x, x_y, x_z, x[..., 3 * d:]], dim=-1)


class RefinementBlock(nn.Module):
    """
    One block of the volume self-refinement transformer: pre-norm
    self-attention + FFN over the 3D feature tokens.

    The model has no (unaligned) reference input, so the former
    cross-attention-to-reference branch is removed; the coarse FDK prior now
    supplies the structural information that branch used to retrieve.
    """
    def __init__(self, embed_dim=768, num_heads=4):
        super().__init__()
        self.norm_q1 = nn.LayerNorm(embed_dim)
        self.norm_q3 = nn.LayerNorm(embed_dim)
        self.self_attn_q = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        ff = embed_dim * 4
        self.ff = nn.Sequential(nn.Linear(embed_dim, ff), nn.GELU(),
                                 nn.Linear(ff, embed_dim))

    def forward(self, V_q):
        # V_q: (B, L, C)
        nq = self.norm_q1(V_q)
        V_q = V_q + self.self_attn_q(nq, nq, nq)[0]
        V_q = V_q + self.ff(self.norm_q3(V_q))
        return V_q


class VolumeRefinementTransformer(nn.Module):
    """T blocks of self-attention for 3D feature self-refinement (+ optional RoPE)."""
    def __init__(self, embed_dim=768, num_heads=4, depth=12, use_rope=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            RefinementBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.rope = RoPE3D(embed_dim) if use_rope else None

    def forward(self, V_f_3D):
        B, C, *spatial = V_f_3D.shape
        V_q = V_f_3D.flatten(2).permute(0, 2, 1)      # (B, L^3, C)
        if self.rope is not None:
            V_q = self.rope(V_q, spatial)
        for block in self.blocks:
            V_q = checkpoint(block, V_q, use_reentrant=False)
        return V_q.permute(0, 2, 1).view(B, C, *spatial)


# fix backprj_feat
class FBPCorrectionSkip(nn.Module):
    def __init__(self, ch):
        super().__init__()
        # Predicts and removes streak artifacts
        self.artifact_predictor = nn.Sequential(
            nn.Conv3d(ch, ch, 3, padding=1),
            nn.InstanceNorm3d(ch),
            nn.LeakyReLU(0.2),
            nn.Conv3d(ch, ch, 3, padding=1),
        )

    def forward(self, backproj_feat):
        artifact = self.artifact_predictor(backproj_feat)
        return backproj_feat - artifact 

# ──────────────────────── MultiScaleDecoder3D ────────────────────────────────

class Conv3DBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv3d(in_c, out_c, 3, padding=1)
        self.norm = nn.GroupNorm(min(32, out_c), out_c)

    def forward(self, x):
        return F.gelu(self.norm(self.conv(x)))


class MultiScaleDecoder3D(nn.Module):
    """
    Hierarchical decoder:  16^3 → 32^3 → 64^3 → 256^3
    Skip connections: skip_mid (32^3, C_b) and skip_fine (64^3, C_a).
    Both source and reference decoder paths share this class.
    """
    def __init__(self, C_f=768, C_b=256, C_a=128):
        super().__init__()
        # 16 → 32  (C_f → C_b, cat with mid skip)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(C_f, C_b, kernel_size=3, padding=1),
            nn.GroupNorm(min(32, C_b), C_b), nn.GELU(),
        )
        #nn.ConvTranspose3d(C_f, C_b, 2, stride=2)
        self.conv1 = Conv3DBlock(C_b + C_b, C_b)
        # 32 → 64  (C_b → C_a, cat with fine skip)
        self.up2   = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(C_b, C_a, kernel_size=3, padding=1),
            nn.GroupNorm(min(32, C_a), C_a), nn.GELU(),
        )
        
        #nn.ConvTranspose3d(C_b, C_a, 2, stride=2)
        
        
        self.conv2 = Conv3DBlock(C_a + C_a, C_a)
        # 64 → 128 → 256  (two ×2 upsamples)
        self.up3   = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(C_a, 64, kernel_size=3, padding=1),
            nn.GroupNorm(min(32, 64), 64), nn.GELU(),
        )
        #nn.ConvTranspose3d(C_a, 64, 2, stride=2)
        self.conv3 = Conv3DBlock(64, 64)
        self.up4   = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.GroupNorm(min(32, 32), 32), nn.GELU(),
        )
        #nn.ConvTranspose3d(64, 32, 2, stride=2)

        self.conv4 = Conv3DBlock(32, 32)
        self.out   = nn.Conv3d(32, 1, 1)

    def forward(self, x, skip_mid, skip_fine):
        # x:         (B, C_f, 16, 16, 16)
        # skip_mid:  (B, C_b, 32, 32, 32)
        # skip_fine: (B, C_a, 64, 64, 64)
        x = self.conv1(torch.cat([self.up1(x), skip_mid],  dim=1))
        x = self.conv2(torch.cat([self.up2(x), skip_fine], dim=1))
        x = self.conv3(self.up3(x))
        x = self.conv4(self.up4(x))
        return self.out(x)


# ──────────────────────────── Main model ─────────────────────────────────────

class twoD2threeDModel(BaseModel):

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names   = ['L_vol', 'L_vol_ref', 'L_render_L1', 'L_render_SSIM', 'L_render', 'L_MAGA', 'L_MAGA_ref', 'L_perceptual', 'L_perceptual_ref']
        self.visual_names = ['pred_vol', 'real_vol']

        C_a, C_b, C_f = 128, 256, 768
        self.C_a, self.C_b, self.C_f = C_a, C_b, C_f

        # ── Network components ──────────────────────────────────────────────
        self.model_names = [
            'encoder', 'modln',
            'aggr', 'aggr_fine', 'aggr_mid',
            'feature_spatial_refine', 'feature_spatial_refine_fine', 'feature_spatial_refine_mid',
            'refine_coarse', 'refine_tf',
            'decoder',
        ]
        fuse_depth   = getattr(opt, 'fuse_depth',   0)
        aggr_layers  = getattr(opt, 'aggr_layers',  2)
        refine_depth = getattr(opt, 'refine_depth', 12)
        use_rope     = getattr(opt, 'use_rope', False)

        # 2D encoder — pyramid_out projects coarse to C_f=768 inside the encoder
        # (Conv2d 1×1 + GroupNorm), eliminating a separate netproj_coarse module.
        _bb = getattr(opt, 'encoder_backbone', '') or ''
        # Fetch the ImageNet-pretrained backbone only when we are about to train
        # from scratch. At test time (and with --continue_train) the trained
        # encoder weights are loaded from the checkpoint, so no download is needed
        # and inference works offline.
        _pretrained = bool(getattr(opt, 'isTrain', False)) and not getattr(opt, 'continue_train', False)
        self.netencoder = MultiScaleCNNEncoder(
            pyramid_out=(C_a, C_b, C_f),
            backbone=_bb,                   # '' = from-scratch ConvNeXt; e.g. 'convnextv2_nano'
            pretrained=_pretrained,
            use_fpn=not bool(_bb),          # pretrained taps are context-rich -> drop the 2D FPN
            instance_norm_input=bool(_bb),  # z-score input for the pretrained backbone
        )
        # Plücker-conditioned layer norm for coarse 2D features (ray conditioning)
        self.netmodln      = ModLN(feature_dim=C_f, ray_input_dim=6)
        # Per-view 2D spatial self-attention — optional; skipped when fuse_depth=0
        # (the pretrained backbone already gives near-global per-view context).
        if fuse_depth > 0:
            self.netfuse = FusionViT(embed_dim=C_f, depth=fuse_depth, num_heads=6)
            self.model_names.append('fuse')
        else:
            self.netfuse = None
        # View-aggregators (attention pooling over N views per voxel)
        self.netaggr       = MultiImageAggregator(embed_dim=C_f, num_layers=aggr_layers)
        self.netaggr_fine  = ViewWeightAggregator(embed_dim=C_a)
        self.netaggr_mid   = ViewWeightAggregator(embed_dim=C_b)
        # Manual refinement: cat(V_bar, mean, max) 3·C_f → C_f
        self.netrefine_coarse = nn.Conv3d(C_f * 3, C_f, 1)
        self.netfeature_spatial_refine = FBPCorrectionSkip(C_f)
        self.netfeature_spatial_refine_fine = FBPCorrectionSkip(C_a)
        self.netfeature_spatial_refine_mid = FBPCorrectionSkip(C_b)

        self.netrefine_tf = VolumeRefinementTransformer(
            embed_dim=C_f, depth=refine_depth, use_rope=use_rope)
        # Decoder — shared between source and reference paths
        self.netdecoder = MultiScaleDecoder3D(C_f=C_f, C_b=C_b, C_a=C_a)

        # ── Renderer ────────────────────────────────────────────────────────
        nSample = int(max(opt.Nx, opt.Ny, opt.Nz) * 1.5)
        self.renderer = DifferentiableDRR(
            opt.Nx, opt.Ny, opt.Nz,
            opt.vol_size, opt.vol_size, opt.vol_size,
            opt.H, opt.W, opt.psize, opt.psize,
            nSample, opt.vol_size,
        )

        # ── Optimizer ───────────────────────────────────────────────────────
        # Single optimizer (FDK-only model; no reference sub-nets). The
        # pretrained 2D-encoder backbone gets its own param group at a reduced
        # LR (--backbone_lr_mult); absent when no pretrained backbone is used.
        # BaseModel is not an nn.Module, so iterate sub-networks directly.
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
        self.optimizers.append(self.optimizer_base)

        self.criterionL1  = torch.nn.L1Loss()
        self.lambda_l1    = opt.lambda_L1
        self.lambda_ssim  = opt.lambda_ssim
        self.lambda_ref   = getattr(opt, 'lambda_ref', 0.5)
        self.lambda_perc  = getattr(opt, 'lambda_perc', 1.0)

        # Pre-built Scharr kernels stored once; moved to the right device/dtype
        # on first use inside compute_multi_scale_loss (not recreated every call).
        smooth = torch.tensor([3.0, 10.0, 3.0])
        deriv  = torch.tensor([-1.0, 0.0, 1.0])
        kx = deriv.view(1,1,3) * smooth.view(1,3,1) * smooth.view(3,1,1)
        ky = smooth.view(1,1,3) * deriv.view(1,3,1) * smooth.view(3,1,1)
        kz = smooth.view(1,1,3) * smooth.view(1,3,1) * deriv.view(3,1,1)
        self._scharr = torch.cat([kx, ky, kz], dim=0).view(3, 1, 3, 3, 3)

        # Default loss values (so get_current_losses() always works)
        self.loss_L_vol          = torch.tensor(0.0)
        self.loss_L_vol_ref      = torch.tensor(0.0)
        self.loss_L_render_L1    = torch.tensor(0.0)
        self.loss_L_render_SSIM  = torch.tensor(0.0)
        self.loss_L_render       = torch.tensor(0.0)
        
        self.loss_L_MAGA         = torch.tensor(0.0)
        self.loss_L_MAGA_ref     = torch.tensor(0.0)
        self.loss_L_perceptual       = torch.tensor(0.0)
        self.loss_L_perceptual_ref   = torch.tensor(0.0)

    # ── Internal helpers ────────────────────────────────────────────────────

    def _compute_plucker(self, angles, B, N, H_feat, W_feat, scale_2d, device):
        """
        Compute Plücker ray encodings for a (H_feat × W_feat) 2D feature grid.
        angles: (B, N) tensor of gantry angles in degrees (CPU).
        scale_2d: H_input / H_feat  (effective pixel-size multiplier).
        Returns: (B*N, 6, H_feat, W_feat) float32 tensor on device.
        """
        psize_eff = self.opt.psize * scale_2d
        rays = []
        ang = angles.cpu().numpy()
        for b in range(B):
            for n in range(N):
                r = map_2d_plucker_ray_directions(H_feat, W_feat, psize_eff, float(ang[b, n]))
                rays.append(r)  # (6, H_feat, W_feat)
        return torch.as_tensor(np.stack(rays), dtype=torch.float32).to(device)

    def _chunked_aggregate(self, feat_2d, angles, B, N, Lx, Ly, Lz, C,
                           scale_3d, scale_2d, device,
                           aggr_net=None, reduce='attn', chunk_size=1024,
                           precomputed_grid=None):
        """
        Back-project 2D features to a (Lx×Ly×Lz) 3D grid and aggregate across views.

        feat_2d          : (B*N, C, H_feat, W_feat) — 2D feature maps.
        precomputed_grid : (B*N, Lx*Ly*Lz, 2) pre-computed grid coords (optional).
                           Pass when calling multiple times with the same angles/grid
                           to avoid recomputing map_3d_grid_to_2d_batch.
        Returns : (B, C, Lx, Ly, Lz).
        """
        H_feat, W_feat = feat_2d.shape[-2:]
        total = Lx * Ly * Lz
        vox_size = self.opt.vol_size * scale_3d

        if precomputed_grid is not None:
            grid_flat = precomputed_grid                         # (B*N, total, 2)
        else:
            ip, iq = map_3d_grid_to_2d_batch(
                Lx, Ly, Lz,
                vox_size, vox_size, vox_size,
                angles.cpu().numpy(),
                H_feat, W_feat,
                self.opt.psize * scale_2d,
            )
            ip = torch.as_tensor(ip, dtype=torch.float32).view(B * N, total)
            iq = torch.as_tensor(iq, dtype=torch.float32).view(B * N, total)
            grid_flat = torch.stack((iq, ip), dim=-1).to(device)   # (B*N, total, 2)

        results = []
        for start in range(0, total, chunk_size):
            end   = min(start + chunk_size, total)
            chunk = end - start

            # grid_sample expects (B*N, 1, chunk, 2)
            gc = grid_flat[:, start:end, :].unsqueeze(1)
            # feat: (B*N, C, H, W) → sample → (B*N, C, 1, chunk)
            s = F.grid_sample(feat_2d.float(), gc.float(),
                              mode='bilinear', padding_mode='zeros', # border, zeros
                              align_corners=True).squeeze(2)   # (B*N, C, chunk)

            s = s.view(B, N, C, chunk).permute(0, 3, 1, 2)    # (B, chunk, N, C)
            s = s.reshape(B * chunk, N, C).contiguous().to(device)

            if reduce == 'attn' and aggr_net is not None:
                agg = checkpoint(aggr_net, s, use_reentrant=False)  # (B*chunk, C)
            elif reduce == 'mean':
                agg = s.mean(dim=1)
            else:  # 'max'
                agg = s.max(dim=1).values

            results.append(agg.view(B, chunk, C))

        out = torch.cat(results, dim=1)                        # (B, total, C)
        return out.view(B, Lx, Ly, Lz, C).permute(0, 4, 1, 2, 3).contiguous()

    def _fast_aggregate_attn(self, feat_2d, angles, B, N, Lx, Ly, Lz, C,
                             scale_3d, scale_2d, aggr_net, device,
                             attn_chunk_size=None):
        """
        Vectorised grid_sample + aggregation for mid / fine scales.

        attn_chunk_size=None (default): one shot over all voxels, no checkpoint.
            Use with ViewWeightAggregator — O(N) scoring, trivial memory.
        attn_chunk_size=<int>: chunked with gradient checkpointing per chunk.
            Use if aggr_net is a heavy transformer and you hit OOM.
        """
        total    = Lx * Ly * Lz
        vox_size = self.opt.vol_size * scale_3d
        H_feat, W_feat = feat_2d.shape[-2:]

        # --- grid (CPU, once) ---
        ip, iq = map_3d_grid_to_2d_batch(
            Lx, Ly, Lz, vox_size, vox_size, vox_size,
            angles.cpu().numpy(), H_feat, W_feat,
            self.opt.psize * scale_2d,
        )
        ip_t = torch.as_tensor(ip, dtype=torch.float32).view(B * N, total)
        iq_t = torch.as_tensor(iq, dtype=torch.float32).view(B * N, total)
        grid_flat = torch.stack((iq_t, ip_t), dim=-1).to(device)  # (B*N, total, 2)

        # --- single vectorised grid_sample over all voxels ---
        sampled = F.grid_sample(
            feat_2d.float(),
            grid_flat.unsqueeze(1).float(),   # (B*N, 1, total, 2)
            mode='bilinear', padding_mode='zeros', align_corners=True,
        ).squeeze(2)                           # (B*N, C, total)

        # rearrange to (B*total, N, C) for the aggregator
        sampled = (sampled.view(B, N, C, total)
                          .permute(0, 3, 1, 2)        # (B, total, N, C)
                          .reshape(B * total, N, C).contiguous())

        if attn_chunk_size is None:
            # One shot — no loop, no checkpoint.  Works well for lightweight
            # aggregators (ViewWeightAggregator) where memory is not an issue.
            out = aggr_net(sampled)                    # (B*total, C)
        else:
            results = []
            for start in range(0, B * total, attn_chunk_size):
                s = sampled[start : start + attn_chunk_size]
                results.append(checkpoint(aggr_net, s, use_reentrant=False))
            out = torch.cat(results, dim=0)            # (B*total, C)

        return (out.view(B, Lx, Ly, Lz, C)
                   .permute(0, 4, 1, 2, 3).contiguous())

    def _encode_views(self, projections):
        """Encode projections (B, N, H, W) -> dict of (B, N, C, H', W') pyramid
        features, chunked over views with gradient checkpointing to bound peak
        memory at large N.  --encoder_view_chunk sets the per-chunk view count;
        0 disables (original single-call, no checkpoint)."""
        B, N = projections.shape[:2]
        chunk = getattr(self.opt, 'encoder_view_chunk', 8)
        if not chunk:
            return self.netencoder(projections.unsqueeze(2))
        chunk = min(int(chunk), N)

        def _enc(x):                                   # x: (B, n, 1, H, W)
            d = self.netencoder(x)
            return d['fine'], d['mid'], d['coarse']

        f, m, c = [], [], []
        for s in range(0, N, chunk):
            x = projections[:, s:s + chunk].unsqueeze(2)
            ef, em, ec = checkpoint(_enc, x, use_reentrant=False)
            f.append(ef); m.append(em); c.append(ec)
        return {'fine':   torch.cat(f, dim=1),
                'mid':    torch.cat(m, dim=1),
                'coarse': torch.cat(c, dim=1)}

    def _encode_and_aggregate_coarse(self, projections, angles, B, N, H, W, device):
        """
        Full coarse 2D→3D pipeline for a set of projections.
        Returns:
          V_f_2D        : (B, C_f, L_fx, L_fy, L_fz) — attn+mean+max fused coarse feature
          encoded       : dict with 'fine', 'mid', 'coarse' maps for decoder skip connections
          coverage_mask : (B, L_fx*L_fy*L_fz) fraction of views that project inside the
                          image FOV for each coarse voxel.  1.0 = always visible,
                          0.0 = never visible (outside FOV from every angle).
                          Scale- and N-free: works for any N at train or test time.
                          Captures both count and angular-distribution effects naturally.
        """
        encoded  = self._encode_views(projections)
        f_coarse = encoded['coarse']                                     # (B, N, C_f, H_c, W_c)
        _, _, _, H_c, W_c = f_coarse.shape
        plucker_scale = H // H_c

        plucker  = self._compute_plucker(angles, B, N, H_c, W_c, plucker_scale, device)
        tokens   = f_coarse.reshape(B * N, self.C_f, H_c, W_c).permute(0, 2, 3, 1).reshape(B * N, H_c * W_c, self.C_f)
        p_tok    = plucker.permute(0, 2, 3, 1).reshape(B * N, H_c * W_c, 6)

        # ModLN ray conditioning (always); optional per-view FusionViT, chunked
        # over views + gradient-checkpointed (its per-view attention activations
        # otherwise dominate coarse-path memory at large N).
        if self.netfuse is None:
            f_fused_2d = (self.netmodln(tokens, p_tok)
                          .reshape(B * N, H_c, W_c, self.C_f).permute(0, 3, 1, 2))
        else:
            def _fuse(t, p):
                return self.netfuse(self.netmodln(t, p))
            vchunk = getattr(self.opt, 'encoder_view_chunk', 8) or (B * N)
            vchunk = min(int(vchunk), B * N)
            fused = [checkpoint(_fuse, tokens[s:s + vchunk], p_tok[s:s + vchunk],
                                use_reentrant=False)
                     for s in range(0, B * N, vchunk)]
            f_fused_2d = (torch.cat(fused, dim=0)
                          .reshape(B * N, H_c, W_c, self.C_f).permute(0, 3, 1, 2))

        L_fx = self.opt.Nx // 16
        L_fy = self.opt.Ny // 16
        L_fz = self.opt.Nz // 16
        agg_kw = dict(scale_3d=16, scale_2d=plucker_scale, device=device)

        _vox = self.opt.vol_size * 16
        _ip, _iq = map_3d_grid_to_2d_batch(
            L_fx, L_fy, L_fz, _vox, _vox, _vox,
            angles.cpu().numpy(), H_c, W_c,
            self.opt.psize * plucker_scale,
        )
        _ip_t = torch.as_tensor(_ip, dtype=torch.float32).view(B * N, L_fx * L_fy * L_fz)
        _iq_t = torch.as_tensor(_iq, dtype=torch.float32).view(B * N, L_fx * L_fy * L_fz)
        coarse_grid = torch.stack((_iq_t, _ip_t), dim=-1).to(device)

        # Per-voxel coverage mask — captures both projection count and angular diversity.
        #
        # Problem with a simple fraction (mean over N):
        #   "seen by all 5 views" == "seen by all 50 views" == 1.0, which is wrong.
        #   50 projections covering the volume is clearly better than 5.
        #
        # Solution: saturating count × angular-adjusted half-life.
        #
        #   count_per_voxel : how many of the N projections actually see this voxel.
        #   angular_spread  : circular variance of the N angles — 1=uniform, 0=all same.
        #
        #   effective half-life = base_hl / (ε + angular_spread)
        #     → uniform angles  : half-life stays at base_hl  (each view is informative)
        #     → clustered angles: half-life grows large        (redundant views, need many more)
        #
        #   coverage = 1 − exp(−count / eff_hl)  ∈ [0, 1]
        #
        # Scale-free: works for any N at train or test time.
        # No N_max needed; saturation handles arbitrarily large N gracefully.
        L_total = L_fx * L_fy * L_fz
        in_fov = (coarse_grid.abs() <= 1.0).all(dim=-1)              # (B*N, L_total)
        count_per_voxel = (in_fov.float()
                           .view(B, N, L_total)
                           .sum(dim=1)
                           .to(device))                               # (B, L_total)

        # Angular spread via circular statistics on the source angles (CPU tensor).
        # mean resultant length R ∈ [0,1]: 1 = all angles identical, 0 = uniform.
        thetas = angles.float() * (math.pi / 180.0)                  # (B, N)
        R = torch.sqrt(thetas.cos().mean(1) ** 2 +
                       thetas.sin().mean(1) ** 2)                    # (B,)  in [0,1]
        angular_spread = (1.0 - R).to(device)                        # (B,)  0=clustered, 1=uniform

        # Effective half-life: clustered angles → larger half-life (views are redundant).
        # base_hl is the half-life for a perfectly uniform angular distribution.
        base_hl = getattr(self.opt, 'proj_coverage_halflife', 10.0)
        eff_hl = base_hl / (0.1 + angular_spread).unsqueeze(1)       # (B, 1)
        coverage_mask = 1.0 - torch.exp(-count_per_voxel / eff_hl)   # (B, L_total) in [0,1]

        aggr_chunk = getattr(self.opt, 'aggr_chunk_size', 32)
        V_bar = self._chunked_aggregate(f_fused_2d, angles, B, N, L_fx, L_fy, L_fz,
                                        self.C_f, aggr_net=self.netaggr, reduce='attn',
                                        precomputed_grid=coarse_grid, chunk_size=aggr_chunk,
                                        **agg_kw)
        V_bar = self.netfeature_spatial_refine(V_bar)
        coarse_mean = self._chunked_aggregate(f_fused_2d, angles, B, N, L_fx, L_fy, L_fz,
                                              self.C_f, reduce='mean',
                                              precomputed_grid=coarse_grid, **agg_kw)
        coarse_max  = self._chunked_aggregate(f_fused_2d, angles, B, N, L_fx, L_fy, L_fz,
                                              self.C_f, reduce='max',
                                              precomputed_grid=coarse_grid, **agg_kw)
        V_f_2D = self.netrefine_coarse(torch.cat([V_bar, coarse_mean, coarse_max], dim=1))
        return V_f_2D, encoded, coverage_mask

    # ── Public API ──────────────────────────────────────────────────────────

    def set_lambda(self, lambda_l1=1.0, lambda_ssim=1.0):
        self.lambda_l1  = lambda_l1
        self.lambda_ssim = lambda_ssim

    def get_pred_drrs(self, pred_vol, angles):
        return self.renderer(pred_vol.permute(0, 3, 2, 1), angles)

    def get_scharr_kernels_3d(self):
        """
        3D Scharr kernels for X, Y, Z gradients.
 
        Built as the outer product of a smoothing kernel [3, 10, 3] on the
        two non-derivative axes and a derivative kernel [-1, 0, 1] on the
        derivative axis. This is the standard separable extension of the 2D
        Scharr operator to 3D.
        """
        smooth = torch.tensor([3.0, 10.0, 3.0])
        deriv = torch.tensor([-1.0, 0.0, 1.0])
 
        # kx: derivative along X (last dim), smoothing along Y and Z
        kx = deriv.view(1, 1, 3) * smooth.view(1, 3, 1) * smooth.view(3, 1, 1)
        # ky: derivative along Y, smoothing along X and Z
        ky = smooth.view(1, 1, 3) * deriv.view(1, 3, 1) * smooth.view(3, 1, 1)
        # kz: derivative along Z (first spatial dim), smoothing along X and Y
        kz = smooth.view(1, 1, 3) * smooth.view(1, 3, 1) * deriv.view(3, 1, 1)
 
        # Shape: (1, 1, 3, 3, 3) -> (out_channels, in_channels/groups, D, H, W)
        return (
            kx.view(1, 1, 3, 3, 3),
            ky.view(1, 1, 3, 3, 3),
            kz.view(1, 1, 3, 3, 3),
        )
        
    def compute_multi_scale_loss(self, pred: torch.Tensor, target: torch.Tensor,
                                 scales=3, pp=1.0) -> torch.Tensor:
        """
        Multi-scale gradient difference loss.
 
        pred, target: (B, C, D, H, W)
        """
        if len(pred.shape) == 4:
            pred = pred.unsqueeze(1)
            target = target.unsqueeze(1)

        channels = pred.shape[1]

        k = self._scharr.to(device=pred.device, dtype=pred.dtype).repeat(channels, 1, 1, 1, 1)
    
        loss = pred.new_zeros(())
        for i in range(scales):
            if i > 0:
                p = F.avg_pool3d(pred, kernel_size=2 ** i)
                t = F.avg_pool3d(target, kernel_size=2 ** i)
            else:
                p, t = pred, target
 
            # Skip if any spatial dim becomes too small for a 3x3x3 conv
            if min(p.shape[-3:]) < 3:
                continue
 
            grad_p = F.conv3d(p, k, padding=1, groups=channels)
            grad_t = F.conv3d(t, k, padding=1, groups=channels)
 
            loss = loss + torch.mean(torch.abs(grad_p - grad_t) ** pp)

        return loss / scales

    def _ssim3d(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Average 2D SSIM across axial, coronal, and sagittal planes.

        pred/target: (B, D, H, W) in [0, 1].
        Averaging over all three planes gives a more isotropic structural
        similarity measure than a single-plane SSIM on (B, D, H, W) where D
        would be silently treated as the channel dimension.
        """
        B, D, H, W = pred.shape
        axial    = ssim(pred.reshape(B * D, 1, H, W),
                        target.reshape(B * D, 1, H, W),
                        data_range=1.0, size_average=True)
        coronal  = ssim(pred.permute(0, 2, 1, 3).reshape(B * H, 1, D, W),
                        target.permute(0, 2, 1, 3).reshape(B * H, 1, D, W),
                        data_range=1.0, size_average=True)
        sagittal = ssim(pred.permute(0, 3, 1, 2).reshape(B * W, 1, D, H),
                        target.permute(0, 3, 1, 2).reshape(B * W, 1, D, H),
                        data_range=1.0, size_average=True)
        return (axial + coronal + sagittal) / 3

    def compute_volume_loss(self, pred_vol, real_vol, vols_ref=None, ref_vol=None):
        self.loss_L_vol_l1 = self.criterionL1(pred_vol, real_vol)
        self.loss_L_ssim   = 1 - self._ssim3d(pred_vol, real_vol)
        alpha = 0.5
        self.loss_L_vol    = (alpha * self.loss_L_ssim
                              + (1 - alpha) * self.loss_L_vol_l1)

        if vols_ref is not None and ref_vol is not None:
            l1_ref   = self.criterionL1(vols_ref, ref_vol)
            ssim_ref = 1 - self._ssim3d(vols_ref, ref_vol)
            self.loss_L_vol_ref = (alpha * ssim_ref
                                   + (1 - alpha) * l1_ref)
        else:
            self.loss_L_vol_ref = torch.tensor(0.0, device=pred_vol.device)

        return 100 * (self.loss_L_vol + self.lambda_ref * self.loss_L_vol_ref)

    def compute_render_loss(self, pred_drrs, projections,
                            angles=None, lambda_l1=None, lambda_ssim=None):
        lambda_l1   = lambda_l1   if lambda_l1   is not None else self.lambda_l1
        lambda_ssim = lambda_ssim if lambda_ssim is not None else self.lambda_ssim

        if (lambda_l1 > 0) or (lambda_ssim > 0):
            pred_drrs   = normalize_min_max(pred_drrs)
            projections = normalize_min_max(projections)
            self.loss_L_render_L1   = self.criterionL1(pred_drrs, projections)
            ssim_val                = ssim(pred_drrs, projections,
                                          data_range=1, size_average=True)
            self.loss_L_render_SSIM = 1.0 - ssim_val
            self.loss_L_render      = (lambda_l1 * self.loss_L_render_L1
                                       + lambda_ssim * self.loss_L_render_SSIM)
        else:
            self.loss_L_render_L1   = torch.tensor(0.0)
            self.loss_L_render_SSIM = torch.tensor(0.0)
            self.loss_L_render      = torch.tensor(0.0)
        return self.loss_L_render

    def generate_visualization_slices(self, pred_vol, real_vol):
        mid_x = pred_vol.shape[1] // 2
        mid_y = pred_vol.shape[2] // 2
        mid_z = pred_vol.shape[3] // 2
        # Normalize both to the GT value range so WandB shows them on the same scale.
        # Without this, a single bright voxel in pred makes everything else look black.
        #v_min = real_vol[0].min()
        #v_max = real_vol[0].max()
        v_min = 0.0
        v_max = 1.0
        
        eps = 1e-8
        def norm(x):
            return (x.clamp(v_min, v_max) - v_min) / (v_max - v_min + eps)
        
        gt_axial = torch.flip(torch.rot90(norm(real_vol[0, :, :, mid_z]), k=3, dims=[0,1]), dims=[-1])
        pred_axial = torch.flip(torch.rot90(norm(pred_vol[0, :, :, mid_z]), k=3, dims=[0,1]), dims=[-1])
        
        gt_coronal = torch.rot90(norm(real_vol[0, :, mid_y, :]), k=1, dims=[0,1])
        pred_coronal = torch.rot90(norm(pred_vol[0, :, mid_y, :]), k=1, dims=[0,1])
        
        gt_sagittal = torch.rot90(norm(real_vol[0, mid_x, :, :]), k=1, dims=[0,1])
        pred_sagittal = torch.rot90(norm(pred_vol[0, mid_x, :, :]), k=1, dims=[0,1]) 

        gt   = torch.cat([gt_axial, gt_coronal, gt_sagittal], dim=0)
        pred = torch.cat([pred_axial, pred_coronal, pred_sagittal], dim=0)
        return {"ground_truth_slices": gt, "predicted_slices": pred}

    def generate_visualization_drrs(self, pred_drrs, real_drrs):
        C   = pred_drrs.shape[1]
        idx = torch.randperm(C)[:3]
        p   = torch.cat([pred_drrs[0, i] for i in idx], dim=0)
        r   = torch.cat([real_drrs[0, i] for i in idx], dim=0)
        return {"real_drr": r, "predicted_drr": p}

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(self, image_data, ref_ct, device='cuda:0'):
        """
        image_data keys:
            'projections'           : (B, N, H, W)
            'projections_reference' : (B, N, H, W)  [optional]
            'angles'                : (B, N) degrees
        ref_ct : (B, Nx, Ny, Nz) reference CT volume
        """
        projections = image_data['projections'].to(device)   # (B, N, H, W)
        angles      = image_data['angles']                   # (B, N), CPU tensor
        B, N, H, W  = projections.shape

        # ── 1-3. Encode + coarse 3D aggregation ─────────────────────────────
        # coverage_mask: (B, L_fx*L_fy*L_fz) — per-voxel fraction of views
        # that project inside the image FOV.  Passed to the refinement
        # transformer so poorly-covered voxels draw more from the reference.
        V_f_2D, encoded, coverage_mask = self._encode_and_aggregate_coarse(
            projections, angles, B, N, H, W, device)
        f_fine = encoded['fine']    # (B, N, C_a, H/8, W/8)
        f_mid  = encoded['mid']     # (B, N, C_b, H/16, W/16)
        _, _, _, H_f, W_f = f_fine.shape
        _, _, _, H_m, W_m = f_mid.shape
        scale_fine = H // H_f
        scale_mid  = H // H_m

        # ── 4. Fine / mid aggregation (skip connections for decoder) ─────────
        L_bx = self.opt.Nx // 8    # 3D grid size for mid scale
        L_by = self.opt.Ny // 8    # 3D grid size for mid scale
        L_bz = self.opt.Nz // 8    # 3D grid size for mid scale
        
        L_ax = self.opt.Nx // 4    # 3D grid size for fine scale
        L_ay = self.opt.Ny // 4    # 3D grid size for fine scale
        L_az = self.opt.Nz // 4    # 3D grid size for fine scale
        V_f_b = self._fast_aggregate_attn(
            f_mid.reshape(B*N, self.C_b, H_m, W_m), angles, B, N,
            L_bx, L_by, L_bz, self.C_b,
            scale_3d=8, scale_2d=scale_mid, aggr_net=self.netaggr_mid, device=device)
        V_f_a = self._fast_aggregate_attn(
            f_fine.reshape(B*N, self.C_a, H_f, W_f), angles, B, N,
            L_ax, L_ay, L_az, self.C_a,
            scale_3d=4, scale_2d=scale_fine, aggr_net=self.netaggr_fine, device=device)

        # ── 5. Spatial refine + self-refinement + decode (FDK-only / no ref) ──
        V_f_b = self.netfeature_spatial_refine_mid(V_f_b)
        V_f_a = self.netfeature_spatial_refine_fine(V_f_a)
        V_f_3D = self.netrefine_tf(V_f_2D)
        vols = self.netdecoder(V_f_3D, V_f_b, V_f_a).squeeze(1)
        self.vols_ref = None
        self.ref_vol  = None
        return vols


# ── Utilities ────────────────────────────────────────────────────────────────

def normalize_min_max(tensor, min_val=0.0, max_val=1.0,
                      normalize_dims=2, eps=1e-8):
    if normalize_dims < 1 or normalize_dims > tensor.ndim:
        raise ValueError(f"normalize_dims must be between 1 and {tensor.ndim}")
    reduce_dims  = tuple(range(-normalize_dims, 0))
    t_min        = tensor.amin(dim=reduce_dims, keepdim=True)
    t_max        = tensor.amax(dim=reduce_dims, keepdim=True)
    normalized   = (tensor - t_min) / (t_max - t_min + eps)
    return normalized * (max_val - min_val) + min_val
