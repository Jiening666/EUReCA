"""
Truly multi-scale CNN feature encoder for grayscale projections.

Each output feature map is computed *at* its spatial resolution by a distinct
stage of the network, not by downsampling a higher-resolution feature. This
gives honest multi-scale representations where:
  - Coarse-scale features have large effective receptive fields, aggregating
    information across wide image regions into abstract semantic content.
  - Fine-scale features have small receptive fields, preserving precise local
    detail (edges, textures).
  - Mid-scale features are balanced between the two.

Architecture (ConvNeXt-style blocks, GroupNorm, 1-channel input):

  Input:          (B, 1, H, W)          H, W must be multiples of 32
    |
    v  Stem (two 3x3 stride-2)
  After stem:     (B,  64, H/4,  W/4)           -- 1/4 res, not returned
    |
    v  Stage 1 (no downsample, just blocks at 1/4 res)
  Stage 1 out:    (B,  64, H/4,  W/4)           -- 1/4 res, optional extra-fine output
    |
    v  Downsample + Stage 2
  Stage 2 out:    (B, 128, H/8,  W/8)   <-- FINE    scale 1/8
    |
    v  Downsample + Stage 3
  Stage 3 out:    (B, 256, H/16, W/16)  <-- MID     scale 1/16
    |
    v  Downsample + Stage 4
  Stage 4 out:    (B, 512, H/32, W/32)  <-- COARSE  scale 1/32

Then a top-down FPN pathway injects deep-semantic content into finer scales,
so each returned feature map carries both its stage's native content and
context from the coarser stages above it.

Final output: dict with 'fine', 'mid', 'coarse' (and optionally 'xfine') feature
maps at 1/8, 1/16, 1/32 of the input resolution respectively.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict


# ---------------------------------------------------------------------------
# ConvNeXt-style residual block.
# ---------------------------------------------------------------------------

class ConvNeXtBlock(nn.Module):
    """
    Depthwise 7x7 -> GroupNorm -> 1x1 expand -> GELU -> 1x1 contract -> layer-scale
    residual add. Layer-scale initialized to 1e-6 so each block starts as
    near-identity, which stabilizes from-scratch training.
    """

    def __init__(self, dim: int, expansion: int = 4, layer_scale_init: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(min(32, dim), dim)
        self.pwconv1 = nn.Conv2d(dim, dim * expansion, kernel_size=1)
        self.pwconv2 = nn.Conv2d(dim * expansion, dim, kernel_size=1)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = F.gelu(x)
        x = self.pwconv2(x)
        return r + self.gamma * x


# ---------------------------------------------------------------------------
# Between-stage downsampler.
# ---------------------------------------------------------------------------

class Downsample(nn.Module):
    """GroupNorm + 2x2 stride-2 conv. Halves H/W, changes channel count."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, in_dim), in_dim)
        self.reduce = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reduce(self.norm(x))


# ---------------------------------------------------------------------------
# Stem: 1-channel input -> 1/4 resolution, preserving fine detail.
# ---------------------------------------------------------------------------

class GrayscaleStem(nn.Module):
    """
    Two 3x3 stride-2 convs instead of a single 4x4 stride-4 patch-embed.
    Gentler compression, better detail preservation for grayscale/scientific data.
    Optional per-image instance normalization handles high-dynamic-range input.
    """

    def __init__(self, out_dim: int, instance_norm_input: bool = False):
        super().__init__()
        self.instance_norm_input = instance_norm_input
        mid = out_dim // 2
        self.conv1 = nn.Conv2d(1, mid, kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.GroupNorm(min(32, mid), mid)
        self.conv2 = nn.Conv2d(mid, out_dim, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.GroupNorm(min(32, out_dim), out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.instance_norm_input:
            mean = x.mean(dim=(2, 3), keepdim=True)
            std = x.std(dim=(2, 3), keepdim=True).clamp(min=1e-5)
            x = (x - mean) / std
        x = F.gelu(self.norm1(self.conv1(x)))
        x = F.gelu(self.norm2(self.conv2(x)))
        return x


# ---------------------------------------------------------------------------
# 2D FPN: top-down fusion across scales.
# ---------------------------------------------------------------------------

class FPN2D(nn.Module):
    """
    Top-down pathway: coarse features are upsampled and merged into mid,
    then the refined mid is upsampled and merged into fine. Each merge goes
    through a 1x1 lateral conv (channel alignment), addition, then a 3x3
    smoothing conv.

    Without this, the fine-scale feature carries only stage-2 content (local
    texture). With it, every scale carries both its native information and
    context from deeper stages. This is essential: it's what makes the fine
    scale actually useful for 3D reconstruction rather than just "pretty edges
    with no structural meaning."
    """

    def __init__(self, channels: Tuple[int, int, int]):
        super().__init__()
        c_fine, c_mid, c_coarse = channels

        self.lat_coarse_to_mid = nn.Conv2d(c_coarse, c_mid,  kernel_size=1)
        self.lat_mid_to_fine   = nn.Conv2d(c_mid,    c_fine, kernel_size=1)

        self.smooth_mid  = nn.Conv2d(c_mid,  c_mid,  kernel_size=3, padding=1)
        self.smooth_fine = nn.Conv2d(c_fine, c_fine, kernel_size=3, padding=1)

        self.norm_mid  = nn.GroupNorm(min(32, c_mid),  c_mid)
        self.norm_fine = nn.GroupNorm(min(32, c_fine), c_fine)

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        fine, mid, coarse = feats

        coarse_up = F.interpolate(coarse, size=mid.shape[-2:], mode="bilinear", align_corners=False)
        mid = mid + 0.1 * self.lat_coarse_to_mid(coarse_up)
        mid = F.gelu(self.norm_mid(self.smooth_mid(mid)))

        mid_up = F.interpolate(mid, size=fine.shape[-2:], mode="bilinear", align_corners=False)
        fine = fine + 0.1 * self.lat_mid_to_fine(mid_up)
        fine = F.gelu(self.norm_fine(self.smooth_fine(fine)))

        return [fine, mid, coarse]


# ---------------------------------------------------------------------------
# Main multi-scale encoder.
# ---------------------------------------------------------------------------

class MultiScaleCNNEncoder(nn.Module):
    """
    Truly multi-scale CNN feature encoder for grayscale projection images.

    Args:
        depths:       blocks per stage, 4 stages total. Default matches ConvNeXt-Tiny.
        dims:         channels per stage. Fine/mid/coarse are dims[1:4].
        pyramid_out:  optional fixed channel counts for the output pyramid. If None,
                      outputs have native dims (dims[1], dims[2], dims[3]). If set,
                      1x1 conv projections align channels to (fine, mid, coarse) targets.
        use_fpn:      apply top-down FPN fusion across scales (recommended).
        return_xfine: also return the stage-1 (1/4 resolution) feature, useful if you
                      want an extra-fine scale for 128^3 voxel grids.
        instance_norm_input: z-score each input image at the stem (recommended for
                      HDR scientific/medical data).

    Input:  (B, N_views, 1, H, W) grayscale. H and W must be multiples of 32.
            Output spatial scales are H/8 × W/8, H/16 × W/16, H/32 × W/32.
    Output: dict with keys 'fine', 'mid', 'coarse' (and 'xfine' if enabled),
            each of shape (B, N_views, C_scale, H_scale, W_scale).
    """

    def __init__(
        self,
        depths: Tuple[int, int, int, int] = (2, 2, 6, 2),
        dims:   Tuple[int, int, int, int] = (64, 128, 256, 512),
        pyramid_out: Optional[Tuple[int, int, int]] = None,
        use_fpn: bool = True,
        return_xfine: bool = False,
        instance_norm_input: bool = False,
        backbone: str = '',
        pretrained: bool = True,
    ):
        super().__init__()
        assert len(depths) == 4 and len(dims) == 4
        self.return_xfine = return_xfine
        self.use_fpn = use_fpn
        self.backbone_name = backbone
        self.instance_norm_input = instance_norm_input

        # ── Backbone: pretrained timm (features_only) OR from-scratch ConvNeXt ──
        if backbone:
            import timm
            # 1-channel input; taps at strides 8 / 16 / 32 -> fine / mid / coarse.
            # in_chans=1 makes timm average the pretrained RGB stem to one channel.
            self.backbone = timm.create_model(
                backbone, pretrained=pretrained, features_only=True,
                out_indices=(1, 2, 3), in_chans=1,
            )
            tap_channels = tuple(self.backbone.feature_info.channels())  # (fine, mid, coarse)
            self.stem = None
        else:
            self.backbone = None
            # Stem: (B, 1, H, W) -> (B, dims[0], H/4, W/4)
            self.stem = GrayscaleStem(out_dim=dims[0], instance_norm_input=instance_norm_input)
            # Stage 1 at 1/4 res (also the optional 'xfine' tap).
            self.stage1 = nn.Sequential(*[ConvNeXtBlock(dims[0]) for _ in range(depths[0])])
            # Stage 2 at 1/8 -> FINE.
            self.down12 = Downsample(dims[0], dims[1])
            self.stage2 = nn.Sequential(*[ConvNeXtBlock(dims[1]) for _ in range(depths[1])])
            # Stage 3 at 1/16 -> MID.
            self.down23 = Downsample(dims[1], dims[2])
            self.stage3 = nn.Sequential(*[ConvNeXtBlock(dims[2]) for _ in range(depths[2])])
            # Stage 4 at 1/32 -> COARSE.
            self.down34 = Downsample(dims[2], dims[3])
            self.stage4 = nn.Sequential(*[ConvNeXtBlock(dims[3]) for _ in range(depths[3])])
            tap_channels = (dims[1], dims[2], dims[3])

        native_pyramid_channels = tap_channels

        # Optional channel projection to a fixed output pyramid (e.g., 256/512/1024).
        # Useful if the downstream 3D pipeline expects specific channel counts
        # regardless of which backbone size was used.
        if pyramid_out is not None:
            c_f, c_m, c_c = pyramid_out
            self.proj_fine   = nn.Conv2d(tap_channels[0], c_f, kernel_size=1)
            self.proj_mid    = nn.Conv2d(tap_channels[1], c_m, kernel_size=1)
            self.proj_coarse = nn.Conv2d(tap_channels[2], c_c, kernel_size=1)
            self.norm_fine   = nn.GroupNorm(min(32, c_f), c_f)
            self.norm_mid    = nn.GroupNorm(min(32, c_m), c_m)
            self.norm_coarse = nn.GroupNorm(min(32, c_c), c_c)
            pyramid_channels = pyramid_out
        else:
            self.proj_fine = self.proj_mid = self.proj_coarse = None
            pyramid_channels = native_pyramid_channels

        self.out_channels = pyramid_channels  # exposed for downstream modules

        # FPN acts on the (possibly projected) pyramid channels.
        if use_fpn:
            self.fpn = FPN2D(channels=pyramid_channels)
        else:
            self.fpn = None

        # Initialize only the new (non-pretrained) modules; never re-init the
        # pretrained backbone weights.
        if self.backbone is None:
            self.apply(self._init_weights)
        else:
            for m in (self.proj_fine, self.proj_mid, self.proj_coarse,
                      getattr(self, 'norm_fine', None), getattr(self, 'norm_mid', None),
                      getattr(self, 'norm_coarse', None), self.fpn):
                if m is not None:
                    m.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GroupNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def _project(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply optional 1x1 channel projections + GroupNorm to pyramid levels."""
        if self.proj_fine is None:
            return feats
        fine, mid, coarse = feats
        fine   = self.norm_fine(self.proj_fine(fine))
        mid    = self.norm_mid(self.proj_mid(mid))
        coarse = self.norm_coarse(self.proj_coarse(coarse))
        return [fine, mid, coarse]

    def _forward_single_batch(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: (B, 1, H, W) single-batch forward (no view dimension).
        Returns dict of feature maps at each scale.
        """
        s1 = None
        if self.backbone is not None:
            # Per-image z-score before a pretrained backbone (HDR/contrast
            # invariance; bridges [0,1] DRRs toward the backbone's input stats).
            if self.instance_norm_input:
                mean = x.mean(dim=(2, 3), keepdim=True)
                std  = x.std(dim=(2, 3), keepdim=True).clamp(min=1e-5)
                x = (x - mean) / std
            fine, mid, coarse = self.backbone(x)   # strides 8 / 16 / 32
        else:
            x = self.stem(x)           # -> (B, dims[0], H/4, W/4)
            s1 = self.stage1(x)        # -> (B, dims[0], H/4, W/4)
            x = self.down12(s1);  fine   = self.stage2(x)   # 1/8  FINE
            x = self.down23(fine); mid    = self.stage3(x)   # 1/16 MID
            x = self.down34(mid);  coarse = self.stage4(x)   # 1/32 COARSE

        # Channel projection to target pyramid dims (if configured).
        fine, mid, coarse = self._project([fine, mid, coarse])

        # Optional FPN top-down fusion.
        if self.fpn is not None:
            fine, mid, coarse = self.fpn([fine, mid, coarse])

        out = {"fine": fine, "mid": mid, "coarse": coarse}
        if self.return_xfine and s1 is not None:
            out["xfine"] = s1
        return out

    def forward(self, views: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        views: (B, N_views, 1, H, W).
        Returns dict of feature maps, each (B, N_views, C_scale, H_scale, W_scale).
        """
        assert views.dim() == 5 and views.shape[2] == 1, \
            f"Expected (B, N, 1, H, W), got {tuple(views.shape)}"
        B, N, _, H, W = views.shape
        assert H % 32 == 0 and W % 32 == 0, "H and W must be multiples of 32."

        x = views.reshape(B * N, 1, H, W)
        feats = self._forward_single_batch(x)

        # Reshape each output back to (B, N, C, H', W').
        out = {}
        for k, v in feats.items():
            out[k] = v.reshape(B, N, v.shape[1], v.shape[2], v.shape[3])
        return out


# ---------------------------------------------------------------------------
# Size variants.
# ---------------------------------------------------------------------------

def ms_cnn_small(**kw) -> MultiScaleCNNEncoder:
    """~4M params. Prototyping, small datasets."""
    return MultiScaleCNNEncoder(depths=(2, 2, 4, 2), dims=(32, 64, 128, 256), **kw)


def ms_cnn_base(**kw) -> MultiScaleCNNEncoder:
    """~11M params. Recommended default."""
    return MultiScaleCNNEncoder(depths=(2, 2, 6, 2), dims=(64, 128, 256, 512), **kw)


def ms_cnn_large(**kw) -> MultiScaleCNNEncoder:
    """~30M params. Only if base plateaus on a large dataset."""
    return MultiScaleCNNEncoder(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), **kw)


# ---------------------------------------------------------------------------
# Sanity check.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import itertools

    configs = [
        ("small, native channels, FPN on",  ms_cnn_small,  dict()),
        ("base,  native channels, FPN on",  ms_cnn_base,   dict()),
        ("base,  fixed pyramid (256/512/1024)", ms_cnn_base,
            dict(pyramid_out=(256, 512, 1024))),
        ("base,  with xfine output",        ms_cnn_base,
            dict(return_xfine=True)),
        ("large, FPN off (raw stage outputs)", ms_cnn_large,
            dict(use_fpn=False)),
    ]

    # Any (H, W) that are multiples of 32 work; 512x512 shown here as an example.
    views = torch.randn(2, 8, 1, 512, 512)  # (B=2, N=8, 1, H=512, W=512)

    for name, ctor, kwargs in configs:
        model = ctor(**kwargs).eval()
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        with torch.no_grad():
            out = model(views)
        print(f"\n{name}  ({n_params:.2f}M params):")
        for k, v in out.items():
            print(f"  {k:6s}: {tuple(v.shape)}")

    # For 512x512 input, 'base, native channels, FPN on' outputs:
    #   fine  : (2, 8, 128,  64,  64)   H/8  x W/8
    #   mid   : (2, 8, 256,  32,  32)   H/16 x W/16
    #   coarse: (2, 8, 512,  16,  16)   H/32 x W/32
