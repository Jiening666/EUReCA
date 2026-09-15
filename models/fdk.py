"""
Differentiable FDK (Feldkamp–Davis–Kress) cone-beam backprojection + a
coverage-gated fusion module, for injecting an in-frame analytic prior into
the reconstruction decoder.

Why FDK injection (vs the reference-CT prior):
  * In-frame — reconstructed from the SOURCE's own projections, so it needs
    NO alignment/correspondence (the part of the reference path that would
    not train).
  * Correct content — it is the current anatomy, just noisy/streaky/
    incomplete; no inter-scan anatomy-change mismatch.
  * N-graded — streaky at few views, ~analytically correct at many, so the
    injected information improves with projection count (and a coverage-
    conditioned gate can trust it more as N grows).  This also gives the
    model an artifact-correction target (FBPConvNet-style), a well-posed
    learning problem.

DifferentiableFDK has no learnable parameters and is meant to run under
no_grad as fixed preprocessing (FDK is a deterministic transform of the
input projections); its output volume is then encoded by a learnable 3D CNN
and injected.  Geometry mirrors threeD_geometry.map_3d_grid_to_2d_batch and
models/renderer.py exactly (source at (SAD,0,0) rotated; x_rot = xr·c + yr·s;
source_dist = SAD − x_rot; detector Np=H rows, Nq=W cols, pixel size psize).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableFDK(nn.Module):
    def __init__(self, Nx, Ny, Nz, res, Np, Nq, psize, SAD=1000.0,
                 filter='ramp', vox_chunk=131072,
                 parker=False, parker_gamma_sign=1.0,
                 pad=False, cutoff=1.0, normalize_output=True):
        super().__init__()
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.res = res                      # isotropic voxel size (mm)
        self.Np, self.Nq = Np, Nq           # detector rows (H), cols (W)
        self.psize = psize
        self.SAD = SAD
        self.filter = filter
        # Parker short-scan redundancy weighting (off by default to preserve the
        # angular-interval weighting the model was trained with).
        self.parker = parker
        self.parker_gamma_sign = parker_gamma_sign
        self.normalize_output = bool(normalize_output)
        self.vox_chunk = vox_chunk

        # ── Cosine cone-beam pre-weight w(u,v) = SAD / sqrt(SAD^2+u^2+v^2) ──
        iq = torch.arange(Nq).float()
        ip = torch.arange(Np).float()
        u = (iq - Nq / 2.0 + 0.5) * psize                # column (fan) coord
        v = (Np / 2.0 - ip - 0.5) * psize                # row (cone) coord
        vv, uu = torch.meshgrid(v, u, indexing='ij')     # (Np, Nq)
        cos_w = SAD / torch.sqrt(SAD ** 2 + uu ** 2 + vv ** 2)
        self.register_buffer('cos_w', cos_w[None, None])  # (1,1,Np,Nq)

        # ── 1-D ramp (Ram-Lak) filter along the Nq (fan) axis ──────────────
        # Standard practice (RTK/ASTRA/TIGRE/skimage): zero-/edge-pad the fan
        # axis to >= 2*Nq (next power of two) before the ramp so the rfft is a
        # *linear* (not circular) convolution — kills FFT wrap-around and the
        # truncation rim at the FOV edge. Filter cutoff defaults to Nyquist
        # (cutoff=1.0 = pure Ram-Lak, the package default); lower it for a
        # smooth raised-cosine roll-off (less noise, slightly softer).
        self.pad   = bool(pad)
        nq_fft     = (1 << int(math.ceil(math.log2(max(2 * Nq, 2))))) if pad else Nq
        self.nq_fft = nq_fft
        freqs = torch.fft.rfftfreq(nq_fft)               # [0, 0.5]
        ramp  = 2.0 * freqs                              # |f|, normalised
        fn    = freqs / 0.5                              # 1.0 = Nyquist
        if filter == 'hann':
            ramp = ramp * (0.5 + 0.5 * torch.cos(math.pi * fn))
        elif cutoff < 1.0:
            w = torch.ones_like(fn)
            roll = fn > cutoff
            w[roll] = 0.5 * (1 + torch.cos(math.pi * (fn[roll] - cutoff) / (1 - cutoff)))
            w[fn > 1.0] = 0.0
            ramp = ramp * w
        self.register_buffer('ramp', ramp[None, None, None, :])  # (1,1,1,F)

        # ── Per-column fan angle γ (signed), for Parker weighting ──────────
        u_col = (torch.arange(Nq).float() - Nq / 2.0 + 0.5) * psize
        gamma_col = torch.atan(u_col / SAD)              # (Nq,)
        self.register_buffer('gamma_col', gamma_col)
        self.gamma_max = float(gamma_col.abs().max())

        # ── Voxel physical coordinates (centred at isocentre) ──────────────
        xx, yy, zz = torch.meshgrid(torch.arange(Nx), torch.arange(Ny),
                                    torch.arange(Nz), indexing='ij')
        xr = (xx.float() - Nx / 2.0) * res
        yr = (yy.float() - Ny / 2.0) * res
        zr = (zz.float() - Nz / 2.0) * res
        self.register_buffer('xr', xr.reshape(-1))       # (V,)
        self.register_buffer('yr', yr.reshape(-1))
        self.register_buffer('zr', zr.reshape(-1))

    @staticmethod
    def _angular_interval_weights(angles_deg):
        """
        Per-view weight = half the angular gap to its two neighbours (radians),
        so the weighted backprojection approximates the angular integral for
        ARBITRARY, possibly sparse / non-uniform angle sets — making the FDK
        magnitude roughly independent of N.  angles_deg: (B, N) → (B, N).
        """
        B, N = angles_deg.shape
        if N == 1:
            return torch.full_like(angles_deg, math.pi)
        a = torch.remainder(angles_deg.float(), 360.0)
        order = a.argsort(dim=1)
        a_sorted = torch.gather(a, 1, order)
        # circular neighbour gaps
        nxt = torch.roll(a_sorted, -1, dims=1)
        gap = torch.remainder(nxt - a_sorted, 360.0)     # forward gap (deg)
        # Clamp large gaps before they become weights.  For a limited arc the
        # wrap-around gap IS the missing wedge (~149° for a 211° scan); split
        # naively onto the two boundary views it over-weights them ~20×, dumping
        # those projections as huge streaks.  Cap at a few× the median local
        # spacing: the missing wedge is then simply not integrated (correct),
        # while genuine non-uniform sampling is still respected.  (Full 360°
        # scans have ~uniform gaps, so the clamp is inert there.)
        med = gap.median(dim=1, keepdim=True).values
        gap = torch.minimum(gap, 3.0 * med)
        prev_gap = torch.roll(gap, 1, dims=1)
        w_sorted = (gap + prev_gap) * 0.5 * (math.pi / 180.0)
        w = torch.empty_like(w_sorted)
        w.scatter_(1, order, w_sorted)                   # back to view order
        return w

    def _filter_projections(self, proj, extra_w=None):
        """proj: (B, N, Np, Nq) → cosine-weighted, ramp-filtered (same shape).
        extra_w: optional (B, N, 1, Nq) per-view/column weight (e.g. Parker),
        applied before the ramp filter."""
        x = proj * self.cos_w
        if extra_w is not None:
            x = x * extra_w
        if self.pad and self.nq_fft > self.Nq:
            total = self.nq_fft - self.Nq
            pl, pr = total // 2, total - total // 2
            left  = x[..., :1].expand(*x.shape[:-1], pl)   # edge-replicate (reduces
            right = x[..., -1:].expand(*x.shape[:-1], pr)  # truncation step at FOV edge)
            x = torch.cat([left, x, right], dim=-1)
            X = torch.fft.rfft(x, dim=-1) * self.ramp.to(torch.complex64)
            y = torch.fft.irfft(X, n=self.nq_fft, dim=-1)
            return y[..., pl:pl + self.Nq]
        X = torch.fft.rfft(x, dim=-1)
        X = X * self.ramp.to(X.dtype)
        return torch.fft.irfft(X, n=self.Nq, dim=-1)

    def _parker_weights(self, angles_deg):
        """
        Parker (1982) short-scan redundancy weights, (B, N, Nq), and the
        per-view fan-beam angle β (B, N) measured along the rotation direction.
        Scan must span ~π + 2·γ_max; the smooth taper at both arc ends makes the
        weighted backprojection ≈ a full 180° integral with no double-counting in
        the redundant overlap (removes the short-scan shading the interval
        weighting leaves behind).
        """
        B, N = angles_deg.shape
        a = angles_deg.float()
        # rotation direction from the median signed inter-view step
        da = torch.remainder(torch.diff(a, dim=1) + 180.0, 360.0) - 180.0
        dirn = torch.sign(torch.median(da, dim=1, keepdim=True).values)
        dirn = torch.where(dirn == 0, torch.ones_like(dirn), dirn)
        # β ∈ [0, arc], increasing along rotation, in radians
        beta = torch.remainder((a - a[:, :1]) * dirn, 360.0) * (math.pi / 180.0)

        gm = self.gamma_max
        g = (self.gamma_col * self.parker_gamma_sign).view(1, 1, -1)   # (1,1,Nq)
        b = beta.view(B, N, 1)                                          # (B,N,1)
        end = math.pi + 2.0 * gm

        w = torch.ones(B, N, self.Nq, device=a.device, dtype=a.dtype)
        # region 1: 0 ≤ β < 2(γm−γ)
        w1 = torch.sin(0.25 * math.pi * b / (gm - g).clamp_min(1e-6)) ** 2
        w = torch.where(b < 2.0 * (gm - g), w1, w)
        # region 3: π−2γ < β ≤ π+2γm
        w3 = torch.sin(0.25 * math.pi * (end - b) / (gm + g).clamp_min(1e-6)) ** 2
        w = torch.where(b > (math.pi - 2.0 * g), w3, w)
        # outside the nominal short-scan span → 0
        w = torch.where((b < 0) | (b > end), torch.zeros_like(w), w)
        return w, beta

    @torch.no_grad()
    def forward(self, projections, angles_deg):
        """
        projections : (B, N, Np, Nq)  attenuation projections (any scale)
        angles_deg  : (B, N)          gantry angles, degrees
        Returns fdk volume (B, 1, Nx, Ny, Nz), per-sample min-max → [0, 1].
        """
        device = projections.device
        B, N, Np, Nq = projections.shape

        # Parker only applies to a genuine SHORT scan (~180deg + fan), so the arc
        # is guarded at BOTH ends (coverage = 360 - largest gap):
        #   upper — on a near-full arc (e.g. full-360 training DRRs) the taper
        #           would zero the excess views;
        #   lower — below a half scan there is no redundant overlap to
        #           compensate, so Parker's premise fails outright.  The region-1
        #           taper sin^2(pi/4 * beta/(gm-g)) is exactly 0 at beta=0, and
        #           beta=0 is the FIRST view by construction, so on a narrow arc
        #           it simply deletes that view: at N=2 90deg apart the surviving
        #           view carries 99.9% of the signal (single-angle streaks).
        #           Region 3 never fires below ~149deg either, so what is left is
        #           an asymmetric taper rather than redundancy compensation.
        # Either way we fall back to interval weighting, which is defined for
        # arbitrary sparse / narrow angle sets.  N <= 2 is excluded outright: the
        # taper needs neighbouring views to hand off to (and at N=1 beta=0 is the
        # only view, which would zero the whole volume).
        use_parker = self.parker and N > 2
        if use_parker:
            a_s = torch.remainder(angles_deg.float(), 360.0).sort(dim=1).values
            gaps = torch.remainder(a_s.roll(-1, dims=1) - a_s, 360.0)
            coverage = 360.0 - gaps.max(dim=1).values            # (B,)
            use_parker = bool(((coverage >= 180.0) & (coverage <= 270.0)).all())

        if use_parker:
            pw, beta = self._parker_weights(angles_deg.to(device))   # (B,N,Nq),(B,N)
            filt = self._filter_projections(projections.float(),
                                            extra_w=pw.unsqueeze(2))  # (B,N,1,Nq)
            # uniform dβ for the angular integral (Parker handles redundancy)
            dbeta = beta.max(dim=1, keepdim=True).values / max(N - 1, 1)
            view_w = dbeta.expand(B, N).reshape(B, N, 1)
        else:
            filt = self._filter_projections(projections.float())     # (B,N,Np,Nq)
            view_w = self._angular_interval_weights(angles_deg).to(device).reshape(B, N, 1)
        filt = filt.reshape(B * N, 1, Np, Nq)

        ang = angles_deg.to(device).float() * (math.pi / 180.0)
        s = torch.sin(ang).reshape(B, N, 1)
        c = torch.cos(ang).reshape(B, N, 1)

        V = self.xr.numel()
        out = torch.zeros(B, V, device=device)
        for st in range(0, V, self.vox_chunk):
            en = min(st + self.vox_chunk, V)
            xr = self.xr[st:en].view(1, 1, -1)           # (1,1,Vc)
            yr = self.yr[st:en].view(1, 1, -1)
            zr = self.zr[st:en].view(1, 1, -1)

            x_rot = xr * c + yr * s                       # (B,N,Vc)
            source_dist = (self.SAD - x_rot).clamp_min(1e-3)
            factor = self.SAD / source_dist
            y_det = (-xr * s + yr * c) * factor
            z_det = zr * factor

            iq = ((y_det / self.psize + Nq / 2 - 0.5) / (Nq - 1)) * 2 - 1
            ip = ((Np / 2 - z_det / self.psize - 0.5) / (Np - 1)) * 2 - 1
            grid = torch.stack((iq, ip), dim=-1).reshape(B * N, en - st, 1, 2)

            samp = F.grid_sample(filt, grid, mode='bilinear',
                                 padding_mode='zeros', align_corners=True)
            samp = samp.reshape(B, N, en - st)
            # FDK distance weight (SAD/U)^2 × angular-interval weight, sum views
            w = view_w * factor ** 2
            out[:, st:en] = (samp * w).sum(dim=1)

        if not self.normalize_output:
            return out.view(B, 1, self.Nx, self.Ny, self.Nz)

        # ── Air-at-zero, robust normalisation to [0,1] ─────────────────────
        # The ramp filter removes the DC component, so the raw FDK volume is
        # ~zero-mean: air/background sits near 0 with NEGATIVE ramp undershoot,
        # tissue is positive.  A (v-lo)/(hi-lo) min-max with lo<0 shifts air up
        # to mid-grey and washes the volume out (looks far worse than real
        # FDK).  Instead clamp the negative undershoot to 0 (air -> black, the
        # GT convention) and scale by a high percentile (robust to streak
        # spikes) so tissue spans [0,1].  Absolute HU is unrecoverable here;
        # this is a range/contrast match consumed by the learnable FDK encoder.
        flat = out.clamp_min(0.0)                          # (B, V); air/ring → 0
        step = max(1, flat.shape[1] // 500000)             # cheap quantile est.
        hi = torch.quantile(flat[:, ::step], 0.99, dim=1, keepdim=True)
        vol = (flat / (hi + 1e-8)).clamp(0, 1)
        return vol.view(B, 1, self.Nx, self.Ny, self.Nz)


class CoverageGatedFusion(nn.Module):
    """
    Inject an (in-frame) prior's features into a decoder skip with a per-voxel
    gate conditioned on the source feature, the prior feature, the projection
    coverage at this scale, AND the iteration/prior-quality level:

        g    = σ( Conv3d([src, prior, coverage, iter_level]) )
        out  = src + g · proj(prior)

    The prior is whatever in-frame volume is injected this pass — FDK on the
    first iteration, the model's own previous output on later iterations.
    Coverage conditioning: prior reliability grows with view count/spread, so
    the gate can lean on it where coverage is high.  Iteration conditioning:
    the prior gets better each refinement iteration, so the gate can learn to
    trust it more at higher iteration levels.

    Live init (small-nonzero proj, gate bias +1) so injection is active from
    step 1 — mirrors the fix that revived the reference branch.
    """
    def __init__(self, ch):
        super().__init__()
        # +1 coverage channel, +1 iteration-level channel
        self.gate = nn.Conv3d(ch * 2 + 2, 1, 3, padding=1)
        self.proj = nn.Conv3d(ch, ch, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, 1.0)
        nn.init.normal_(self.proj.weight, std=0.05)
        nn.init.zeros_(self.proj.bias)

    def forward(self, src, prior, coverage, iter_level=0.0):
        # src, prior: (B, C, Lx, Ly, Lz); coverage: (B, 1, Lx, Ly, Lz) in [0,1]
        # iter_level: scalar in [0,1] (0 = first/FDK iteration), broadcast in.
        B = src.shape[0]
        il = torch.full((B, 1, *src.shape[2:]), float(iter_level),
                        device=src.device, dtype=src.dtype)
        g = torch.sigmoid(self.gate(torch.cat([src, prior, coverage, il], dim=1)))
        inject = g * self.proj(prior)
        self.last_gate = g.detach()
        num = inject.detach().reshape(B, -1).norm(dim=1)
        den = src.detach().reshape(B, -1).norm(dim=1).clamp_min(1e-8)
        self.last_inj_ratio = num / den
        return src + inject
