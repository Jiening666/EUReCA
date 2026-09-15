import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class DifferentiableDRR(torch.nn.Module):
    def __init__(self, Nx, Ny, Nz, resx, resy, resz, Np, Nq, psize_x, psize_y, nSample, sample_step, SAD=1000.0):
        super().__init__()
        self.SAD = SAD

        # CT geometry
        self.Nx   = Nx
        self.Ny   = Ny
        self.Nz   = Nz
        self.resx = resx
        self.resy = resy
        self.resz = resz
        self.isox = Nx / 2.0
        self.isoy = Ny / 2.0
        self.isoz = Nz / 2.0

        # DRR geometry
        self.Np          = Np
        self.Nq          = Nq
        self.psize_x     = psize_x
        self.psize_y     = psize_y
        self.nSample     = nSample
        self.sample_step = sample_step

    def _build_detector_pixels(self, device):
        """
        Replicates your kernel's detector pixel layout.
        Returns source (x2,y2,z2) in the un-rotated frame: [Np, Nq, 3]
        """
        # ip = row index (0..Np-1), iq = col index (0..Nq-1)
        ip = torch.arange(self.Np, device=device, dtype=torch.float32)
        iq = torch.arange(self.Nq, device=device, dtype=torch.float32)
        grid_p, grid_q = torch.meshgrid(ip, iq, indexing='ij')  # [Np, Nq]

        x2 = torch.zeros_like(grid_p)
        y2 = (grid_q + 0.5 - self.Nq / 2.0) * self.psize_x   # your kernel line
        z2 = (self.Np / 2.0 - grid_p - 0.5) * self.psize_y   # your kernel line

        # Stack: [Np, Nq, 3]
        det_local = torch.stack([x2, y2, z2], dim=-1)
        return det_local

    def _rotate_points(self, pts, sin_a, cos_a):
        """
        Apply your kernel's rotation: x' = x*cos - y*sin, y' = x*sin + y*cos
        pts: [..., 3], sin_a/cos_a: [B] scalars
        Returns [..., 3] rotated, broadcast over batch dim B.
        """
        # pts shape: [Np, Nq, 3] → broadcast with [B, 1, 1]
        x = pts[..., 0]
        y = pts[..., 1]
        z = pts[..., 2]

        # sin_a, cos_a: [B] → [B, 1, 1]
        s = sin_a[:, None, None]
        c = cos_a[:, None, None]

        xr = x * c - y * s
        yr = x * s + y * c
        zr = z  # unchanged

        return torch.stack([xr, yr, zr], dim=-1)  # [B, Np, Nq, 3]

    def _build_sample_points(self, sin_a, cos_a):
        """
        Build all ray sample points in world space.
        Returns pts: [B, Np, Nq, nSample, 3], dt: [B, Np, Nq, 1]
        """
        B = sin_a.shape[0]
        device = sin_a.device

        # Source (fixed, then rotated): x1=SAD, y1=0, z1=0
        src_local = torch.tensor([self.SAD, 0.0, 0.0], device=device)
        src_local = src_local.expand(self.Np, self.Nq, 3)    # [Np,Nq,3]

        # Detector pixels (un-rotated frame)
        det_local = self._build_detector_pixels(device)      # [Np,Nq,3]

        # Rotate both into world frame
        # expand to [B, Np, Nq, 3]
        src_local_b = src_local.unsqueeze(0).expand(B, -1, -1, -1)
        det_local_b = det_local.unsqueeze(0).expand(B, -1, -1, -1)

        xr1 = src_local_b[..., 0] * cos_a[:,None,None] - src_local_b[..., 1] * sin_a[:,None,None]
        yr1 = src_local_b[..., 0] * sin_a[:,None,None] + src_local_b[..., 1] * cos_a[:,None,None]
        zr1 = src_local_b[..., 2]
        src_world = torch.stack([xr1, yr1, zr1], dim=-1)    # [B,Np,Nq,3]

        xr2 = det_local_b[..., 0] * cos_a[:,None,None] - det_local_b[..., 1] * sin_a[:,None,None]
        yr2 = det_local_b[..., 0] * sin_a[:,None,None] + det_local_b[..., 1] * cos_a[:,None,None]
        zr2 = det_local_b[..., 2]
        det_world = torch.stack([xr2, yr2, zr2], dim=-1)    # [B,Np,Nq,3]

        # Ray direction and length (for oblique factor)
        diff  = src_world - det_world                        # [B,Np,Nq,3]
        dist  = diff.norm(dim=-1, keepdim=True)              # [B,Np,Nq,1]
        slope = diff / dist                                   # unit direction

        # Sample parameter: your kernel goes tis in [-nSample/2, nSample/2)
        tis = torch.arange(-self.nSample//2, self.nSample//2,
                           device=device, dtype=torch.float32)  # [S]

        # pts = det_world + tis * slope * sample_step
        # [B,Np,Nq,1,3] + [S,1]*[B,Np,Nq,1,3]
        pts = det_world.unsqueeze(-2) + \
              tis[None, None, None, :, None] * slope.unsqueeze(-2) * self.sample_step
        # pts: [B, Np, Nq, S, 3]

        return pts, dist  # dist: [B, Np, Nq, 1]

    def _world_to_grid(self, pts):
        """
        Replicates your kernel's voxel index computation, then maps to [-1,1]
        for grid_sample.
        
        Your kernel:
            xrs_idx = xxx/resx + isox
            xx = floor(xrs_idx)
        We need normalised coords for grid_sample (which expects [-1,1]).
        
        grid_sample convention: -1 = first voxel, +1 = last voxel
        """
        x = pts[..., 0]
        y = pts[..., 1]
        z = pts[..., 2]

        # Convert world → voxel index (same as your kernel)
        xi = x / self.resx + self.isox   # [B,Np,Nq,S]
        yi = y / self.resy + self.isoy
        zi = z / self.resz + self.isoz

        # Normalise voxel index to [-1, 1] for grid_sample
        # grid_sample: index 0 → -1, index N-1 → +1
        xn = (xi / (self.Nx - 1)) * 2 - 1
        yn = (yi / (self.Ny - 1)) * 2 - 1
        zn = (zi / (self.Nz - 1)) * 2 - 1

        # grid_sample 5D expects [x, y, z] order (W, H, D)
        # Your volume is [B, Nz, Ny, Nx] → grid_sample maps (x→W, y→H, z→D)
        grid = torch.stack([xn, yn, zn], dim=-1)  # [B, Np, Nq, S, 3]
        return grid

    def forward_(self, vol, angles_deg):
        """
        vol:        [B, Nz, Ny, Nx]
        angles_deg: [B, C]  — C angles processed together, vol shared (not copied)

        Returns: proj [B, C, Np, Nq]
        """
        B, C = angles_deg.shape
        device = vol.device

        angle_rad = angles_deg * (torch.pi / 180.0)         # [B, C]
        s = torch.sin(angle_rad)[:, :, None, None]           # [B, C, 1, 1]
        c = torch.cos(angle_rad)[:, :, None, None]

        det_local = self._build_detector_pixels(device)      # [Np, Nq, 3]
        dl = det_local[None, None]                           # [1, 1, Np, Nq, 3]

        # Rotate detector and source into world frame: [B, C, Np, Nq, 3]
        det_x = dl[..., 0] * c - dl[..., 1] * s
        det_y = dl[..., 0] * s + dl[..., 1] * c
        det_z = dl[..., 2].expand(B, C, self.Np, self.Nq)
        det_world = torch.stack([det_x, det_y, det_z], dim=-1)   # [B,C,Np,Nq,3]

        # Source rotates as (SAD,0,0): [B, C, 1, 1, 3] broadcast over Np,Nq
        src_world = torch.stack([
            self.SAD * c.expand(B, C, self.Np, self.Nq),
            self.SAD * s.expand(B, C, self.Np, self.Nq),
            torch.zeros(B, C, self.Np, self.Nq, device=device),
        ], dim=-1)                                               # [B,C,Np,Nq,3]

        diff  = src_world - det_world                            # [B,C,Np,Nq,3]
        dist  = diff.norm(dim=-1, keepdim=True)                  # [B,C,Np,Nq,1]
        slope = diff / dist

        tis = torch.arange(-self.nSample // 2, self.nSample // 2,
                           device=device, dtype=torch.float32)
        pts = (det_world.unsqueeze(-2)
               + tis[None, None, None, None, :, None]
               * slope.unsqueeze(-2) * self.sample_step)         # [B,C,Np,Nq,S,3]

        # Normalise to [-1, 1] for grid_sample
        x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
        xn = (x / self.resx + self.isox) / (self.Nx - 1) * 2 - 1
        yn = (y / self.resy + self.isoy) / (self.Ny - 1) * 2 - 1
        zn = (z / self.resz + self.isoz) / (self.Nz - 1) * 2 - 1
        grid = torch.stack([xn, yn, zn], dim=-1)                 # [B,C,Np,Nq,S,3]

        # grid_sample: vol [B,1,Nz,Ny,Nx] shared — NOT copied per angle
        vol_5d    = vol.unsqueeze(1).float()
        grid_flat = grid.reshape(B, C * self.Np * self.Nq * self.nSample, 1, 1, 3)

        sampled = F.grid_sample(vol_5d, grid_flat,
                                mode='bilinear', padding_mode='zeros',
                                align_corners=True)   # [B,1,C*Np*Nq*S,1,1]

        mu = (sampled.squeeze(1).squeeze(-1).squeeze(-1)
              .reshape(B, C, self.Np, self.Nq, self.nSample))

        oblique = self.SAD / dist.squeeze(-1)                    # [B,C,Np,Nq]
        return mu.sum(dim=-1) * self.sample_step * oblique       # [B,C,Np,Nq]


    def forward(self, vol, angles_deg, angle_chunk_size=1):
        """
        vol:        [B, Nz, Ny, Nx]
        angles_deg: [B, N]

        Returns: proj [B, N, Np, Nq]
        """
        B, N = angles_deg.shape
        chunks = []
        for i in range(0, N, angle_chunk_size):
            angle_chunk = angles_deg[:, i:i + angle_chunk_size]   # [B, C]
            proj_chunk = checkpoint(self.forward_, vol, angle_chunk, use_reentrant=False)
            chunks.append(proj_chunk)                              # [B, C, Np, Nq]
        return torch.cat(chunks, dim=1)                            # [B, N, Np, Nq]