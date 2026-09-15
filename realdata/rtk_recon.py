"""
rtk_recon.py — RTK FDK reconstruction matched to DifferentiableFDK's grid.

Reconstructs a scan with both RTK (CPU) and DifferentiableFDK at the SAME
(Nx,Ny,Nz) grid, aligns RTK into the same volume frame (using the asymmetric
high-density structures), and writes both as .nii.gz for direct comparison.

Usage: python rtk_recon.py [scan] [n_views] [angle_mode]

angle_mode:
  raw        use stored angles directly
  offset     use stored angles - 90 deg
  legacy     use -stored angles - 90 deg (old default)
"""
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
import numpy as np
import nibabel as nib
import itk
from itk import RTK as rtk
import torch
from models.fdk import DifferentiableFDK

SAD, SID, PSIZE = 1000.0, 1540.0, 0.7273
DET = PSIZE * SID / SAD                        # detector-plane pixel = 1.12 mm
Nx, Ny, Nz, VOX = 256, 256, 128, 2.0           # match DifferentiableFDK
IT = itk.Image[itk.F, 3]

scan   = sys.argv[1] if len(sys.argv) > 1 else "anon012_prostatebed"
n_use  = int(sys.argv[2]) if len(sys.argv) > 2 else 135
angle_mode = sys.argv[3] if len(sys.argv) > 3 else "raw"
root   = f"D:/Jiening/prostate/processed/{scan}"
out    = f"{root}/_diagnostic"; os.makedirs(out, exist_ok=True)

projs = np.load(f"{root}/projections.npy").astype(np.float32)
angs  = np.load(f"{root}/angles.npy").astype(np.float32)
idx = np.linspace(0, len(projs) - 1, min(n_use, len(projs))).round().astype(int)
projs, angs = projs[idx], angs[idx]
Nproj, Np, Nq = projs.shape
print(f"{scan}: {Nproj} views")

def map_angles(angles, mode):
    if mode == "raw":
        return np.mod(angles, 360.0).astype(np.float32)
    if mode == "offset":
        return np.mod(angles - 90.0, 360.0).astype(np.float32)
    if mode == "legacy":
        return np.mod(-angles - 90.0, 360.0).astype(np.float32)
    raise ValueError(f"unknown angle_mode {mode!r}; use raw, offset, or legacy")


# ---------------- DifferentiableFDK (your convention) ----------------
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
p = projs / max(float(projs.max()), 1e-9)
a = map_angles(angs, angle_mode)
print(f"angle_mode={angle_mode}: {a.min():.1f}..{a.max():.1f} deg")
mine = DifferentiableFDK(Nx, Ny, Nz, VOX, Np, Nq, PSIZE, SAD=SAD,
                         parker=True, pad=True).to(dev)(
        torch.from_numpy(p)[None].to(dev), torch.from_numpy(a)[None].to(dev)
    ).squeeze().cpu().numpy()                   # (Nx,Ny,Nz)
mine_name = f"mine_fdk_{angle_mode}.nii.gz"
nib.save(nib.Nifti1Image(mine.astype(np.float32), np.diag([VOX, VOX, VOX, 1.0])),
         f"{out}/{mine_name}")
print(f"saved {mine_name} {mine.shape}")

# ---------------- RTK FDK ----------------
pim = itk.image_from_array(np.ascontiguousarray(projs))
pim.SetSpacing([DET, DET, 1.0])
pim.SetOrigin([-(Nq - 1) * DET / 2.0, -(Np - 1) * DET / 2.0, 0.0])
geo = rtk.ThreeDCircularProjectionGeometry.New()
for ang in angs:
    geo.AddProjection(SAD, SID, float((180.0 - ang) % 360.0))     # RTK Varian convention
pk = rtk.ParkerShortScanImageFilter[IT].New()
pk.SetInput(pim); pk.SetGeometry(geo); pk.Update()
# RTK rotation axis = y -> longitudinal voxels go on size_y (=Nz here)
vs = rtk.ConstantImageSource[IT].New()
vs.SetSpacing([VOX, VOX, VOX]); vs.SetSize([Nx, Nz, Ny])
vs.SetOrigin([-(Nx - 1) * VOX / 2, -(Nz - 1) * VOX / 2, -(Ny - 1) * VOX / 2])
vs.SetConstant(0.0)
fdk = rtk.FDKConeBeamReconstructionFilter[IT].New()
fdk.SetInput(0, vs.GetOutput()); fdk.SetInput(1, pk.GetOutput()); fdk.SetGeometry(geo)
t = time.time(); fdk.Update()
rtk_raw = itk.array_from_image(fdk.GetOutput())          # (z=Ny, y=Nz, x=Nx)
print(f"RTK FDK done {rtk_raw.shape}  {time.time()-t:.0f}s")
# put longitudinal (RTK y) last -> (Nx?,Nx?,Nz) = (256,256,128) to match mine
rtk_vol = np.transpose(rtk_raw, (0, 2, 1)).clip(0, None)  # (Ny->256, Nx->256, Nz->128)

# ---------------- align RTK transverse axes to mine ----------------
ref = mine - mine.mean()
def corr(v):
    v = v - v.mean()
    return float((ref * v).sum() / (np.linalg.norm(ref) * np.linalg.norm(v) + 1e-9))
best, best_c = rtk_vol, -2
for k in range(4):                                # rot in transverse (axes 0,1)
    r = np.rot90(rtk_vol, k, axes=(0, 1))
    for fl in (r, r[::-1], r[:, ::-1], r[::-1, ::-1]):
        if fl.shape == mine.shape:
            c = corr(fl)
            if c > best_c:
                best, best_c = fl.copy(), c
print(f"best RTK->mine alignment corr = {best_c:.3f}")
nib.save(nib.Nifti1Image(best.astype(np.float32), np.diag([VOX, VOX, VOX, 1.0])),
         f"{out}/rtk_fdk.nii.gz")
print(f"saved rtk_fdk.nii.gz {best.shape}  (aligned to mine_fdk.nii.gz)")
