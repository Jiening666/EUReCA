"""Quantify what our 256x256x128 @2mm grid costs relative to the clinical recon.

Three separate error sources are measured on real scans:

  (1) FOV CROP      our 512 mm in-plane FOV vs the clinical 538 mm: how much
                    non-air anatomy falls outside our grid entirely.
  (2) Z PAD/CROP    the clinical recon spans only its illuminated band; our
                    grid spans 256 mm -> how many of our slices have no
                    reference at all (already handled by zmask, reported here).
  (3) RESAMPLE      1.051 -> 2.0 mm in-plane is a ~1.9x DOWNSAMPLE. Linear
                    interpolation without a low-pass prefilter aliases high
                    frequencies (bone edges). Measured by a round trip:
                    native -> our grid -> back to native, compared on the
                    shared FOV, with and without an anti-alias prefilter.

The round-trip number is the ceiling on how faithfully ANY method scored on
our grid can represent the clinical image, and tells us whether the prefilter
is worth adding to the GT resample.

    python realdata/quantify_resample_error.py --scan_dir <raw scan> [...]
"""
import argparse
import glob
import os.path as osp

import numpy as np


def load_native(scan_dir, ct_min=-1000.0):
    import pydicom
    files = sorted(glob.glob(osp.join(scan_dir, 'recon_vol', '*.dcm')))
    sl = [pydicom.dcmread(f) for f in files]
    sl.sort(key=lambda d: float(d.ImagePositionPatient[2]))
    arr = np.stack([s.pixel_array.astype(np.float32) for s in sl])
    hu = arr * float(getattr(sl[0], 'RescaleSlope', 1.0)) + \
        float(getattr(sl[0], 'RescaleIntercept', 0.0))
    py, px = [float(v) for v in sl[0].PixelSpacing]
    ipp0 = np.array([float(v) for v in sl[0].ImagePositionPatient])
    z = np.array([float(s.ImagePositionPatient[2]) for s in sl])
    dz = float(np.median(np.diff(z))) if len(sl) > 1 else 1.0
    return hu, (py, px, dz), ipp0, z


def to_our_grid(hu, spac, ipp0, z, Nx=256, Ny=256, Nz=128, vox=2.0,
                ct_min=-1000.0, prefilter=False):
    from scipy import ndimage
    py, px, dz = spac
    src = hu
    if prefilter:
        # anti-alias before decimation: sigma so the source is band-limited to
        # the target Nyquist (standard sigma = sqrt(r^2-1)/2 in source pixels)
        rx, ry = vox / px, vox / py
        sx = np.sqrt(max(rx * rx - 1.0, 0.0)) / 2.0
        sy = np.sqrt(max(ry * ry - 1.0, 0.0)) / 2.0
        rz = vox / abs(dz)
        sz = np.sqrt(max(rz * rz - 1.0, 0.0)) / 2.0
        src = ndimage.gaussian_filter(hu, sigma=(sz, sy, sx), mode='nearest')
    gx = (np.arange(Nx) - (Nx - 1) / 2.0) * vox
    gy = (np.arange(Ny) - (Ny - 1) / 2.0) * vox
    gz = (np.arange(Nz) - (Nz - 1) / 2.0) * vox
    ix = (gx - ipp0[0]) / px
    iy = (gy - ipp0[1]) / py
    iz = (gz - z[0]) / dz
    IZ, IY, IX = np.meshgrid(iz, iy, ix, indexing='ij')
    out = ndimage.map_coordinates(src, [IZ, IY, IX], order=1,
                                  mode='constant', cval=ct_min)
    return np.transpose(out, (2, 1, 0)), (gx, gy, gz)


def back_to_native(vol_ours, grids, hu_shape, spac, ipp0, z, vox=2.0, ct_min=-1000.0):
    from scipy import ndimage
    gx, gy, gz = grids
    py, px, dz = spac
    nz, ny, nx = hu_shape
    nxs = ipp0[0] + np.arange(nx) * px
    nys = ipp0[1] + np.arange(ny) * py
    nzs = z
    JX = (nxs - gx[0]) / vox
    JY = (nys - gy[0]) / vox
    JZ = (nzs - gz[0]) / vox
    Z, Y, X = np.meshgrid(JZ, JY, JX, indexing='ij')
    src = np.transpose(vol_ours, (2, 1, 0))          # (Nz,Ny,Nx)
    return ndimage.map_coordinates(src, [Z, Y, X], order=1,
                                   mode='constant', cval=ct_min)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scan_dir', required=True)
    p.add_argument('--vox', type=float, default=2.0)
    p.add_argument('--body_hu', type=float, default=-500.0,
                   help='HU above which a voxel counts as anatomy (not air)')
    args = p.parse_args()

    hu, spac, ipp0, z = load_native(args.scan_dir)
    py, px, dz = spac
    nz, ny, nx = hu.shape
    print(f'clinical: {hu.shape} (Z,Y,X)  spacing {px:.3f}x{py:.3f}x{dz:.2f} mm  '
          f'FOV {nx*px:.0f}x{ny*py:.0f}x{nz*abs(dz):.0f} mm')
    print(f'our grid: 256x256x128 @{args.vox} mm  FOV '
          f'{256*args.vox:.0f}x{256*args.vox:.0f}x{128*args.vox:.0f} mm')

    # ---- (1) in-plane FOV crop -------------------------------------------
    xs = ipp0[0] + np.arange(nx) * px
    ys = ipp0[1] + np.arange(ny) * py
    half = 128 * args.vox                                   # 256 mm
    inx = np.abs(xs) <= half
    iny = np.abs(ys) <= half
    body = hu > args.body_hu
    total_body = int(body.sum())
    keep = body[:, iny][:, :, inx]
    lost = total_body - int(keep.sum())
    print(f'\n(1) in-plane crop: {lost}/{total_body} anatomy voxels outside our FOV '
          f'= {100.0*lost/max(total_body,1):.3f}%')

    # ---- (2) z extent -----------------------------------------------------
    gz = (np.arange(128) - 63.5) * args.vox
    zcov = (gz >= z.min()) & (gz <= z.max())
    print(f'(2) z: clinical spans {z.min():.0f}..{z.max():.0f} mm; our grid covers '
          f'{gz.min():.0f}..{gz.max():.0f} mm -> {int(zcov.sum())}/128 slices have a '
          f'reference ({128-int(zcov.sum())} are padding)')

    # ---- (3) resampling round trip ---------------------------------------
    for pref in (False, True):
        ours, grids = to_our_grid(hu, spac, ipp0, z, vox=args.vox, prefilter=pref)
        back = back_to_native(ours, grids, hu.shape, spac, ipp0, z, vox=args.vox)
        m = np.zeros_like(hu, bool)
        yy, xx = np.ix_(iny, inx)
        m[:, yy, xx] = True
        m &= (hu > args.body_hu)
        err = (back - hu)[m]
        rmse = float(np.sqrt((err ** 2).mean()))
        mae = float(np.abs(err).mean())
        # PSNR on the [-1000,1000] window both pipelines use
        psnr = 10 * np.log10(1.0 / max(float(((err / 2000.0) ** 2).mean()), 1e-12))
        print(f'(3) round trip {"WITH" if pref else "without"} anti-alias prefilter: '
              f'RMSE {rmse:6.1f} HU   MAE {mae:6.1f} HU   PSNR {psnr:5.2f} dB')


if __name__ == '__main__':
    main()
