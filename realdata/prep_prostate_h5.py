"""Prepare a Halcyon HyperSight iTools h5 projection dump for model testing.

Input  (per scan folder, see prostate_dump/README.md):
    h5/Proj_#####.hdf   dataset 'Projection' (rows, 3060) float32, ALREADY
                        log-normalised (dumped after Analytical Spectrum
                        Correction, before Filter & Decimation); acquisition
                        metadata in HDF5 attrs of AcqParams / ProjParams.
    recon_vol/*.dcm     iTools clinical reconstruction (real HU).

Output (our standard evaluation format, --out_dir):
    projs.npy   (N, 384, 768) float32  line integrals on the ISO-plane
                                       virtual detector (psize 0.7273 mm)
    angles.npy  (N,)          float32  degrees in OUR convention
                                       (source at (DSO cos t, DSO sin t, 0))
    gt.npy      (256,256,128) float32  clinical recon, HU -> [0,1] on our grid
    zmask.npy   (128,)        bool     slices actually covered by the detector
    meta.json   geometry + provenance + angle-convention record

Geometry notes (measured from a real dump, P01):
  SAD 1000, SID 1540 -> magnification 1.54. ImagerResY 1.12 mm at the imager
  = 0.7273 mm at iso == our psize exactly (rows need NO rescaling); ImagerResX
  0.28 mm = 0.1818 mm at iso = psize/4 (columns bin by 4).
  Detector rows 212 -> 154.2 mm of z at iso, i.e. LESS than our 256 mm volume:
  the covered band is recorded in zmask (BladeY1/BladeY2 are asymmetric,
  e.g. -48.5..+106.4 mm), so metrics can be restricted to covered slices.

Angle convention: iTools stores ProjParams.GantryRtn (IEC, deg) and
SourceAngle = GantryRtn - 90. Which sign/offset maps onto our synthetic
convention is resolved empirically by realdata/verify_prostate_geometry.py
(FDK vs the clinical recon); the chosen mapping is passed via --angle_mode
and recorded in meta.json.
"""
import argparse
import json
import glob
import os.path as osp
from pathlib import Path

import numpy as np


ANGLE_MODES = {
    # name           : lambda gantry_deg -> our theta (deg)
    # VERIFIED on P01 (verify_prostate_geometry + refine_real_geometry):
    # gantry_p180 wins (NCC 0.87 with ZERO residual shift once the detector
    # row placement is correct); plain gantry / neg_gantry / +-90 variants all
    # need large incoherent translations to fit -> wrong.
    'gantry_p180':    lambda g: g + 180.0,
    'gantry':         lambda g: g,
    'neg_gantry':     lambda g: -g,
    'gantry_p90':     lambda g: g + 90.0,
    'neg_gantry_p90': lambda g: -g + 90.0,
    'gantry_m90':     lambda g: g - 90.0,
    'neg_gantry_m90': lambda g: -g - 90.0,
    'source':         lambda g: g - 90.0,          # iTools SourceAngle
    'neg_source':     lambda g: -(g - 90.0),
}


def _attrs(h5obj):
    out = {}
    for k, v in h5obj.attrs.items():
        try:
            vv = v[0] if hasattr(v, '__len__') and len(v) == 1 else v
            out[k] = vv.decode() if isinstance(vv, bytes) else (
                float(vv) if isinstance(vv, (np.floating, np.integer)) else vv)
        except Exception:
            pass
    return out


def load_projections(scan_dir, psize=0.7273, out_H=384, out_W=768,
                     angle_mode='gantry', verbose=True):
    import h5py
    from scipy import ndimage

    files = sorted(glob.glob(osp.join(scan_dir, 'h5', 'Proj_*.hdf')))
    if not files:
        raise SystemExit(f'no Proj_*.hdf under {scan_dir}/h5')

    with h5py.File(files[0], 'r') as h:
        acq = _attrs(h['AcqParams'])
    SAD = float(acq.get('SAD', 1000.0))
    SID = float(acq.get('SID', 1540.0))
    resX = float(acq.get('ImagerResX', 0.28))    # column pitch at imager (mm)
    resY = float(acq.get('ImagerResY', 1.12))    # row pitch at imager (mm)
    mag = SID / SAD
    zoom_r = (resY / mag) / psize                # rows  -> ~1.0
    zoom_c = (resX / mag) / psize                # cols  -> ~0.25

    # Row placement.
    # The dump stores ONLY the illuminated rows (verified across all 15 scans:
    # the non-zero band spans the whole array), and the row count varies per
    # scan (212..384) because the blades crop the detector at acquisition.
    # The band's z centre is where the DETECTOR sits: ImagerLng at the imager
    # plane -> ImagerLng/mag at iso. Measured on all 15 scans this agrees with
    # the blade centre 0.5*(BladeY1+BladeY2) to <=0.16 mm, but ImagerLng is the
    # physically primary quantity (blades only crop), and the offset is
    # per-scan (-9.5 .. +29.1 mm) -- never hard-code it.
    # SIGN: row 0 of our detector convention is superior (+z), so a band
    # centred at +z must be placed at LOWER row indices -> negative shift.
    # Empirically confirmed on P01 (row-placement calibration: residual z
    # misalignment 0 mm at row_shift = -40 = -(29.09/0.7273)).
    bY1 = float(acq.get('BladeY1', 0.0)); bY2 = float(acq.get('BladeY2', 0.0))
    imager_lng = float(acq.get('ImagerLng', 0.0))
    band_centre_iso = imager_lng / mag           # mm at iso
    blade_centre_iso = 0.5 * (bY1 + bY2)
    row_shift = -int(round(band_centre_iso / psize))

    projs, gantry = [], []
    _src_rows = None
    for i, f in enumerate(files):
        with h5py.File(f, 'r') as h:
            img = np.asarray(h['Projection'][()], dtype=np.float32)
            pp = _attrs(h['ProjParams'])
        if _src_rows is None:
            _src_rows = img.shape[0]
        g = float(pp.get('GantryRtn', np.nan))
        z = ndimage.zoom(img, (zoom_r, zoom_c), order=1)
        zh, zw = z.shape
        out = np.zeros((out_H, out_W), np.float32)
        # columns: centre
        left_src = max(0, (zw - out_W) // 2); left_dst = max(0, (out_W - zw) // 2)
        w = min(zw, out_W)
        # rows: centre the physical band, then apply its z offset
        top_dst = (out_H - zh) // 2 + row_shift
        top_src = 0
        if top_dst < 0:
            top_src = -top_dst; top_dst = 0
        hgt = min(zh - top_src, out_H - top_dst)
        if hgt > 0 and w > 0:
            out[top_dst:top_dst + hgt, left_dst:left_dst + w] = \
                z[top_src:top_src + hgt, left_src:left_src + w]
        projs.append(out)
        gantry.append(g)
        if verbose and i % 100 == 0:
            print(f'  proj {i}/{len(files)}  gantry={g:.2f}')

    gantry = np.asarray(gantry, np.float32)
    theta = np.asarray(ANGLE_MODES[angle_mode](gantry), np.float32) % 360.0
    # illuminated band extent at iso, from the stored row count + its centre
    band_half = 0.5 * (zoom_r * _src_rows) * psize
    meta = dict(acq=acq, SAD=SAD, SID=SID, mag=mag, psize=psize,
                zoom_rows=zoom_r, zoom_cols=zoom_c, n_proj=len(files),
                src_rows=_src_rows, row_shift=row_shift,
                band_centre_iso=band_centre_iso, blade_centre_iso=blade_centre_iso,
                imager_lng=imager_lng, band_half_iso=band_half,
                band_extent_iso=[band_centre_iso - band_half, band_centre_iso + band_half],
                blade_y=[bY1, bY2], angle_mode=angle_mode,
                gantry_first=float(gantry[0]), gantry_last=float(gantry[-1]))
    return np.stack(projs), theta, gantry, meta


def load_clinical_recon(scan_dir, Nx=256, Ny=256, Nz=128, vox=2.0,
                        ct_min=-1000.0, ct_max=1000.0):
    """iTools DICOM series -> (gt01 (Nx,Ny,Nz), zmask_info) on OUR grid.

    Our grid is isocentre-centred, x/y/z = 2 mm, (256,256,128). DICOM gives
    patient coords via ImagePositionPatient / PixelSpacing; for Halcyon CBCT
    the recon is isocentre-centred, so we resample by physical coordinates.
    """
    import pydicom
    from scipy import ndimage

    files = sorted(glob.glob(osp.join(scan_dir, 'recon_vol', '*.dcm')))
    if not files:
        raise SystemExit(f'no dcm under {scan_dir}/recon_vol')
    sl = [pydicom.dcmread(f) for f in files]
    sl.sort(key=lambda d: float(d.ImagePositionPatient[2]))
    arr = np.stack([s.pixel_array.astype(np.float32) for s in sl])      # (Z,Y,X)
    inter = float(getattr(sl[0], 'RescaleIntercept', 0.0))
    slope = float(getattr(sl[0], 'RescaleSlope', 1.0))
    hu = arr * slope + inter

    py, px = [float(v) for v in sl[0].PixelSpacing]                      # (row, col) mm
    ipp0 = np.array([float(v) for v in sl[0].ImagePositionPatient], np.float64)
    z_positions = np.array([float(s.ImagePositionPatient[2]) for s in sl])
    dz = float(np.median(np.diff(z_positions))) if len(sl) > 1 else 1.0
    ny_src, nx_src = arr.shape[1], arr.shape[2]
    x_extent = [float(ipp0[0]), float(ipp0[0] + (nx_src - 1) * px)]
    y_extent = [float(ipp0[1]), float(ipp0[1] + (ny_src - 1) * py)]

    # physical coords of our target grid (iso-centred)
    gx = (np.arange(Nx) - (Nx - 1) / 2.0) * vox
    gy = (np.arange(Ny) - (Ny - 1) / 2.0) * vox
    gz = (np.arange(Nz) - (Nz - 1) / 2.0) * vox
    # DICOM voxel indices for those coords
    ix = (gx - ipp0[0]) / px
    iy = (gy - ipp0[1]) / py
    iz = (gz - z_positions[0]) / dz
    IZ, IY, IX = np.meshgrid(iz, iy, ix, indexing='ij')                  # (Nz,Ny,Nx)
    samp = ndimage.map_coordinates(hu, [IZ, IY, IX], order=1,
                                   mode='constant', cval=ct_min)
    vol_hu = np.transpose(samp, (2, 1, 0))                               # -> (Nx,Ny,Nz)
    gt01 = np.clip((vol_hu - ct_min) / (ct_max - ct_min), 0.0, 1.0).astype(np.float32)
    info = dict(dicom_shape=list(arr.shape), pixel_spacing=[py, px], dz=dz,
                ipp0=ipp0.tolist(), z_range=[float(z_positions[0]), float(z_positions[-1])],
                hu_range=[float(hu.min()), float(hu.max())],
                x_extent=x_extent, y_extent=y_extent,
                fov_mm=[float(nx_src * px), float(ny_src * py)])
    return gt01, info


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scan_dir', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--angle_mode', default='gantry_p180', choices=list(ANGLE_MODES))
    p.add_argument('--psize', type=float, default=0.7273)
    p.add_argument('--proj_H', type=int, default=384)
    p.add_argument('--proj_W', type=int, default=768)
    p.add_argument('--vox', type=float, default=2.0)
    p.add_argument('--skip_gt', action='store_true')
    args = p.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    projs, theta, gantry, meta = load_projections(
        args.scan_dir, args.psize, args.proj_H, args.proj_W, args.angle_mode)
    np.save(out / 'projs.npy', projs)
    np.save(out / 'angles.npy', theta)
    np.save(out / 'gantry.npy', gantry)
    print(f'projections {projs.shape}  angles {theta.min():.1f}..{theta.max():.1f} '
          f'(gantry {gantry.min():.1f}..{gantry.max():.1f})')

    Nz = 128
    gz = (np.arange(Nz) - (Nz - 1) / 2.0) * args.vox
    # Coverage = the illuminated band actually stored (row count x pitch about
    # the detector centre). Equals the blade opening to <1 mm on all 15 scans,
    # but is measured from the data rather than assumed.
    band_lo, band_hi = meta['band_extent_iso']
    zmask = (gz >= band_lo) & (gz <= band_hi)

    if not args.skip_gt:
        gt, ginfo = load_clinical_recon(args.scan_dir, vox=args.vox)
        np.save(out / 'gt.npy', gt)
        meta['clinical_recon'] = ginfo
        print(f'clinical recon -> {gt.shape}  HU {ginfo["hu_range"]}')
        # The clinical recon spans only its own z range (it follows the
        # illuminated band, e.g. -40..+90 mm, not the blades' -48.5..+106.4);
        # slices outside it are constant padding in the reference, so they
        # must NOT count as "covered" or every metric is computed partly
        # against fabricated air.
        zlo, zhi = ginfo['z_range']
        zmask &= (gz >= zlo) & (gz <= zhi)
        meta['z_mask_source'] = 'blades AND clinical-recon extent'

    np.save(out / 'zmask.npy', zmask)
    meta['z_covered_slices'] = int(zmask.sum())

    # ---- 3-D VALIDITY MASK -------------------------------------------------
    # Metrics must be restricted to the PHYSICALLY INTERSECTED region: our grid
    # AND the clinical recon's own support AND the illuminated band. The
    # clinical in-plane FOV varies a lot (282 mm for HN at 0.55 mm spacing vs
    # 538 mm for pelvis), so on some scans most of our 512 mm grid holds only
    # fabricated air -- scoring air against air inflates every method's PSNR
    # and compresses the differences between them.
    gx = (np.arange(256) - 127.5) * args.vox
    gy = (np.arange(256) - 127.5) * args.vox
    if not args.skip_gt:
        xe = meta['clinical_recon']['x_extent']; ye = meta['clinical_recon']['y_extent']
        inx = (gx >= min(xe)) & (gx <= max(xe))
        iny = (gy >= min(ye)) & (gy <= max(ye))
    else:
        inx = np.ones(256, bool); iny = np.ones(256, bool)
    valid = (inx[:, None, None] & iny[None, :, None] & zmask[None, None, :])
    np.save(out / 'validmask.npy', valid)
    frac = float(valid.mean())
    meta['valid_voxels'] = int(valid.sum())
    meta['valid_fraction'] = frac
    print(f'valid (intersected) region: {valid.sum():,} voxels = {100*frac:.1f}% of the '
          f'grid  [x {int(inx.sum())}/256, y {int(iny.sum())}/256, z {int(zmask.sum())}/128]')
    print(f'z coverage: {zmask.sum()}/{Nz} slices  (band {band_lo:.1f}..{band_hi:.1f} mm, '
          f'centre {meta["band_centre_iso"]:+.1f} mm from ImagerLng, '
          f'{meta["src_rows"]} src rows'
          + (f'; recon {meta["clinical_recon"]["z_range"][0]:.1f}..'
             f'{meta["clinical_recon"]["z_range"][1]:.1f} mm)' if not args.skip_gt else ')'))

    (out / 'meta.json').write_text(json.dumps(meta, indent=1, default=str))
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
