import numpy as np
import math

def map_3d_grid_to_2d_batch(Nx, Ny, Nz,
                            resx, resy, resz,
                            batch_angles,
                            Np, Nq, psize, SAD=1000.0):
    """Vectorized version — replaces nested Python loops with NumPy broadcasting.
    Same outputs as the original; ~10x faster for large grids (e.g. 64^3).
    """
    xx, yy, zz = np.meshgrid(np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing='ij')
    xr = (xx - Nx / 2.0) * resx  # (Nx, Ny, Nz)
    yr = (yy - Ny / 2.0) * resy
    zr = (zz - Nz / 2.0) * resz

    angle_rad = batch_angles * math.pi / 180
    s = np.sin(angle_rad)[:, :, None, None, None]   # (B, N, 1, 1, 1)
    c = np.cos(angle_rad)[:, :, None, None, None]

    x_rot = xr * c + yr * s                          # (B, N, Nx, Ny, Nz)
    y_rot = -xr * s + yr * c

    source_dist = np.where(SAD - x_rot == 0, 1e-6, SAD - x_rot)
    factor  = SAD / source_dist
    y_det   = y_rot * factor
    z_det   = zr * factor                            # zr broadcasts

    iq_batch = ((y_det / psize + Nq / 2 - 0.5) / (Nq - 1)) * 2 - 1
    ip_batch = ((Np / 2 - z_det / psize - 0.5) / (Np - 1)) * 2 - 1
    return ip_batch, iq_batch


def map_3d_grid_to_2d_batch_original(Nx, Ny, Nz,
                            resx, resy, resz, 
                            batch_angles, 
                            Np, Nq, psize, SAD=1000.0):
    """
    Maps a full 3D grid of CT indices to 2D projection indices for a batch.
    
    Returns:
        ip_batch: (Batch, N, Nx, Ny, Nz) array of vertical detector indices normalized
        iq_batch: (Batch, N, Nx, Ny, Nz) array of horizontal detector indices normalized 
    """
    # 1. Create 3D Grid of indices (using 'ij' indexing to match your [z, y, x] order)
    # Using 0 to N-1 for indices
    #zz, yy, xx = np.meshgrid(np.arange(Nz), np.arange(Ny), np.arange(Nx), indexing='ij')
    xx, yy, zz = np.meshgrid(np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing='ij')
    
    isox = Nx / 2.0
    isoy = Ny / 2.0
    isoz = Nz / 2.0

    # 2. Convert to Physical CT Space (relative to isocenter)
    # Reversing: xrs_idx = xxx/resx + proj_isox 
    xxx = (xx - isox) * resx
    yyy = (yy - isoy) * resy
    zzz = (zz - isoz) * resz
    
    # xr = xxx.reshape(Nz, Ny, Nx)
    # yr = yyy.reshape(Nz, Ny, Nx)
    # zr = zzz.reshape(Nz, Ny, Nx)
    
    xr = xxx.reshape(Nx, Ny, Nz)
    yr = yyy.reshape(Nx, Ny, Nz)
    zr = zzz.reshape(Nx, Ny, Nz)
    
    angle_rad = batch_angles * math.pi / 180
    sinAngle_batch = np.sin(angle_rad)
    cosAngle_batch = np.cos(angle_rad)

    batch_size = batch_angles.shape[0]
    n_images = batch_angles.shape[1]
    # ip_batch = np.zeros((*batch_angles.shape, Nz, Ny, Nx))
    # iq_batch = np.zeros((*batch_angles.shape, Nz, Ny, Nx))
    ip_batch = np.zeros((*batch_angles.shape, Nx, Ny, Nz))
    iq_batch = np.zeros((*batch_angles.shape, Nx, Ny, Nz))

    for b in range(batch_size):
        for n in range(n_images):
            # 4. De-rotate from Gantry angle back to the Source-Detector coordinate system
            s, c = sinAngle_batch[b, n], cosAngle_batch[b, n]
            # These are the coordinates in the "unrotated" beam frame
            x_rot = xr * c + yr * s
            y_rot = -xr * s + yr * c
            z_rot = zr

            # 5. Perspective Projection
            # Source at (SAD, 0, 0), Detector at x=0
            # Calculate scaling factor based on distance from source
            source_dist = SAD - x_rot
            # Avoid division by zero for points at the source focal point
            source_dist[source_dist == 0] = 1e-6 
            
            factor = SAD / source_dist
            y_det = y_rot * factor
            z_det = z_rot * factor

            # 6. Map to Pixel Indices
            # Reversing your kernel's detector geometry:
            
            # iq_batch[b, n] = (y_det / psize) + (Nq / 2) - 0.5
            # ip_batch[b, n] = (Np / 2) - (z_det / psize) - 0.5
            
            # output range [-1,1] (for F.grid_sample) 
            iq_batch[b, n] = ((y_det / psize + Nq/2 - 0.5) / (Nq - 1)) * 2 - 1
            ip_batch[b, n] = ((Np/2 - z_det/psize - 0.5) / (Np - 1)) * 2 - 1

    return ip_batch, iq_batch
        
def map_3d_point_array_to_2d_batch(points, resx, resy, resz, 
                            isox, isoy, isoz, 
                            sinAngle_batch, cosAngle_batch, 
                            Np, Nq, psize, SAD=1000.0):
    """
    Maps a sequence of 3d points of CT indices to 2D projection indices for a batch.
    points is expected to be of shape [num_points, 3] in (z, y, x) order, with indices from 0 to N-1.
    
    Returns:
        ip_batch: (Batch, Nz, Ny, Nx) array of vertical detector indices
        iq_batch: (Batch, Nz, Ny, Nx) array of horizontal detector indices
    """
    # 1. Create 3D Grid of indices (using 'ij' indexing to match your [z, y, x] order)
    # Using 0 to N-1 for indices
    zz, yy, xx = points[:, 0], points[:, 1], points[:, 2]

    # 2. Convert to Physical CT Space (relative to isocenter)
    # Reversing: xrs_idx = xxx/resx + proj_isox 
    xxx = (xx - isox) * resx
    yyy = (yy - isoy) * resy
    zzz = (zz - isoz) * resz
    
    xr = xxx
    yr = yyy
    zr = zzz

    batch_size = sinAngle_batch.shape[0]
    ip_batch = np.zeros((batch_size, len(points)))
    iq_batch = np.zeros((batch_size, len(points)))

    for b in range(batch_size):
        # 4. De-rotate from Gantry angle back to the Source-Detector coordinate system
        s, c = sinAngle_batch[b], cosAngle_batch[b]
        # These are the coordinates in the "unrotated" beam frame
        x_rot = xr * c + yr * s
        y_rot = -xr * s + yr * c
        z_rot = zr

        # 5. Perspective Projection
        # Source at (SAD, 0, 0), Detector at x=0
        # Calculate scaling factor based on distance from source
        source_dist = SAD - x_rot
        # Avoid division by zero for points at the source focal point
        source_dist[source_dist == 0] = 1e-9 
        
        factor = SAD / source_dist
        y_det = y_rot * factor
        z_det = z_rot * factor

        # 6. Map to Pixel Indices
        # Reversing your kernel's detector geometry:
        iq_batch[b] = (y_det / psize) + (Nq / 2) - 0.5
        ip_batch[b] = (Np / 2) - (z_det / psize) - 0.5

    return ip_batch, iq_batch


def map_2d_plucker_ray_directions(Np, Nq, psize, angle, SAD=1000.0):
    """
    Maps each pixel in the 2D projection to a 3D ray direction in the CT coordinate system for a batch.
    angle in degree, when angle=0, the source is at (SAD, 0, 0) and the detector is in the Y-Z plane.
    Returns:
        ray_directions: (Batch, Np, Nq, 3) array of unit ray direction vectors in CT space
    """
    # Create grid of pixel centers
    ip = np.arange(Np) + 0.5 - (Np / 2)
    iq = np.arange(Nq) + 0.5 - (Nq / 2)
    iq_grid, ip_grid = np.meshgrid(iq, ip)  # Shape: (Np, Nq)

    # Calculate detector coordinates
    y_det = iq_grid * psize
    z_det = -ip_grid * psize
    
    angle_deg = angle
    angle_rad = angle_deg * 3.1415926 / 180
    s = math.sin(angle_rad)
    c = math.cos(angle_rad)

    # Assuming source rotates on the X-Y plane
    source_pos = np.array([SAD * c, SAD * s, 0]) 
    
    # 3. Rotate Detector Pixel Coordinates to CT Space
    # (This logic depends on your specific system geometry)
    x_rot = -y_det * s
    y_rot = y_det * c
    z_rot = z_det
    
    pixel_pos = np.stack([x_rot, y_rot, z_rot], axis=-1)

    # 4. Calculate Direction Vector (d)
    d = pixel_pos - source_pos
    d_norm = np.linalg.norm(d, axis=-1, keepdims=True)
    d_unit = d / (d_norm + 1e-9)

    # 5. Calculate Moment Vector (m = source x d)
    # We use the cross product of the source position and the direction
    m = np.cross(source_pos, d_unit)
    
    m = m / (np.linalg.norm(m, axis=-1, keepdims=True) + 1e-9)

    # 6. Stack into (d, m) Plücker format
    plucker_rays = np.concatenate([d_unit, m], axis=-1) # plucker_rays.shape = (Np, Nq, 6)
    plucker_rays = plucker_rays.transpose(2, 0, 1) # shape becomes (6, Np, Nq)

    return plucker_rays