from numba import cuda, jit
import torch
import torch.nn.functional as F
import numpy as np
import math
import random
from PIL import Image
from scipy import ndimage
import os
from typing import NamedTuple
from pathlib import Path
import shutil
import cv2

import sys

import copy

import nibabel as nib

#import kornia
import time
import threading
from dataclasses import dataclass
PI = 3.1415926


class DRRParam(NamedTuple):    
    psize_x: float # mm
    psize_y: float # mm
    Np: int
    Nq: int
    nSample: int
    sample_step: float
    detector_shift: float
    image_shift: float

# read CT
@dataclass
class CTParam:
    width: int
    height: int
    num_slices: int
    resx: float = 1.0
    resy: float = 1.0
    resz: float = 1.0
    isox: float = 0.0
    isoy: float = 0.0
    isoz: float = 0.0

    @property
    def res(self):
        """Returns resolution as a tuple."""
        return (self.resx, self.resy, self.resz)
    
    @property
    def iso(self):
        return (self.isox, self.isoy, self.isoz)
    
    @res.setter
    def res(self, values: tuple):
        """Allows: my_ct.res = (0.5, 0.5, 2.0)"""
        self.resx, self.resy, self.resz = map(float, values)

    @property
    def Nxyz(self):
        return (self.width, self.height, self.num_slices)


class CT_DRRGenerator:
    def __init__(self, opt, ct=None, ct_file=None, device='cuda'):
        if isinstance(device, str):
            device = torch.device(device)
        # Force Numba to use the same GPU as PyTorch
        if device.type == 'cuda':
            current_device_id = device.index if device.index is not None else torch.cuda.current_device()
            try:
                cuda.select_device(current_device_id)
            except Exception as e:
                # Fallback if Numba context is already set
                print(f"Numba device selection warning: {e}")
        ##############################################################
        
        if (ct is not None) and (ct_file is not None):
            raise ValueError("Provide either ct or ct_file, not both.")
        

        self.device = device
        self.opt = opt
        self.drr_size = [opt.H, opt.W]
        
        
        if ct_file is not None:
            ct, ct_param = self.load_ct(ct_file)
        else:
            assert ct.shape == (opt.Nx, opt.Ny, opt.Nz_temp), f"Expected CT shape ({opt.Nx}, {opt.Ny}, {opt.Nz_temp}), got {ct.shape}"
            ct_param = CTParam(
                width=opt.Nx,
                height=opt.Ny,
                num_slices=opt.Nz_temp,
                resx=opt.vol_size,
                resy=opt.vol_size,
                resz=opt.vol_size,
                isox=(opt.Nx-1)/2.0,
                isoy=(opt.Ny-1)/2.0,
                isoz=(opt.Nz_temp-1)/2.0,
            )
            ct = ct - ct.min()

        
        self.ct = ct
        self.ct_param = ct_param
        
        ################################### Setting DRR and DTI settings:
        # DRR Parameters
        sample_step = min(self.ct_param.res[0], self.ct_param.res[2])*0.5
        nSample = int(2.0 * math.sqrt(2) * max(self.ct_param.Nxyz) * max(self.ct_param.res) / sample_step)
        self.drr_Param = DRRParam(
            Np=self.drr_size[0],
            Nq=self.drr_size[1],
            psize_x=opt.resw, #0.388 / 1.5,
            psize_y=opt.resh, #0.388 / 1.5,
            nSample=nSample,
            sample_step=sample_step,
            detector_shift=0.0,
            image_shift=0.0)
    
    def normalize_min_max(self,
        tensor: torch.Tensor,
        min_val: float = 0.0,
        max_val: float = 1.0,
        normalize_dims: int = 2,
        eps: float = 1e-8
    ) -> torch.Tensor:
        if normalize_dims < 1 or normalize_dims > tensor.ndim:
            raise ValueError(f"normalize_dims must be between 1 and {tensor.ndim}, got {normalize_dims}")
        
        # Determine which dimensions to reduce over (the last normalize_dims dimensions)
        reduce_dims = tuple(range(-normalize_dims, 0))
        
        # Compute min and max over the specified dimensions, keeping dimensions for broadcasting
        tensor_min = tensor.amin(dim=reduce_dims, keepdim=True)
        tensor_max = tensor.amax(dim=reduce_dims, keepdim=True)
        
        # Normalize to [0, 1]
        tensor_range = tensor_max - tensor_min
        normalized = (tensor - tensor_min) / (tensor_range + eps)
        
        # Scale to [min_val, max_val]
        scaled = normalized * (max_val - min_val) + min_val
        
        return scaled    
          

    def load_ct(self,ct_file):
        img = nib.load(ct_file)
        data = img.get_fdata()

        # Coerce to 3D: some NIfTIs carry a trailing singleton (e.g. (X,Y,Z,1))
        # or a genuine 4th dim. self.ct is later permuted as a 3D volume, so
        # collapse extra dims here (matches the refine_resample coercion).
        if data.ndim != 3:
            data = np.squeeze(data)
            while data.ndim > 3:
                data = data[..., 0]
            if data.ndim != 3:
                raise ValueError(f"Expected a 3D CT but got shape "
                                 f"{img.shape} for {ct_file}")

        header = img.header
        spacing = header.get_zooms()
        dims = data.shape                     # use coerced 3D shape, not img.shape

        iso = [(d-1.0)/2.0 for d in dims]
        
        thisCTParam = CTParam(
            width=dims[0],
            height=dims[1],
            num_slices=dims[2],
            resx=spacing[0],
            resy=spacing[1],
            resz=spacing[2],
            isox=iso[0],
            isoy=iso[1],
            isoz=iso[2],
        )
        
        #print("data.min()",data.min())
        data = data - data.min()
        #data =data.transpose(2, 1, 0)
        
        #print(thisCTParam)
        #print(data.shape)
        return data, thisCTParam
    
    def generate_projections(self, angles, transform=None):
        """
        Generates one batch of data.
        """
        # For testing: use first 4dct as template
        
        actual_batch_size = len(angles)
        batch_angles = torch.tensor(angles,device=self.device)
        # indices = self.indices_ct[self.current_idx : end_idx]
        
        #angle_rad = (batch_angles - 90.0) * PI / 180
        angle_rad = batch_angles * PI / 180 # actual source angles, not relative to 90
        sinAngle_batch = torch.sin(angle_rad)
        cosAngle_batch = torch.cos(angle_rad)
        
        if transform is not None:
            matrices = torch.tensor(transform, device=self.device, dtype=torch.float32)
        else:
            matrices = torch.eye(4,device=self.device)
        matrices = matrices.unsqueeze(0).expand(actual_batch_size, *matrices.shape)
        
        stream = cuda.stream()
        
        # CUDA Grid Config
        threadsperblock = (4, 16, 16) 
        # Grid: (Batch, Row blocks, Col blocks)
        
        blockspergrid = (
            math.ceil(actual_batch_size/ threadsperblock[0]),
            math.ceil(self.drr_size[0] / threadsperblock[1]),
            math.ceil(self.drr_size[1] / threadsperblock[2])
        )
        #sample_step = min(self.ct_Param.res[0], self.ct_Param.res[2])*0.5
        
        ######################### DRR ####################################
        batch_vols = torch.tensor(self.ct, device=self.device).permute(2,1,0).contiguous()
        batch_vols = batch_vols.unsqueeze(0).expand(actual_batch_size, *batch_vols.shape)
        
        drr_out = torch.zeros((actual_batch_size, self.drr_size[0], self.drr_size[1]), device=self.device)
        
        forward_projection_pixelwise_batch_CUDA[blockspergrid, threadsperblock,stream](
            *self.ct_param.Nxyz,*self.ct_param.iso,*self.ct_param.res, self.drr_Param.sample_step,
            self.drr_Param.Np, self.drr_Param.Nq, self.drr_Param.nSample, self.drr_Param.psize_x, self.drr_Param.psize_y,
            batch_vols,
            sinAngle_batch,cosAngle_batch,
            matrices,
            drr_out)
        
        
        ############# post-process outputs ##########################    
        stream.synchronize()
        return drr_out


def generate_drr(opt, ct_file=None, ct=None , d_angle=2, transform=None, output_folder ='img', output_format='both'):
    """
    Args:
        ct_file: Path to CT volume
        ct: CT volume data
        n_aug: Number of augmentation folders
        d_angle: Degree step for projections
        transform: 4x4 transformation matrix
        output_format: 'png' (normalized), 'raw' (unnormalized .npy), or 'both'
        output_folder: full dir path to store outputs, e.g. "img_aug" (will create subfolders like "img_aug/aug_0", "img_aug/aug_1", etc.)
    """
    output_dir = Path(output_folder)
    
    assert (ct_file is not None) or (ct is not None), "Provide either ct_file or ct data"
    
    if ct_file is not None:
        #print(f"Loading CT from file: {ct_file}")
        start_time = time.perf_counter()
        dataloader = CT_DRRGenerator(opt, ct_file=ct_file)
        end_time = time.perf_counter()
        print(f"Initialized CT_DRRGenerator in {end_time - start_time:.2f} seconds")
    else:
        #print(f"Using provided CT data")
        dataloader = CT_DRRGenerator(opt, ct=ct)
    
    start = 0 # np.random.uniform(0, d_angle)
    angles = np.arange(start, 360 + start, d_angle) % 360
    start_time = time.perf_counter()
    drrs = dataloader.generate_projections(angles, transform=transform)
    end_time = time.perf_counter()
    print(f"Generated {len(angles)} DRRs in {end_time - start_time:.2f} seconds")
    
    if output_dir.exists() and output_dir.is_dir():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.perf_counter()
    for i in range(drrs.shape[0]):
        raw_img = drrs[i].cpu().numpy()
        if output_format in ['png', 'both']:
            img = ((raw_img-raw_img.min())/(raw_img.max()-raw_img.min()) * 255).astype(np.uint8)
            output_path = output_dir / f"proj_{angles[i]:.2f}.png"
            cv2.imwrite(str(output_path), img)
        if output_format in ['raw', 'both']:
            output_path = output_dir / f"proj_{angles[i]:.2f}.npy"
            np.save(str(output_path), raw_img)
    end_time = time.perf_counter()
    print(f"Saved DRRs in {end_time - start_time:.2f} seconds")


@cuda.jit(max_registers=64)
def forward_projection_pixelwise_batch_CUDA(Nx, Ny, Nz, tumor_isox, tumor_isoy, tumor_isoz, resx, resy, resz,sample_step, Np, Nq, nSample, psize_x, psize_y, cbct_batch, sinAngle_batch, cosAngle_batch, matrix_batch, proj_batch):
    '''
    Batch processing where:
    - CBCT volumes are unique per batch.
    - Gantry angles are unique per batch.
    - Transformation matrices are unique per batch.
    
    Inputs:
    - cbct_batch: (Batch, Nz, Ny, Nx) 
    - gantry_angle_batch: (Batch,) - <--- Now accessed via index [b]
    - matrix_batch: (Batch, 4, 4)
    - proj_batch: (Batch, Np, Nq)
    '''

    SAD = 1000 # mm
    # Nx, Ny, Nz = ct_param.Nxyz
    # tumor_isox, tumor_isoy, tumor_isoz = ct_param.iso
    # resx,resy,resz = ct_param.res

    

    # default is tumor iso is image iso
    proj_isox = tumor_isox
    proj_isoy = tumor_isoy
    proj_isoz = tumor_isoz
    image_shift = 0.0
    
    # psize = drr_param.psize
    # Np = drr_param.Np
    # Nq = drr_param.Nq
    # nSample = drr_param.nSample
    
    # 3D Grid: Batch x Pixel_Row x Pixel_Col
    b, ip, iq = cuda.grid(3)
    if (b < proj_batch.shape[0]) and (ip < Np) and (iq < Nq):
        
        # --- Batch Specific Angle ---
        # Retrieve the specific angle for this batch item
        
        #current_angle = gantry_angle_batch[b]
        
        #angle_deg = current_angle - 90.0
        #angle_rad = angle_deg * 3.1415926 / 180
        #sinAngle = math.sin(angle_rad)
        #cosAngle = math.cos(angle_rad)
        sinAngle = sinAngle_batch[b]
        cosAngle = cosAngle_batch[b]
        
        #sample_step = min(resx, resz)

        # --- Geometry Setup ---
        
        # Imaging source
        x1 = SAD
        y1 = 0
        z1 = 0

        # Detector pixel
        x2 = 0
        y2 = (iq + 0.5 - Nq / 2) * psize_x + image_shift
        z2 = (Np / 2 - ip - 0.5) * psize_y

        # Coordinates in fixed system
        xr1 = x1 * cosAngle - y1 * sinAngle
        yr1 = x1 * sinAngle + y1 * cosAngle
        zr1 = z1

        xr2 = x2 * cosAngle - y2 * sinAngle
        yr2 = x2 * sinAngle + y2 * cosAngle
        zr2 = z2

        dist = math.sqrt((xr2 - xr1) * (xr2 - xr1) + (yr2 - yr1) * (yr2 - yr1) + (zr2 - zr1) * (zr2 - zr1))

        slopex = (xr1 - xr2) / dist
        slopey = (yr1 - yr2) / dist
        slopez = (zr1 - zr2) / dist

        # Couch center adjustment
        #imager_depth_shift = 0.0

        # --- Batch Specific Matrix ---
        # Retrieve the specific matrix for this batch item
        m00 = matrix_batch[b, 0, 0]
        m01 = matrix_batch[b, 0, 1]
        m02 = matrix_batch[b, 0, 2]
        m03 = matrix_batch[b, 0, 3]
        
        m10 = matrix_batch[b, 1, 0]
        m11 = matrix_batch[b, 1, 1]
        m12 = matrix_batch[b, 1, 2]
        m13 = matrix_batch[b, 1, 3]
        
        m20 = matrix_batch[b, 2, 0]
        m21 = matrix_batch[b, 2, 1]
        m22 = matrix_batch[b, 2, 2]
        m23 = matrix_batch[b, 2, 3]

        # --- Ray Tracing ---
        temp_proj_val = 0.0

        for tis in range(int(-nSample/2), int(nSample/2)):
            
            # Use current batch matrix params (m03, m13, etc)
            xr = xr2 + tis * slopex * sample_step
            yr = yr2 + tis * slopey * sample_step
            zr = zr2 + tis * slopez * sample_step

            # Transformation
            xxx = m00 * xr + m01 * yr + m02 * zr + m03
            yyy = m10 * xr + m11 * yr + m12 * zr + m13
            zzz = m20 * xr + m21 * yr + m22 * zr + m23
            
            xrs = xxx / resx 
            yrs = yyy / resy
            zrs = zzz / resz

            xrs_idx = xrs + proj_isox 
            yrs_idx = yrs + proj_isoy
            zrs_idx = zrs + proj_isoz

            xx = int(math.floor(xrs_idx + 0.001))
            yy = int(math.floor(yrs_idx + 0.001))
            zz = int(math.floor(zrs_idx + 0.001))

            if xx >= 0 and xx + 1 < Nx and yy >= 0 and yy + 1 < Ny and zz >= 0 and zz + 1 < Nz:
                
                # --- Batch Specific CBCT Access ---
                v1 = cbct_batch[b, zz, yy, xx]
                v2 = cbct_batch[b, zz, yy, xx + 1]
                v3 = cbct_batch[b, zz, yy + 1, xx]
                v4 = cbct_batch[b, zz, yy + 1, xx + 1]
                
                v5 = cbct_batch[b, zz + 1, yy, xx]
                v6 = cbct_batch[b, zz + 1, yy, xx + 1]
                v7 = cbct_batch[b, zz + 1, yy + 1, xx]
                v8 = cbct_batch[b, zz + 1, yy + 1, xx + 1]
                
                wx = xx + 1 - xrs_idx
                wy = yy + 1 - yrs_idx
                wz = zz + 1 - zrs_idx
                
                c00 = v1 * wx + v2 * (1 - wx)
                c01 = v3 * wx + v4 * (1 - wx)
                c10 = v5 * wx + v6 * (1 - wx)
                c11 = v7 * wx + v8 * (1 - wx)
                
                c0 = c00 * wy + c01 * (1 - wy)
                c1 = c10 * wy + c11 * (1 - wy)
                
                c = c0 * wz + c1 * (1 - wz)
                
                temp_proj_val += c

        oblique_factor = SAD / dist
        proj_batch[b, ip, iq] = temp_proj_val * sample_step * oblique_factor
        
        

        
        
