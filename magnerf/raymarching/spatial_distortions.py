from typing import Optional, Union

import torch
import torch.nn as nn


class SpatialDistortion(nn.Module):
    """Apply spatial distortions"""

    def forward(
        self, positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            positions: Sample to distort (shape: batch-size, ..., 3)
        Returns:
            distorted sample - same shape
        """

import numpy as np
import struct
def write_pointcloud(filename,xyz_points,rgb_points=None):
    assert xyz_points.shape[1] == 3,'Input XYZ points should be Nx3 float array'
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8)*255
    assert xyz_points.shape == rgb_points.shape,'Input RGB colors should be Nx3 float array and have same size as input XYZ points'

    # Write header of .ply file
    fid = open(filename,'wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n'%xyz_points.shape[0], 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    fid.write(bytes('property uchar red\n', 'utf-8'))
    fid.write(bytes('property uchar green\n', 'utf-8'))
    fid.write(bytes('property uchar blue\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))

    # Write 3D points to .ply file
    for i in range(xyz_points.shape[0]):
        fid.write(bytearray(struct.pack("fffccc",xyz_points[i,0],xyz_points[i,1],xyz_points[i,2],
                                        rgb_points[i,0].tobytes(),rgb_points[i,1].tobytes(),
                                        rgb_points[i,2].tobytes())))
    fid.close()


class SceneContraction(SpatialDistortion):
    """Contract unbounded space using the contraction was proposed in MipNeRF-360.
        We use the following contraction equation:
        .. math::
            f(x) = \\begin{cases}
                x & ||x|| \\leq 1 \\\\
                (2 - \\frac{1}{||x||})(\\frac{x}{||x||}) & ||x|| > 1
            \\end{cases}
        If the order is not specified, we use the Frobenius norm, this will contract the space to a sphere of
        radius 1. If the order is L_inf (order=float("inf")), we will contract the space to a cube of side length 2.
        If using voxel based encodings such as the Hash encoder, we recommend using the L_inf norm.
        Args:
            order: Order of the norm. Default to the Frobenius norm. Must be set to None for Gaussians.
    """

    def __init__(self,
                 order: Optional[Union[float, int]] = None,
                 global_translation: Optional[torch.Tensor] = None,
                 global_scale: Optional[torch.Tensor] = None,
                 ) -> None:
        super().__init__()
        self.order = order

        if global_translation is None:
            global_translation = torch.tensor([0.0, 0.0, 0.0])
        self.global_translation = nn.Parameter(global_translation, requires_grad=False)
        if global_scale is None:
            global_scale = torch.tensor([1.0, 1.0, 1.0])

        #self.global_scale = nn.Parameter(global_scale, requires_grad=False)

        #aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]) * 4.0
        #aabb = global_scale
        self.aabb = nn.Parameter(torch.tensor(global_scale), requires_grad=False)
        

    def forward(self, positions):
        # Apply global scale and translation
        #positions = (
        #    positions * self.global_scale[None, None, :]
        #    + self.global_translation[None, None, :]
        #)

        aabb_min, aabb_max = torch.split(self.aabb, 3, dim=-1)
        positions = (positions - aabb_min) / (aabb_max - aabb_min)
        positions = positions * 2 - 1  # aabb is at [-1, 1]

        mag = torch.linalg.norm(positions, ord=self.order, dim=-1)
        mask = mag >= 1
        x_new = positions.clone()
        #x_new[mask] = (2 - (1 / mag[mask][..., None])) * (positions[mask] / mag[mask][..., None])
        x_new[mask] = (1.5 - (1 / mag[mask][..., None])) * (positions[mask] / mag[mask][..., None])

        x_new = (x_new / 1.5) * 2

        return x_new

    def __str__(self):
        return (f"SceneContraction(global_translation={self.global_translation}, "
                f"global_scale={self.global_scale})")
