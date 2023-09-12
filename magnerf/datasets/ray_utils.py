from typing import Tuple, Optional

import numpy as np
import torch
import scipy

from .intrinsics import Intrinsics

__all__ = (
    "create_meshgrid",
    "get_ray_directions",
    "stack_camera_dirs",
    "get_rays",
    "ndc_rays_blender",
    "center_poses",
    "generate_spiral_path",
    "generate_hemispherical_orbit",
    "generate_spherical_poses",
    "normalize",
    "average_poses",
    "viewmatrix",
)


def create_meshgrid(height: int,
                    width: int,
                    dev: str = 'cpu',
                    add_half: bool = True,
                    flat: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = torch.arange(width, dtype=torch.float32, device=dev)
    ys = torch.arange(height, dtype=torch.float32, device=dev)
    if add_half:
        xs += 0.5
        ys += 0.5
    # generate grid by stacking coordinates
    yy, xx = torch.meshgrid([ys, xs], indexing="ij")  # both HxW
    if flat:
        return xx.flatten(), yy.flatten()
    return xx, yy


def stack_camera_dirs(x: torch.Tensor, y: torch.Tensor, intrinsics: Intrinsics, opengl_camera: bool):
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    x = x.float()
    y = y.float()
    return torch.stack([
        (x - intrinsics.center_x) / intrinsics.focal_x,
        (y - intrinsics.center_y) / intrinsics.focal_y
        * (-1.0 if opengl_camera else 1.0),
        torch.full_like(x, fill_value=-1.0 if opengl_camera else 1.0)
    ], -1)  # (H, W, 3)


def get_ray_directions(intrinsics: Intrinsics, opengl_camera: bool, add_half: bool = True) -> torch.Tensor:
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        height, width, focal: image height, width and focal length
    Outputs:
        directions: (height, width, 3), the direction of the rays in camera coordinate
    """
    xx, yy = create_meshgrid(intrinsics.height, intrinsics.width, add_half=add_half)

    return stack_camera_dirs(xx, yy, intrinsics, opengl_camera)


def get_rays(directions: torch.Tensor,
             c2w: torch.Tensor,
             ndc: bool,
             ndc_near: float = 1.0,
             intrinsics: Optional[Intrinsics] = None,
             normalize_rd: bool = True):
    """Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Args:
        directions:
        c2w:
        ndc:
        ndc_near:
        intrinsics:
        normalize_rd:

    Returns:

    """
    directions = directions.view(-1, 3)  # [n_rays, 3]
    if len(c2w.shape) == 2:
        c2w = c2w[None, ...]
    rd = (directions[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
    ro = torch.broadcast_to(c2w[:, :3, 3], directions.shape)
    if ndc:
        assert intrinsics is not None, "intrinsics must not be None when NDC active."
        ro, rd = ndc_rays_blender(
            intrinsics=intrinsics, near=ndc_near, rays_o=ro, rays_d=rd)
    if normalize_rd:
        rd /= torch.linalg.norm(rd, dim=-1, keepdim=True)
    return ro, rd


def ndc_rays_blender(intrinsics: Intrinsics, near: float, rays_o: torch.Tensor,
                     rays_d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    ndc_coef_x = - (2 * intrinsics.focal_x) / intrinsics.width
    ndc_coef_y = - (2 * intrinsics.focal_y) / intrinsics.height
    o0 = ndc_coef_x * rays_o[..., 0] / rays_o[..., 2]
    o1 = ndc_coef_y * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = ndc_coef_x * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = ndc_coef_y * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses: np.ndarray) -> np.ndarray:
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)
    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)
    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)
    # 4. Compute the x axis
    x = normalize(np.cross(z, y_))  # (3)
    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(x, z)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """
    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation
    pose_avg_homo = pose_avg_homo
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg_homo


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    return np.stack([-vec0, vec1, vec2, pos], axis=1)


def generate_spiral_path(poses: np.ndarray,
                         near_fars: np.ndarray,
                         n_frames=120,
                         n_rots=2,
                         zrate=.5,
                         dt=0.75,
                         percentile=70,
                         center=None) -> np.ndarray:
    """Calculates a forward facing spiral path for rendering.

    From https://github.com/google-research/google-research/blob/342bfc150ef1155c5254c1e6bd0c912893273e8d/regnerf/internal/datasets.py
    and https://github.com/apchenstu/TensoRF/blob/main/dataLoader/llff.py

    :param poses: [N, 3, 4]
    :param near_fars:
    :param n_frames:
    :param n_rots:
    :param zrate:
    :param dt:
    :return:
    """
    # center pose
    if center is not None:
        c2w = center
    else:
        c2w = average_poses(poses)  # [3, 4]

    # Get average pose
    up = normalize(poses[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset as a weighted average
    # of near and far bounds in disparity space.
    close_depth, inf_depth = np.min(near_fars) * 1.0, np.max(near_fars) * 5.0
    focal = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))

    # Get radii for spiral path using 90th percentile of camera positions.
    positions = poses[:, :3, 3]
    radii = np.percentile(np.abs(positions), percentile, 0)
    radii = np.concatenate([radii, [1.]])

    # Generate poses for spiral path.
    render_poses = []
    for theta in np.linspace(0., 2. * np.pi * n_rots, n_frames, endpoint=False):
        t = radii * [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]
        position = c2w @ t
        lookat = c2w @ np.array([0, 0, -focal, 1.0])
        z_axis = normalize(position - lookat)
        render_poses.append(viewmatrix(z_axis, up, position))
    return np.stack(render_poses, axis=0)


def generate_hemispherical_orbit(poses: torch.Tensor, n_frames=120, opengl_cam=True):
    """Calculates a render path which orbits around the z-axis.
    Based on https://github.com/google-research/google-research/blob/342bfc150ef1155c5254c1e6bd0c912893273e8d/regnerf/internal/datasets.py
    """
    origins = poses[:, :3, 3]
    radius = torch.sqrt(torch.mean(torch.sum(origins ** 2, dim=-1)))

    # Assume that z-axis points up towards approximate camera hemisphere
    sin_phi = torch.mean(origins[:, 2], dim=0) / radius + 0.4
    cos_phi = torch.sqrt(1 - sin_phi ** 2)
    render_poses = []

    up = torch.tensor([0., 0., 1.])
    for theta in np.linspace(0., 2. * np.pi, n_frames, endpoint=False):
        camorigin = radius * torch.tensor(
            [cos_phi * np.cos(theta), cos_phi * np.sin(theta), sin_phi])
        render_poses.append(torch.from_numpy(viewmatrix(camorigin, up, camorigin)))

    render_poses = torch.stack(render_poses, dim=0)

    return render_poses


trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()

rot_phi = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()

rot_theta = lambda th: torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()


def generate_spherical_poses(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


def focus_point_fn(poses: np.ndarray) -> np.ndarray:
  """Calculate nearest point to all focal axes in poses."""
  directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
  m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
  mt_m = np.transpose(m, [0, 2, 1]) @ m
  focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
  return focus_pt


def generate_ellipse_path(poses: np.ndarray,
                          n_frames: int = 120,
                          z_variation: float = 0.,
                          z_phase: float = 0.) -> np.ndarray:
  """Generate an elliptical render path based on the given poses."""
  # Calculate the focal point for the path (cameras point toward this).
  center = focus_point_fn(poses)
  # Path height sits at z=0 (in middle of zero-mean capture pattern).
  offset = np.array([center[0], center[1], 0])

  # Calculate scaling for ellipse axes based on input camera positions.
  sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
  # Use ellipse that is symmetric about the focal point in xy.
  low = -sc + offset
  high = sc + offset
  # Optional height variation need not be symmetric
  z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
  z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)

  def get_positions(theta):
    # Interpolate between bounds with trig functions to get ellipse in x-y.
    # Optionally also interpolate in z to change camera height along path.
    return np.stack([
        low[0] + (high - low)[0] * (np.cos(theta) * .5 + .5),
        low[1] + (high - low)[1] * (np.sin(theta) * .5 + .5),
        z_variation * (z_low[2] + (z_high - z_low)[2] *
                       (np.cos(theta + 2 * np.pi * z_phase) * .5 + .5)),
    ], -1)

  theta = np.linspace(0, 2. * np.pi, n_frames + 1, endpoint=True)
  positions = get_positions(theta)

  # Throw away duplicated last position.
  positions = positions[:-1]

  # Set path's up vector to axis closest to average of input pose up vectors.
  avg_up = poses[:, :3, 1].mean(0)
  avg_up = avg_up / np.linalg.norm(avg_up)
  ind_up = np.argmax(np.abs(avg_up))
  up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

  return np.stack([viewmatrix(p - center, up, p) for p in positions])

def viewmatrix(lookdir: np.ndarray, up: np.ndarray,
                position: np.ndarray) -> np.ndarray:
    """Construct lookat view matrix."""
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m

def poses_to_points(poses, dist, look_at_neg):
    """Converts from pose matrices to (position, lookat, up) format."""
    pos = poses[:, :3, -1]
    if look_at_neg:
        lookat = poses[:, :3, -1] + dist * poses[:, :3, 2]
        up = poses[:, :3, -1] - dist * poses[:, :3, 1]
    else:
        lookat = poses[:, :3, -1] - dist * poses[:, :3, 2]
        up = poses[:, :3, -1] + dist * poses[:, :3, 1]
    return np.stack([pos, lookat, up], 1)


def points_to_poses(points):
    """Converts from (position, lookat, up) format to pose matrices."""
    pose = np.array([viewmatrix(p - l, u - p, p) for p, l, u in points])
    return torch.from_numpy(pose).float()


def interp(points, n, k, s):
    """Runs multidimensional B-spline interpolation on the input points."""
    sh = points.shape
    pts = np.reshape(points, (sh[0], -1))
    k = min(k, sh[0] - 1)
    tck, _ = scipy.interpolate.splprep(pts.T, k=k, s=s)
    u = np.linspace(0, 1, n, endpoint=False)
    new_points = np.array(scipy.interpolate.splev(u, tck))
    new_points = np.reshape(new_points.T, (n, sh[1], sh[2]))
    return new_points


def rotx(θ):
    """ Returns a rotation matrix for the given angle about the X axis """
    return np.array(((1,0,0,0),
                    (0,np.cos(θ),-np.sin(θ),0),
                    (0,np.sin(θ),np.cos(θ),0),
                    (0,0,0,1)))

def roty(θ):
    """ Returns a rotation matrix for the given angle about the X axis """
    return np.array(((np.cos(θ),0,np.sin(θ),0),
                    (0,1,0,0),
                    (-np.sin(θ),0,np.cos(θ),0),
                    (0,0,0,1)))

def rotz(θ):
    """ Returns a rotation matrix for the given angle about the X axis """
    return np.array(((np.cos(θ), -np.sin(θ),0,0),
                    (np.sin(θ), np.cos(θ),0,0),
                    (0,0,1,0),
                    (0,0,0,1)))

def trans(x=0, y=0, z=0):
    """ Returns a translation matrix for the given offset """
    return np.array(((1,0,0,x),
                    (0,1,0,y),
                    (0,0,1,z),
                    (0,0,0,1)))


def generate_interpolated_path(poses: np.ndarray,
                               n_interp: int,
                               spline_degree: int = 5,
                               smoothness: float = .03,
                               rot_weight: float = .1,
                               look_at_neg: bool = False,
                               n_points: int = None):
  """Creates a smooth spline path between input keyframe camera poses.
  Spline is calculated with poses in format (position, lookat-point, up-point).
  Args:
    poses: (n, 3, 4) array of input pose keyframes.
    n_interp: returned path will have n_interp * (n - 1) total poses.
    spline_degree: polynomial degree of B-spline.
    smoothness: parameter for spline smoothing, 0 forces exact interpolation.
    rot_weight: relative weighting of rotation/translation in spline solve.
  Returns:
    Array of new camera poses with shape (n_interp * (n - 1), 3, 4).
  """
  if n_points is None:
    n_points = n_interp * (poses.shape[0] - 1)
  points = poses_to_points(poses, dist=rot_weight, look_at_neg=look_at_neg)
  new_points = interp(points,
                      n_points,
                      k=spline_degree,
                      s=smoothness)
  return points_to_poses(new_points)



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
        fid.write(bytearray(struct.pack("fffccc",xyz_points[i,0],xyz_points[i,1],xyz_points[i,2], rgb_points[i,0].tobytes(),rgb_points[i,1].tobytes(), rgb_points[i,2].tobytes())))
    fid.close()

