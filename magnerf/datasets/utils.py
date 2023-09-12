"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""
import numpy as np
import scipy
import torch

import collections

Rays = collections.namedtuple("Rays", ("origins", "viewdirs"))


def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*(None if x is None else fn(x) for x in tup))

def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector."""
    return v / np.linalg.norm(v)

def generate_interpolated_path(poses: np.ndarray,
                               n_interp: int,
                               spline_degree: int = 5,
                               smoothness: float = .03,
                               rot_weight: float = .1,
                               look_at_neg: bool = False):
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

  points = poses_to_points(poses, dist=rot_weight, look_at_neg=look_at_neg)
  new_points = interp(points,
                      n_interp * (points.shape[0] - 1),
                      k=spline_degree,
                      s=smoothness)
  return points_to_poses(new_points)