import os
import sys
import logging as log
from typing import Optional

import numpy as np
import torch

from PIL import Image

from .ray_utils import (
    generate_interpolated_path,
    create_meshgrid,
    stack_camera_dirs,
    roty,
    rotx,
    rotz,
    trans,
)
from .base_dataset import BaseDataset
from .intrinsics import Intrinsics

_PATH = os.path.abspath(__file__)

sys.path.insert(0, os.path.join(os.path.dirname(_PATH), ".", "pycolmap", "pycolmap"))
from scene_manager import SceneManager


def _load_colmap_temporal(
    data_dir: str,
    factor: int = 1,
    hold_every: int = 8,
    timestep: str = "001",
    frame_offset: int = 2,
):
    colmap_dir = os.path.join(data_dir, "sparse/0/")

    manager = SceneManager(colmap_dir)
    manager.load_cameras()
    manager.load_images()
    manager.load_points3D()
    pts_ret = manager.get_filtered_points3D(return_colors=True)

    # Assume shared intrinsics between all cameras.
    cam = manager.cameras[1]
    fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    K[:2, :] /= factor

    intrinsics = Intrinsics(
        width=int(cam.width / factor),
        height=int(cam.height / factor),
        focal_x=fx / factor,
        focal_y=fy / factor,
        center_x=cx / factor,
        center_y=cy / factor,
    )

    # Extract extrinsic matrices in world-to-camera format.
    imdata = manager.images
    w2c_mats = []
    bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
    for k in imdata:
        im = imdata[k]
        rot = im.R()
        trans = im.tvec.reshape(3, 1)
        w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
        w2c_mats.append(w2c)
    w2c_mats = np.stack(w2c_mats, axis=0)

    # Convert extrinsics to camera-to-world.
    camtoworlds = np.linalg.inv(w2c_mats)

    # Image names from COLMAP. No need for permuting the poses according to
    # image names anymore.
    image_names = [imdata[k].name for k in imdata]

    # Get distortion parameters.
    type_ = cam.camera_type

    if type_ == 0 or type_ == "SIMPLE_PINHOLE":
        params = None
        camtype = "perspective"

    elif type_ == 1 or type_ == "PINHOLE":
        params = None
        camtype = "perspective"

    if type_ == 2 or type_ == "SIMPLE_RADIAL":
        params = {k: 0.0 for k in ["k1", "k2", "k3", "p1", "p2"]}
        params["k1"] = cam.k1
        camtype = "perspective"

    elif type_ == 3 or type_ == "RADIAL":
        params = {k: 0.0 for k in ["k1", "k2", "k3", "p1", "p2"]}
        params["k1"] = cam.k1
        params["k2"] = cam.k2
        camtype = "perspective"

    elif type_ == 4 or type_ == "OPENCV":
        params = {k: 0.0 for k in ["k1", "k2", "k3", "p1", "p2"]}
        params["k1"] = cam.k1
        params["k2"] = cam.k2
        params["p1"] = cam.p1
        params["p2"] = cam.p2
        camtype = "perspective"

    elif type_ == 5 or type_ == "OPENCV_FISHEYE":
        params = {k: 0.0 for k in ["k1", "k2", "k3", "k4"]}
        params["k1"] = cam.k1
        params["k2"] = cam.k2
        params["k3"] = cam.k3
        params["k4"] = cam.k4
        camtype = "fisheye"

    # assert params is None, f"Only support pinhole camera model. Params: {params}"

    # Previous Nerf results were generated with images sorted by filename,
    # ensure metrics are reported on the same test set.
    inds = np.argsort(image_names)
    image_names = [image_names[i] for i in inds]
    camtoworlds = camtoworlds[inds]

    # Load images.
    image_dir = os.path.join(data_dir, f"video_frames_{frame_offset}")
    print(camtoworlds.shape)

    names = sorted(os.listdir(image_dir))

    # image_paths = [
    #    os.path.join(image_dir, f'image_{int(timestep):04d}.png')
    # ]

    image_paths = [os.path.join(image_dir, names[int(timestep) - 1])]

    images = []
    for x in image_paths:
        pil_image = Image.open(x).convert("RGB")
        if factor > 1:
            width, height = pil_image.size
            newsize = (int(width / factor), int(height / factor))
            pil_image = pil_image.resize(newsize, resample=Image.BILINEAR)
        images.append(np.array(pil_image, dtype="uint8"))

    images = np.stack(images, axis=0)
    images = np.repeat(images, len(camtoworlds), axis=0)

    # Select the split.
    all_indices = np.array([len(camtoworlds) - frame_offset])
    split_indices = {
        "test": all_indices,
        "train": all_indices,
    }
    return images, camtoworlds, intrinsics, split_indices


def _load_colmap(data_dir: str, factor: int = 1, hold_every: int = 8):
    assert factor in [1, 2, 4, 8]

    colmap_dir = os.path.join(data_dir, "sparse/0/")

    manager = SceneManager(colmap_dir)
    manager.load_cameras()
    manager.load_images()
    manager.load_points3D()
    pts_ret = manager.get_filtered_points3D(return_colors=True)

    # Assume shared intrinsics between all cameras.
    cam = manager.cameras[1]
    fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    K[:2, :] /= factor

    intrinsics = Intrinsics(
        width=int(cam.width / factor),
        height=int(cam.height / factor),
        focal_x=fx / factor,
        focal_y=fy / factor,
        center_x=cx / factor,
        center_y=cy / factor,
    )

    # Extract extrinsic matrices in world-to-camera format.
    imdata = manager.images
    w2c_mats = []
    bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
    for k in imdata:
        im = imdata[k]
        rot = im.R()
        trans = im.tvec.reshape(3, 1)
        w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
        w2c_mats.append(w2c)
    w2c_mats = np.stack(w2c_mats, axis=0)

    # Convert extrinsics to camera-to-world.
    camtoworlds = np.linalg.inv(w2c_mats)

    # Image names from COLMAP. No need for permuting the poses according to
    # image names anymore.
    image_names = [imdata[k].name for k in imdata]

    # # Switch from COLMAP (right, down, fwd) to Nerf (right, up, back) frame.
    # poses = poses @ np.diag([1, -1, -1, 1])

    # Get distortion parameters.
    type_ = cam.camera_type

    if type_ == 0 or type_ == "SIMPLE_PINHOLE":
        params = None
        camtype = "perspective"

    elif type_ == 1 or type_ == "PINHOLE":
        params = None
        camtype = "perspective"

    if type_ == 2 or type_ == "SIMPLE_RADIAL":
        params = {k: 0.0 for k in ["k1", "k2", "k3", "p1", "p2"]}
        params["k1"] = cam.k1
        camtype = "perspective"

    elif type_ == 3 or type_ == "RADIAL":
        params = {k: 0.0 for k in ["k1", "k2", "k3", "p1", "p2"]}
        params["k1"] = cam.k1
        params["k2"] = cam.k2
        camtype = "perspective"

    elif type_ == 4 or type_ == "OPENCV":
        params = {k: 0.0 for k in ["k1", "k2", "k3", "p1", "p2"]}
        params["k1"] = cam.k1
        params["k2"] = cam.k2
        params["p1"] = cam.p1
        params["p2"] = cam.p2
        camtype = "perspective"

    elif type_ == 5 or type_ == "OPENCV_FISHEYE":
        params = {k: 0.0 for k in ["k1", "k2", "k3", "k4"]}
        params["k1"] = cam.k1
        params["k2"] = cam.k2
        params["k3"] = cam.k3
        params["k4"] = cam.k4
        camtype = "fisheye"

    # assert params is None, f"Only support pinhole camera model. Params: {params}"

    # Previous Nerf results were generated with images sorted by filename,
    # ensure metrics are reported on the same test set.
    inds = np.argsort(image_names)
    image_names = [image_names[i] for i in inds]
    camtoworlds = camtoworlds[inds]

    # Load images.
    # if factor > 1:
    #    image_dir_suffix = f"_{int(factor)}"
    # else:
    #    image_dir_suffix = ""

    colmap_image_dir = os.path.join(data_dir, "images")
    image_dir = os.path.join(data_dir, "images")
    for d in [image_dir, colmap_image_dir]:
        if not os.path.exists(d):
            raise ValueError(f"Image folder {d} does not exist.")
    # Downsampled images may have different names vs images used for COLMAP,
    # so we need to map between the two sorted lists of files.
    colmap_files = sorted(os.listdir(colmap_image_dir))
    image_files = sorted(os.listdir(image_dir))
    colmap_to_image = dict(zip(colmap_files, image_files))
    image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]
    print("loading images")
    images = []
    for x in image_paths:
        pil_image = Image.open(x).convert("RGB")
        if factor > 1:
            width, height = pil_image.size
            newsize = (int(width / factor), int(height / factor))
            pil_image = pil_image.resize(newsize, resample=Image.BILINEAR)
        images.append(np.array(pil_image, dtype="uint8"))

    # images = [imageio.imread(x) for x in image_paths]

    images = np.stack(images, axis=0)

    # Select the split.
    all_indices = np.arange(images.shape[0])
    split_indices = {
        "test": all_indices[all_indices % hold_every == 0],
        "train": all_indices,
    }
    return images, camtoworlds, intrinsics, split_indices


def similarity_from_cameras(c2w, strict_scaling):
    """
    reference: nerf-factory
    Get a similarity transform to normalize dataset
    from c2w (OpenCV convention) cameras
    :param c2w: (N, 4)
    :return T (4,4) , scale (float)
    """
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]

    # (1) Rotate the world so that z+ is the up axis
    # we estimate the up axis by averaging the camera up axes
    ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up)

    up_camspace = np.array([0.0, -1.0, 0.0])
    c = (up_camspace * world_up).sum()
    cross = np.cross(world_up, up_camspace)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    if c > -1:
        R_align = np.eye(3) + skew + (skew @ skew) * 1 / (1 + c)
    else:
        # In the unlikely case the original data has y+ up axis,
        # rotate 180-deg about x axis
        R_align = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    #  R_align = np.eye(3) # DEBUG
    R = R_align @ R
    fwds = np.sum(R * np.array([0, 0.0, 1.0]), axis=-1)
    t = (R_align @ t[..., None])[..., 0]

    # (2) Recenter the scene using camera center rays
    # find the closest point to the origin for each camera's center ray
    nearest = t + (fwds * -t).sum(-1)[:, None] * fwds

    # median for more robustness
    translate = -np.median(nearest, axis=0)

    #  translate = -np.mean(t, axis=0)  # DEBUG

    transform = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = R_align

    # (3) Rescale the scene using camera distances
    scale_fn = np.max if strict_scaling else np.median
    scale = 1.0 / scale_fn(np.linalg.norm(t + translate, axis=-1))

    return transform, scale


class TripodDataset(BaseDataset):
    def __init__(
        self,
        datadir,
        split: str,
        batch_size: Optional[int] = None,
        downsample: int = 4,
        hold_every: int = 8,
        contraction: bool = False,
        frame_offset: int = 1,
        near_plane_offset: float = 0.0,
        start_interp_index: int = 5,
        aabb=[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
        use_app: bool = False,
        near_plane: float = 0.2,
        far_plane: float = 30.0,
        bg_color_aug: str = "black",
    ):
        self.downsample = downsample
        self.hold_every = hold_every
        self.bg_color_aug = bg_color_aug
        if not isinstance(contraction, bool):
            contraction = eval(contraction)

        data_dir = "/".join(datadir.split("/")[:-1])
        self.timestep = datadir.split("/")[-1]
        scene_name = datadir.split("/")[-2]

        if split == "render":
            if self.timestep != "000":
                imgs, poses, intrinsics, split_indices = _load_colmap_temporal(
                    data_dir, downsample, hold_every, self.timestep, frame_offset
                )
            else:
                imgs, poses, intrinsics, split_indices = _load_colmap(
                    data_dir, downsample, hold_every
                )
            T, sscale = similarity_from_cameras(poses, strict_scaling=False)
            poses = np.einsum("nij, ki -> nkj", poses, T)
            poses[:, :3, 3] *= sscale
            poses = torch.from_numpy(poses).float()

            sta_id = start_interp_index
            end_id = len(poses) - frame_offset

            render_poses_0 = poses[sta_id:sta_id+1, :3].repeat(30, 1, 1)
            render_poses_1 = generate_interpolated_path(
                torch.cat([render_poses_0[0:1], poses[end_id : end_id + 1, :3]]),
                n_interp=30,
                look_at_neg=False,
            )
            render_poses_2 = poses[end_id : end_id + 1, :3].repeat(30, 1, 1)
            render_poses_3 = generate_interpolated_path(
                torch.cat([poses[end_id : end_id + 1, :3], render_poses_0[0:1]]),
                n_interp=30,
                look_at_neg=False,
            )
            self.poses = torch.cat(
                [
                    render_poses_0,
                    render_poses_0,
                    render_poses_1,
                    render_poses_2,
                    render_poses_2,
                    render_poses_3,
                ],
                dim=0,
            )
            if scene_name == "standing":
                rot = (
                    torch.from_numpy(rotx(-1 * np.pi / 36)).float()
                    @ torch.from_numpy(roty(-1 * np.pi / 30)).float()
                )
                rot = rot @ torch.from_numpy(rotz(-1 * np.pi / 30)).float()

                rot = rot @ torch.from_numpy(trans(x=0.6, z=-0.0, y=-0.5)).float()
                pose_a = poses[end_id : end_id + 1, :3]
                pose_b = poses[end_id : end_id + 1, :3] @ rot
                a2b = generate_interpolated_path(
                    torch.cat([pose_a, pose_b]), n_interp=30, look_at_neg=False
                )
                rot = (
                    torch.from_numpy(roty(1 * np.pi / 36)).float()
                    @ torch.from_numpy(rotz(1 * np.pi / 30)).float()
                )
                rot = rot @ torch.from_numpy(trans(x=-0.5, z=0.2, y=0.2)).float()
                pose_c = poses[end_id : end_id + 1, :3] @ rot
                c2a = generate_interpolated_path(
                    torch.cat([pose_c, pose_a]), n_interp=30, look_at_neg=False
                )
                b2c = generate_interpolated_path(
                    torch.cat([pose_b, pose_c]), n_interp=30, look_at_neg=False
                )
                self.poses = torch.cat([pose_a.repeat(60, 1, 1), a2b, b2c, c2a], dim=0)

            if scene_name == "breathing":
                rot = (
                    torch.from_numpy(rotx(-1 * np.pi / 18)).float()
                    @ torch.from_numpy(roty(-np.pi / 15)).float()
                )
                rot = rot @ torch.from_numpy(trans(x=2.0, z=-1.0, y=-1.5)).float()
                pose_a = poses[end_id : end_id + 1, :3]
                pose_b = poses[end_id : end_id + 1, :3] @ rot
                a2b = generate_interpolated_path(
                    torch.cat([pose_a, pose_b]), n_interp=30, look_at_neg=False
                )
                rot = (
                    torch.from_numpy(roty(np.pi / 30)).float()
                    @ torch.from_numpy(rotz(np.pi / 15)).float()
                )
                rot = rot @ torch.from_numpy(trans(x=-0.0, z=1.0, y=0.5)).float()
                pose_c = poses[end_id : end_id + 1, :3] @ rot
                c2a = generate_interpolated_path(
                    torch.cat([pose_c, pose_a]), n_interp=30, look_at_neg=False
                )
                b2c = generate_interpolated_path(
                    torch.cat([pose_b, pose_c]), n_interp=30, look_at_neg=False
                )
                self.poses = torch.cat(
                    [
                        pose_a.repeat(60, 1, 1),
                        a2b,
                        pose_b.repeat(60, 1, 1),
                        b2c,
                        pose_c.repeat(60, 1, 1),
                        c2a,
                    ],
                    dim=0,
                )

            if scene_name == "fork":
                pose_a = poses[21 : 21 + 1, :3]
                pose_b = poses[1 : 1 + 1, :3]
                pose_c = poses[4 : 4 + 1, :3]
                a2b = generate_interpolated_path(
                    torch.cat([pose_a, pose_b]), n_interp=30, look_at_neg=False
                )
                b2c = generate_interpolated_path(
                    torch.cat([pose_b, pose_c]), n_interp=30, look_at_neg=False
                )
                c2a = generate_interpolated_path(
                    torch.cat([pose_c, pose_a]), n_interp=30, look_at_neg=False
                )
                self.poses = torch.cat([pose_a.repeat(30, 1, 1), a2b, b2c, c2a], dim=0)

            imgs = None
        else:
            if self.timestep != "000" and split == "train":
                imgs, poses, intrinsics, split_indices = _load_colmap_temporal(
                    data_dir, downsample, hold_every, self.timestep, frame_offset
                )
            else:
                imgs, poses, intrinsics, split_indices = _load_colmap(
                    data_dir, downsample, hold_every
                )

            T, sscale = similarity_from_cameras(poses, strict_scaling=False)
            poses = np.einsum("nij, ki -> nkj", poses, T)
            poses[:, :3, 3] *= sscale

            self.poses = torch.from_numpy(poses).float()

            indices = split_indices[split]
            imgs = imgs[indices]
            self.poses = self.poses[indices]
            imgs = torch.from_numpy(imgs).to(torch.uint8)

            if split == "train":
                imgs = imgs.view(-1, imgs.shape[-1])
            else:
                imgs = imgs.view(
                    -1, intrinsics.height * intrinsics.width, imgs.shape[-1]
                )

        self.num_images = len(os.listdir(f"{data_dir}/images"))
        self.test_img_id = self.num_images - frame_offset

        self.near_fars = torch.tensor([near_plane, far_plane])
        if split != "train":
            self.near_fars[0] += near_plane_offset

        self.global_translation = torch.tensor([0.0, 0.0, 0.0])
        self.global_scale = aabb
        bbox = torch.tensor([aabb[:3], aabb[3:]])

        if type(use_app) == str:
            use_app = bool(eval(use_app))

        self.use_app_emb = use_app
        self.OPENGL_CAMERA = False

        super().__init__(
            datadir=datadir,
            split=split,
            scene_bbox=bbox,
            batch_size=batch_size,
            imgs=imgs,
            rays_o=None,
            rays_d=None,
            intrinsics=intrinsics,
            is_ndc=False,
            is_contracted=contraction,
        )

        log.info(
            f"TripodDataset. {contraction=}. Loaded {split} set from {datadir}. "
            f"{self.num_images} poses of shape {self.img_h}x{self.img_w}. "
            f"Images loaded: {imgs is not None}. Near-far[:3]: {self.near_fars[:3]}. "
            f"Sampling without replacement={self.use_permutation}. {intrinsics}"
        )

    def __getitem__(self, index):
        h = self.intrinsics.height
        w = self.intrinsics.width
        dev = "cpu"
        out = {"near_fars": self.near_fars.view(-1, 2)}

        if self.split == "train":
            index = self.get_rand_ids(index)
            image_id = torch.div(index, h * w, rounding_mode="floor").long()
            y = torch.remainder(index, h * w).div(w, rounding_mode="floor")
            x = torch.remainder(index, h * w).remainder(w)
            x = x + 0.5
            y = y + 0.5

            if self.bg_color_aug == "random":
                out["bg_color"] = torch.rand_like(torch.tensor([[0.0, 0.0, 0.0]]))
            elif self.bg_color_aug == "white":
                out["bg_color"] = torch.tensor([[1.0, 1.0, 1.0]])
            else:
                out["bg_color"] = torch.tensor([[0.0, 0.0, 0.0]])
        else:
            image_id = torch.LongTensor([index])
            x, y = create_meshgrid(height=h, width=w, dev=dev, add_half=True, flat=True)
            if self.bg_color_aug == "random":
                out["bg_color"] = torch.tensor([[0.0, 0.0, 0.0]])
            elif self.bg_color_aug == "white":
                out["bg_color"] = torch.tensor([[1.0, 1.0, 1.0]])
            else:
                out["bg_color"] = torch.tensor([[0.0, 0.0, 0.0]])

        if self.imgs is not None:
            out["imgs"] = (
                self.imgs[index] / 255.0
            )  # (num_rays, 3)   this converts to f32
        else:
            out["imgs"] = None

        c2w = self.poses[image_id]  # (num_rays, 3, 4)
        camera_dirs = stack_camera_dirs(
            x, y, self.intrinsics, self.OPENGL_CAMERA
        )  # [num_rays, 3]
        # [num_rays, 3]
        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        viewdirs = directions / torch.linalg.norm(directions, dim=-1, keepdims=True)

        out["rays_o"] = origins
        out["rays_d"] = viewdirs

        if self.use_app_emb:
            out["timestamps"] = (image_id + 1) % self.num_images
            if self.timestep != "000":
                out["timestamps"] = (
                    self.test_img_id * torch.ones_like(image_id) + 1
                ) % self.num_images

        return out

