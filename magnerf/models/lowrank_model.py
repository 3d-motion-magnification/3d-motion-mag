from typing import List, Sequence, Optional, Union, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from magnerf.models.density_fields import KPlaneDensityField
from magnerf.models.kplane_field import KPlaneField
from magnerf.ops.activations import init_density_activation
from magnerf.raymarching.ray_samplers import (
    UniformLinDispPiecewiseSampler, UniformSampler,
    ProposalNetworkSampler, RayBundle, RaySamples
)
from magnerf.raymarching.spatial_distortions import SceneContraction, SpatialDistortion
from magnerf.utils.timer import CudaTimer
from magnerf.models.kplane_field import interpolate_ms_features, normalize_aabb

def exp_map_SO3xR3(tangent_vector):
    """Compute the exponential map of the direct product group `SO(3) x R^3`.

    This can be used for learning pose deltas on SE(3), and is generally faster than `exp_map_SE3`.

    Args:
        tangent_vector: Tangent vector; length-3 translations, followed by an `so(3)` tangent vector.
    Returns:
        [R|t] transformation matrices.
    """
    # code for SO3 map grabbed from pytorch3d and stripped down to bare-bones
    log_rot = tangent_vector[:, 3:]
    nrms = (log_rot * log_rot).sum(1)
    rot_angles = torch.clamp(nrms, 1e-4).sqrt()
    rot_angles_inv = 1.0 / rot_angles
    fac1 = rot_angles_inv * rot_angles.sin()
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())
    skews = torch.zeros((log_rot.shape[0], 3, 3), dtype=log_rot.dtype, device=log_rot.device)
    skews[:, 0, 1] = -log_rot[:, 2]
    skews[:, 0, 2] = log_rot[:, 1]
    skews[:, 1, 0] = log_rot[:, 2]
    skews[:, 1, 2] = -log_rot[:, 0]
    skews[:, 2, 0] = -log_rot[:, 1]
    skews[:, 2, 1] = log_rot[:, 0]
    skews_square = torch.bmm(skews, skews)

    ret = torch.zeros(tangent_vector.shape[0], 3, 4, dtype=tangent_vector.dtype, device=tangent_vector.device)
    ret[:, :3, :3] = (
        fac1[:, None, None] * skews
        + fac2[:, None, None] * skews_square
        + torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]
    )

    # Compute the translation
    ret[:, :3, 3] = tangent_vector[:, :3]
    return ret


def pose_multiply(pose_a, pose_b):
    """Multiply two pose matrices, A @ B.

    Args:
        pose_a: Left pose matrix, usually a transformation applied to the right.
        pose_b: Right pose matrix, usually a camera pose that will be transformed by pose_a.

    Returns:
        Camera pose matrix where pose_a was applied to pose_b.
    """
    R1, t1 = pose_a[..., :3, :3], pose_a[..., :3, 3:]
    R2, t2 = pose_b[..., :3, :3], pose_b[..., :3, 3:]
    R = R1.matmul(R2)
    t = t1 + R1.matmul(t2)
    return torch.cat([R, t], dim=-1)


class LowrankModel(nn.Module):
    def __init__(self,
                 grid_config: Union[str, List[Dict]],
                 # boolean flags
                 is_ndc: bool,
                 is_contracted: bool,
                 aabb: torch.Tensor,
                 # Model arguments
                 multiscale_res: Sequence[int],
                 density_activation: Optional[str] = 'trunc_exp',
                 concat_features_across_scales: bool = False,
                 linear_decoder: bool = True,
                 linear_decoder_layers: Optional[int] = 1,
                 # Spatial distortion
                 global_translation: Optional[torch.Tensor] = None,
                 global_scale: Optional[torch.Tensor] = None,
                 # proposal-sampling arguments
                 num_proposal_iterations: int = 1,
                 use_same_proposal_network: bool = False,
                 proposal_net_args_list: List[Dict] = None,
                 num_proposal_samples: Optional[Tuple[int]] = None,
                 num_samples: Optional[int] = None,
                 single_jitter: bool = False,
                 proposal_warmup: int = 5000,
                 proposal_update_every: int = 5,
                 use_proposal_weight_anneal: bool = True,
                 proposal_weights_anneal_max_num_iters: int = 1000,
                 proposal_weights_anneal_slope: float = 10.0,
                 # appearance embedding (phototourism)
                 use_appearance_embedding: bool = False,
                 appearance_embedding_dim: int = 0,
                 num_images: Optional[int] = None,
                 opaque_background: bool = False,
                 **kwargs,
                 ):
        super().__init__()
        if isinstance(grid_config, str):
            self.config: List[Dict] = eval(grid_config)
        else:
            self.config: List[Dict] = grid_config
        self.multiscale_res = multiscale_res
        self.is_ndc = is_ndc
        self.is_contracted = is_contracted
        self.concat_features_across_scales = concat_features_across_scales
        self.linear_decoder = linear_decoder
        self.linear_decoder_layers = linear_decoder_layers
        self.density_act = init_density_activation(density_activation)
        self.timer = CudaTimer(enabled=False)

        self.spatial_distortion: Optional[SpatialDistortion] = None
        if self.is_contracted:
            self.spatial_distortion = SceneContraction(
                order=float('inf'), global_scale=global_scale,
                global_translation=global_translation)

        self.field = KPlaneField(
            aabb,
            grid_config=self.config,
            concat_features_across_scales=self.concat_features_across_scales,
            multiscale_res=self.multiscale_res,
            use_appearance_embedding=use_appearance_embedding,
            appearance_embedding_dim=appearance_embedding_dim,
            spatial_distortion=self.spatial_distortion,
            density_activation=self.density_act,
            linear_decoder=self.linear_decoder,
            linear_decoder_layers=self.linear_decoder_layers,
            num_images=num_images,
            lagrangian=kwargs.get('lagrangian')
        )
        if num_images:
            self.pose_adjustment = torch.nn.Parameter(torch.zeros((num_images, 6)))

        # Initialize proposal-sampling nets
        self.density_fns = []
        self.num_proposal_iterations = num_proposal_iterations
        self.proposal_net_args_list = proposal_net_args_list
        self.proposal_warmup = proposal_warmup
        self.proposal_update_every = proposal_update_every
        self.use_proposal_weight_anneal = use_proposal_weight_anneal
        self.proposal_weights_anneal_max_num_iters = proposal_weights_anneal_max_num_iters
        self.proposal_weights_anneal_slope = proposal_weights_anneal_slope
        self.proposal_networks = torch.nn.ModuleList()
        if use_same_proposal_network:
            assert len(self.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.proposal_net_args_list[0]
            network = KPlaneDensityField(
                aabb, spatial_distortion=self.spatial_distortion,
                density_activation=self.density_act, linear_decoder=self.linear_decoder, **prop_net_args)
            self.proposal_networks.append(network)
            self.density_fns.extend([network.get_density for _ in range(self.num_proposal_iterations)])
        else:
            for i in range(self.num_proposal_iterations):
                prop_net_args = self.proposal_net_args_list[min(i, len(self.proposal_net_args_list) - 1)]
                network = KPlaneDensityField(
                    aabb, spatial_distortion=self.spatial_distortion,
                    density_activation=self.density_act, linear_decoder=self.linear_decoder, **prop_net_args,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.get_density for network in self.proposal_networks])

        update_schedule = lambda step: np.clip(
            np.interp(step, [0, self.proposal_warmup], [0, self.proposal_update_every]),
            1,
            self.proposal_update_every,
        )
        if self.is_contracted or self.is_ndc:
            initial_sampler = UniformLinDispPiecewiseSampler(single_jitter=single_jitter)
        else:
            initial_sampler = UniformSampler(single_jitter=single_jitter)

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=num_samples,
            num_proposal_samples_per_ray=num_proposal_samples,
            num_proposal_network_iterations=self.num_proposal_iterations,
            single_jitter=single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler
        )
        if not isinstance(opaque_background, bool):
            opaque_background = eval(opaque_background)
        self.opaque_background = opaque_background

    def step_before_iter(self, step):
        if self.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.proposal_weights_anneal_max_num_iters
            # https://arxiv.org/pdf/2111.12077.pdf eq. 18
            train_frac = np.clip(step / N, 0, 1)
            bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
            anneal = bias(train_frac, self.proposal_weights_anneal_slope)
            self.proposal_sampler.set_anneal(anneal)

    def step_after_iter(self, step):
        if self.use_proposal_weight_anneal:
            self.proposal_sampler.step_cb(step)

    @staticmethod
    def render_rgb(rgb: torch.Tensor, weights: torch.Tensor, bg_color: Optional[torch.Tensor]):
        comp_rgb = torch.sum(weights * rgb, dim=-2)
        accumulated_weight = torch.sum(weights, dim=-2)
        if bg_color is None:
            pass
        else:
            comp_rgb = comp_rgb + (1.0 - accumulated_weight) * bg_color
        return comp_rgb

    @staticmethod
    def render_depth(weights: torch.Tensor, ray_samples: RaySamples, rays_d: torch.Tensor):
        steps = (ray_samples.starts + ray_samples.ends) / 2
        one_minus_transmittance = torch.sum(weights, dim=-2)
        depth = torch.sum(weights * steps, dim=-2) + one_minus_transmittance * rays_d[..., -1:]
        return depth

    @staticmethod
    def render_accumulation(weights: torch.Tensor):
        accumulation = torch.sum(weights, dim=-2)
        return accumulation

    def forward(self, rays_o, rays_d, bg_color, near_far: torch.Tensor,
                timestamps=None, c2w=None, camera_dirs=None, indices=None):
        """
        rays_o : [batch, 3]
        rays_d : [batch, 3]
        timestamps : [batch]
        near_far : [batch, 2]
        """
        # Fix shape for near-far
        nears, fars = torch.split(near_far, [1, 1], dim=-1)
        if nears.shape[0] != rays_o.shape[0]:
            ones = torch.ones_like(rays_o[..., 0:1])
            nears = ones * nears
            fars = ones * fars

        if indices is not None:
            camera_opt_to_camera = exp_map_SO3xR3(self.pose_adjustment[indices, :])
            c2w = c2w.to(camera_opt_to_camera.device)
            camera_dirs = camera_dirs.to(camera_opt_to_camera.device)
            c2w = pose_multiply(c2w, camera_opt_to_camera)
            directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
            rays_d = directions / torch.linalg.norm(directions, dim=-1, keepdims=True)
            rays_o = torch.broadcast_to(c2w[:, :3, -1], rays_d.shape)

        ray_bundle = RayBundle(origins=rays_o, directions=rays_d, nears=nears, fars=fars)
        # Note: proposal sampler mustn't use timestamps (=camera-IDs) with appearance-embedding,
        #       since the appearance embedding should not affect density. We still pass them in the
        #       call below, but they will not be used as long as density-field resolutions
        #       are be 3D.

        ray_samples, weights_list, ray_samples_list = self.proposal_sampler.generate_ray_samples(
            ray_bundle, timestamps=timestamps, density_fns=self.density_fns)

        field_out = self.field(ray_samples.get_positions(), ray_bundle.directions, timestamps)
        rgb, density = field_out["rgb"], field_out["density"]

        weights = ray_samples.get_weights(density, opaque_background=self.opaque_background)
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.render_rgb(rgb=rgb, weights=weights, bg_color=bg_color)
        depth = self.render_depth(weights=weights, ray_samples=ray_samples, rays_d=ray_bundle.directions)
        accumulation = self.render_accumulation(weights=weights)
        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
        }

        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list
        for i in range(self.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.render_depth(
                weights=weights_list[i], ray_samples=ray_samples_list[i], rays_d=ray_bundle.directions)
        return outputs


    def get_params(self, lr: float):
        model_params = self.field.get_params()
        pn_params = [pn.get_params() for pn in self.proposal_networks]
        field_params = model_params["field"] + [p for pnp in pn_params for p in pnp["field"]]
        nn_params = model_params["nn"] + [p for pnp in pn_params for p in pnp["nn"]]
        other_params = model_params["other"] + [p for pnp in pn_params for p in pnp["other"]]
        return [
            {"params": field_params, "lr": lr},
            {"params": nn_params, "lr": lr},
            {"params": other_params, "lr": lr},
        ]

    def partial_forward_0(self, rays_o, rays_d, bg_color, near_far: torch.Tensor, timestamps=None):
        """
        rays_o : [batch, 3]
        rays_d : [batch, 3]
        timestamps : [batch]
        near_far : [batch, 2]
        """
        # Fix shape for near-far
        nears, fars = torch.split(near_far, [1, 1], dim=-1)
        if nears.shape[0] != rays_o.shape[0]:
            ones = torch.ones_like(rays_o[..., 0:1])
            nears = ones * nears
            fars = ones * fars

        ray_bundle = RayBundle(origins=rays_o, directions=rays_d, nears=nears, fars=fars)
        # Note: proposal sampler mustn't use timestamps (=camera-IDs) with appearance-embedding,
        #       since the appearance embedding should not affect density. We still pass them in the
        #       call below, but they will not be used as long as density-field resolutions
        #       are be 3D.
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler.generate_ray_samples(
            ray_bundle, timestamps=timestamps, density_fns=self.density_fns)

        return ray_samples, ray_bundle


    def partial_forward_1(self, ray_samples, ray_bundle):
        field_out = self.field(ray_samples.get_positions(), ray_bundle.directions)
        rgbs, density = field_out["rgb"], field_out["density"]
        weights = ray_samples.get_weights(density)
        
        return rgbs, weights


    def partial_forward_2(self, rgbs, weights, bg_color):
        rgb = self.render_rgb(rgb=rgbs, weights=weights, bg_color=bg_color)

        return rgb


    def partial_forward_lagrangian_1(self, pts):
        if self.field.spatial_distortion is not None:
            pts = self.spatial_distortion(pts)
            pts = pts / 2  # from [-2, 2] to [-1, 1]
        else:
            pts = normalize_aabb(pts, self.field.aabb)

        pts = pts.reshape(-1, pts.shape[-1])

        delta = self.field.motion_net(pts)

        return delta, pts

    def partial_forward_lagrangian_2(self, pts, directions, n_rays, n_samples, ray_samples, bg_color):

        features = interpolate_ms_features(
            pts, ms_grids=self.field.grids,  # noqa
            grid_dimensions=self.field.grid_config[0]["grid_dimensions"],
            concat_features=self.field.concat_features, num_levels=None)
        if len(features) < 1:
            features = torch.zeros((0, 1)).to(features.device)
        if self.field.linear_decoder:
            density_before_activation = self.sigma_net(features)  # [batch, 1]
        else:
            features = self.field.sigma_net(features)
            features, density_before_activation = torch.split(
                features, [self.field.geo_feat_dim, 1], dim=-1)

        density = self.field.density_activation(
            density_before_activation.to(pts)
        ).view(n_rays, n_samples, 1)

        directions = (directions + 1.0) / 2.0
        encoded_directions = self.field.direction_encoder(directions)

        color_features = [encoded_directions, features.view(-1, self.field.geo_feat_dim)]

        color_features = torch.cat(color_features, dim=-1)
        rgb = self.field.color_net(color_features).to(directions).view(n_rays, n_samples, 3)

        weights = ray_samples.get_weights(density)

        rgb = self.render_rgb(rgb=rgb, weights=weights, bg_color=bg_color)

        return rgb
