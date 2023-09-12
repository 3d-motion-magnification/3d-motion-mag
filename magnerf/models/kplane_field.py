import itertools
import logging as log
from typing import Optional, Union, List, Dict, Sequence, Iterable, Collection, Callable

import torch
import torch.nn as nn
import tinycudann as tcnn
import numpy as np

from magnerf.ops.interpolation import grid_sample_wrapper
from magnerf.raymarching.spatial_distortions import SpatialDistortion


def get_normalized_directions(directions):
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0


def normalize_aabb(pts, aabb):
    '''
    x = pts[..., 0:1]
    new_x = torch.where(x > 0.75, 10 * torch.ones_like(x), x)
    new_x = torch.where(x < -0.75, 10 * torch.ones_like(x), new_x)
    y = pts[..., 1:2]
    new_y = torch.where(x > 1.0, 10 * torch.ones_like(x), y)
    new_y = torch.where(x < -1.0, 10 * torch.ones_like(x), new_y)
    z = pts[..., 1:2]
    new_z = torch.where(z > 2, 10 * torch.ones_like(z), z)
    new_z = torch.where(z < 1, 10 * torch.ones_like(z), new_z)
    pts = torch.cat([new_x, new_y, new_z], -1)
    '''
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0


def init_grid_param(
        grid_nd: int,
        in_dim: int,
        out_dim: int,
        reso: Sequence[int],
        a: float = 0.1,
        b: float = 0.2):
    assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
    has_time_planes = in_dim == 4
    assert grid_nd <= in_dim
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
    grid_coefs = nn.ParameterList()
    for ci, coo_comb in enumerate(coo_combs):
        new_grid_coef = nn.Parameter(torch.empty(
            [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]
        ))
        if has_time_planes and 3 in coo_comb:  # Initialize time planes to 1
            nn.init.ones_(new_grid_coef)
        else:
            nn.init.uniform_(new_grid_coef, a=a, b=b)

        grid_coefs.append(new_grid_coef)

    return grid_coefs


def init_delta_grid_param(
        grid_nd: int,
        in_dim: int,
        out_dim: int,
        reso: Sequence[int],
        a: float = 0.1,
        b: float = 0.5):
    assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
    assert grid_nd <= in_dim
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
    grid_coefs = nn.ParameterList()
    for ci, coo_comb in enumerate(coo_combs):
        new_grid_coef = nn.Parameter(torch.empty(
            [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]
        ))
        nn.init.zeros_(new_grid_coef)
        grid_coefs.append(new_grid_coef)

    return grid_coefs



def init_grid_param_motion(
        grid_nd: int,
        in_dim: int,
        out_dim: int,
        reso: Sequence[int],
        zeros: bool = True):
    assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
    assert grid_nd <= in_dim
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
    grid_coefs = nn.ParameterList()
    for ci, coo_comb in enumerate(coo_combs):
        new_grid_coef = nn.Parameter(torch.empty(
            [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]
        ))
        nn.init.uniform_(new_grid_coef, -0.001, 0.001)
        grid_coefs.append(new_grid_coef)

    return grid_coefs


def interpolate_ms_features(pts: torch.Tensor,
                            ms_grids: Collection[Iterable[nn.Module]],
                            grid_dimensions: int,
                            concat_features: bool,
                            num_levels: Optional[int],
                            ) -> torch.Tensor:
    hadamard = False
    coo_combs = list(itertools.combinations(
        range(pts.shape[-1]), grid_dimensions)
    )
    if num_levels is None:
        num_levels = len(ms_grids)
    multi_scale_interp = [] if concat_features else 0.
    grid: nn.ParameterList
    for scale_id, grid in enumerate(ms_grids[:num_levels]):
        interp_space = 1. if hadamard else []
        for ci, coo_comb in enumerate(coo_combs):
            # interpolate in plane
            feature_dim = grid[ci].shape[1]  # shape of grid[ci]: 1, out_dim, *reso
            interp_out_plane = (
                grid_sample_wrapper(grid[ci], pts[..., coo_comb])
                .view(-1, feature_dim)
            )
            # compute product over planes
            interp_space = interp_space * interp_out_plane if hadamard else interp_space + [interp_out_plane]
        if not hadamard:
            interp_space = torch.cat(interp_space, dim = -1)

        # combine over scales
        if concat_features:
            multi_scale_interp.append(interp_space)
        else:
            multi_scale_interp = multi_scale_interp + interp_space

    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
    return multi_scale_interp


def interpolate_ms_features_v2(pts: torch.Tensor,
                            ms_grids: Collection[Iterable[nn.Module]],
                            grid_dimensions: int,
                            concat_features: bool,
                            num_levels: Optional[int],
                            ) -> torch.Tensor:
    coo_combs = list(itertools.combinations(
        range(pts.shape[-1] - 1), grid_dimensions)
    )
    if num_levels is None:
        num_levels = len(ms_grids)
    multi_scale_interp = [] if concat_features else 0.
    grid: nn.ParameterList
    for scale_id, grid in enumerate(ms_grids[:num_levels]):
        interp_space = 1.
        for ci, _coo_comb in enumerate(coo_combs):
            coo_comb = _coo_comb + (3, )
            # interpolate in plane
            feature_dim = grid[ci].shape[1]  # shape of grid[ci]: 1, out_dim, *reso
            interp_out_plane = (
                grid_sample_wrapper(grid[ci], pts[..., coo_comb])
                .view(-1, feature_dim)
            )
            # compute product over planes
            interp_space = interp_space * interp_out_plane

        # combine over scales
        if concat_features:
            multi_scale_interp.append(interp_space)
        else:
            multi_scale_interp = multi_scale_interp + interp_space

    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
    return multi_scale_interp


class KPlaneField(nn.Module):
    def __init__(
        self,
        aabb,
        grid_config: Union[str, List[Dict]],
        concat_features_across_scales: bool,
        multiscale_res: Optional[Sequence[int]],
        use_appearance_embedding: bool,
        appearance_embedding_dim: int,
        spatial_distortion: Optional[SpatialDistortion],
        density_activation: Callable,
        linear_decoder: bool,
        linear_decoder_layers: Optional[int],
        num_images: Optional[int],
        lagrangian: bool = False,
    ) -> None:
        super().__init__()

        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.spatial_distortion = spatial_distortion
        self.grid_config = grid_config

        self.multiscale_res_multipliers: List[int] = multiscale_res or [1]
        self.concat_features = concat_features_across_scales
        self.density_activation = density_activation
        self.linear_decoder = linear_decoder
        self.lagrangian = lagrangian

        # 1. Init planes
        self.grids = nn.ModuleList()
        self.feature_dim = 0
        for res in self.multiscale_res_multipliers:
            # initialize coordinate grid
            config = self.grid_config[0].copy()
            # Resolution fix: multi-res only on spatial planes
            config["resolution"] = [
                r * res for r in config["resolution"][:3]
            ] + config["resolution"][3:]
            gp = init_grid_param(
                grid_nd=config["grid_dimensions"],
                in_dim=config["input_coordinate_dim"],
                out_dim=config["output_coordinate_dim"],
                reso=config["resolution"],
            )
            # shape[1] is out-dim - Concatenate over feature len for each scale
            if self.concat_features:
                self.feature_dim += gp[-1].shape[1]
            else:
                self.feature_dim = gp[-1].shape[1]
            self.grids.append(gp)
            self.feature_dim = self.feature_dim * 3

        log.info(f"Initialized model grids: {self.grids}")

        self.delta_grids = nn.ModuleList()
        for res in self.multiscale_res_multipliers:
            config = self.grid_config[0].copy()
            config["resolution"] = [
                r * res for r in config["resolution"][:3]
            ] + config["resolution"][3:]
            gp = init_delta_grid_param(
                grid_nd=config["grid_dimensions"],
                in_dim=config["input_coordinate_dim"],
                out_dim=config["output_coordinate_dim"],
                reso=config["resolution"],
            )
            self.delta_grids.append(gp)

        self.use_delta_features = False
        self.delta_add_back = True

        if self.lagrangian:
            self.motion_net = tcnn.NetworkWithInputEncoding(
                n_input_dims=3,
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 5,
                },
                encoding_config={
                    "otype": "Frequency",
                    "n_frequencies": 3,
                }
            )

        # 2. Init appearance code-related parameters
        self.use_average_appearance_embedding = True  # for test-time
        self.use_appearance_embedding = use_appearance_embedding
        self.num_images = num_images
        self.appearance_embedding = None
        if use_appearance_embedding:
            assert self.num_images is not None
            self.appearance_embedding_dim = appearance_embedding_dim
            # this will initialize as normal_(0.0, 1.0)
            self.appearance_embedding = nn.Embedding(self.num_images, self.appearance_embedding_dim)
            self.test_appearance_idx = -1
        else:
            self.appearance_embedding_dim = 0


        self.direction_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 6,
            },
        )

        # 3. Init decoder network
        if self.linear_decoder:
            assert linear_decoder_layers is not None
            # The NN learns a basis that is used instead of spherical harmonics
            # Input is an encoded view direction, output is weights for
            # combining the color features into RGB
            # This architecture is based on instant-NGP
            self.color_basis = tcnn.Network(
                n_input_dims=3 + self.appearance_embedding_dim,#self.direction_encoder.n_output_dims,
                n_output_dims=3 * self.feature_dim,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 128,
                    "n_hidden_layers": linear_decoder_layers,
                },
            )
            # sigma_net just does a linear transformation on the features to get density
            self.sigma_net = tcnn.Network(
                n_input_dims=self.feature_dim,
                n_output_dims=1,
                network_config={
                    "otype": "CutlassMLP",
                    "activation": "None",
                    "output_activation": "None",
                    "n_neurons": 128,
                    "n_hidden_layers": 0,
                },
            )
        else:
            self.geo_feat_dim = 15
            self.sigma_net = tcnn.Network(
                n_input_dims=self.feature_dim,
                n_output_dims=self.geo_feat_dim + 1,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                },
            )
            self.in_dim_color = (
                    self.direction_encoder.n_output_dims
                    + self.geo_feat_dim
                    + self.appearance_embedding_dim
            )
            self.color_net = tcnn.Network(
                n_input_dims=self.in_dim_color,
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "Sigmoid",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                },
            )
            self.feature_norm = None

        max_resolution = 2048
        base_resolution = 16
        n_levels = 16
        per_level_scale = np.exp(
            (np.log(max_resolution) - np.log(base_resolution)) / (n_levels - 1)
        ).tolist()
        log2_hashmap_size = 19

        if False:
            self.mlp_base = tcnn.NetworkWithInputEncoding(
                n_input_dims=3,
                n_output_dims=1 + self.geo_feat_dim,
                encoding_config={
                    "otype": "HashGrid",
                    "n_levels": n_levels,
                    "n_features_per_level": 2,
                    "log2_hashmap_size": log2_hashmap_size,
                    "base_resolution": base_resolution,
                    "per_level_scale": per_level_scale,
                },
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 3,
                },
            )

    def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        """Computes and returns the densities."""
        if self.spatial_distortion is not None:
            pts = self.spatial_distortion(pts)
            pts = pts / 2  # from [-2, 2] to [-1, 1]
        else:
            pts = normalize_aabb(pts, self.aabb)

        n_rays, n_samples = pts.shape[:2]

        if timestamps is not None:
            timestamps = timestamps[:, None].expand(-1, n_samples)[..., None]  # [n_rays, n_samples, 1]
            pts = torch.cat((pts, timestamps), dim=-1)  # [n_rays, n_samples, 4]

        #for multires_grids in self.grids:
        #    for i, grid in enumerate(multires_grids):
        #        if i in [2, 4, 5]:
        #            grid.data[:, :, 0] = torch.ones_like(grid.data[:, :, 0])

        pts = pts.reshape(-1, pts.shape[-1])

        if self.lagrangian:
            delta = self.motion_net(pts)
            pts = pts + delta

        features = interpolate_ms_features(
            pts, ms_grids=self.grids,  # noqa
            grid_dimensions=self.grid_config[0]["grid_dimensions"],
            concat_features=self.concat_features, num_levels=None)

        if self.use_delta_features:
            delta_features = interpolate_ms_features(
                pts, ms_grids=self.delta_grids,  # noqa
                grid_dimensions=self.grid_config[0]["grid_dimensions"],
                concat_features=self.concat_features, num_levels=None)
            if self.delta_add_back:
                features = delta_features + features
            else:
                features = delta_features

        if len(features) < 1:
            features = torch.zeros((0, 1)).to(features.device)
        if self.linear_decoder:
            density_before_activation = self.sigma_net(features)  # [batch, 1]
        else:
            features = self.sigma_net(features)
            #features = self.mlp_base((pts + 1.0) / 2)

            features, density_before_activation = torch.split(
                features, [self.geo_feat_dim, 1], dim=-1)

        if self.feature_norm is not None:
            features = self.feature_norm(features)

        density = self.density_activation(
            density_before_activation.to(pts)
        ).view(n_rays, n_samples, 1)

        #density = self.density_activation(
        #    self.mlp_base((pts + 1.0) / 2)
        #).view(n_rays, n_samples, 1)

        return density, features

    def forward(self,
                pts: torch.Tensor,
                directions: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None):
        camera_indices = None
        if self.use_appearance_embedding:
            if timestamps is None:
                raise AttributeError("timestamps (appearance-ids) are not provided.")
            camera_indices = timestamps
            timestamps = None

        density, features = self.get_density(pts, timestamps)
        n_rays, n_samples = pts.shape[:2]

        directions = directions.view(-1, 1, 3).expand(pts.shape).reshape(-1, 3)
        if not self.linear_decoder:
            directions = get_normalized_directions(directions)
            encoded_directions = self.direction_encoder(directions)

        if self.linear_decoder:
            color_features = [features]
        else:
            color_features = [encoded_directions, features.view(-1, self.geo_feat_dim)]

        if self.use_appearance_embedding:
            if camera_indices.dtype == torch.float32:
                # Interpolate between two embeddings. Currently they are hardcoded below.
                #emb1_idx, emb2_idx = 100, 121  # trevi
                emb1_idx, emb2_idx = 11, 142  # sacre
                emb_fn = self.appearance_embedding
                emb1 = emb_fn(torch.full_like(camera_indices, emb1_idx, dtype=torch.long))
                emb1 = emb1.view(emb1.shape[0], emb1.shape[2])
                emb2 = emb_fn(torch.full_like(camera_indices, emb2_idx, dtype=torch.long))
                emb2 = emb2.view(emb2.shape[0], emb2.shape[2])
                embedded_appearance = torch.lerp(emb1, emb2, camera_indices)
            elif self.training:
                #if torch.sum(camera_indices < 0) > 0:
                #    embedded_appearance = torch.ones(
                #        (camera_indices.shape[0], self.appearance_embedding_dim), device=directions.device
                #    ) * self.appearance_embedding.weight.mean(dim=0)
                #else:
                #    embedded_appearance = self.appearance_embedding(camera_indices)
                embedded_appearance = self.appearance_embedding(camera_indices)
            else:
                if hasattr(self, "test_appearance_embedding"):
                    embedded_appearance = self.test_appearance_embedding(camera_indices)
                #elif self.use_average_appearance_embedding:
                #    embedded_appearance = torch.ones(
                #        (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                #    ) * self.appearance_embedding.weight.mean(dim=0)
                else:
                    #embedded_appearance = torch.zeros(
                    #    (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                    #)
                    embedded_appearance = torch.ones(
                        (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                    ) * self.appearance_embedding.weight[self.test_appearance_idx]

            if self.training:
                # expand embedded_appearance from n_rays, dim to n_rays*n_samples, dim
                ea_dim = embedded_appearance.shape[-1]
                embedded_appearance = embedded_appearance.view(-1, 1, ea_dim).expand(n_rays, n_samples, -1).reshape(-1, ea_dim)
            if not self.linear_decoder:
                color_features.append(embedded_appearance)

        color_features = torch.cat(color_features, dim=-1)

        if self.linear_decoder:
            if self.use_appearance_embedding:
                basis_values = self.color_basis(torch.cat([directions, embedded_appearance], dim=-1))
            else:
                basis_values = self.color_basis(directions)  # [batch, color_feature_len * 3]
            basis_values = basis_values.view(color_features.shape[0], 3, -1)  # [batch, 3, color_feature_len]
            rgb = torch.sum(color_features[:, None, :] * basis_values, dim=-1)  # [batch, 3]
            rgb = rgb.to(directions)
            rgb = torch.sigmoid(rgb).view(n_rays, n_samples, 3)
        else:
            rgb = self.color_net(color_features).to(directions).view(n_rays, n_samples, 3)

        rgb = rgb * (1 + 2 * 0.001) - 0.001

        return {"rgb": rgb, "density": density}

    def get_params(self):
        field_params = {k: v for k, v in self.grids.named_parameters(prefix="grids")}
        nn_params = [
            self.sigma_net.named_parameters(prefix="sigma_net"),
            self.direction_encoder.named_parameters(prefix="direction_encoder"),
        ]
        if self.linear_decoder:
            nn_params.append(self.color_basis.named_parameters(prefix="color_basis"))
        else:
            nn_params.append(self.color_net.named_parameters(prefix="color_net"))
        nn_params = {k: v for plist in nn_params for k, v in plist}
        other_params = {k: v for k, v in self.named_parameters() if (
            k not in nn_params.keys() and k not in field_params.keys()
        )}
        return {
            "nn": list(nn_params.values()),
            "field": list(field_params.values()),
            "other": list(other_params.values()),
        }
