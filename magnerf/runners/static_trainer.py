import math
import os
from collections import defaultdict
from typing import Dict, MutableMapping, Union, Any

import pandas as pd
import torch
import torch.utils.data

from magnerf.datasets import SyntheticNerfDataset, HandheldDataset
from magnerf.models.lowrank_model import LowrankModel
from magnerf.utils.ema import EMA
from magnerf.utils.my_tqdm import tqdm
from magnerf.utils.parse_args import parse_optint
from .base_trainer import BaseTrainer, init_dloader_random, initialize_model
from .regularization import (
    PlaneTV, HistogramLoss, L1ProposalNetwork, DepthTV, DistortionLoss, L1GridChange, TVGridChange
)


class StaticTrainer(BaseTrainer):
    def __init__(self,
                 tr_loader: torch.utils.data.DataLoader,
                 ts_dset: torch.utils.data.TensorDataset,
                 tr_dset: torch.utils.data.TensorDataset,
                 num_steps: int,
                 logdir: str,
                 expname: str,
                 train_fp16: bool,
                 save_every: int,
                 valid_every: int,
                 save_outputs: bool,
                 device: Union[str, torch.device],
                 **kwargs
                 ):
        self.test_dataset = ts_dset
        self.train_dataset = tr_dset
        self.is_ndc = self.test_dataset.is_ndc
        self.is_contracted = self.test_dataset.is_contracted

        super().__init__(
            train_data_loader=tr_loader,
            num_steps=num_steps,
            logdir=logdir,
            expname=expname,
            train_fp16=train_fp16,
            save_every=save_every,
            valid_every=int(valid_every),
            save_outputs=save_outputs,
            device=device,
            **kwargs
        )

    def eval_step(self, data, **kwargs) -> MutableMapping[str, torch.Tensor]:
        """
        Note that here `data` contains a whole image. we need to split it up before tracing
        for memory constraints.
        """
        super().eval_step(data, **kwargs)
        batch_size = self.eval_batch_size

        if "timestamps" not in data:
            data["timestamps"] = None

        channels = {"rgb", "depth", "proposal_depth"}
        with torch.cuda.amp.autocast(enabled=self.train_fp16), torch.no_grad():
            rays_o = data["rays_o"]
            rays_d = data["rays_d"]
            # near_far and bg_color are constant over mini-batches
            near_far = data["near_fars"].to(self.device)
            bg_color = data["bg_color"]
            if isinstance(bg_color, torch.Tensor):
                bg_color = bg_color.to(self.device)
            timestamps = data['timestamps']
            if isinstance(timestamps, torch.Tensor):
                timestamps = timestamps.to(self.device)

            preds = defaultdict(list)
            for b in range(math.ceil(rays_o.shape[0] / batch_size)):
                rays_o_b = rays_o[b * batch_size: (b + 1) * batch_size].to(self.device)
                rays_d_b = rays_d[b * batch_size: (b + 1) * batch_size].to(self.device)
                outputs = self.model(rays_o_b, rays_d_b, near_far=near_far,
                                     bg_color=bg_color, timestamps=timestamps)
                for k, v in outputs.items():
                    if k in channels or "depth" in k:
                        preds[k].append(v.cpu())
        return {k: torch.cat(v, 0) for k, v in preds.items()}

    def train_step(self, data: Dict[str, Union[int, torch.Tensor]], **kwargs):
        return super().train_step(data, **kwargs)

    def post_step(self, progress_bar):
        super().post_step(progress_bar)

    def pre_epoch(self):
        super().pre_epoch()
        self.train_dataset.reset_iter()

    @torch.no_grad()
    def validate(self):
        dataset = self.test_dataset
        per_scene_metrics = defaultdict(list)
        pb = tqdm(total=len(dataset), desc=f"Test scene {dataset.name}")
        for img_idx, data in enumerate(dataset):
            ts_render = self.eval_step(data)
            out_metrics, _, _ = self.evaluate_metrics(
                data["imgs"], ts_render, dset=dataset, img_idx=img_idx,
                name=None, save_outputs=self.save_outputs)
            for k, v in out_metrics.items():
                per_scene_metrics[k].append(v)
            pb.set_postfix_str(f"PSNR={out_metrics['psnr']:.2f}", refresh=False)
            pb.update(1)
        pb.close()
        val_metrics = [
            self.report_test_metrics(per_scene_metrics, extra_name="")
        ]
        df = pd.DataFrame.from_records(val_metrics)
        df.to_csv(os.path.join(self.log_dir, f"test_metrics_step{self.global_step}.csv"))

    def get_save_dict(self):
        base_save_dict = super().get_save_dict()
        return base_save_dict

    def load_model(self, checkpoint_data, training_needed: bool = True):
        super().load_model(checkpoint_data, training_needed)

    def init_epoch_info(self):
        ema_weight = 0.9  # higher places higher weight to new observations
        loss_info = defaultdict(lambda: EMA(ema_weight))
        return loss_info

    def init_model(self, **kwargs) -> LowrankModel:
        return initialize_model(self, **kwargs)

    def get_regularizers(self, **kwargs):
        regularizers = [
            PlaneTV(kwargs.get('plane_tv_weight', 0.0), what='field'),
            PlaneTV(kwargs.get('plane_tv_weight_proposal_net', 0.0), what='proposal_network'),
            HistogramLoss(kwargs.get('histogram_loss_weight', 0.0)),
            L1ProposalNetwork(kwargs.get('l1_proposal_net_weight', 0.0)),
            DepthTV(kwargs.get('depth_tv_weight', 0.0)),
            DistortionLoss(kwargs.get('distortion_loss_weight', 0.0)),
            L1GridChange(kwargs.get('gridchange_loss_weight', 0.0)),
            TVGridChange(kwargs.get('tv_gridchange_loss_weight', 0.0)),
        ]

        return regularizers

    @property
    def calc_metrics_every(self):
        return 5


def decide_dset_type(dd) -> str:
    if ("chair" in dd or "drums" in dd or "ficus" in dd or "hotdog" in dd
            or "lego" in dd or "materials" in dd or "mic" in dd
            or "ship" in dd):
        return "synthetic"
    elif ("fern" in dd or "flower" in dd or "fortress" in dd
          or "horns" in dd or "leaves" in dd or "orchids" in dd
          or "room" in dd or "trex" in dd):
        return "llff"
    else:
        raise RuntimeError(f"data_dir {dd} not recognized.")


def init_tr_data(data_downsample: float, data_dir: str, **kwargs):
    batch_size = int(kwargs['batch_size'])
    dset_type = kwargs["dset_type"] if "dset_type" in kwargs else "synthetic"

    if dset_type == "synthetic":
        max_tr_frames = parse_optint(kwargs.get('max_tr_frames'))
        radius = kwargs["radius"] if "radius" in kwargs else 1.3
        near_far = kwargs["near_far"] if "near_far" in kwargs else None
        dset = SyntheticNerfDataset(
            data_dir, split='train', downsample=data_downsample,
            max_frames=max_tr_frames, batch_size=batch_size, radius=radius, near_far=near_far)

    elif dset_type == "handheld":
        hold_every = parse_optint(kwargs.get('hold_every'))
        dset = HandheldDataset(
            data_dir, split='train', downsample=int(data_downsample), hold_every=hold_every,
            batch_size=batch_size,
            contraction=kwargs.get('contract', 'True'),
            train_cutoff=int(kwargs.get('train_cutoff', -1)),
            dynamic_start=int(kwargs.get('dynamic_start', 0)),
            aabb=kwargs.get('bbox'),
            use_app=kwargs.get('use_app_embedding', False),
            near_plane=float(kwargs.get('near_plane', 0.2)),
            far_plane=float(kwargs.get('far_plane', 30.0)),
            bg_color_aug=kwargs.get('bg_color_aug', 'black'),
        )
    else:
        raise ValueError(f"Dataset type {dset_type} invalid.")
    dset.reset_iter()

    tr_loader = torch.utils.data.DataLoader(
        dset, num_workers=4, prefetch_factor=4, pin_memory=True,
        batch_size=None, worker_init_fn=init_dloader_random)

    return {
        "tr_dset": dset,
        "tr_loader": tr_loader,
    }


def init_ts_data(data_downsample: float, data_dir: str, split: str, **kwargs):

    #dset_type = decide_dset_type(data_dir)
    dset_type = kwargs["dset_type"] if "dset_type" in kwargs else "synthetic"

    if dset_type == "synthetic":
        max_ts_frames = parse_optint(kwargs.get('max_ts_frames'))
        dset = SyntheticNerfDataset(
            data_dir, split=split, downsample=1, max_frames=max_ts_frames)
    elif dset_type == "handheld":
        hold_every = parse_optint(kwargs.get('hold_every'))
        dset = HandheldDataset(
            data_dir, split=split, downsample=data_downsample, hold_every=hold_every,
            contraction=kwargs.get('contract', 'True'),
            train_cutoff=int(kwargs.get('train_cutoff', -1)),
            dynamic_start=int(kwargs.get('dynamic_start', 0)),
            aabb=kwargs.get('bbox'),
            start_interp_index=int(kwargs.get('start_interp_index', 5)),
            use_app=kwargs.get('use_app_embedding', 'False'),
            near_plane_offset=float(kwargs.get('near_plane_offset', 0.0)),
            near_plane=float(kwargs.get('near_plane', 0.2)),
            far_plane=float(kwargs.get('far_plane', 30.0)),
            bg_color_aug=kwargs.get('bg_color_aug', 'black'),
        )
    else:
        raise ValueError(f"Dataset type {dset_type} invalid.")
    return {"ts_dset": dset}


def load_data(data_downsample, data_dir, validate_only, render_only, **kwargs):
    od: Dict[str, Any] = {}
    if not validate_only:
        od.update(init_tr_data(data_downsample, data_dir, **kwargs))
    else:
        od.update(tr_loader=None, tr_dset=None)
    test_split = 'render' if render_only else 'test'
    od.update(init_ts_data(data_downsample, data_dir, split=test_split, **kwargs))
    return od
