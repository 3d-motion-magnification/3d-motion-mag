import os
import warnings
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

import argparse
import importlib.util
import logging
import pprint
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from typing import List, Dict, Any
import tempfile

import numpy as np
import imageio
import tqdm
from scipy import signal
import subprocess

from plenoptic.simulate import Steerable_Pyramid_Freq


def bandpass_filter(data_3d, mask):
    d_data_3d = data_3d

    data_f = torch.fft.fft(d_data_3d)
    data_f = data_f * mask.to(data_f.device)

    out = torch.fft.ifft(data_f)

    return out.real


def bandpass_filter_np(data_3d, mask):
    d_data_3d = data_3d - data_3d[..., 0:1]
    data_f = np.fft.fft(d_data_3d)
    data_f[~mask] = 0
    filtered = np.fft.ifft(data_f)

    return filtered.real


def create_bandpass_mask(lowcut, highcut, n, fs):
    fl, fh = lowcut / fs, highcut / fs
    fl, fh = fl * 2, fh * 2
    B = signal.firwin(n, cutoff=[fl, fh], window="hamming", pass_zero="bandpass")
    B = torch.FloatTensor(B[:n])

    mask = torch.fft.fft(torch.fft.ifftshift(B))
    mask = mask.unsqueeze(0).unsqueeze(0)

    return mask


def create_bandpass_mask_ideal(lowcut, highcut, n, fs):
    freq = np.arange(n) / n
    freq = freq * fs

    mask = (freq > lowcut) * (freq <= highcut)
    mask = torch.from_numpy(mask)

    return mask


def amplify_one_channel_vid(
    f_vid, alpha, low, high, fs, ideal=False, suppress_others=False, flt_thrs=0.0
):
    if ideal:
        mask = create_bandpass_mask_ideal(
            lowcut=low, highcut=high, n=f_vid.shape[-1], fs=fs
        )
    else:
        mask = create_bandpass_mask(lowcut=low, highcut=high, n=f_vid.shape[-1], fs=fs)

    mask = mask.repeat(f_vid.shape[0], f_vid.shape[1], 1)

    ref_phs = f_vid.angle()[..., 0:1]

    delta = f_vid.angle() - ref_phs
    delta = ((np.pi + delta) % (2 * np.pi)) - np.pi

    flt_phs = bandpass_filter(delta, mask)

    thrs = torch.quantile(flt_phs.abs(), q=flt_thrs)
    flt_phs[flt_phs.abs() < thrs] = 0.0

    amp_phs = alpha * flt_phs
    amp_phs = ((np.pi + amp_phs) % (2 * np.pi)) - np.pi

    if suppress_others:
        amp_vid = f_vid[..., 0:1] * torch.exp(1j * amp_phs)
    else:
        amp_vid = f_vid * torch.exp(1j * amp_phs)

    return amp_vid


def process_feature_images(f_im, alpha, low, high, fs, device, args):
    f_im = torch.cat(f_im, dim=0)

    pyr = Steerable_Pyramid_Freq(
        height="auto",
        image_shape=[f_im.shape[-2], f_im.shape[-1]],
        order=args.orientation,
        is_complex=True,
        downsample=True,
        twidth=0.75,
    )
    pyr.to(device)
    mbsize = 30

    for channel in tqdm.trange(f_im.shape[1], disable=True):
        all_coeffs = {}

        for mb in range(0, len(f_im), mbsize):
            coeff = pyr.forward(
                f_im[mb : mb + mbsize, channel : channel + 1].to(device)
            )
            for key in coeff.keys():
                all_coeffs.setdefault(key, []).append(coeff[key])
            del coeff
        for key in all_coeffs.keys():
            all_coeffs[key] = torch.cat(all_coeffs[key])

        for key in all_coeffs.keys():
            _vid = all_coeffs[key]
            if "residual_highpass" in key or "residual_lowpass" in key:
                continue
            else:
                amp_vid = amplify_one_channel_vid(
                    _vid[:, 0].permute(1, 2, 0),
                    alpha,
                    low,
                    high,
                    fs,
                    ideal=args.ideal,
                    suppress_others=args.suppress_others,
                    flt_thrs=args.flt_thrs,
                )
            all_coeffs[key] = amp_vid.permute(2, 0, 1)[:, None]
            del amp_vid

        for mb in range(0, len(f_im), mbsize):
            mb_coeffs = {}
            for key in all_coeffs.keys():
                mb_coeffs[key] = all_coeffs[key][mb : mb + mbsize]

            out = pyr.recon_pyr(mb_coeffs)
            out = out.cpu()

            f_im[mb : mb + mbsize, channel : channel + 1] = out

            del out, mb_coeffs
        del all_coeffs

    return torch.unbind(f_im.unsqueeze(1), 0)


def get_freer_gpu():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_fname = os.path.join(tmpdir, "tmp")
        os.system(f'nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >"{tmp_fname}"')
        if os.path.isfile(tmp_fname):
            memory_available = [
                int(x.split()[2]) for x in open(tmp_fname, "r").readlines()
            ]
            if len(memory_available) > 0:
                return np.argmax(memory_available)
    return None  # The grep doesn't work with all GPUs. If it fails we ignore it.


gpu = get_freer_gpu()
if gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"CUDA_VISIBLE_DEVICES set to {gpu}")
else:
    print(f"Did not set GPU.")

import torch
import torch.utils.data
from magnerf.runners import static_trainer
from magnerf.utils.parse_args import parse_optfloat


def setup_logging(log_level=logging.INFO):
    handlers = [logging.StreamHandler(sys.stdout)]
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s|%(levelname)8s| %(message)s",
        handlers=handlers,
        force=True,
    )


def load_data(
    model_type: str,
    data_downsample,
    data_dirs,
    validate_only: bool,
    render_only: bool,
    **kwargs,
):
    data_downsample = parse_optfloat(data_downsample, default_val=1.0)

    return static_trainer.load_data(
        data_downsample,
        data_dirs,
        validate_only=validate_only,
        render_only=render_only,
        **kwargs,
    )


def init_trainer(model_type: str, **kwargs):
    from magnerf.runners import static_trainer

    return static_trainer.StaticTrainer(**kwargs)


def vis_grid_features(new_field_feature_imgs, base_logdir):
    planes = torch.cat(new_field_feature_imgs["delta_grids.0.0"]).cpu()
    out = planes.permute(0, 2, 3, 1)[..., :3]
    out = (out - out.min()) / (out.max() - out.min())
    out = np.uint8(out * 255)
    imageio.mimwrite(f"{base_logdir}/xy.mp4", out, fps=10, quality=8)

    planes = torch.cat(new_field_feature_imgs["delta_grids.0.1"]).cpu()
    out = planes.permute(0, 2, 3, 1)[..., :3]
    out = (out - out.min()) / (out.max() - out.min())
    out = np.uint8(out * 255)
    imageio.mimwrite(f"{base_logdir}/xz.mp4", out, fps=10, quality=8)

    planes = torch.cat(new_field_feature_imgs["delta_grids.0.2"]).cpu()
    out = planes.permute(0, 2, 3, 1)[..., :3]
    out = (out - out.min()) / (out.max() - out.min())
    out = np.uint8(out * 255)
    imageio.mimwrite(f"{base_logdir}/yz.mp4", out, fps=10, quality=8)


def main():
    setup_logging()
    logger = logging.getLogger()
    logger.disabled = True

    p = argparse.ArgumentParser(description="")

    p.add_argument("--render-only", action="store_true")
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--spacetime-only", action="store_true")
    p.add_argument("--config-path", type=str, required=True)
    p.add_argument("--log-dir", type=str, default=None)
    p.add_argument("--timestep", type=str, default="000")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_video", action="store_true")

    p.add_argument(
        "--depth", "-n", default=4, type=int, help="Depth of Pyramid. Integer"
    )
    p.add_argument(
        "--orientation", "-k", default=8, type=int, help="Orientation. Integer"
    )

    p.add_argument("--num_timesteps", type=int, default=30)
    p.add_argument("--num_frames", type=int, default=30)
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--low", type=float, default=2.5)
    p.add_argument("--high", type=float, default=3.5)
    p.add_argument("--fs", type=int, default=30)
    p.add_argument("--view_num", type=int, default=-1)
    p.add_argument("--ideal", action="store_true")
    p.add_argument("--suppress_others", action="store_true")
    p.add_argument("--start_idx", type=int, default=1)
    p.add_argument("--vid_fps", type=int, default=30)
    p.add_argument("--flt_thrs", type=float, default=0.0)

    p.add_argument("override", nargs=argparse.REMAINDER)

    args = p.parse_args()

    num_frames = args.num_frames
    num_timesteps = args.num_timesteps
    alpha = args.alpha
    low, high, fs = args.low, args.high, args.fs

    start_idx = args.start_idx
    device = "cuda"

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Import config
    spec = importlib.util.spec_from_file_location(
        os.path.basename(args.config_path), args.config_path
    )
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    config: Dict[str, Any] = cfg.config
    # Process overrides from argparse into config
    # overrides can be passed from the command line as key=value pairs. E.g.
    # python plenoxels/main.py --config-path plenoxels/config/cfg.py max_ts_frames=200
    # note that all values are strings, so code should assume incorrect data-types for anything
    # that's derived from config - and should not a string.
    overrides: List[str] = args.override
    overrides_dict = {ovr.split("=")[0]: ovr.split("=")[1] for ovr in overrides}

    timestep = "000"
    expname = config["expname"]
    data_dir = config["data_dirs"]
    base_logdir = config["logdir"]
    if "runname" in config:
        runname = config["runname"]
    else:
        runname = "."

    if "expname" in overrides_dict:
        expname = overrides_dict["expname"]
    if "runname" in overrides_dict:
        runname = overrides_dict["runname"]
    if "data_dirs" in overrides_dict:
        data_dir = overrides_dict["data_dirs"]
    if "logdir" in overrides_dict:
        base_logdir = overrides_dict["logdir"]

    overrides_dict["data_dirs"] = f"{data_dir}/{expname}/{timestep}"

    expname = f"{expname}/{runname}"

    ckpt_base_logdir = f"{base_logdir}/{expname}/output"

    if "angle" in overrides_dict:
        name2 = overrides_dict["expname2"]
        overrides_dict["data_dirs"] = f"{data_dir}/{name2}/{timestep}"
        base_logdir = f'{base_logdir}/{overrides_dict["angle"]}'
        overrides_dict["logdir"] = base_logdir
        overrides_dict["test_split"] = f"test_angle_{overrides_dict['angle']}"

    config.update(overrides_dict)

    base_logdir = f"{base_logdir}/{expname}/output"
    render_dir = f"{base_logdir}/render/phase"
    os.makedirs(f"{render_dir}", exist_ok=True)

    frame_dir = f"{render_dir}/{low:.1f}_{high:.1f}_x{alpha:.1f}"
    if args.ideal:
        frame_dir = f"{frame_dir}_ideal"
    if args.suppress_others:
        frame_dir = f"{frame_dir}_supp"
    if args.flt_thrs > 0.0:
        frame_dir = f"{frame_dir}_thrs_{args.flt_thrs}"

    frame_dir = f"{frame_dir}/{args.view_num:03d}"

    os.makedirs(f"{frame_dir}", exist_ok=True)

    for f in os.listdir(frame_dir):
        if f.endswith(".png"):
            os.remove(f"{frame_dir}/{f}")

    model_type = "static"
    validate_only = args.validate_only
    render_only = args.render_only
    spacetime_only = args.spacetime_only
    if validate_only and render_only:
        raise ValueError("render_only and validate_only are mutually exclusive.")
    if render_only and spacetime_only:
        raise ValueError("render_only and spacetime_only are mutually exclusive.")
    if validate_only and spacetime_only:
        raise ValueError("validate_only and spacetime_only are mutually exclusive.")

    if "bbox" in config and isinstance(config["bbox"], str):
        config.update({"bbox": list(map(float, config["bbox"].split(",")))})
    if isinstance(config["num_samples"], str):
        config.update({"num_samples": int(config["num_samples"])})

    pprint.pprint(config)

    data = load_data(
        model_type,
        validate_only=validate_only,
        render_only=render_only or spacetime_only,
        **config,
    )
    config.update(data)
    trainer = init_trainer(model_type, **config)

    dset = trainer.test_dataset

    field_feature_imgs = {}
    field_feature_imgs2 = {}

    for i in range(start_idx, num_timesteps + start_idx):
        timestep = f"{i:03d}"
        checkpoint_path = os.path.join(ckpt_base_logdir, f"{timestep}", "model.pth")
        trainer = init_trainer(model_type, **config)
        trainer.load_model(torch.load(checkpoint_path), training_needed=False)

        for n, p in trainer.model.field.named_parameters():
            if "delta" not in n and len(p.shape) == 4:
                field_feature_imgs2.setdefault(n, []).append(p.data.clone().cpu())

        for n, p in trainer.model.field.named_parameters():
            if "delta_grids" in n:
                n2 = n.replace("delta_", "")
                x = field_feature_imgs2[n2][0]
                field_feature_imgs.setdefault(n, []).append(p.data.clone().cpu() + x)

    trainer.model.field.use_delta_features = True
    trainer.model.field.delta_add_back = False

    new_field_feature_imgs = {}
    for n, f_im in field_feature_imgs.items():
        if low < 0 and high < 0:
            new_field_feature_imgs[n] = f_im
        else:
            print(f"Processing {n}")
            processed = process_feature_images(f_im, alpha, low, high, fs, device, args)
            new_field_feature_imgs[n] = processed
    vis_grid_features(new_field_feature_imgs, frame_dir)

    for i in range(num_frames):
        for n, p in trainer.model.field.named_parameters():
            if "delta_grids" in n:
                new_data = new_field_feature_imgs[n][i % num_timesteps].data
                p.data = new_data.to(device)

        img_h, img_w = dset.img_h, dset.img_w

        view = args.view_num if args.view_num > -1 else i % len(dset.poses)
        ts_render = trainer.eval_step(dset[view])
        preds_rgb = (
            ts_render["rgb"]
            .reshape(img_h, img_w, 3)
            .cpu()
            .clamp(0, 1)
            .mul(255.0)
            .byte()
            .numpy()
        )
        imageio.imwrite(f"{frame_dir}/{i:03d}.jpg", preds_rgb)

    if args.save_video:
        subprocess.call(
            [
                "ffmpeg",
                "-framerate",
                f"{args.vid_fps}",
                "-i",
                f"{frame_dir}/%03d.jpg",
                "-vcodec",
                "libx264",
                "-y",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                f"{frame_dir}/../{args.view_num:03d}.mp4",
            ]
        )


if __name__ == "__main__":
    main()
