import argparse
import importlib.util
import logging
import os
import pprint
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from typing import List, Dict, Any
import tempfile

import numpy as np
import imageio


def get_freer_gpu():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_fname = os.path.join(tmpdir, "tmp")
        os.system(f'nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >"{tmp_fname}"')
        if os.path.isfile(tmp_fname):
            memory_available = [int(x.split()[2]) for x in open(tmp_fname, 'r').readlines()]
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
from magnerf.utils.create_rendering import render_to_path
from magnerf.utils.parse_args import parse_optfloat


def setup_logging(log_level=logging.INFO):
    handlers = [logging.StreamHandler(sys.stdout)]
    logging.basicConfig(level=log_level,
                        format='%(asctime)s|%(levelname)8s| %(message)s',
                        handlers=handlers,
                        force=True)


def load_data(model_type: str, data_downsample, data_dirs, validate_only: bool, render_only: bool, **kwargs):
    data_downsample = parse_optfloat(data_downsample, default_val=1.0)

    return static_trainer.load_data(
        data_downsample, data_dirs, validate_only=validate_only,
        render_only=render_only, **kwargs)


def init_trainer(model_type: str, **kwargs):
    from magnerf.runners import static_trainer
    return static_trainer.StaticTrainer(**kwargs)


def save_config(config):
    log_dir = os.path.join(config['logdir'], config['expname'])
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(log_dir, 'config.py'), 'wt') as out:
        out.write('config = ' + pprint.pformat(config))

    with open(os.path.join(log_dir, 'config.csv'), 'w') as f:
        for key in config.keys():
            f.write("%s\t%s\n" % (key, config[key]))


def vis_grid_features(trainer, base_logdir):
    new_field_feature_imgs = {}
    for n, p in trainer.model.field.named_parameters():
        if 'grids' in n and 'delta' not in n:
            new_field_feature_imgs[n] = p.data

    planes = new_field_feature_imgs['grids.0.0'].squeeze(0).cpu()
    out = planes[:3].permute(1, 2, 0)
    out = (out - out.min()) / (out.max() - out.min())
    out = np.uint8(out * 255)
    mask = np.zeros_like(out)
    h, w = mask.shape[0], mask.shape[1]
    mask[(h//4):-(h//4), (w//4):-(w//4)] = 1.0
    imageio.imwrite(f'{base_logdir}/xy.png', out)
    #imageio.imwrite(f'{base_logdir}/xy_m.png', out * mask)

    planes = new_field_feature_imgs['grids.0.1'].squeeze(0).cpu()
    out = planes[:3].permute(1, 2, 0)
    out = (out - out.min()) / (out.max() - out.min())
    out = np.uint8(out * 255)
    imageio.imwrite(f'{base_logdir}/xz.png', out)
    #imageio.imwrite(f'{base_logdir}/xz_m.png', out * mask)

    planes = new_field_feature_imgs['grids.0.2'].squeeze(0).cpu()
    out = planes[:3].permute(1, 2, 0)
    out = (out - out.min()) / (out.max() - out.min())
    out = np.uint8(out * 255)
    imageio.imwrite(f'{base_logdir}/yz.png', out)
    #imageio.imwrite(f'{base_logdir}/yz_m.png', out * mask)


def main():
    setup_logging()

    p = argparse.ArgumentParser(description="")

    p.add_argument('--render-only', action='store_true')
    p.add_argument('--validate-only', action='store_true')
    p.add_argument('--spacetime-only', action='store_true')
    p.add_argument('--config-path', type=str, required=True)
    p.add_argument('--log-dir', type=str, default=None)
    p.add_argument('--timestep', type=str, default="000")
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--c2f', action='store_true')
    p.add_argument('override', nargs=argparse.REMAINDER)

    args = p.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Import config
    spec = importlib.util.spec_from_file_location(os.path.basename(args.config_path), args.config_path)
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

    base_logdir = config["logdir"]
    expname = config["expname"]
    if "runname" in config:
        runname = config["runname"]
    else:
        runname = '.'
    data_dir = config["data_dirs"]

    if "expname" in overrides_dict:
        expname = overrides_dict["expname"]
    if "runname" in overrides_dict:
        runname = overrides_dict["runname"]
    if "data_dirs" in overrides_dict:
        data_dir = overrides_dict["data_dirs"]

    overrides_dict["data_dirs"] = f"{data_dir}/{expname}/{args.timestep}"

    expname = f"{expname}/{runname}"
    base_logdir = f"{base_logdir}/{expname}/output"
    overrides_dict["expname"] = f"{expname}/output/{args.timestep}"
    config.update(overrides_dict)

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

    if 'bbox' in config and isinstance(config["bbox"], str):
        config.update({'bbox': list(map(float, config["bbox"].split(',')))})
    if isinstance(config["num_samples"], str):
        config.update({'num_samples': int(config["num_samples"])})

    pprint.pprint(config)
    if not validate_only or render_only:
        save_config(config)

    data = load_data(model_type, validate_only=validate_only, render_only=render_only or spacetime_only, **config)

    config.update(data)
    config.update({'lr': float(config["lr"])})
    trainer = init_trainer(model_type, **config)

    finetune_iters = int(overrides_dict.get('finetune_iters', 10001))
    finetune_lr = config.get('finetune_lr', None)
    print(finetune_lr)
    if args.timestep != '000' and not render_only:
        config.update(
            {'num_steps': finetune_iters, 
             'save_every': finetune_iters-1,
             'valid_every': finetune_iters-1,
             }
        )
        if finetune_lr is not None:
            config.update({'lr': finetune_lr})

        trainer = init_trainer(model_type, **config)
        checkpoint_path = os.path.join(base_logdir, "000", "model.pth")

        ckpt = torch.load(checkpoint_path)
        trainer.model.load_state_dict(ckpt['model'])

        for n, p in trainer.model.proposal_networks.named_parameters():
            p.requires_grad = False
        for n, p in trainer.model.field.named_parameters():
            if not 'delta_grids' in n:
                p.requires_grad = False
        for n, p in trainer.model.named_parameters():
            print(f'{n} | {p.shape} | Trainable: {p.requires_grad}')

        trainer.model.field.use_delta_features = True

    if validate_only:
        checkpoint_path = os.path.join(base_logdir, f"{(int(args.timestep)):03d}", "model.pth")
        ckpt = torch.load(checkpoint_path)
        trainer.model.load_state_dict(ckpt['model'])
        trainer.validate()

    elif render_only:
        checkpoint_path = os.path.join(base_logdir, f"{(int(args.timestep)):03d}", "model.pth")
        ckpt = torch.load(checkpoint_path)
        trainer.model.load_state_dict(ckpt['model'])

        render_to_path(trainer, extra_name="")
        vis_grid_features(trainer, base_logdir)

    else:
        if args.timestep != '000' or not args.c2f:
            trainer.train()
        else:
            factors = [1, 1, 1, 1, 1]
            steps = [100, 500, 1000, 2000, 20000]
            for f, s in zip(factors, steps):
                print(f'Training at factor {f} for {s} steps')
                config.update({'data_downsample': int(f)})
                data = load_data(model_type, validate_only=validate_only, render_only=render_only or spacetime_only, **config)
                for k,v in data.items():
                    setattr(trainer, k, v)
                trainer.num_steps = s
                trainer.train()
                vis_grid_features(trainer, base_logdir)
            trainer.save_model()

if __name__ == "__main__":
    main()
