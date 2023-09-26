config = {
    "expname": "standing",
    "runname": "",
    'dset_type': 'tripod',

    "logdir": "./logs/tripod",
    "device": "cuda:0",

    # Data settings
    "data_downsample": 1,
    "data_dirs": "./data/tripod",
    "hold_every": 25,

    "bbox": "-3.0,-3.0,2.65,4.0,2.5,6.5",

    "use_app_embedding": True,
    "near_plane": 2.5,
    "far_plane": 20.0,
    "opaque_background": False,
    "contract": False,
    "bg_color_aug": "random",
    "frame_offset": 1,

    # Optimization settings
    "num_steps": 30_001,
    "batch_size": 8192,
    "eval_batch_size": 8192,
    "scheduler_type": "warmup_cosine",
    "optim_type": "adam",
    "lr": 2e-2,

    # Regularization
    'plane_tv_weight': 2e-4,
    'plane_tv_weight_proposal_net': 2e-4,
    'distortion_loss_weight': 5.0,
    'histogram_loss_weight': 1.0,
    "l1_proposal_net_weight": 0,
    'gridchange_loss_weight': 1,

    # Training settings
    "train_fp16": True,
    "save_every": 10_000,
    "valid_every": 5_000,
    "save_outputs": True,

    # Raymarching settings
    "num_samples": 64,
    "single_jitter": False,
    # proposal sampling
    "num_proposal_samples": [256, 128],
    "num_proposal_iterations": 2,
    "use_same_proposal_network": False,
    "use_proposal_weight_anneal": True,
    "proposal_net_args_list": [
        {"resolution": [128, 128, 128], "num_input_coords": 3, "num_output_coords": 8},
        {"resolution": [256, 256, 256], "num_input_coords": 3, "num_output_coords": 8},
    ],

    # Model settings
    'appearance_embedding_dim': 8,
    'multiscale_res': [1],
    'density_activation': 'trunc_exp',
    'concat_features_across_scales': True,
    'linear_decoder': False,
    'grid_config': [{
        'grid_dimensions': 2,
        'input_coordinate_dim': 3,
        'output_coordinate_dim': 32,
        'resolution': [512, 512, 512]
    }],
}
