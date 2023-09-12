config = {
    "expname": "baby",
    "runname": "test",
    'dset_type': 'handheld',
    "logdir": "./logs/handheld",
    "device": "cuda:0",

    # Data settings
    "data_downsample": 1,
    "data_dirs": "./data/handheld",
    "hold_every": 25,
    "finetune_iters": 1000,

    "bbox": "-4.0,-5.0,1.0,5.0,4.5,8.0",
    "near_plane": 0.2,
    "far_plane": 100.0,
    "use_app_embedding": False,
    "opaque_background": False,
    "contract": False,
    "bg_color_aug": "random",
    "dynamic_start": 154,

    # Optimization settings
    "num_steps": 20_001,
    "batch_size": 8192,
    "eval_batch_size": 8192,
    "scheduler_type": "warmup_cosine",
    "optim_type": "adam",
    "lr": 2e-2,

    # Regularization
    'plane_tv_weight': 2e-4,
    'plane_tv_weight_proposal_net': 2e-4,
    'distortion_loss_weight': 1.0,
    'histogram_loss_weight': 1.0,
    "l1_proposal_net_weight": 0,
    'gridchange_loss_weight': 0.5,

    # Training settings
    "train_fp16": True,
    "save_every": 10_000,
    "valid_every": 10_000,
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
