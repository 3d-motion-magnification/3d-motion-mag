python -u magnerf/mag_render.py \
        --config-path magnerf/configs/fork.py \
        --alpha 70 \
        --low 10.5 \
        --high 12.5 \
        --start_idx 1 \
        --num_timesteps 30 \
        --fs 30 \
        --num_frames 120 \
        --render-only \
        --save_video \
        --vid_fps 30 \
        --ideal \
        data_downsample=2
