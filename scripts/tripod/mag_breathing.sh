python -u magnerf/mag_render.py \
        --config-path magnerf/configs/breathing.py \
        --alpha 20 \
        --low 0.8 \
        --high 1.2 \
        --start_idx 1 \
        --num_timesteps 30 \
        --fs 30 \
        --num_frames 270 \
        --render-only \
        --save_video \
        --vid_fps 30 \
        --suppress_others \
        --ideal \
        --flt_thrs 0.5 \
        data_downsample=2
