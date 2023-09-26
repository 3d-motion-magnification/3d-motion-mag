python -u magnerf/mag_render.py \
        --config-path magnerf/configs/standing.py \
        --alpha 50 \
        --low 0.5 \
        --high 1.5 \
        --num_timesteps 15 \
        --fs 15 \
        --num_frames 150 \
        --render-only \
        --save_video \
        --vid_fps 30 \
        --ideal \
        --suppress_others \
        --flt_thrs 0.1 \
        data_downsample=2
