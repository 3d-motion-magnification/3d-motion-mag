python -u magnerf/mag_render.py \
        --config-path magnerf/configs/jackson.py \
        --alpha 50 \
        --low 0.1 \
        --high 1.1 \
        --num_timesteps 30 \
        --fs 30 \
        --num_frames 300 \
        --render-only \
        --save_video \
        --vid_fps 30 \
        --ideal \
        --suppress_others \
        --flt_thrs 0.5 \
        data_downsample=1