python -u magnerf/mag_render.py \
        --config-path magnerf/configs/ronaldo.py \
        --alpha 75 \
        --low 0.9 \
        --high 1.1 \
        --num_timesteps 30 \
        --fs 30 \
        --num_frames 360 \
        --render-only \
        --save_video \
        --vid_fps 30 \
        --ideal \
        --suppress_others \
        --flt_thrs 0.5 \
        data_downsample=1
