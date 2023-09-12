python -u magnerf/mag_render.py \
        --config-path magnerf/configs/baby.py \
        --alpha 300 \
        --low 0.1 \
        --high 1.1 \
        --num_timesteps 60 \
        --fs 30 \
        --num_frames 360 \
        --render-only \
        --save_video \
        --vid_fps 30 \
        --ideal \
        --suppress_others \
        --flt_thrs 0.99 \
        runname=g0_5 \
        data_downsample=2