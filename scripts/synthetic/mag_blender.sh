python -u magnerf/mag_render.py \
    --config-path magnerf/configs/blender.py \
    --alpha 50 \
    --low 1.5 \
    --high 4.5 \
    --fs 30 \
    --num_frames 120 \
    --render-only \
    --save_video \
    --start_idx 0 \
    expname=lego_3Hz_0_005 \
    data_downsample=1