for i in {000..029}; do
    python -u magnerf/train.py \
        --config-path magnerf/configs/blender.py \
        --timestep $i \
        expname=lego_3Hz_0_005
done