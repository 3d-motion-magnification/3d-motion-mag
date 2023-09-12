for i in {000..060}; do
    python -u magnerf/train.py \
        --config-path magnerf/configs/baby.py \
        --timestep $i
done