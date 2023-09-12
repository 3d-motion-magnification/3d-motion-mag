for i in {000..030}; do
    python -u magnerf/train.py \
        --config-path magnerf/configs/jackson.py \
        --timestep $i
done