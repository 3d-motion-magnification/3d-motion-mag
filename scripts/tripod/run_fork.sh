for i in {000..030}; do
    python -u magnerf/train.py \
        --config-path magnerf/configs/fork.py \
        --timestep $i
done