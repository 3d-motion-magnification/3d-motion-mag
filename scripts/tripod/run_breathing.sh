for i in {000..030}; do
    python -u magnerf/train.py \
        --config-path magnerf/configs/breathing.py \
        --timestep $i
done