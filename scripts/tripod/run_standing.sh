for i in {000..015}; do
    python -u magnerf/train.py \
        --config-path magnerf/configs/standing.py \
        --timestep $i
done