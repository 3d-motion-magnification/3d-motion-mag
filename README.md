# 3D Motion Magnification: Visualizing Subtle Motions with Time-Varying Neural Fields

This repo contains implementation of the method described in the paper, with demonstrations on both synthetic _multi-view_ data generated in Blender and _handheld monocular_ video captured in the real world.

[Project page](https://3d-motion-magnification.github.io/)

## Preparing environment

```
conda create -n magnerf python=3.11
conda activate magnerf
conda install pytorch torchvision cudatoolkit=11.8 -c pytorch -c nvidia

cd magnerf/datasets
git clone https://github.com/trueprice/pycolmap.git
cd ../..

git clone https://github.com/LabForComputationalVision/plenoptic.git
cd plenoptic
pip install -e .
```

## Blender multi-view scenes

Link to data: [Google Drive](https://drive.google.com/drive/folders/1d8lWyzSRb3KH5AagDWbOaw2BbCgxcJD-?usp=sharing). Blender multi-view scenes are under `data/synthetic`.


See `scripts/synthetic/run_blender.sh` for commands to begin training. 
Set appropriate `data_dirs` and `expname` (scene folder name) in the config file under `magnerf/configs`.

See `scripts/synthetic/mag_blender.sh` for commands to generate magnified rendering after training. The output will be saved under `logs/blender/$expname/output/render`.


## Handheld monocular video captures

Link to data: [Google Drive](https://drive.google.com/drive/folders/1d8lWyzSRb3KH5AagDWbOaw2BbCgxcJD-?usp=sharing). Handheld scenes are under `data/handheld`.


COLMAP is used to pre-process the data and generate camera poses.


See the example of `scripts/handheld/run_baby.sh` for commands to begin training. Set appropriate `data_dirs` and `expname` (scene folder name) in the config file under `magnerf/configs`.


See the example of `scripts/handheld/mag_baby.sh` for commands to generate magnified rendering after training. The output will be saved under `logs/handheld/baby/output/render`.


## Notes
- Unlike synthetic scenes generated Blender, handheld scenes are captured in the wild with varying scene layouts and compositions.
Therefore, when applying this method to your own scenes, it is important to set appropriate values for typical NeRF-specific parameters like _bbox_, _near_plane_, _far_plane_ in the config files under `magnerf/configs`.
Magnification will not work well if the initial static NeRF reconstruction is not good.


- This method is developed on a machine with a NVIDIA A4000 GPU with 16GB memory and 32GB RAM.
When the number of time steps is large, the magnification procedure performed in _mag_render.py_ can become memory intensive, since the implementation loads feature planes from all time steps into RAM. If memory issues arise, consider reducing the feature plane resolution before NeRF training, reducing the overall number of time steps, or using a machine with more RAM.


## Citation

```
@inproceedings{feng2023motionmag,
    author    = {Feng, Brandon Y. and AlZayer, Hadi and Rubinstein, Michael and Freeman, William T. and Huang, Jia-Bin},
    title     = {Visualizing Subtle Motions with Time-Varying Neural Fields},
    booktitle = {International Conference on Computer Vision (ICCV)},
    year      = {2023},
}
```

---
The code base is adapted from [K-Planes: Explicit Radiance Fields in Space, Time, and Appearance](https://sarafridov.github.io/K-Planes) and redistributed under the [BSD 3-clause license](LICENSE).