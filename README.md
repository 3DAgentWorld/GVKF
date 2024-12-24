## GVKF: Gaussian Voxel Kernel Functions for Highly Efficient Surface Reconstruction in Open Scenes [NeurIPS 2024]

## 1. Pose Estimation
For custom datasets without camera poses, refer to [Colmap](https://colmap.github.io/). 

## 2. Environment
### 2.1 Conda
- basic image: pytorch1.12-py3.8-cuda11.3
- packages
```
conda create -n gvkf python=3.8
conda activate gvkf
pip install -r requirements.txt
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```
- Others
```
# To export complete sky of Mesh, use MT algorithm from GOF
cd submodules/tetra-triangulation
conda install cmake
conda install conda-forge::gmp
conda install conda-forge::cgal
# make sure valid nvcc path
cmake .
make 
pip install -e .
```
### 2.2 Docker

```
# Only test on Windows, you may try this but not sure 100% works
docker pull song21/gvkf:v_release
# start docker, windows for example:
docker run -it --name gvkf --gpus all -p 8030:22 -v D:\Docker_projs\codes\gvkf:/workspace/gvkf:rw song21/gvkf:v_release
# enter into docker container
docker exec -it gvkf /bin/bash
# In docker bash, run SSH server
service ssh start
# connect to docker via SSH
ssh -p 8030 root@127.0.0.1
# default password
password: 111111
# then, activate conda
conda activate gvkf
```

## 3. Datasets
We support datasets post-processed by COLMAP, meaning all of them should have the following format like:
```
./datasets/
|---dataset_name
|------images
|------sparse
|--------0
```
Refer to the website of [Mip-NeRF 360](https://jonbarron.info/mipnerf360/), [Tanks and Temples](https://www.tanksandtemples.org/download/), and [Waymo](https://waymo.com/open/download/) for downloading datasets. For Waymo, we utilize the poses provided by [COLMAP](https://colmap.github.io/); detailed usage instructions can be found in the documentation.

For faster and more accurate pose estimation, stay tuned for the release of our upcoming work.

## 4. Usage
### For training
-  `scripts/train_gvkf.py` : train dataset with COLMAP poses
-  `scripts/train_mip360.py` : COLMAP poses is provided for MipNeRF-360

### For extracting mesh
- `scripts/mesh_extract.py` : extract mesh from trained Gaussians

### For evaluating image
- `scripts/image_eval.py` : render image and evaluate PSNR/SSIM/LPIPS from trained GS

## 5. Mesh Visualization 

- We recommend using Blender for mesh visualization, and removing redundant meshes in "Edit mode". 

- If there are any requirements for importing/exporting camera from COLMAP (OpenCV) to blender (opengl), or setting mesh visualization results like our project page, please refer to [VisAnything](https://github.com/3DAgentWorld/VisAnything).

## Acknowledgements


- This work is built on [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [Scaffod-GS](https://github.com/city-super/Scaffold-GS), thanks for these great works.

- For better mesh results, some rendering implementation and MT meshing borrow from concurrent [GOF](https://github.com/autonomousvision/gaussian-opacity-fields/tree/main) in this version, thanks for this great work. Also, hoping our math analysis in the paper can help to understand this rendering formula better.

- Mesh visualization result takes from [VisAnything](https://github.com/3DAgentWorld/VisAnything), Gaussian visualization takes from [GS-Monitor](https://github.com/RongLiu-Leo/Gaussian-Splatting-Monitor), thanks for these great works.