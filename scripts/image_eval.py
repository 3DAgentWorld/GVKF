import os

gpu=0
if_render = True
if_metrics = True
iteration = 30000
scene_ind=0

scenes = ["bicycle", "bonsai", "counter", "flowers", "garden", "stump", "treehill", "kitchen", "room"]
factors = [4, 2, 2, 4, 4, 4, 4, 2, 2]

current_scene=scenes[scene_ind]
current_factor=factors[scene_ind]

dataset_dir = f'datasets/360_v2/{current_scene}'
output_dir = "new_exps/exp_mip360/release-test"
images_i = f'images_{current_factor}'


appearance_dim = 0
ratio = 1


if if_render:  # --skip_train
    cmd_r = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu}" \
            f" python render_neuralGS.py -m {output_dir}/{current_scene} --iteration {iteration}" \
            f" -s {dataset_dir} -i {images_i} --skip_train --eval" \
            f" --appearance_dim {appearance_dim} --ratio {ratio}"
    print(cmd_r)
    os.system(cmd_r)

if if_metrics:
    cmd_m = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python metrics.py -m {output_dir}/{current_scene}"
    print(cmd_m)
    os.system(cmd_m)