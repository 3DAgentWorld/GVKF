import os

scene_ind=4
gpu=0
output_dir = "new_exps/exp_mip360/release-test"

######### leave default

scenes = ["bicycle", "bonsai", "counter", "flowers", "garden", "stump", "treehill", "kitchen", "room"]
factors = [4, 2, 2, 4, 4, 4, 4, 2, 2]

current_scene=scenes[scene_ind]
current_factor=factors[scene_ind]


voxel_size = 0.01 
update_init_factor = 16
appearance_dim = 0
ratio = 1

cmd = f"CUDA_LAUNCH_BLOCKING=1 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu}" \
    f" python train_neuralGS.py -s datasets/360_v2/{current_scene} -m {output_dir}/{current_scene}" \
    f" -i images_{current_factor} --port {6109+int(gpu)}" \
    f" --voxel_size {voxel_size} --update_init_factor {update_init_factor}" \
    f" --appearance_dim {appearance_dim} --ratio {ratio}"

print(cmd)
os.system(cmd)




