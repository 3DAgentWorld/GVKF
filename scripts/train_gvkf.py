import os


gpu=0
output_dir = "new_exps/exp_own/release-test"

# modify your datasets, for example:
# for waymo, like this
current_scene = "waymo10061"
dataset_dir='datasets/waymo_data/individual_files_training_segment-10061305430875486848_1080_000_1100_000_with_camera_labels/Q_concentric_5_20_1_False'

# for tant, like this
# current_scene = "Barn"
# dataset_dir = 'datasets/tant/trainingset/Barn'

######### leave default

voxel_size = 0.01 
update_init_factor = 16
appearance_dim = 0
ratio = 1

cmd = f"CUDA_LAUNCH_BLOCKING=1 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu}" \
    f" python train_neuralGS.py -s {dataset_dir} -m {output_dir}/{current_scene}" \
    f" -i images --port {6109+int(gpu)}" \
    f" --voxel_size {voxel_size} --update_init_factor {update_init_factor}" \
    f" --appearance_dim {appearance_dim} --ratio {ratio}"

print(cmd)
os.system(cmd)




