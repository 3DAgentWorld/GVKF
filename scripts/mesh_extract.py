import os

gpu = 0
iteration = 30000

# if mip 360
output_dir = "new_exps/exp_mip360/release-test"
current_scene = "bicycle"
dataset_dir = 'datasets/360_v2/bicycle'
images_i = 'images_4'


# if others

# for waymo
# output_dir = "new_exps/exp_own/release-test"
# current_scene = "waymo10061"
# dataset_dir = 'datasets/waymo_data/individual_files_training_segment-10061305430875486848_1080_000_1100_000_with_camera_labels/Q_concentric_5_20_1_False'
# images_i = 'images'

# for tant
# output_dir = "new_exps/exp_own/release-test"
# current_scene = "Barn"
# dataset_dir = 'datasets/tant/trainingset/Barn'
# images_i = 'images'

######### leave default

f_c = '6-6-6'
appearance_dim = 0
ratio = 1


cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu}" \
    f" python extract_mesh_neuralGS.py -m {output_dir}/{current_scene} --iteration {iteration}" \
    f" -s {dataset_dir} -i {images_i}" \
    f" --appearance_dim {appearance_dim} --ratio {ratio}  --f_c {f_c}"

print(cmd)
os.system(cmd)
            





