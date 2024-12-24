import torch
from scene import NeuralScene
import os
from os import makedirs
from gaussian_renderer import regress_neuralGS
import random
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import NGaussianModel
import numpy as np
import trimesh
from tetranerf.utils.extension import cpp
from utils.tetmesh import marching_tetrahedra
from gaussian_renderer import generate_neural_gaussians
import sys


def voxel_downsample(points, voxel_size):
    """
    使用体素网格下采样的方法对点云进行下采样。

    参数:
    points (torch.Tensor): 形状为 (N, 3) 的点云张量。
    voxel_size (float): 体素的大小。

    返回:
    torch.Tensor: 下采样后的点云张量。
    """

    coords = (points / voxel_size).floor().int()


    unique_coords, indices = torch.unique(coords, return_inverse=True, dim=0)


    voxel_dict = {}
    for idx in range(len(indices)):
        voxel_idx = indices[idx].item()
        if voxel_idx not in voxel_dict:
            voxel_dict[voxel_idx] = points[idx]

    sampled_points = torch.stack(list(voxel_dict.values()))

    return sampled_points


@torch.no_grad()
def get_sdfs(points, views, gaussians, pipeline, background, kernel_size):
    # traversal cameras for max sdf
    max_sdfs = -10*torch.ones((points.shape[0]), dtype=torch.float32, device="cuda") 

    with torch.no_grad():
        for _, view in enumerate(tqdm(views, desc="traversal all cam for maximum sdf")):
            sdfs = regress_neuralGS(points, view, gaussians, pipeline, background, kernel_size=kernel_size)
            max_sdfs = torch.max(sdfs, max_sdfs)

    if torch.isnan(max_sdfs).any():
        import ipdb;
        print('find NAN, something wrong in cuda')
        ipdb.set_trace()
       
    return max_sdfs

@torch.no_grad()
def mt(model_path, name, iteration, views, gaussians, pipeline, background,
                                           kernel_size, f_c):
    
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), 'meshes')
                                   
    # import ipdb;ipdb.set_trace()
    if f_c is not None:
        render_path = render_path + f'_box{f_c}'
    print(render_path)
    makedirs(render_path, exist_ok=True)

    # generate tetra points here
    # pick one view for neural GS
    xyz, color, opacity, scaling_final, rot, neural_opacity, mask, rot_origin = \
        generate_neural_gaussians(views[0], gaussians, visible_mask=None)
    # xyz in first view
    # print(f'origin xyz num: {xyz.shape}')
    # xyz = voxel_downsample(xyz, voxel_size=0.05)
    points, points_scale = gaussians.get_tetra_points(xyz, scaling_final, rot_origin, opacity, f_c)

    print(f'all sample num: {points.shape}')
    cells = cpp.triangulate(points)
    # evaluate alpha
    sdf = get_sdfs(points, views, gaussians, pipeline, background, kernel_size)

    vertices = points.cuda()[None]
    tets = cells.cuda().long()

    print(vertices.shape, tets.shape, sdf.shape)

    # sdf = alpha_to_sdf_nonlinear(alpha)

    torch.cuda.empty_cache()
    verts_list, scale_list, faces_list, _ = marching_tetrahedra(vertices, tets, sdf[None], points_scale[None])
    torch.cuda.empty_cache()

    end_points, end_sdf = verts_list[0]
    end_scales = scale_list[0]

    faces = faces_list[0].cpu().numpy()
    points = (end_points[:, 0, :] + end_points[:, 1, :]) / 2.

    left_points = end_points[:, 0, :]
    right_points = end_points[:, 1, :]
    left_sdf = end_sdf[:, 0, :]
    right_sdf = end_sdf[:, 1, :]
    left_scale = end_scales[:, 0, 0]
    right_scale = end_scales[:, 1, 0]
    distance = torch.norm(left_points - right_points, dim=-1)
    scale = left_scale + right_scale

    for i in range(3): # borrowed from gof for smoother surface
        print(f"meshing {i}/3, the more the better")
        mid_points = (left_points + right_points) / 2
        sdf = get_sdfs(mid_points, views, gaussians, pipeline, background, kernel_size)[None]
        mid_sdf = sdf.squeeze().unsqueeze(-1)

        ind_low = ((mid_sdf < 0) & (left_sdf < 0)) | ((mid_sdf > 0) & (left_sdf > 0))

        left_sdf[ind_low] = mid_sdf[ind_low]
        right_sdf[~ind_low] = mid_sdf[~ind_low]
        left_points[ind_low.flatten()] = mid_points[ind_low.flatten()]
        right_points[~ind_low.flatten()] = mid_points[~ind_low.flatten()]

        points = (left_points + right_points) / 2
    
    mesh = trimesh.Trimesh(vertices=points.cpu().numpy(), faces=faces, process=False)

        # filter
    mask = (distance <= scale).cpu().numpy()
    face_mask = mask[faces].all(axis=1)
    mesh.update_vertices(mask)
    mesh.update_faces(face_mask)
    mesh.export(os.path.join(render_path, f"mesh_export.ply"))

        



def extract_mesh(dataset: ModelParams, iteration: int, pipeline: PipelineParams, f_c):
    with torch.no_grad():
        # print(dataset.feat_dim)
        # import ipdb;ipdb.set_trace()
        gaussians = NGaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth,
                                   dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank,
                                   dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist,
                                   dataset.add_cov_dist,
                                   dataset.add_color_dist)
        scene = NeuralScene(dataset, gaussians, load_iteration=iteration, ply_path=None, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        kernel_size = dataset.kernel_size

        cams = scene.getTrainCameras()
        mt(dataset.model_path, "test", iteration, cams, gaussians, pipeline,
                                               background, kernel_size, f_c)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Meshing script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--f_c", default=None, type=str)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    print("Rendering " + args.model_path)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
    # import ipdb;ipdb.set_trace()
    extract_mesh(lp.extract(args), args.iteration, pp.extract(args), args.f_c)
