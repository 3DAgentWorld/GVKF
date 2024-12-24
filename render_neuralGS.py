import torch
from scene import NeuralScene
import os
from os import makedirs
from gaussian_renderer import render_neuralGS
import random
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import NGaussianModel
import numpy as np
import torchvision
import sys
from PIL import Image

def save_img_f32(depthmap, pth):
  """Save an image (probably a depthmap) to disk as a float32 TIFF."""
  with open(pth, 'wb') as f:
    Image.fromarray(np.nan_to_num(depthmap).astype(np.float32)).save(f, 'TIFF')

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, kernel_size, scale_factor):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"test_preds_{scale_factor}")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"gt_{scale_factor}")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"depth_{scale_factor}")


    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering_all = render_neuralGS(view, gaussians, pipeline, background, kernel_size=kernel_size)["render"]
        rendering = rendering_all[:3, :, :]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        # depth_map = rendering_all[6, :, :]
        # save_img_f32(depth_map.cpu().numpy(),
        #              os.path.join(depth_path, 'depth_{0:05d}'.format(idx) + ".tiff"))


def render_trained_neuralGS(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train, skip_test):
    with torch.no_grad():
        # print(dataset.feat_dim)
        # import ipdb;ipdb.set_trace()
        gaussians = NGaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth,
                                   dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank,
                                   dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist,
                                   dataset.add_cov_dist,
                                   dataset.add_color_dist)
        scene = NeuralScene(dataset, gaussians, load_iteration=iteration, ply_path=None, shuffle=False)
        scale_factor = dataset.resolution
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        kernel_size = dataset.kernel_size

        # cams = scene.getTrainCameras()
        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline,
                       background, kernel_size, scale_factor=scale_factor)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline,
                       background, kernel_size, scale_factor=scale_factor)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Meshing script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    print("Rendering " + args.model_path)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

    render_trained_neuralGS(lp.extract(args), args.iteration, pp.extract(args), args.skip_train, args.skip_test)
