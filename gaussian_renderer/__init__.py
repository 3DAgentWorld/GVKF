#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.neural_gaussian_model import NGaussianModel
from utils.sh_utils import eval_sh
from einops import repeat



def generate_neural_gaussians(viewpoint_camera, pc: NGaussianModel, visible_mask):
    ## view frustum filtering for acceleration
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device=pc.get_anchor.device)

    feat = pc._anchor_feat[visible_mask]
    anchor = pc.get_anchor[visible_mask]

    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]
    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist  # direction normalized vector

    # process inf
    # inf_mask = torch.isinf(ob_view)
    # rows_with_inf = inf_mask.any(dim=1)
    # ob_view[rows_with_inf] = 0
    ## view-adaptive feature
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)  # Man, 3+1

        bank_weight = pc.get_featurebank_gskernel(cat_view).unsqueeze(dim=1)  # [Man, 1, 3]

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:, ::4, :1].repeat([1, 4, 1]) * bank_weight[:, :, :1] + \
               feat[:, ::2, :1].repeat([1, 2, 1]) * bank_weight[:, :, 1:2] + \
               feat[:, ::1, :1] * bank_weight[:, :, 2:]
        feat = feat.squeeze(dim=-1)  # [Man, c]

    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1)  # [Man, c+3+1]
    cat_local_view_wodist = torch.cat([feat, ob_view], dim=1)  # [Man, c+3]
    if pc.appearance_dim > 0:
        camera_indicies = torch.ones_like(cat_local_view[:, 0], dtype=torch.long,
                                          device=ob_dist.device) * viewpoint_camera.uid
        # camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * 10
        appearance = pc.get_appearance(camera_indicies)

    # get offset's opacity
    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_gskernel(cat_local_view)  # [Man, offset*1]
    else:
        neural_opacity = pc.get_opacity_gskernel(cat_local_view_wodist)
    # neural_opacity = pc.get_opacity_gskernel(feat)

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])  # Man*offset, 1
    # print(f'-- neural opacity shape {neural_opacity.shape}') # num xyz, -1,1

    mask = (neural_opacity > 0)

    mask = mask.view(-1)

    # import ipdb; ipdb.set_trace()
    # select opacity
    opacity = neural_opacity[mask]  # Man*offset,1 -->  Oxyz,1

    # get offset's color
    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            color = pc.get_color_gskernel(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = pc.get_color_gskernel(torch.cat([cat_local_view_wodist, appearance], dim=1))
    else:
        if pc.add_color_dist:
            color = pc.get_color_gskernel(cat_local_view)
        else:
            color = pc.get_color_gskernel(cat_local_view_wodist)
    color = color.reshape([anchor.shape[0] * pc.n_offsets, 3])  # [mask]

    # get offset's cov
    if pc.add_cov_dist:
        scale_rot = pc.get_cov_gskernel(cat_local_view)
    else:
        scale_rot = pc.get_cov_gskernel(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0] * pc.n_offsets, 7])  # [mask]

    # offsets
    offsets = grid_offsets.view([-1, 3])  # [mask]

    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)  # Man,6+3
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)',
                                   k=pc.n_offsets)  # Man,9 --> Man*offset,9 = xyz, 9
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)  # xyz,9 --> xyz,9+3+7+3
    masked = concatenated_all[mask]  # xyz,9373 --> Oxyz, 9373
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)

    # post-process cov # grid scaling(345) * gs scaling（gskernel）
    scaling = scaling_repeat[:, 3:] * torch.sigmoid(scale_rot[:, :3])  # * (1+torch.sigmoid(repeat_dist))
    rot_origin = scale_rot[:, 3:7]
    rot = pc.rotation_activation(rot_origin)

    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:, :3]  # offset * grid scaling (012)
    xyz = repeat_anchor + offsets

 
    return xyz, color, opacity, scaling, rot, neural_opacity, mask, rot_origin


def render_neuralGS(viewpoint_camera, pc: NGaussianModel, pipe, bg_color: torch.Tensor, kernel_size: float,
                    scaling_modifier=1.0,
                    override_color=None, subpixel_offset=None, visible_mask=None, retain_grad=False):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    # is_training = pc.get_color_gskernel.training
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device=pc.get_anchor.device)

    xyz, color, opacity, scaling, rot, neural_opacity, mask, _ = generate_neural_gaussians(viewpoint_camera, pc,
                                                                                        visible_mask)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    # shape: Oxyz,1
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if subpixel_offset is None:
        subpixel_offset = torch.zeros((int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 2),
                                      dtype=torch.float32, device="cuda")

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=kernel_size,
        subpixel_offset=subpixel_offset,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = xyz
    # means2D = screenspace_points
    # opacity = opacity

    # view2gaussian_precomp = None

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    if torch.isnan(xyz).any() or torch.isnan(color).any() or torch.isnan(opacity).any() or torch.isnan(
            scaling).any() or torch.isnan(rot).any():
        print('appear NAN before rasterizer')
        import ipdb;
        ipdb.set_trace()

    rendered_image, radii = rasterizer(
        means3D=xyz.cuda(),
        means2D=screenspace_points.cuda(),
        shs=None,
        colors_precomp=color.cuda(),
        opacities=opacity.cuda(),
        scales=scaling.cuda(),
        rotations=rot.cuda(),
        cov3D_precomp=None,
        view2gaussian_precomp=None)

    visb_f = radii > 0  # int32 = xyz.shape (float32)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": visb_f,
            "radii": None,
            "selection_mask": mask,
            "neural_opacity": neural_opacity,
            "scaling": scaling
            }



def regress_neuralGS(querys, viewpoint_camera, pc: NGaussianModel, pipe, bg_color: torch.Tensor, kernel_size: float,
                       scaling_modifier=1.0, override_color=None, subpixel_offset=None, visible_mask=None,
                       retain_grad=False):

    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device=pc.get_anchor.device)
    xyz, color, opacity, scaling, rot, neural_opacity, mask, _ = generate_neural_gaussians(viewpoint_camera, pc,
                                                                                        visible_mask)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if subpixel_offset is None:
        subpixel_offset = torch.zeros((int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 2),
                                      dtype=torch.float32, device="cuda")

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=kernel_size,
        subpixel_offset=subpixel_offset,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    sdf_regress = rasterizer.regress_sdf(
        querys=querys,
        means3D=xyz,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=color,
        opacities=opacity,
        scales=scaling,
        rotations=rot,
        cov3D_precomp=None,
        view2gaussian_precomp=None)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return sdf_regress


def prefilter_voxel(viewpoint_camera, pc: NGaussianModel, not_pre_filter, pipe, bg_color: torch.Tensor, kernel_size: float,
                    scaling_modifier=1.0,
                    override_color=None, subpixel_offset=None):
    """
    Render the scene.
    Background tensor (bg_color) must be on GPU!
    """
    if not_pre_filter: 
        return torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device=pc.get_anchor.device)
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # todo: author dimission, we will release following part soon; mayhave bugs if running
    screenspace_points = torch.zeros_like(pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True,
                                          device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if subpixel_offset is None:
        subpixel_offset = torch.zeros((int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 2),
                                      dtype=torch.float32, device="cuda")

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=kernel_size,
        subpixel_offset=subpixel_offset,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_anchor

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    scales = pc.get_scaling

    rots = torch.zeros((means3D.shape[0], 4), device="cuda")
    rots[:, 0] = 1
    rotations = pc.rotation_activation(rots)

    # rotations = pc.get_rotation

    radii_pure = rasterizer.visible_filter(means3D=means3D,
                                           scales=scales[:, :3],
                                           rotations=rotations,
                                           cov3D_precomp=cov3D_precomp)

    return radii_pure > 0
