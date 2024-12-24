import torch
# from pytorch3d.renderer import FoVPerspectiveCameras as P3DCameras
# from pytorch3d.renderer.cameras import _get_sfm_calibration_matrix
# from pytorch3d.transforms import quaternion_apply, quaternion_invert
import numpy as np
from utils.graphics_utils import fov2focal


def convert_camera_from_gs_to_pytorch3d(gs_cameras, device='cuda'):
    """
    From Gaussian Splatting camera parameters,
    computes R, T, K matrices and outputs pytorch3d-compatible camera object.

    Args:
        gs_cameras (List of GSCamera): List of Gaussian Splatting cameras.
        device (_type_, optional): _description_. Defaults to 'cuda'.

    Returns:
        p3d_cameras: pytorch3d-compatible camera object.
    """

    N = len(gs_cameras)

    R = torch.Tensor(np.array([gs_camera.R for gs_camera in gs_cameras])).to(device)
    T = torch.Tensor(np.array([gs_camera.T for gs_camera in gs_cameras])).to(device)
    fx = torch.Tensor(np.array([fov2focal(gs_camera.FoVx, gs_camera.image_width) for gs_camera in gs_cameras])).to(
        device)
    fy = torch.Tensor(np.array([fov2focal(gs_camera.FoVy, gs_camera.image_height) for gs_camera in gs_cameras])).to(
        device)
    image_height = torch.tensor(np.array([gs_camera.image_height for gs_camera in gs_cameras]), dtype=torch.int).to(
        device)
    image_width = torch.tensor(np.array([gs_camera.image_width for gs_camera in gs_cameras]), dtype=torch.int).to(
        device)
    cx = image_width / 2.  # torch.zeros_like(fx).to(device)
    cy = image_height / 2.  # torch.zeros_like(fy).to(device)

    w2c = torch.zeros(N, 4, 4).to(device)
    w2c[:, :3, :3] = R.transpose(-1, -2)
    w2c[:, :3, 3] = T
    w2c[:, 3, 3] = 1

    c2w = w2c.inverse()
    c2w[:, :3, 1:3] *= -1
    c2w = c2w[:, :3, :]

    distortion_params = torch.zeros(N, 6).to(device)
    camera_type = torch.ones(N, 1, dtype=torch.int32).to(device)

    # Pytorch3d-compatible camera matrices
    # Intrinsics
    image_size = torch.Tensor(
        [image_width[0], image_height[0]],
    )[
        None
    ].to(device)
    scale = image_size.min(dim=1, keepdim=True)[0] / 2.0
    c0 = image_size / 2.0
    p0_pytorch3d = (
            -(
                    torch.Tensor(
                        (cx[0], cy[0]),
                    )[
                        None
                    ].to(device)
                    - c0
            )
            / scale
    )
    focal_pytorch3d = (
            torch.Tensor([fx[0], fy[0]])[None].to(device) / scale
    )
    K = _get_sfm_calibration_matrix(
        1, "cpu", focal_pytorch3d, p0_pytorch3d, orthographic=False
    )
    K = K.expand(N, -1, -1)

    # Extrinsics
    line = torch.Tensor([[0.0, 0.0, 0.0, 1.0]]).to(device).expand(N, -1, -1)
    cam2world = torch.cat([c2w, line], dim=1)
    world2cam = cam2world.inverse()
    R, T = world2cam.split([3, 1], dim=-1)
    R = R[:, :3].transpose(1, 2) * torch.Tensor([-1.0, 1.0, -1]).to(device)
    T = T.squeeze(2)[:, :3] * torch.Tensor([-1.0, 1.0, -1]).to(device)

    p3d_cameras = P3DCameras(device=device, R=R, T=T, K=K, znear=0.0001)

    return p3d_cameras


def get_points_depth_in_depth_map(h, w, fov_camera, depth, points_in_camera_space):
    # the camera is p3d camera
    depth_view = depth.unsqueeze(0).unsqueeze(-1).permute(0, 3, 1, 2)
    pts_projections = fov_camera.get_projection_transform().transform_points(points_in_camera_space)

    factor = -1 * min(h, w)
    # todo: Parallelize these two lines with a tensor [image_width, image_height]
    pts_projections[..., 0] = factor / w * pts_projections[..., 0]
    pts_projections[..., 1] = factor / h * pts_projections[..., 1]
    pts_projections = pts_projections[..., :2].view(1, -1, 1, 2)

    map_z = torch.nn.functional.grid_sample(input=depth_view,
                                            grid=pts_projections,
                                            mode='bilinear',
                                            padding_mode='border'  # 'reflection', 'zeros'
                                            )[0, 0, :, 0]
    return map_z


def sample_points_efficient(xyz, n_points, scaling=0.1, mask=None):
    """
    高效地根据给定的参数采样点，并返回采样点在原始xyz中的索引。

    参数:
    xyz (torch.Tensor): 形状为(N, 3)的张量，表示空间中的点集。
    scaling (float): 用于控制正态分布的缩放。
    n_points (int): 想要获取的采样点总数。
    mask (torch.Tensor, optional): 一维的bool张量，用于遮盖一些xyz点。默认为None。

    返回:
    (torch.Tensor, torch.Tensor): 一个元组，包含采样得到的点集和这些点在原始xyz中的索引。
    """
    if mask is not None:
        # 应用mask过滤点和索引
        valid_indices = torch.arange(xyz.shape[0])[mask]
        filtered_xyz = xyz[mask]
    else:
        valid_indices = torch.arange(xyz.shape[0])
        filtered_xyz = xyz

    # 假设所有点的权重相等（简化示例），创建一个均匀的概率分布
    probabilities = torch.ones(filtered_xyz.shape[0]) / filtered_xyz.shape[0]
    cum_probs = probabilities.cumsum(dim=-1)

    # 基于累积概率分布进行加权采样，以获得随机索引
    random_indices = torch.multinomial(cum_probs, num_samples=n_points, replacement=True)
    final_indices = valid_indices[random_indices]

    # 为选中的点添加正态分布的随机偏移
    noise = torch.randn(final_indices.shape[0], 3) * scaling
    sampled_xyz = xyz[final_indices] + noise

    return sampled_xyz, final_indices


def cal_sdf_basedon_depth(h, w, fov_camera, sdf_samples, depth):
    sdf_samples_in_camera_space = fov_camera.get_world_to_view_transform().transform_points(sdf_samples)
    sdf_samples_z = sdf_samples_in_camera_space[..., 2] + 0.  # 点到相机的距离
    proj_mask = sdf_samples_z > fov_camera.znear
    sdf_samples_map_z = get_points_depth_in_depth_map(h, w, fov_camera, depth, sdf_samples_in_camera_space[proj_mask])
    sdf_estimation = sdf_samples_map_z - sdf_samples_z[proj_mask]  # 点深度-点到相机距离
    return sdf_estimation

def weighted_sdf_loss(sdf_estimate, gt_sdf, epsilon=0.1):
    # 为接近表面的预测赋予更高的权重
    weights = torch.exp(-torch.abs(gt_sdf) / epsilon)
    loss = torch.mean(weights * (sdf_estimate - gt_sdf) ** 2)
    return loss

if __name__ == "__main__":
    # 生成示例数据
    N = 10  # 总点数
    xyz = torch.randn(N, 3)  # 随机生成N个3D点
    scaling = 0.1  # 控制正态分布缩放的因子
    n_points = 4  # 想要采样的点数
    mask = torch.rand(N) > 0.1  # 随机生成一个mask，约一半的点可被采样

    # 调用采样函数
    print(xyz)
    print(mask)
    sampled_xyz, sampled_indices = sample_points_efficient(xyz, scaling, n_points, mask)
