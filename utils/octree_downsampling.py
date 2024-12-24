import numpy as np
import torch


def get_bounding_box(points):
    """
    计算点云在三个轴上的最大值和最小值。

    参数:
    points: Tensor, 形状为 (N, 3)，表示点云

    返回:
    xmin, xmax, ymin, ymax, zmin, zmax: 分别表示在三个轴上的最小值和最大值
    """
    xmin = np.min(points[:, 0])
    xmax = np.max(points[:, 0])
    ymin = np.min(points[:, 1])
    ymax = np.max(points[:, 1])
    zmin = np.min(points[:, 2])
    zmax = np.max(points[:, 2])

    return [xmin, xmax, ymin, ymax, zmin, zmax]


class OctreeNode:
    def __init__(self, boundary, depth=0):
        self.boundary = boundary  # Node boundary [xmin, xmax, ymin, ymax, zmin, zmax]
        self.depth = depth
        self.points = []  # This will now store tuples of (point, index)
        self.children = []

    def insert(self, point, index):
        if not self._in_bounds(point):
            return False

        if len(self.points) < 1 or self.depth >= MAX_DEPTH:
            self.points.append((point, index))
            return True

        if not self.children:
            self._subdivide()

        return any(child.insert(point, index) for child in self.children)

    def _in_bounds(self, point):
        xmin, xmax, ymin, ymax, zmin, zmax = self.boundary
        x, y, z = point
        return (xmin <= x <= xmax) and (ymin <= y <= ymax) and (zmin <= z <= zmax)

    def _subdivide(self):
        xmin, xmax, ymin, ymax, zmin, zmax = self.boundary
        xmid = (xmin + xmax) / 2
        ymid = (ymin + ymax) / 2
        zmid = (zmin + zmax) / 2

        self.children = [
            OctreeNode([xmin, xmid, ymin, ymid, zmin, zmid], self.depth + 1),
            OctreeNode([xmin, xmid, ymin, ymid, zmid, zmax], self.depth + 1),
            OctreeNode([xmin, xmid, ymid, ymax, zmin, zmid], self.depth + 1),
            OctreeNode([xmin, xmid, ymid, ymax, zmid, zmax], self.depth + 1),
            OctreeNode([xmid, xmax, ymin, ymid, zmin, zmid], self.depth + 1),
            OctreeNode([xmid, xmax, ymin, ymid, zmid, zmax], self.depth + 1),
            OctreeNode([xmid, xmax, ymid, ymax, zmin, zmid], self.depth + 1),
            OctreeNode([xmid, xmax, ymid, ymax, zmid, zmax], self.depth + 1),
        ]

    def collect_points(self, depth):
        # Collects original points and their indices from all nodes up to the specified depth
        if self.depth <= depth:
            for point, index in self.points:
                yield point, index
            if self.children:
                for child in self.children:
                    yield from child.collect_points(depth)


def create_octree(points, boundary, max_depth):
    global MAX_DEPTH
    MAX_DEPTH = max_depth
    root = OctreeNode(boundary)
    for idx, point in enumerate(points):
        root.insert(point, idx)
    return root


def sample_indices(anchors, depth):
    points = anchors.detach().cpu().numpy()
    boundary = get_bounding_box(points)
    octree = create_octree(points, boundary, max_depth=9)
    downsampled_points, indices = zip(*list(octree.collect_points(depth=depth)))
    indices_array = np.array(indices)
    return downsampled_points, indices_array


# # Example usage
# points = np.random.rand(100, 3) * 10  # 100 random 3D points
# boundary = [0, 10, 0, 10, 0, 10]  # Space boundary
# octree = create_octree(points, boundary, max_depth=9)
#
# # Collect points from a specified depth (for downsampling)
# downsampled_points, indices = zip(*list(octree.collect_points(depth=3)))
# indices_array = np.array(indices)
# print("Downsampled Points:\n", len(downsampled_points))
# print("Indices Array:\n", indices_array)
