import kaolin
import torch
from sklearn.neighbors import KDTree


# Laplacian regularization using umbrella operator (Fujiwara / Desbrun).
# https://mgarland.org/class/geom04/material/smoothing.pdf
def laplace_regularizer_const(mesh_verts, mesh_faces):
    term = torch.zeros_like(mesh_verts)
    norm = torch.zeros_like(mesh_verts[..., 0:1])

    v0 = mesh_verts[mesh_faces[:, 0], :]
    v1 = mesh_verts[mesh_faces[:, 1], :]
    v2 = mesh_verts[mesh_faces[:, 2], :]

    #     d01 = torch.norm(v1-v0, dim=-1, keepdim=True).clamp(min=1e-3)
    #     d02 = torch.norm(v2-v0, dim=-1, keepdim=True).clamp(min=1e-3)
    #     d12 = torch.norm(v2-v1, dim=-1, keepdim=True).clamp(min=1e-3)
    #     term.scatter_add_(0, mesh_faces[:, 0:1].repeat(1,3), (v1 - v0) / d01  + (v2 - v0) / d02)
    #     term.scatter_add_(0, mesh_faces[:, 1:2].repeat(1,3), (v0 - v1) / d01 + (v2 - v1) / d12)
    #     term.scatter_add_(0, mesh_faces[:, 2:3].repeat(1,3), (v0 - v2) / d02 + (v1 - v2) / d12)

    term.scatter_add_(0, mesh_faces[:, 0:1].repeat(1, 3), (v1 - v0) + (v2 - v0))
    term.scatter_add_(0, mesh_faces[:, 1:2].repeat(1, 3), (v0 - v1) + (v2 - v1))
    term.scatter_add_(0, mesh_faces[:, 2:3].repeat(1, 3), (v0 - v2) + (v1 - v2))
    two = torch.ones_like(v0) * 2.0
    norm.scatter_add_(0, mesh_faces[:, 0:1], two)
    norm.scatter_add_(0, mesh_faces[:, 1:2], two)
    norm.scatter_add_(0, mesh_faces[:, 2:3], two)
    term = term / torch.clamp(norm, min=1.0)

    return torch.mean(term ** 2)


def loss_f(mesh_verts, mesh_faces, points, it, iterations, laplacian_weight):
    areas = kaolin.ops.mesh.face_areas(mesh_verts.unsqueeze(0), mesh_faces)[0]
    pred_points = kaolin.ops.mesh.sample_points(mesh_verts.unsqueeze(0), mesh_faces, len(points) // 2,
                                                areas=areas.unsqueeze(0))[0][0]
    chamfer = kaolin.metrics.pointcloud.chamfer_distance(pred_points.unsqueeze(0), points.unsqueeze(0)).mean()
    if it > iterations // 4:
        lap = laplace_regularizer_const(mesh_verts, mesh_faces)
        return chamfer + lap * laplacian_weight
    return chamfer


def get_tetrahedra_verts(grid_resolution):
    return None


def get_tetrahedras(grid_resolution):
    return None


def normalize_points(points):
    center = (points.max(0)[0] + points.min(0)[0]) / 2
    max_l = (points.max(0)[0] - points.min(0)[0]).max()
    points = ((points - center) / max_l)
    return points


def filter_pcd_outliers(points, neighbour_rate=1e-3):
    X = points.cpu().numpy()
    tree = KDTree(X)
    dist, ind = tree.query(X, k=int(len(X) * neighbour_rate))
    dist = dist[:, -1]
    ind = ind[:, -1]
    mean_dist, std_dist = dist.mean(), dist.std()
    threshold_dist = mean_dist + 3 * std_dist
    normal_points = ind[dist < threshold_dist]
    points = points[torch.tensor(normal_points).long().to(points.device)]
    return points
