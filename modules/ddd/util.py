import mesh_to_sdf
import trimesh
from sklearn.neighbors import KDTree

from modules.common.util import *


def get_tetrahedras_grid(grid_resolution, offset_x=0.5, offset_y=0.5, offset_z=0.5,
                         scale_x=2.0, scale_y=2.0, scale_z=2.0):
    coords = torch.linspace(start=0, end=1, steps=grid_resolution)
    x, y, z = torch.meshgrid(torch.arange(grid_resolution), torch.arange(grid_resolution),
                             torch.arange(grid_resolution))
    x, y, z = x.reshape(-1), y.reshape(-1), z.reshape(-1)

    vertexes = torch.stack([(coords[x] - offset_x) * scale_x,
                            (coords[y] - offset_y) * scale_y,
                            (coords[z] - offset_z) * scale_z], dim=-1)

    # six tetrahedras vertexes divides unit cube
    ###
    # 0 4 1 3
    # 1 4 5 3
    # 4 7 5 3
    # 4 6 7 3
    # 2 6 4 3
    # 0 2 4 3
    ###
    x, y, z = torch.meshgrid(torch.arange(grid_resolution - 1), torch.arange(grid_resolution - 1),
                             torch.arange(grid_resolution - 1))
    x, y, z = x.reshape(-1), y.reshape(-1), z.reshape(-1)
    v0 = get_vertex_id(x, y, z, grid_res=grid_resolution)
    v1 = get_vertex_id(x, y + 1, z, grid_res=grid_resolution)
    v2 = get_vertex_id(x, y, z + 1, grid_res=grid_resolution)
    v3 = get_vertex_id(x, y + 1, z + 1, grid_res=grid_resolution)
    v4 = get_vertex_id(x + 1, y, z, grid_res=grid_resolution)
    v5 = get_vertex_id(x + 1, y + 1, z, grid_res=grid_resolution)
    v6 = get_vertex_id(x + 1, y, z + 1, grid_res=grid_resolution)
    v7 = get_vertex_id(x + 1, y + 1, z + 1, grid_res=grid_resolution)
    tetrahedras = torch.cat([
        torch.stack([v0, v4, v1, v3], dim=-1),
        torch.stack([v1, v4, v5, v3], dim=-1),
        torch.stack([v4, v7, v5, v3], dim=-1),
        torch.stack([v4, v6, v7, v3], dim=-1),
        torch.stack([v2, v6, v4, v3], dim=-1),
        torch.stack([v0, v2, v4, v3], dim=-1)
    ], dim=0)

    return vertexes, tetrahedras


def get_vertex_id(x, y, z, grid_res):
    return x * grid_res * grid_res + y * grid_res + z


def normalize_points(points):
    if len(points.shape) == 2:
        center = (points.max(0)[0] + points.min(0)[0]) / 2
        points -= center
        max_l = (torch.abs(points).max(0)[0]).max()
        points /= max_l
    else:
        center = (points.max(dim=1)[0] + points.min(dim=1)[0]) / 2
        points -= center
        max_l = (torch.abs(points).max(dim=1)[0]).max(dim=-1)[0]
        points /= max_l
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
    points = points[torch.tensor(normal_points).type_as(points).long()]
    return points


def voxelize_points3d(points, features, grid_res, mask=None):
    b, n_points, dim = features.shape
    points, features = points.view(b * n_points, 3), features.view(b * n_points, dim)
    # retrieve indexes inside grid
    indexes = torch.tensor_split(torch.floor((points + 1.0) / 2 * (grid_res - 1) + 0.5).long(), 3, dim=-1)
    indexes = [torch.arange(b).type_as(indexes[0]).view(b, 1).repeat(1, n_points).view(b * n_points)] + [
        index.squeeze(-1) for index in indexes]
    # setup grid
    grid = torch.zeros(b, grid_res, grid_res, grid_res, dim).type_as(features)
    denom = torch.zeros(b, grid_res, grid_res, grid_res, dim).type_as(features)

    if mask is not None:
        mask = mask.view(b * n_points)
        features = torch.where(mask.unsqueeze(-1), features, torch.zeros_like(features))
        feature_ones = torch.where(mask.unsqueeze(-1), torch.ones_like(features), torch.zeros_like(features))
    else:
        feature_ones = torch.ones_like(features)

    grid.index_put_(indexes, features, accumulate=True)
    denom.index_put_(indexes, feature_ones, accumulate=True)
    grid /= torch.clamp(torch.sqrt(denom), min=1.0)

    return grid


def devoxelize_points3d(points, grid, mask=None):
    b, grid_res, _, _, dim = grid.shape
    b, n_points, _ = points.shape
    points = points.view(b * n_points, 3)
    # retrieve values in grid scale
    grid_points = (points + 1.0) / 2 * (grid_res - 1)
    xval, yval, zval = torch.tensor_split(grid_points, 3, dim=-1)
    xval, yval, zval = xval.squeeze(-1), yval.squeeze(-1), zval.squeeze(-1)
    # retrieve indexes of voxel in grid
    x, y, z = torch.tensor_split(torch.clamp(torch.floor(grid_points), max=grid_res - 2).long(), 3, dim=-1)
    x, y, z = x.squeeze(-1), y.squeeze(-1), z.squeeze(-1)
    bb = torch.arange(b).type_as(x).view(b, 1).repeat(1, n_points).view(b * n_points)
    # apply trilinear interpolation for each point
    xd, yd, zd = (xval - x.type_as(xval)).unsqueeze(-1), (yval - y.type_as(yval)).unsqueeze(-1), (
            zval - z.type_as(zval)).unsqueeze(-1)
    first_plane = ((grid[bb, x, y, z] * zd + grid[bb, x, y, z + 1] * (1 - zd)) * yd
                   + (grid[bb, x, y + 1, z] * zd + grid[bb, x, y + 1, z + 1] * (1 - zd)) * (1 - yd))
    second_plane = ((grid[bb, x + 1, y, z] * zd + grid[bb, x + 1, y, z + 1] * (1 - zd)) * yd
                    + (grid[bb, x + 1, y + 1, z] * zd + grid[bb, x + 1, y + 1, z + 1] * (1 - zd)) * (1 - yd))
    features = first_plane * xd + second_plane * (1 - xd)
    features = features.view(b, n_points, dim)
    if mask is not None:
        features = torch.where(mask.unsqueeze(-1), features, torch.zeros_like(features))
    return features


def get_surface_tetrahedras(vertexes, tets, sdf, features):
    vertex_outside = sdf > 0
    tet_sdf = vertex_outside[tets.reshape(len(tets) * 4)].reshape(len(tets), 4).byte()
    tet_sum = torch.sum(tet_sdf, dim=-1)
    surface_tets = tets[(tet_sum > 0) & (tet_sum < 4)]

    tet_vertexes_msk = torch.zeros(len(vertexes)).type_as(vertexes).bool()
    tet_vertexes_msk[surface_tets.view(len(surface_tets) * 4).unique()] = True
    tet_vertexes_ids = torch.arange(len(vertexes)).type_as(tet_vertexes_msk).long()[tet_vertexes_msk]

    # we need to recalculate tetrahedras ids
    indicators = torch.cumsum(tet_vertexes_msk.long(), dim=0) - 1
    surface_tets = indicators[surface_tets.view(len(surface_tets) * 4)].view(len(surface_tets), 4)

    return vertexes[tet_vertexes_ids], surface_tets, sdf[tet_vertexes_ids], \
           features[tet_vertexes_ids], vertexes[~tet_vertexes_msk], \
           sdf[~tet_vertexes_msk]


def get_tetrahedras_edges(tets):
    n = torch.max(tets) + 1 if len(tets) > 0 else torch.ones(1).type_as(tets)
    codes = torch.cat([
        tets[:, 0] * n + tets[:, 1],
        tets[:, 1] * n + tets[:, 0],

        tets[:, 1] * n + tets[:, 2],
        tets[:, 2] * n + tets[:, 1],

        tets[:, 2] * n + tets[:, 0],
        tets[:, 0] * n + tets[:, 2],

        tets[:, 0] * n + tets[:, 3],
        tets[:, 3] * n + tets[:, 0],

        tets[:, 1] * n + tets[:, 3],
        tets[:, 3] * n + tets[:, 1],

        tets[:, 2] * n + tets[:, 3],
        tets[:, 3] * n + tets[:, 2]
    ], dim=0).unique()

    edges = torch.stack([codes.div(n, rounding_mode='trunc'), codes % n], dim=0)
    return edges


def get_mesh_edges(faces):
    n = torch.max(faces) + 1 if len(faces) > 0 else torch.ones(1).type_as(faces)
    codes = torch.cat([
        faces[:, 0] * n + faces[:, 1],
        faces[:, 1] * n + faces[:, 0],

        faces[:, 1] * n + faces[:, 2],
        faces[:, 2] * n + faces[:, 1],

        faces[:, 2] * n + faces[:, 0],
        faces[:, 0] * n + faces[:, 2]
    ], dim=0).unique()

    edges = torch.stack([codes.div(n, rounding_mode='trunc'), codes % n], dim=0)
    return edges


def calculate_gaussian_curvature(vertices, faces):
    if len(vertices) == 0 or len(faces) == 0:
        return torch.zeros_like(vertices)
    v1, v2, v3 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    angles = torch.zeros(len(vertices)).type_as(vertices)
    n_faces = len(faces)
    angles.index_put_((faces[:, 0],),
                      torch.arccos((torch.matmul((v2 - v1).unsqueeze(1), (v3 - v1).unsqueeze(2)).view(n_faces)
                                    / (torch.norm(v2 - v1, dim=-1) * torch.norm(v3 - v1, dim=-1))
                                    .clamp(min=1e-6)).clamp(min=-1 + 1e-6, max=1 - 1e-6)),
                      accumulate=True)
    angles.index_put_((faces[:, 1],),
                      torch.arccos((torch.matmul((v1 - v2).unsqueeze(1), (v3 - v2).unsqueeze(2)).view(n_faces)
                                    / (torch.norm(v1 - v2, dim=-1) * torch.norm(v3 - v2, dim=-1))
                                    .clamp(min=1e-6)).clamp(min=-1 + 1e-6, max=1 - 1e-6)),
                      accumulate=True)
    angles.index_put_((faces[:, 2],),
                      torch.arccos((torch.matmul((v1 - v3).unsqueeze(1), (v2 - v3).unsqueeze(2)).view(n_faces)
                                    / (torch.norm(v1 - v3, dim=-1) * torch.norm(v2 - v3, dim=-1))
                                    .clamp(min=1e-6)).clamp(min=-1 + 1e-6, max=1 - 1e-6)),
                      accumulate=True)

    return 2 * torch.pi - angles


def read_obj(in_file, with_normals=False):
    vertices = []
    faces = []
    normals = []
    face_normals = []
    with open(in_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.split(' ')
            if tokens[0] == 'v':
                vertices.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])
            elif with_normals and tokens[0] == 'vn':
                normals.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])
            elif tokens[0] == 'f':
                items = [v.split('/') for v in tokens[1:] if len(v.replace(' ', '')) > 0]
                if len(items) < 3:
                    continue
                vertex_ids = [int(v[0]) for v in items]
                triangles = [[vertex_ids[0], vertex_ids[i], vertex_ids[i + 1]] for i in range(1, len(vertex_ids) - 1)]
                faces.extend(triangles)
                if with_normals:
                    normal_ids = [int(v[2]) for v in items if len(v) > 2]
                    tri_normals = [[normal_ids[0], normal_ids[i], normal_ids[i + 1]] for i in
                                   range(1, len(normal_ids) - 1)]
                    face_normals.extend(tri_normals)

    vertices = torch.tensor(vertices, dtype=torch.float)
    faces = torch.tensor(faces, dtype=torch.long) - 1
    if with_normals:
        normals = torch.tensor(normals, dtype=torch.float)
        face_normals = torch.tensor(face_normals, dtype=torch.long) - 1
        mean_normals = (normals[face_normals[:, 0]] + normals[face_normals[:, 1]] + normals[face_normals[:, 2]]) / 3
        return vertices, faces, mean_normals
    return vertices, faces


def tetrahedras2mesh(vertices, tetrahedras):
    # we must create 4 triangles for each tetrahedron

    faces_1 = torch.stack([tetrahedras[:, 0], tetrahedras[:, 1], tetrahedras[:, 2]], dim=-1)
    faces_2 = torch.stack([tetrahedras[:, 0], tetrahedras[:, 1], tetrahedras[:, 3]], dim=-1)
    faces_3 = torch.stack([tetrahedras[:, 1], tetrahedras[:, 2], tetrahedras[:, 3]], dim=-1)
    faces_4 = torch.stack([tetrahedras[:, 2], tetrahedras[:, 0], tetrahedras[:, 3]], dim=-1)
    faces = torch.cat([faces_1, faces_2, faces_3, faces_4], dim=0)
    return vertices, faces


def encode_3d_features2sequence(points, features, grid_res, seq_len):
    points = (points + 1.0) / 2
    indexes = torch.floor(points * (grid_res - 1) + 0.5).long()
    hashes = (indexes[:, :, 0] * 31 + indexes[:, :, 1] * 31 ** 2 + indexes[:, :, 2] * 31 ** 3) % seq_len

    b, n_points, dim = features.shape
    __features = torch.zeros(b, seq_len, dim).type_as(features)
    __denom = torch.zeros(b, seq_len, dim).type_as(features)

    batch_index = torch.arange(b).type_as(hashes).view(b, 1).repeat(1, n_points).view(-1)
    __features.index_put((batch_index, hashes.view(-1)), features, accumulate=True)
    __denom.index_put((batch_index, hashes.view(-1)), torch.ones_like(features), accumulate=True)
    __features /= torch.clamp(__denom ** 0.5, min=1.0)

    return __features


def calculate_sdf(points, vertices, faces, scan_count=10, scan_resolution=128, true_sdf=None):
    if true_sdf is not None:
        sdf = devoxelize_points3d(points, true_sdf.unsqueeze(-1).unsqueeze(0)).view(1, -1)
        return sdf
    mesh = trimesh.Trimesh(vertices=tn(vertices[0]), faces=tn(faces))
    sdf = mesh_to_sdf.mesh_to_sdf(mesh, tn(points[0]),
                                  surface_point_method='scan', sign_method='depth',
                                  bounding_radius=None, scan_count=scan_count, scan_resolution=scan_resolution,
                                  sample_point_count=100000, normal_sample_count=15)
    sdf = torch.tensor(sdf).type_as(vertices).view(1, -1)
    return sdf
