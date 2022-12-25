import mesh_to_sdf
import trimesh
from sklearn.neighbors import KDTree

from modules.common.util import *


def render_pixels(rgb, density, dists):
    # rgb (b spp 3)
    # density (b spp)
    # dists (b spp+1)
    b, spp = density.shape
    device = density.device
    weighted_intervals = (dists[:, 1:] - dists[:, :-1]) * density  # b spp
    cum_transmittance = torch.cat(
        [torch.ones(b, 1, device=device), torch.exp(-torch.cumsum(weighted_intervals, dim=1))[:, :-1]], dim=-1)  # b spp
    weights = cum_transmittance * (1 - torch.exp(-weighted_intervals))
    pixels = torch.sum(weights.unsqueeze(-1) * rgb, dim=1)  # b 3
    return pixels, weights, cum_transmittance


def sample_dists(near, far, spp):
    device = near.device
    b = near.shape[0]

    dist = near.unsqueeze(1) + (far - near).unsqueeze(1) \
           * torch.linspace(start=0, end=1, steps=spp + 1, device=device).unsqueeze(0)  # b spp+1
    dist += torch.rand(b, spp + 1, device=device) * ((far - near) / spp).unsqueeze(1)
    return dist


def dist_to_rays(ray_o, ray_d, dist):
    # (b 1 3) * (b spp+1 1) so it will copy d 3 times and ray_d, ray_o spp+1 times
    points = ray_o.unsquueze(1) + ray_d.unsqueeze(1) * dist.unsquueze(2)
    return points


def get_images_rays(h, w, focal, poses):
    y, x = torch.meshgrid(torch.arange(h).type_as(focal), torch.arange(w).type_as(focal), indexing='ij')
    x = (x + 0.5 - w / 2).reshape(1, h, w) / (focal.reshape(-1, 1, 1) * w)
    y = (y + 0.5 - h / 2).reshape(1, h, w) / (focal.reshape(-1, 1, 1) * h)
    dirs = torch.stack([x, -y, -torch.ones_like(x)], dim=-1)  # b h w 3
    ray_o = poses[:, :3, -1]  # b 3
    ray_d = torch.matmul(dirs.unsqueeze(-2), poses[:, None, None, :3, :3].transpose(-1, -2)).squeeze(-2)
    return ray_o, ray_d


def get_image_coords(h, w, focal_x, focal_y, device=torch.device('cpu')):
    # returns coordinates of each pixel in camera coordinate system
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    x = (x + 0.5 - w / 2) / (focal_x * w)
    y = (y + 0.5 - h / 2) / (focal_y * h)
    dirs = torch.stack([
        x, -y, -torch.ones((h, w), device=device)
    ], dim=-1)  # h w 3
    return dirs


def get_rays(poses, pixel_coords):
    ray_o = poses[:, :3, -1]  # b 3
    ray_d = torch.matmul(poses[:, :3, :3], pixel_coords[..., None]).squeeze(-1)  # b 3 3
    return ray_o, ray_d


def look_at_matrix(eye, target):
    fwd = target - eye
    side = torch.linalg.cross(fwd, torch.tensor([0, 1, 0]).type_as(eye).view(1, 3).repeat(len(fwd), 1))
    up = torch.linalg.cross(side, fwd)
    fwd, side, up = normalize_vector(fwd), normalize_vector(side), normalize_vector(up)
    side = torch.cat([side, -torch.matmul(side.unsqueeze(-2), eye.unsqueeze(-1)).squeeze(-1)], dim=-1)
    up = torch.cat([up, -torch.matmul(up.unsqueeze(-2), eye.unsqueeze(-1)).squeeze(-1)], dim=-1)
    fwd = torch.cat([-fwd, torch.matmul(fwd.unsqueeze(-2), eye.unsqueeze(-1)).squeeze(-1)], dim=-1)
    pad = torch.tensor([0, 0, 0, 1]).type_as(eye).view(1, 4).repeat(len(eye), 1)
    w2c = torch.stack([side, up, fwd, pad], dim=1)  # b 4 4
    return w2c


def projection_matrix(near, far, fov=90, degrees=True, aspect_ratio=1.0):
    # transform FOV of camera to rop and right coordinates of image plane
    if not isinstance(fov, torch.Tensor):
        fov = torch.ones_like(near) * fov
    if degrees:
        fov = fov / 180 * torch.pi
    h = torch.tan(fov / 2) * near
    t, r = h, h * aspect_ratio

    z, e = torch.zeros_like(near), torch.ones_like(near)
    row1 = torch.stack([near / r, z, z, z], dim=1)
    row2 = torch.stack([z, near / t, z, z], dim=1)
    row3 = torch.stack([z, z, -(near + far) / (far - near), -2 * far * near / (far - near)], dim=1)
    row4 = torch.stack([z, z, -e, z], dim=1)
    matrix = torch.stack([row1, row2, row3, row4], dim=1)
    return matrix


def get_random_poses(dist):
    n_poses = len(dist)
    target = torch.zeros(n_poses, 3).type_as(dist)
    eye = normalize_vector(torch.rand(n_poses, 3).type_as(dist) - 0.5) * dist.unsqueeze(-1)
    return torch.linalg.inv(look_at_matrix(eye, target))


def get_random_view(dist):
    n_poses = len(dist)
    target = torch.zeros(n_poses, 3).type_as(dist)
    eye = normalize_vector(torch.rand(n_poses, 3).type_as(dist) - 0.5) * dist.unsqueeze(-1)
    return look_at_matrix(eye, target)


def get_tetrahedras_grid(grid_resolution, offset_x=0.5, offset_y=0.5, offset_z=0.5,
                         scale_x=2.0, scale_y=2.0, scale_z=2.0, less=True):
    # define vertexes positions
    coords = torch.linspace(start=0, end=1, steps=grid_resolution)
    x, y, z = torch.meshgrid(torch.arange(grid_resolution), torch.arange(grid_resolution),
                             torch.arange(grid_resolution))
    x, y, z = x.reshape(-1), y.reshape(-1), z.reshape(-1)

    vertexes = torch.stack([(coords[x] - offset_x) * scale_x,
                            (coords[y] - offset_y) * scale_y,
                            (coords[z] - offset_z) * scale_z], dim=-1)
    # define vertexes for each cube
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

    if not less:
        # six tetrahedras subdivision
        ###
        # 0 4 1 3
        # 1 4 5 3
        # 4 7 5 3
        # 4 6 7 3
        # 2 6 4 3
        # 0 2 4 3
        ###
        tetrahedras = torch.cat([
            torch.stack([v0, v4, v1, v3], dim=-1),
            torch.stack([v1, v4, v5, v3], dim=-1),
            torch.stack([v4, v7, v5, v3], dim=-1),
            torch.stack([v4, v6, v7, v3], dim=-1),
            torch.stack([v2, v6, v4, v3], dim=-1),
            torch.stack([v0, v2, v4, v3], dim=-1)
        ], dim=0)
    else:
        # five tetrahedras subdivision
        ###
        # 0 4 1 2
        # 1 7 3 2
        # 4 2 6 7
        # 4 5 1 7
        # 4 1 3 7
        ###
        tetrahedras = torch.cat([
            torch.stack([v0, v4, v1, v2], dim=-1),
            torch.stack([v1, v7, v3, v2], dim=-1),
            torch.stack([v4, v2, v6, v7], dim=-1),
            torch.stack([v4, v5, v1, v7], dim=-1),
            torch.stack([v4, v1, v3, v7], dim=-1),
        ], dim=0)

    return vertexes, tetrahedras


def get_vertex_id(x, y, z, grid_res):
    return x * grid_res * grid_res + y * grid_res + z


def normalize_points(points):
    if len(points.shape) == 2:
        center = (points.max(0)[0] + points.min(0)[0]) / 2
        points = points - center
        max_l = (torch.abs(points).max(0)[0]).max()
        points = points / max_l
    else:
        center = (points.max(dim=1)[0] + points.min(dim=1)[0]) / 2
        points = points - center
        max_l = (torch.abs(points).max(dim=1)[0]).max(dim=-1)[0]
        points = points / max_l
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
    indexes = torch.tensor_split(torch.clamp(torch.floor((points + 1.0) / 2 * (grid_res - 1) + 0.5),
                                             min=0, max=grid_res - 1).long(), 3, dim=-1)
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

    grid = grid.index_put_(indexes, features, accumulate=True)
    denom = denom.index_put_(indexes, feature_ones, accumulate=True)
    grid = grid / torch.clamp(torch.sqrt(denom), min=1.0)

    return grid


def devoxelize_points3d(points, grid, mask=None):
    b_grid, grid_res, _, _, dim = grid.shape
    b, n_points, _ = points.shape
    points = points.view(b * n_points, 3)
    # retrieve values in grid scale
    grid_points = (points + 1.0) / 2 * (grid_res - 1)
    xval, yval, zval = torch.tensor_split(grid_points, 3, dim=-1)
    xval, yval, zval = xval.squeeze(-1), yval.squeeze(-1), zval.squeeze(-1)
    # retrieve indexes of voxel in grid
    x, y, z = torch.tensor_split(torch.clamp(torch.floor(grid_points), min=0, max=grid_res - 2).long(), 3, dim=-1)
    x, y, z = x.squeeze(-1), y.squeeze(-1), z.squeeze(-1)
    bb = torch.arange(b).type_as(x).view(b, 1).repeat(1, n_points).view(b * n_points)
    if b_grid < b:
        bb = torch.zeros_like(bb)
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


def get_only_surface_tetrahedras(tets, sdf):
    vertex_outside = sdf > 0
    tet_sdf = vertex_outside[tets.reshape(len(tets) * 4)].reshape(len(tets), 4).byte()
    tet_sum = torch.sum(tet_sdf, dim=-1)
    surface_tets = tets[(tet_sum > 0) & (tet_sum < 4)]
    return surface_tets


def get_surface_tetrahedras(vertexes, tets, sdf, features):
    vertex_outside = sdf > 0
    tet_sdf = vertex_outside[tets.reshape(len(tets) * 4)].reshape(len(tets), 4).byte()
    tet_sum = torch.sum(tet_sdf, dim=-1)
    surface_tets = tets[(tet_sum > 0) & (tet_sum < 4)]
    non_surface_tets = tets[(tet_sum == 0) | (tet_sum == 4)]

    tet_vertexes_msk = torch.zeros(len(vertexes)).type_as(vertexes).bool()
    tet_vertexes_msk[surface_tets.view(len(surface_tets) * 4).unique()] = True
    tet_vertexes_ids = torch.arange(len(vertexes)).type_as(surface_tets).long()[tet_vertexes_msk]

    # we need to recalculate tetrahedras ids
    indicators = torch.cumsum(tet_vertexes_msk.long(), dim=0) - 1
    surface_tets = indicators[surface_tets.view(len(surface_tets) * 4)].view(len(surface_tets), 4)

    indicators = torch.cumsum((~tet_vertexes_msk).long(), dim=0) - 1
    non_surface_tets = indicators[non_surface_tets.view(len(non_surface_tets) * 4)].view(len(non_surface_tets), 4)

    return vertexes[tet_vertexes_ids], surface_tets, sdf[tet_vertexes_ids], \
           features[tet_vertexes_ids], vertexes[~tet_vertexes_msk], non_surface_tets, \
           sdf[~tet_vertexes_msk]


def get_tetrahedras_edges(tets, unique=False):
    n = torch.max(tets) + 1 if len(tets) > 0 else torch.ones(1).type_as(tets)
    v0, v1, v2, v3 = tets[:, 0], tets[:, 1], tets[:, 2], tets[:, 3]
    if unique:
        codes = torch.cat([
            torch.minimum(v0, v1) * n + torch.maximum(v0, v1),
            torch.minimum(v0, v2) * n + torch.maximum(v0, v2),
            torch.minimum(v0, v3) * n + torch.maximum(v0, v3),
            torch.minimum(v1, v2) * n + torch.maximum(v1, v2),
            torch.minimum(v1, v3) * n + torch.maximum(v1, v3),
            torch.minimum(v2, v3) * n + torch.maximum(v2, v3)
        ], dim=0).unique()
    else:
        codes = torch.cat([
            v0 * n + v1,
            v1 * n + v0,

            v0 * n + v2,
            v2 * n + v0,

            v0 * n + v3,
            v3 * n + v0,

            v1 * n + v2,
            v2 * n + v1,

            v1 * n + v3,
            v3 * n + v1,

            v2 * n + v3,
            v3 * n + v2
        ], dim=0).unique()

    edges = torch.stack([codes.div(n, rounding_mode='trunc'), codes % n], dim=0)
    return edges


def get_mesh_edges(faces, unique=False):
    n = torch.max(faces) + 1 if len(faces) > 0 else torch.ones(1).type_as(faces)
    v0, v1, v2 = faces[:, 0], faces[:, 1], faces[:, 2]
    if unique:
        codes = torch.cat([
            torch.minimum(v0, v1) * n + torch.maximum(v0, v1),
            torch.minimum(v1, v2) * n + torch.maximum(v1, v2),
            torch.minimum(v0, v2) * n + torch.maximum(v0, v2),
        ], dim=0).unique()
    else:
        codes = torch.cat([
            v0 * n + v1,
            v1 * n + v0,

            v1 * n + v2,
            v2 * n + v1,

            v2 * n + v0,
            v0 * n + v2
        ], dim=0).unique()

    edges = torch.stack([codes.div(n, rounding_mode='trunc'), codes % n], dim=0)
    return edges


def calculate_gaussian_curvature(vertices, faces):
    if len(vertices) == 0 or len(faces) == 0:
        return torch.zeros_like(vertices)
    v1, v2, v3 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    angles = torch.zeros(len(vertices)).type_as(vertices)
    n_faces = len(faces)
    angles = angles.index_put_((faces[:, 0],),
                               torch.arccos((torch.matmul((v2 - v1).unsqueeze(1), (v3 - v1).unsqueeze(2)).view(n_faces)
                                             / (torch.norm(v2 - v1, dim=-1) * torch.norm(v3 - v1, dim=-1))
                                             .clamp(min=1e-6)).clamp(min=-1 + 1e-6, max=1 - 1e-6)),
                               accumulate=True)
    angles = angles.index_put_((faces[:, 1],),
                               torch.arccos((torch.matmul((v1 - v2).unsqueeze(1), (v3 - v2).unsqueeze(2)).view(n_faces)
                                             / (torch.norm(v1 - v2, dim=-1) * torch.norm(v3 - v2, dim=-1))
                                             .clamp(min=1e-6)).clamp(min=-1 + 1e-6, max=1 - 1e-6)),
                               accumulate=True)
    angles = angles.index_put_((faces[:, 2],),
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


def save_obj(path, vertices, faces):
    vertices, faces = tn(vertices).tolist(), tn(faces).tolist()
    lines = []
    for v in vertices:
        lines.append(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}')
    for f in faces:
        lines.append(f'f {f[0]} {f[1]} {f[2]}')

    with open(path, 'w') as file:
        file.write('\n'.join(lines))


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
    __features = __features.index_put_((batch_index, hashes.view(-1)), features, accumulate=True)
    __denom = __denom.index_put_((batch_index, hashes.view(-1)), torch.ones_like(features), accumulate=True)
    __features = __features / torch.clamp(__denom ** 0.5, min=1.0)

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


def face_normals(vertices, faces):
    v0 = vertices[0, faces[:, 1]] - vertices[0, faces[:, 0]]
    v1 = vertices[0, faces[:, 2]] - vertices[0, faces[:, 0]]
    face_normals = torch.cross(v0, v1, dim=1)

    face_normals_length = face_normals.norm(dim=1, keepdim=True)
    face_normals = face_normals / (face_normals_length + 1e-7)

    return face_normals


def laplace_smoothing(vertices, edges):
    values = torch.zeros_like(vertices)
    norms = torch.zeros_like(vertices)

    v0, v1 = edges[0], edges[1]
    neighbour_vertices = vertices[v1]
    values = values.index_put_((v0,), neighbour_vertices)
    norms = norms.index_put_((v0,), torch.ones_like(neighbour_vertices))
    values = values / torch.clamp(norms, min=1.0)
    return values


def delta_laplace_loss_tetrahedras(delta_vertexes, tetrahedras):
    edges = get_tetrahedras_edges(tetrahedras)
    values = laplace_smoothing(delta_vertexes, edges)
    loss = torch.mean(torch.sum((delta_vertexes - values) ** 2, dim=1), dim=0)
    return loss


def delta_laplace_loss_mesh(delta_vertices, faces):
    edges = get_mesh_edges(faces)
    values = laplace_smoothing(delta_vertices, edges)
    loss = torch.mean(torch.sum((delta_vertices - values) ** 2, dim=1), dim=0)
    return loss


def get_mesh_edges_with_extra_vertex(faces):
    n = torch.max(faces) + 1 if len(faces) > 0 else torch.ones(1).type_as(faces)
    v0, v1, v2 = faces[:, 0], faces[:, 1], faces[:, 2]
    codes = torch.cat([
        torch.minimum(v0, v1) * n * n + torch.maximum(v0, v1) * n + v2,
        torch.minimum(v1, v2) * n * n + torch.maximum(v1, v2) * n + v1,
        torch.minimum(v0, v2) * n * n + torch.maximum(v0, v2) * n + v0,
    ], dim=0).unique(sorted=True)

    edges = torch.stack([codes.div(n * n, rounding_mode='trunc'),
                         (codes % (n * n)).div(n, rounding_mode='trunc'), codes % n], dim=0)
    return edges


def smoothness_loss(vertices, faces):
    edges = get_mesh_edges_with_extra_vertex(faces).transpose(0, 1)
    indexes = torch.arange(len(edges) - 1).type_as(edges)[torch.all(torch.eq(edges[:-1, :2], edges[1:, :2]), dim=1)]
    indexes = torch.stack([indexes, indexes + 1], dim=1).view(-1)
    edges = edges[indexes].view(-1, 2, 3)  # two consequent edges it's same edge from different faces

    v0, v1 = vertices[edges[:, 0, 0]], vertices[edges[:, 0, 1]]
    v2, v3 = vertices[edges[:, 0, 2]], vertices[edges[:, 1, 2]]

    first_normal = torch.cross(v1 - v0, v2 - v0, dim=1)
    first_normal = first_normal / (first_normal.norm(dim=1, keepdim=True) + 1e-7)

    # change sign as we want to keep same orientation
    second_normal = - torch.cross(v1 - v0, v3 - v0, dim=1)
    second_normal = second_normal / (second_normal.norm(dim=1, keepdim=True) + 1e-7)

    cos = torch.abs(torch.sum(first_normal * second_normal, dim=-1))
    loss = torch.mean((1 - cos) ** 2)
    return loss


def sdf_sign_reg(sdf, edges):
    if sdf.numel() == 0 or edges.numel() == 0:
        return torch.tensor(0).type_as(sdf)
    edges = edges.transpose(0, 1)
    edges = edges[torch.sign(sdf[edges[:, 0]]) != torch.sign(sdf[edges[:, 1]])]
    if edges.numel() == 0:
        return torch.tensor(0).type_as(sdf)
    v0, v1 = edges[:, 0], edges[:, 1]
    loss = -torch.mean(torch.nan_to_num(torch.where(sdf[v0] > 0, torch.log(torch.sigmoid(sdf[v1]) + 1e-6),
                                                    torch.log(1 - torch.sigmoid(sdf[v1]) + 1e-6))
                                        + torch.where(sdf[v1] > 0, torch.log(torch.sigmoid(sdf[v0]) + 1e-6),
                                                      torch.log(1 - torch.sigmoid(sdf[v0])) + 1e-6),
                                        nan=0, posinf=0, neginf=0))
    return loss


def sdf_value_reg(sdf, edges, resolution):
    if sdf.numel() == 0 or edges.numel() == 0:
        return torch.tensor(0).type_as(sdf)
    v0, v1 = edges[0], edges[1]
    # sdf values must differ not bigger than resolution
    loss = torch.mean((torch.abs(sdf[v0] - sdf[v1]) - 1 / (2 * resolution)).clamp(min=0.0))
    return loss


def continuous_mesh_reg(vertices, faces, vertexes, tets):
    tet_ids = torch.arange(len(tets)).type_as(tets)
    n = torch.max(tets) + 1 if len(tets) > 0 else torch.ones(1).type_as(tets)
    m = torch.max(tet_ids) + 1 if len(tet_ids) > 0 else torch.ones(1).type_as(tets)

    tets = torch.sort(tets, dim=1)[0]
    v0, v1, v2, v3 = tets[:, 0], tets[:, 1], tets[:, 2], tets[:, 3]
    tet_faces = torch.cat([
        v0 * n * n * m + v1 * n * m + v2 * m + tet_ids,
        v0 * n * n * m + v1 * n * m + v3 * m + tet_ids,
        v1 * n * n * m + v2 * n * m + v3 * m + tet_ids,
        v0 * n * n * m + v2 * n * m + v3 * m + tet_ids,
    ], dim=0).unique(sorted=True)

    pair_mask = (tet_faces[:-1].div(m, rounding_mode='trunc')) == (tet_faces[1:].div(m, rounding_mode='trunc'))
    indexes = torch.arange(len(tet_faces) - 1).type_as(tet_faces)[pair_mask]
    indexes = torch.stack([indexes, indexes + 1], dim=1).view(-1)
    tet_faces = tet_faces[indexes].view(-1, 2) % m


def get_close_faces(point, vertices, faces, dist=0.1):
    close_vertices = torch.all(torch.abs(vertices - point.unsqueeze(0)) < dist, dim=-1)
    face_mask = close_vertices[faces[:, 0]] | close_vertices[faces[:, 1]] | close_vertices[faces[:, 2]]
    return faces[face_mask]


def viscosity_sdf_reg(sdf, h, eps=1e-2):
    grid3 = len(sdf)
    grid_res = int(grid3 ** (1 / 3) + 0.5)
    sdf = sdf.view(grid_res, grid_res, grid_res)

    pad = torch.ones(grid_res, grid_res).type_as(sdf)
    x_sdf = torch.cat([pad.unsqueeze(0), sdf, pad.unsqueeze(0)], dim=0)
    y_sdf = torch.cat([pad.unsqueeze(1), sdf, pad.unsqueeze(1)], dim=1)
    z_sdf = torch.cat([pad.unsqueeze(2), sdf, pad.unsqueeze(2)], dim=2)

    df = torch.stack([
        x_sdf[2:, :, :] - x_sdf[:-2, :, :],
        y_sdf[:, 2:, :] - y_sdf[:, :-2, :],
        z_sdf[:, :, 2:] - z_sdf[:, :, :-2]], dim=-1) / (2 * h)

    ddf = torch.stack([
        x_sdf[2:, :, :] + x_sdf[:-2, :, :] - 2 * x_sdf[1:-1, :, :],
        y_sdf[:, 2:, :] + y_sdf[:, :-2, :] - 2 * y_sdf[:, 1:-1, :],
        z_sdf[:, :, 2:] + z_sdf[:, :, :-2] - 2 * z_sdf[:, :, 1:-1]], dim=-1) / h ** 2

    loss = torch.mean(((torch.norm(df, dim=-1) - 1) * torch.sign(sdf) - eps * torch.sum(ddf, dim=-1)) ** 2)
    return loss


def coarea_sdf_reg(sdf, beta=1e-6):
    grid3 = len(sdf)
    grid_res = int(grid3 ** (1 / 3) + 0.5)
    sdf = sdf.view(grid_res, grid_res, grid_res)

    xx, yy, zz = torch.meshgrid(torch.arange(grid_res - 1).type_as(sdf).long(),
                                torch.arange(grid_res - 1).type_as(sdf).long(),
                                torch.arange(grid_res - 1).type_as(sdf).long())
    xx, yy, zz = xx.view(-1), yy.view(-1), zz.view(-1)

    # trilinear interpolation for center of the voxel
    f_mid = (sdf[xx, yy, zz] + sdf[xx, yy, zz + 1] + sdf[xx, yy + 1, zz] + sdf[xx, yy + 1, zz + 1] +
             sdf[xx + 1, yy, zz] + sdf[xx + 1, yy, zz + 1] + sdf[xx + 1, yy + 1, zz] + sdf[xx + 1, yy + 1, zz + 1]) / 8

    # gradient of trilinear interpolation at the center of the voxel
    df_mid = torch.stack([
        sdf[xx, yy, zz] + sdf[xx, yy, zz + 1] + sdf[xx, yy + 1, zz] + sdf[xx, yy + 1, zz + 1] +
        -sdf[xx + 1, yy, zz] - sdf[xx + 1, yy, zz + 1] - sdf[xx + 1, yy + 1, zz] - sdf[xx + 1, yy + 1, zz + 1],
        sdf[xx, yy, zz] + sdf[xx, yy, zz + 1] - sdf[xx, yy + 1, zz] - sdf[xx, yy + 1, zz + 1] +
        sdf[xx + 1, yy, zz] + sdf[xx + 1, yy, zz + 1] - sdf[xx + 1, yy + 1, zz] - sdf[xx + 1, yy + 1, zz + 1],
        sdf[xx, yy, zz] - sdf[xx, yy, zz + 1] + sdf[xx, yy + 1, zz] - sdf[xx, yy + 1, zz + 1] +
        sdf[xx + 1, yy, zz] - sdf[xx + 1, yy, zz + 1] + sdf[xx + 1, yy + 1, zz] - sdf[xx + 1, yy + 1, zz + 1]
    ], dim=-1) / 4.0

    loss = torch.mean(torch.exp(torch.abs(f_mid * (-1)) / beta) / (2 * beta) * torch.norm(df_mid, dim=-1))
    return loss
