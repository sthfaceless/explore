from builtins import ModuleNotFoundError

import torch_geometric

from modules.common.model import *
from modules.ddd.util import *


def nonlinear(x):
    return torch.nn.functional.leaky_relu(x)


class PointVoxelCNN(torch.nn.Module):

    def __init__(self, input_dim, dim, grid_res=32, kernel_size=3, num_groups=32, do_points_map=True):
        super(PointVoxelCNN, self).__init__()

        self.input_dim = input_dim
        self.dim = dim
        self.grid_res = grid_res

        self.do_points_map = do_points_map
        if do_points_map:
            self.point_layer = torch.nn.Linear(input_dim, dim)

        self.norm1 = norm(input_dim, num_groups)
        self.conv1 = torch.nn.Conv3d(in_channels=input_dim, out_channels=dim, kernel_size=kernel_size,
                                     padding=kernel_size // 2)
        self.norm2 = norm(dim, num_groups)
        self.conv2 = torch.nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=kernel_size,
                                     padding=kernel_size // 2)
        if input_dim != dim:
            self.skip_conv = torch.nn.Conv3d(in_channels=input_dim, out_channels=dim, kernel_size=kernel_size,
                                             padding=kernel_size // 2)

    def voxelize(self, points, features):
        # map grid
        input_grid = voxelize_points3d(points, features, self.grid_res).movedim(-1, 1)
        grid = self.conv1(nonlinear(self.norm1(input_grid)))
        grid = self.conv2(nonlinear(self.norm2(grid)))
        skip = input_grid if self.input_dim == self.dim else self.skip_conv(input_grid)
        out_grid = (grid + skip).movedim(1, -1) / 2 ** 0.5
        return out_grid

    def devoxelize(self, points, grid):
        return devoxelize_points3d(points, grid)

    def map_points(self, features):
        if not self.do_points_map:
            raise ModuleNotFoundError('Initialized model without points mapping')
        h = self.point_layer(features)
        return h

    def forward(self, points, features):
        # points (b, 3)
        # features (b, input_dim)
        return (self.map_points(features) + self.devoxelize(points, self.voxelize(points, features))) / 2 ** 0.5

    def bvoxelize(self, points, features):
        return [self.voxelize(p.unsqueeze(0), f.unsqueeze(0)).squeeze(0) for p, f in zip(points, features)]

    def bdevoxelize(self, points, grid):
        return [self.devoxelize(p.unsqueeze(0), g.unsqueeze(0)).squeeze(0) for p, g in zip(points, grid)]

    def bforward(self, points, features):
        return [self.forward(p.unsqueeze(0), f.unsqueeze(0)).squeeze(0) for p, f in zip(points, features)]


class MultiPointVoxelCNN(torch.nn.Module):

    def __init__(self, input_dim, dims=(64, 128, 256), grids=(32, 16, 8), do_points_map=True,
                 kernel_size=3, num_groups=32):
        super(MultiPointVoxelCNN, self).__init__()
        self.models = torch.nn.ModuleList(
            [PointVoxelCNN(input_dim=input_dim, dim=dim, grid_res=res, do_points_map=do_points_map,
                           kernel_size=kernel_size, num_groups=num_groups) for dim, res in zip(dims, grids)])

    def voxelize(self, points, features):
        return [model.voxelize(points, features) for model in self.models]

    def devoxelize(self, points, grids):
        return torch.cat([model.devoxelize(points, grid) for model, grid in zip(self.models, grids)], dim=-1)

    def forward(self, points, features):
        return torch.cat([module(points, features) for module in self.models], dim=-1)

    def bvoxelize(self, points, features):
        return [[model.voxelize(p.unsqueeze(0), f.unsqueeze(0)).squeeze(0) for model in self.models]
                for p, f in zip(points, features)]

    def bdevoxelize(self, points, bgrids):
        return [torch.cat([model.devoxelize(p.unsqueeze(0), g.unsqueeze(0)).squeeze(0)
                           for g, model in zip(grids, self.models)], dim=-1) for p, grids in zip(points, bgrids)]

    def bforward(self, points, features):
        return [self.forward(p.unsqueeze(0), f.unsqueeze(0)).squeeze(0) for p, f in zip(points, features)]


class SDFDiscriminator(torch.nn.Module):

    def __init__(self, input_dim, hidden_dims=(32, 64, 128, 256), num_groups=16):
        super(SDFDiscriminator, self).__init__()
        self.input_conv = torch.nn.Conv3d(in_channels=input_dim, out_channels=hidden_dims[0], kernel_size=5)
        self.norm1 = norm(hidden_dims[0], num_groups)
        self.downsample_conv = torch.nn.Conv3d(in_channels=hidden_dims[0], out_channels=hidden_dims[1],
                                               stride=2, kernel_size=3)
        self.conv1 = torch.nn.Conv3d(in_channels=hidden_dims[1], out_channels=hidden_dims[2], kernel_size=3)
        self.conv2 = torch.nn.Conv3d(in_channels=hidden_dims[2], out_channels=hidden_dims[3], kernel_size=3)
        self.out_norm = norm(hidden_dims[-1], num_groups)
        self.out_layer = torch.nn.Linear(hidden_dims[-1], 1)

    def forward(self, x):
        h = self.input_conv(x)  # 16 -> 12
        h = self.downsample_conv(nonlinear(self.norm1(h)))  # 12 -> 5
        h = self.conv1(nonlinear(h))  # 5 -> 3
        h = self.conv2(nonlinear(h))  # 3 -> 1
        h = h.reshape(h.shape[0], -1)
        h = self.out_layer(nonlinear(self.out_norm(h)))
        out = torch.sigmoid(h)
        return out


class TetConv(torch.nn.Module):

    def __init__(self, input_dim, out_dim, gcn_dims=(256, 128), mlp_dims=(128, 64)):
        super(TetConv, self).__init__()
        self.gcn1 = torch.nn.Module()
        self.gcn1.input = torch.nn.Linear(input_dim, gcn_dims[0])
        self.gcn1.first = torch_geometric.nn.GCNConv(gcn_dims[0], gcn_dims[0])
        self.gcn1.second = torch_geometric.nn.GCNConv(gcn_dims[0], gcn_dims[0])
        self.gcn2 = torch.nn.Module()
        self.gcn2.input = torch.nn.Linear(gcn_dims[0], gcn_dims[1])
        self.gcn2.first = torch_geometric.nn.GCNConv(gcn_dims[1], gcn_dims[1])
        self.gcn2.second = torch_geometric.nn.GCNConv(gcn_dims[1], gcn_dims[1])
        self.gcn_out = torch.nn.Linear(gcn_dims[1], mlp_dims[0])
        self.layer1 = torch.nn.Linear(mlp_dims[0], mlp_dims[1])
        self.layer2 = torch.nn.Linear(mlp_dims[1], out_dim)

    def forward(self, x_verts, tets):
        delta_v, delta_s, features = [], [], []
        for x_vert, tet in zip(x_verts, tets):
            edges = get_tetrahedras_edges(tet)
            x_vert = self.gcn1.input(x_vert)
            h_vert = self.gcn1.first(nonlinear(x_vert), edges)
            h_vert = self.gcn1.second(nonlinear(h_vert), edges)
            x_vert = (x_vert + h_vert) / 2 ** 0.5

            x_vert = self.gcn2.input(nonlinear(x_vert))
            h_vert = self.gcn2.first(nonlinear(x_vert), edges)
            h_vert = self.gcn2.second(nonlinear(h_vert), edges)
            x_vert = (x_vert + h_vert) / 2 ** 0.5
            h_vert = self.gcn_out(x_vert)

            h_vert = self.layer1(nonlinear(h_vert))
            h_vert = self.layer2(nonlinear(h_vert))
            delta_v.append(h_vert[:, :3])
            delta_s.append(h_vert[:, 3])
            features.append(h_vert[:, 4:])
        return delta_v, delta_s, features


class MeshConv(torch.nn.Module):

    def __init__(self, input_dim, out_dim, gcn_dims=(256, 128), mlp_dims=(128, 64)):
        super(MeshConv, self).__init__()
        self.gcn1 = torch.nn.Module()
        self.gcn1.input = torch.nn.Linear(input_dim, gcn_dims[0])
        self.gcn1.first = torch_geometric.nn.GCNConv(gcn_dims[0], gcn_dims[0])
        self.gcn1.second = torch_geometric.nn.GCNConv(gcn_dims[0], gcn_dims[0])
        self.gcn2 = torch.nn.Module()
        self.gcn2.input = torch.nn.Linear(gcn_dims[0], gcn_dims[1])
        self.gcn2.first = torch_geometric.nn.GCNConv(gcn_dims[1], gcn_dims[1])
        self.gcn2.second = torch_geometric.nn.GCNConv(gcn_dims[1], gcn_dims[1])
        self.gcn_out = torch.nn.Linear(gcn_dims[1], mlp_dims[0])
        self.layer1 = torch.nn.Linear(mlp_dims[0], mlp_dims[1])
        self.layer2 = torch.nn.Linear(mlp_dims[1], out_dim)

    def forward(self, x_verts, faces):
        delta_v, alphas = [], []
        for x_vert, face in zip(x_verts, faces):
            edges = get_mesh_edges(face)
            x_vert = self.gcn1.input(x_vert)
            h_vert = self.gcn1.first(nonlinear(x_vert), edges)
            h_vert = self.gcn1.second(nonlinear(h_vert), edges)
            x_vert = (x_vert + h_vert) / 2 ** 0.5

            x_vert = self.gcn2.input(nonlinear(x_vert))
            h_vert = self.gcn2.first(nonlinear(x_vert), edges)
            h_vert = self.gcn2.second(nonlinear(h_vert), edges)
            x_vert = (x_vert + h_vert) / 2 ** 0.5
            h_vert = self.gcn_out(x_vert)

            h_vert = self.layer1(nonlinear(h_vert))
            h_vert = self.layer2(nonlinear(h_vert))
            delta_v.append(h_vert[:, :3])
            alphas.append(h_vert[:, 3])
        return delta_v, alphas


class SimpleMLP(torch.nn.Module):

    def __init__(self, input_dim, out_dim, hidden_dims=(256, 128, 64)):
        super(SimpleMLP, self).__init__()
        self.input_layer = torch.nn.Linear(input_dim, hidden_dims[0])
        self.layers = torch.nn.ModuleList([torch.nn.Linear(dim, nxt_dim) for dim, nxt_dim in
                                           zip(hidden_dims[:-1], hidden_dims[1:])])
        self.out_layer = torch.nn.Linear(hidden_dims[-1], out_dim)

    def forward(self, x):
        h = self.input_layer(x)
        for layer in self.layers:
            h = layer(nonlinear(h))
        out = self.out_layer(nonlinear(h))
        return out

    def bforward(self, bx):
        return [self.forward(x.unsqueeze(0)).squeeze(0) for x in bx]
