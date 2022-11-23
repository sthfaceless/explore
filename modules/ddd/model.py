from builtins import ModuleNotFoundError

import torch_geometric

from modules.common.model import *
from modules.ddd.util import *


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


class SFDDiscriminator(torch.nn.Module):

    def __init__(self, hidden_dims=(32, 64, 128, 256), num_groups=16):
        super(SFDDiscriminator, self).__init__()
        self.input_conv = torch.nn.Conv3d(in_channels=1, out_channels=hidden_dims[0], kernel_size=5)
        self.norm1 = norm(hidden_dims[0], num_groups)
        self.downsample_conv = torch.nn.Conv3d(in_channels=hidden_dims[0], out_channels=hidden_dims[1],
                                               stride=2, kernel_size=3)
        self.conv1 = torch.nn.Conv3d(in_channels=hidden_dims[1], out_channels=hidden_dims[2], kernel_size=3)
        self.conv2 = torch.nn.Conv3d(in_channels=hidden_dims[2], out_channels=hidden_dims[3], kernel_size=3)
        self.out_norm = norm(hidden_dims[-1], num_groups)
        self.out_layer = torch.nn.Linear(hidden_dims[-1], 1)

    def forward(self, x, emb):
        h = self.input_conv(x)  # 16 -> 12
        h = self.downsample_conv(nonlinear(self.norm1(h)))  # 12 -> 5
        h = self.conv1(nonlinear(h))  # 5 -> 3
        h = self.conv2(nonlinear(h))  # 3 -> 1
        h = h.reshape(h.shape[0], -1) + emb
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
        outs = []
        for item_id, tet in enumerate(tets):
            edges = get_tetrahedras_edges(tet)
            x_vert = self.gcn1.input(x_verts[item_id])
            h_vert = self.gcn1.first(nonlinear(x_vert), edges)
            h_vert = self.gcn1.second(nonlinear(h_vert), edges)
            x_vert = (x_vert + h_vert) / 2 ** 0.5

            x_vert = self.gcn2.input(nonlinear(x_vert), edges)
            h_vert = self.gcn2.first(nonlinear(x_vert), edges)
            h_vert = self.gcn2.second(nonlinear(h_vert), edges)
            x_vert = (x_vert + h_vert) / 2 ** 0.5
            h_vert = self.gcn_out(x_vert)

            h_vert = self.layer1(nonlinear(h_vert))
            h_vert = self.layer2(nonlinear(h_vert))
            outs.append(h_vert)
        out = torch.stack(outs, dim=0)
        delta_v = out[:, :, :3]
        delta_sdf = out[:, :, 3]
        features = out[:, :, 4:]
        return delta_v, delta_sdf, features


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
        outs = []
        for item_id, face in enumerate(faces):
            edges = get_mesh_edges(face)
            x_vert = self.gcn1.input(x_verts[item_id])
            h_vert = self.gcn1.first(nonlinear(x_vert), edges)
            h_vert = self.gcn1.second(nonlinear(h_vert), edges)
            x_vert = (x_vert + h_vert) / 2 ** 0.5

            x_vert = self.gcn2.input(nonlinear(x_vert), edges)
            h_vert = self.gcn2.first(nonlinear(x_vert), edges)
            h_vert = self.gcn2.second(nonlinear(h_vert), edges)
            x_vert = (x_vert + h_vert) / 2 ** 0.5
            h_vert = self.gcn_out(x_vert)

            h_vert = self.layer1(nonlinear(h_vert))
            h_vert = self.layer2(nonlinear(h_vert))
            outs.append(h_vert)
        out = torch.stack(outs, dim=0)
        v = out[:, :, :3]
        alpha = out[:, :, 3]
        return v, alpha
