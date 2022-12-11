from builtins import ModuleNotFoundError

import torch.nn
import torch_geometric

from modules.common.model import *
from modules.ddd.util import *


def nonlinear(x):
    return torch.nn.functional.leaky_relu(x)


class PointVoxelCNN(torch.nn.Module):

    def __init__(self, input_dim, dim, grid_res=32, kernel_size=3, num_groups=32, do_points_map=True, with_norm=True):
        super(PointVoxelCNN, self).__init__()

        self.input_dim = input_dim
        self.dim = dim
        self.grid_res = grid_res

        if input_dim != dim:
            self.input_conv = torch.nn.Conv3d(in_channels=input_dim, out_channels=dim, kernel_size=kernel_size,
                                              padding=kernel_size // 2)

        self.conv1 = torch.nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=kernel_size,
                                     padding=kernel_size // 2)
        self.conv2 = torch.nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=kernel_size,
                                     padding=kernel_size // 2)

        self.with_norm = with_norm
        if with_norm:
            self.norm1 = norm(dim, num_groups)
            self.norm2 = norm(dim, num_groups)

        self.do_points_map = do_points_map
        if do_points_map:
            self.point_layer = torch.nn.Module()
            if input_dim != dim:
                self.point_layer.input = torch.nn.Conv1d(in_channels=input_dim, out_channels=dim, kernel_size=1)
            self.point_layer.conv1 = torch.nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1)
            self.point_layer.conv2 = torch.nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1)
            if with_norm:
                self.point_layer.norm1 = norm(dim, num_groups)
                self.point_layer.norm2 = norm(dim, num_groups)

    def voxelize(self, points, features, mask=None):
        # map grid
        input_grid = voxelize_points3d(points, features, self.grid_res, mask=mask).movedim(-1, 1)
        if self.input_dim != self.dim:
            input_grid = self.input_conv(input_grid)
        if self.with_norm:
            input_grid = self.norm1(input_grid)
        grid = self.conv1(nonlinear(input_grid))
        if self.with_norm:
            grid = self.norm2(grid)
        grid = self.conv2(nonlinear(grid))
        out_grid = (grid + input_grid) / 2 ** 0.5
        return out_grid.movedim(1, -1)

    def devoxelize(self, points, grid, mask=None):
        return devoxelize_points3d(points, grid, mask=mask)

    def map_points(self, features, mask=None):
        if not self.do_points_map:
            raise ModuleNotFoundError('Initialized model without points mapping')
        features = features.movedim(-1, 1)
        if self.input_dim != self.dim:
            features = self.point_layer.input(features)
        if self.with_norm:
            features = self.point_layer.norm1(features)
        h = self.point_layer.conv1(nonlinear(features))
        if self.with_norm:
            h = self.point_layer.norm2(h)
        h = self.point_layer.conv2(nonlinear(h))
        out = (h + features) / 2 ** 0.5
        if mask is not None:
            out = out * mask.int().unsqueeze(-1)  # zero outing all extra features
        return out.movedim(1, -1)

    def forward(self, points, features, grid=None, mask=None):
        # points (b, 3)
        # features (b, input_dim)
        if grid is None:
            grid = self.voxelize(points, features, mask=mask)
        return (self.map_points(features, mask=mask) + self.devoxelize(points, grid, mask=mask)) / 2 ** 0.5


class MultiPointVoxelCNN(torch.nn.Module):

    def __init__(self, input_dim, dim, dims=(64, 128, 256), grids=(32, 16, 8), do_points_map=True,
                 kernel_size=3, num_groups=32, with_norm=True):
        super(MultiPointVoxelCNN, self).__init__()
        self.input_dim = input_dim
        self.dim = dim
        self.models = torch.nn.ModuleList(
            [PointVoxelCNN(input_dim=input_dim, dim=dim, grid_res=res, do_points_map=do_points_map, with_norm=with_norm,
                           kernel_size=kernel_size, num_groups=num_groups) for dim, res in zip(dims, grids)])

        self.with_norm = with_norm
        self.do_points_map = do_points_map
        if do_points_map:
            self.point_layer = torch.nn.Module()
            if input_dim != dim:
                self.point_layer.input = torch.nn.Conv1d(in_channels=input_dim, out_channels=dim, kernel_size=1)
            self.point_layer.conv1 = torch.nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1)
            self.point_layer.conv2 = torch.nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1)
            if with_norm:
                self.point_layer.norm1 = norm(dim, num_groups)
                self.point_layer.norm2 = norm(dim, num_groups)

        if with_norm:
            self.out_norm = norm(sum(dims), num_groups=num_groups * len(dims))
        self.out_layer = torch.nn.Conv1d(sum(dims), dim, kernel_size=1)

    def voxelize(self, points, features, mask=None):
        return [model.voxelize(points, features, mask=mask) for model in self.models]

    def devoxelize(self, points, grids, mask=None):
        h = torch.cat([model.devoxelize(points, grid, mask=mask) for model, grid in zip(self.models, grids)], dim=-1)
        h = h.movedim(-1, 1)
        if self.with_norm:
            h = self.out_norm(h)
        out = self.out_layer(nonlinear(h))
        return out.movedim(1, -1)

    def map_points(self, features):
        if not self.do_points_map:
            raise ModuleNotFoundError('Initialized model without points mapping')
        features = features.movedim(-1, 1)
        if self.input_dim != self.dim:
            features = self.point_layer.input(features)
        if self.with_norm:
            features = self.point_layer.norm1(features)
        h = self.point_layer.conv1(nonlinear(features))
        if self.with_norm:
            h = self.point_layer.norm2(h)
        h = self.point_layer.conv2(nonlinear(h))
        out = (h + features) / 2 ** 0.5
        return out

    def forward(self, points, features, grids=None, mask=None):
        if grids is None:
            grids = self.voxelize(points, features, mask=mask)
        volume_features = self.devoxelize(points, grids, mask=mask)
        return self.map_points(features) + volume_features


class SDFDiscriminator(torch.nn.Module):

    def __init__(self, input_dim, hidden_dims=(32, 64, 128, 256), num_groups=16, with_norm=True):
        super(SDFDiscriminator, self).__init__()
        self.input_conv = torch.nn.Conv3d(in_channels=input_dim, out_channels=hidden_dims[0], kernel_size=5)

        self.downsample_conv = torch.nn.Conv3d(in_channels=hidden_dims[0], out_channels=hidden_dims[1],
                                               stride=2, kernel_size=3)
        self.conv1 = torch.nn.Conv3d(in_channels=hidden_dims[1], out_channels=hidden_dims[2], kernel_size=3)
        self.conv2 = torch.nn.Conv3d(in_channels=hidden_dims[2], out_channels=hidden_dims[3], kernel_size=3)

        self.out_layer = torch.nn.Linear(hidden_dims[-1], 1)

        self.with_norm = with_norm
        if with_norm:
            self.norm1 = norm(hidden_dims[0], num_groups)
            self.out_norm = norm(hidden_dims[-1], num_groups)

    def forward(self, x):
        h = self.input_conv(x)  # 16 -> 12
        if self.with_norm:
            h = self.norm1(h)
        h = self.downsample_conv(nonlinear(h))  # 12 -> 5
        h = self.conv1(nonlinear(h))  # 5 -> 3
        h = self.conv2(nonlinear(h))  # 3 -> 1
        h = h.reshape(h.shape[0], -1)
        if self.with_norm:
            h = self.out_norm(h)
        h = self.out_layer(nonlinear(h))
        out = torch.sigmoid(h)
        return out


class GCNConv(torch.nn.Module):

    def __init__(self, input_dim, out_dim, gcn_dims=(256, 128), mlp_dims=(128, 64), with_norm=True):
        super(GCNConv, self).__init__()
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

        self.with_norm = with_norm
        if with_norm:
            self.gcn1.norm1 = torch.nn.BatchNorm1d(gcn_dims[0])
            self.gcn1.norm2 = torch.nn.BatchNorm1d(gcn_dims[0])
            self.gcn2.norm1 = torch.nn.BatchNorm1d(gcn_dims[1])
            self.gcn2.norm2 = torch.nn.BatchNorm1d(gcn_dims[1])
            self.norm1 = torch.nn.BatchNorm1d(mlp_dims[0])
            self.norm2 = torch.nn.BatchNorm1d(mlp_dims[1])

    def apply_gcn(self, x_in, gcn, edges):
        x_in = gcn.input(x_in)
        x_vert = x_in
        if self.with_norm:
            x_vert = gcn.norm1(x_vert)
        h_vert = gcn.first(nonlinear(x_vert), edges)
        if self.with_norm:
            h_vert = gcn.norm2(h_vert)
        h_vert = gcn.second(nonlinear(h_vert), edges)
        x_vert = (x_vert + h_vert) / 2 ** 0.5
        return x_vert + x_in

    def forward(self, x_vert, edges):

        x_vert = self.apply_gcn(x_vert, self.gcn1, edges)
        x_vert = self.apply_gcn(x_vert, self.gcn2, edges)
        h_vert = self.gcn_out(x_vert)

        if self.with_norm:
            h_vert = self.norm1(h_vert)
        h_vert = self.layer1(nonlinear(h_vert))
        if self.with_norm:
            h_vert = self.norm2(h_vert)
        h_vert = self.layer2(nonlinear(h_vert))
        return h_vert


class SimpleMLP(torch.nn.Module):

    def __init__(self, input_dim, out_dim, hidden_dims=(256, 128, 64), num_groups=32, with_norm=True):
        super(SimpleMLP, self).__init__()
        self.input_layer = torch.nn.Conv1d(in_channels=input_dim, out_channels=hidden_dims[0], kernel_size=1)
        self.layers = torch.nn.ModuleList([torch.nn.Conv1d(in_channels=dim, out_channels=nxt_dim, kernel_size=1)
                                           for dim, nxt_dim in zip(hidden_dims[:-1], hidden_dims[1:])])
        self.out_layer = torch.nn.Conv1d(in_channels=hidden_dims[-1], out_channels=out_dim, kernel_size=1)
        self.with_norm = with_norm
        if with_norm:
            self.norms = torch.nn.ModuleList(
                [norm(dim, num_groups) for dim, nxt_dim in zip(hidden_dims[:-1], hidden_dims[1:])])
            self.out_norm = norm(hidden_dims[-1], num_groups)

    def forward(self, x):
        h = self.input_layer(x.movedim(-1, 1))
        for layer_idx, layer in enumerate(self.layers):
            if self.with_norm:
                h = self.norms[layer_idx](h)
            h = layer(nonlinear(h))
        if self.with_norm:
            h = self.out_norm(h)
        out = self.out_layer(nonlinear(h))
        return out.movedim(1, -1)
