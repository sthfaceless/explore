import torch.nn

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
            self.point_input = torch.nn.Linear(input_dim, dim)
            self.point_1 = torch.nn.Linear(dim, dim)
            self.point_2 = torch.nn.Linear(dim, dim)

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
        features = self.point_input(features)
        h = self.point_1(nonlinear(features))
        h = self.point_2(nonlinear(h))
        return (h + features) / 2 ** 0.5

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


class PointRegression(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim=256, n_blocks=8, grid_res=256, hash_dim=1024,
                 out_dims=(512, 256, 64), num_heads=8, dropout=0.0):
        super(PointRegression, self).__init__()
        self.grid_res = grid_res
        self.hash_dim = hash_dim

        self.input_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.embeds = torch.nn.Embedding(num_embeddings=hash_dim, embedding_dim=hidden_dim)
        self.blocks = torch.nn.ModuleList([])
        for block_id in range(n_blocks):
            block = torch.nn.Module()
            block.norm1 = torch.nn.LayerNorm(hidden_dim)
            block.dropout1 = torch.nn.Dropout(dropout)
            block.attn = Attention(input_dim=hidden_dim, embed_dim=hidden_dim, num_heads=num_heads)
            block.norm2 = torch.nn.LayerNorm(hidden_dim)
            block.dropout2 = torch.nn.Dropout(dropout)
            block.ff1 = torch.nn.Linear(hidden_dim, hidden_dim * 4)
            block.ff2 = torch.nn.Linear(hidden_dim * 4, hidden_dim)
            self.blocks.append(block)

        self.out_layer = torch.nn.Linear(hidden_dim, 1)
        self.out_agg = torch.nn.Linear(hash_dim, out_dims[0])
        self.out_blocks = torch.nn.ModuleList([])
        for prev_dim, dim in zip(out_dims, list(out_dims[1:]) + [1]):
            self.out_blocks.append(torch.nn.Linear(prev_dim, dim))

    def forward(self, points, features):

        points = (points + 1.0) / 2
        indexes = torch.floor(points * (self.grid_res - 1) + 0.5).long()
        hashes = (indexes[:, :, 0] * 31 + indexes[:, :, 1] * 31 ** 2 + indexes[:, :, 2] * 31 ** 3) % self.hash_dim

        features = self.input_layer(features)
        b, n_points, dim = features.shape
        __features = torch.zeros(b, self.hash_dim, dim).type_as(features)
        __denom = torch.zeros(b, self.hash_dim, dim).type_as(features)
        batch_index = torch.arange(b).type_as(hashes).view(b, 1).repeat(1, n_points).view(-1)
        __features.index_put((batch_index, hashes.view(-1)), features, accumulate=True)
        __denom.index_put((batch_index, hashes.view(-1)), torch.ones_like(features), accumulate=True)
        __features /= torch.clamp(__denom ** 0.5, min=1.0)

        h = __features  # b features seq_len
        embed = self.embeds(torch.arange(self.hash_dim).type_as(hashes))[None, ...]
        for block in self.blocks:
            h = h + embed
            h = block.norm1(block.dropout1(block.attn(h)) + h)
            h = block.norm2(block.dropout2(block.ff2(nonlinear(block.ff1(nonlinear(h))))) + h)

        h = self.out_layer(nonlinear(h)).view(b, self.hash_dim)
        h = self.out_agg(h)
        for block in self.out_blocks:
            h = block(nonlinear(h))

        return h
