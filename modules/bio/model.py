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

    def voxelize(self, points, features, mask=None):
        # map grid
        input_grid = voxelize_points3d(points, features, self.grid_res, mask=mask).movedim(-1, 1)
        grid = self.conv1(nonlinear(self.norm1(input_grid)))
        grid = self.conv2(nonlinear(self.norm2(grid)))
        skip = input_grid if self.input_dim == self.dim else self.skip_conv(input_grid)
        out_grid = (grid + skip).movedim(1, -1) / 2 ** 0.5
        return out_grid

    def devoxelize(self, points, grid, mask=None):
        return devoxelize_points3d(points, grid, mask=mask)

    def map_points(self, features, mask=None):
        if not self.do_points_map:
            raise ModuleNotFoundError('Initialized model without points mapping')
        features = self.point_input(features)
        h = self.point_1(nonlinear(features))
        h = self.point_2(nonlinear(h))
        out = (h + features) / 2 ** 0.5
        if mask is not None:
            out = out * mask.int().unsqueeze(-1)  # zero outing all extra features
        return out

    def forward(self, points, features, grid=None, mask=None):
        # points (b, 3)
        # features (b, input_dim)
        if grid is None:
            grid = self.voxelize(points, features, mask=mask)
        return (self.map_points(features, mask=mask) + self.devoxelize(points, grid, mask=mask)) / 2 ** 0.5


class MultiPointVoxelCNN(torch.nn.Module):

    def __init__(self, input_dim, dims=(64, 128, 256), grids=(32, 16, 8), do_points_map=True,
                 kernel_size=3, num_groups=32):
        super(MultiPointVoxelCNN, self).__init__()
        self.models = torch.nn.ModuleList(
            [PointVoxelCNN(input_dim=input_dim, dim=dim, grid_res=res, do_points_map=do_points_map,
                           kernel_size=kernel_size, num_groups=num_groups) for dim, res in zip(dims, grids)])

    def voxelize(self, points, features, mask=None):
        return [model.voxelize(points, features, mask=mask) for model in self.models]

    def devoxelize(self, points, grids, mask=None):
        return torch.cat([model.devoxelize(points, grid, mask=mask) for model, grid in zip(self.models, grids)], dim=-1)

    def forward(self, points, features, grids=None, mask=None):
        outs = []
        for module_id, module in enumerate(self.models):
            grid = grids[module_id] if grids is not None else None
            outs.append(module(points, features, grid=grid, mask=mask))
        return torch.cat(outs, dim=-1)


class PointRegression(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim=256, n_blocks=2, out_dims=(512, 256, 64), seq_len=400,
                 num_heads=8, dropout=0.0, agg_dim=4):
        super(PointRegression, self).__init__()
        self.seq_len = seq_len
        self.agg_dim = agg_dim

        self.input_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.embeds = torch.nn.Embedding(num_embeddings=seq_len + 1, embedding_dim=hidden_dim, padding_idx=seq_len)
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

        self.out_layer = torch.nn.Linear(hidden_dim, agg_dim)
        self.out_agg = torch.nn.Linear(seq_len * agg_dim, out_dims[0])

        self.out_blocks = torch.nn.ModuleList([])
        for prev_dim, dim in zip(out_dims, list(out_dims[1:]) + [1]):
            self.out_blocks.append(torch.nn.Linear(prev_dim, dim))

    def forward(self, seq, mask=None):

        # input mapping to internal dim
        h = self.input_layer(seq)

        # additional positional embedding
        b, seq_len, dim = seq.shape
        pos_idx = torch.arange(seq_len).type_as(seq).long().view(1, seq_len).repeat(b, 1)
        if mask is not None:
            pos_idx = torch.where(mask, pos_idx, torch.ones_like(pos_idx) * seq_len)
        embed = self.embeds(pos_idx)
        h += embed

        # transformer encoder blocks
        for block in self.blocks:
            h = block.norm1(block.dropout1(block.attn(h, mask=mask)) + h)
            h = block.norm2(block.dropout2(block.ff2(nonlinear(block.ff1(nonlinear(h))))) + h)

        # out regression head
        h = self.out_layer(h).view(b, self.seq_len * self.agg_dim)
        h = self.out_agg(h)
        for block in self.out_blocks:
            h = block(nonlinear(h))
        out = h.view(b)  # squeeze last regression dim
        return out
