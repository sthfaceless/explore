import torch.nn

from modules.common.util import *


def nonlinear(x):
    return torch.nn.functional.silu(x)


def norm(dims, num_groups=32):
    while dims % num_groups != 0:
        num_groups -= 1
    return torch.nn.GroupNorm(num_channels=dims, num_groups=num_groups, eps=1e-6)


class SinActivation(torch.nn.Module):

    def forward(self, x):
        return torch.sin(x)


class UpSample2d(torch.nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size=3, transpose=False, scale_factor=2.0):
        super(UpSample2d, self).__init__()
        self.transpose = transpose
        if self.transpose:
            self.conv = torch.nn.ConvTranspose2d(in_channels=in_dim, out_channels=out_dim,
                                                 kernel_size=kernel_size,
                                                 stride=2, padding=kernel_size // 2, output_padding=1)
        else:
            self.scale_factor = scale_factor
            self.conv = torch.nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size,
                                        stride=1, padding=kernel_size // 2)

    def forward(self, x):
        if self.transpose:
            h = self.conv(x)
        else:
            h = torch.nn.functional.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
            h = self.conv(h)
        return h


class DownSample2d(torch.nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size=3, scale_factor=0.5):
        super(DownSample2d, self).__init__()
        self.scale_factor = scale_factor
        self.conv = torch.nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size,
                                    padding=kernel_size // 2, stride=1)

    def forward(self, x):
        h = self.conv(x)
        h = torch.nn.functional.interpolate(h, scale_factor=self.scale_factor, mode='bilinear')
        return h


class Attention(torch.nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads=8, head_dim=None, dropout=0., bias=False):
        super().__init__()
        if head_dim is not None:
            self.head_dim = head_dim
            self.num_heads = embed_dim // head_dim
        else:
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = torch.nn.LayerNorm(embed_dim)
        self.to_q = torch.nn.Linear(input_dim, embed_dim, bias=bias)
        self.to_kv = torch.nn.Linear(input_dim, embed_dim * 2, bias=bias)
        self.to_out = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, input_dim, bias=bias),
            torch.nn.Dropout(dropout)
        )

    def forward(self, q, v=None, mask=None):
        # normalize inputs
        q = self.norm(q)
        if v is None:
            v = q
        else:
            v = self.norm(v)

        # map inputs
        q = self.to_q(q)
        k, v = self.to_kv(v).chunk(2, dim=-1)

        # b n (h d) -> b h n d
        b, n, dim = q.shape
        q, k, v = map(lambda t: t.view(b, n, self.num_heads, self.head_dim).transpose(1, 2), [q, k, v])

        # b h n n
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if mask is not None:
            dots = torch.where(mask.view(b, 1, 1, n), dots, torch.ones_like(dots) * float('-inf'))
        attn = torch.nn.functional.softmax(dots, dim=-1)
        # (b h n n) * (b h n d)
        out = torch.matmul(attn, v)

        # b h n d -> b n (h d)
        out = out.transpose(1, 2).reshape(b, n, self.num_heads * self.head_dim)
        out = self.to_out(out)

        return out


class Attention2D(torch.nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.0, num_groups=32):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.norm = norm(dim, num_groups)
        self.q = torch.nn.Conv2d(dim, dim, kernel_size=1)
        self.k = torch.nn.Conv2d(dim, dim, kernel_size=1)
        self.v = torch.nn.Conv2d(dim, dim, kernel_size=1)
        self.out = torch.nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, q, v=None):
        q_in = q
        q = self.norm(q)
        if v is None:
            v = q
        else:
            v = self.norm(v)
        q, k, v = self.q(q), self.k(v), self.v(v)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)  # b,c,hw
        k = k.reshape(b, c, h * w).transpose(-1, -2)  # b,hw,c
        attn_weights = torch.nn.functional.softmax(torch.matmul(k, q) * (int(c) ** (-0.5)), dim=-2)  # b,hw,hw

        # attend to values
        v = v.reshape(b, c, h * w)
        out = torch.matmul(v, attn_weights).reshape(b, c, h, w)  # b, c, hw

        out = self.out(out)

        return (out + q_in) / 2 ** (1 / 2)


class MHAAttention2D(torch.nn.Module):
    def __init__(self, dim, num_heads=None, head_channel=32, dropout=0.0, num_groups=32):
        super().__init__()
        self.dim = dim
        if num_heads:
            self.num_heads = num_heads
            self.head_dim = dim // self.num_heads
        else:
            self.num_heads = dim // head_channel
            self.head_dim = head_channel
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.norm = norm(dim, num_groups)
        self.q = torch.nn.Conv2d(dim, dim, kernel_size=1)
        self.k = torch.nn.Conv2d(dim, dim, kernel_size=1)
        self.v = torch.nn.Conv2d(dim, dim, kernel_size=1)
        self.out = torch.nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, q, v=None):
        q_in = q
        q = self.norm(q)
        if v is None:
            v = q
        else:
            v = self.norm(v)
        q, k, v = self.q(q), self.k(v), self.v(v)

        # compute attention
        b, c, h, w = q.shape
        # b, c, h, w -> b, n, d, hw -> b, n, hw, d
        q, k, v = map(lambda t: t.reshape(b, self.num_heads, self.head_dim, h * w).transpose(-1, -2), [q, k, v])
        # b, n, hw, hw
        attn_weights = torch.nn.functional.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.scale, dim=-1)

        # attend to values (b n hw d)
        out = torch.matmul(attn_weights, v)
        # b n hw d -> b n*d, h, w
        out = out.transpose(-1, -2).reshape(b, self.dim, h, w)
        out = self.out(out)

        return (out + q_in) / 2 ** (1 / 2)


class LocalMHAAttention2D(torch.nn.Module):
    def __init__(self, dim, num_heads=None, head_channel=32, dropout=0.0, num_groups=32, kernel_size=8):
        super().__init__()
        self.dim = dim
        if num_heads:
            self.num_heads = num_heads
            self.head_dim = dim // self.num_heads
        else:
            self.num_heads = dim // head_channel
            self.head_dim = head_channel
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout
        self.kernel_size = kernel_size

        self.norm = norm(dim, num_groups)
        self.q = torch.nn.Conv2d(dim, dim, kernel_size=1)
        self.k = torch.nn.Conv2d(dim, dim, kernel_size=1)
        self.v = torch.nn.Conv2d(dim, dim, kernel_size=1)
        self.out = torch.nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, q, v=None):
        q_in = q
        q = self.norm(q)
        if v is None:
            v = q
        else:
            v = self.norm(v)
        q, k, v = self.q(q), self.k(v), self.v(v)

        # compute attention
        b, c, h, w = q.shape
        # b, c, h, w -> b, n, h/k * w/k, k*k, d
        q, k, v = map(lambda t: t.reshape(
            b, self.num_heads, self.head_dim,
            h // self.kernel_size, self.kernel_size, w // self.kernel_size, self.kernel_size)
                      .transpose(-2, -3).movedim(2, -1)
                      .reshape(b, self.num_heads, h * w // self.kernel_size ** 2, self.kernel_size ** 2, self.head_dim),
                      [q, k, v])
        # b, n, h/k * w/k, k*k, k*k
        attn_weights = torch.nn.functional.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.scale, dim=-1)

        # attend to values (b n h/k * w/k k*k d)
        out = torch.matmul(attn_weights, v)
        # b n h/k * w/k k*k d ->
        out = out.movedim(-1, 2).reshape(
            b, self.dim, h // self.kernel_size, w // self.kernel_size, self.kernel_size, self.kernel_size) \
            .transpose(-2, -3).reshape(b, self.dim, h, w)
        out = self.out(out)

        return (out + q_in) / 2 ** (1 / 2)


class ResBlock2d(torch.nn.Module):

    def __init__(self, hidden_dim, kernel_size=3, num_groups=32, in_dim=-1):
        super(ResBlock2d, self).__init__()

        if in_dim == -1:
            in_dim = hidden_dim
        else:
            self.input_mapper = torch.nn.Conv2d(in_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.ln_1 = norm(in_dim, num_groups)
        self.layer_1 = torch.nn.Conv2d(in_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.ln_2 = norm(hidden_dim, num_groups)
        self.layer_2 = torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, input):
        x = self.layer_1(nonlinear(self.ln_1(input)))
        skip = input if self.in_dim == self.hidden_dim else self.input_mapper(input)
        x = (self.layer_2(nonlinear(self.ln_2(x))) + skip) / 2 ** (1 / 2)
        return x


class TimestepResBlock2D(torch.nn.Module):

    def __init__(self, hidden_dim, timestep_dim, kernel_size=3, num_groups=32, in_dim=-1, attn=False, num_heads=4):
        super(TimestepResBlock2D, self).__init__()

        if in_dim == -1:
            in_dim = hidden_dim
        else:
            self.res_mapper = torch.nn.Conv2d(in_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.ln_1 = norm(in_dim, num_groups)
        self.layer_1 = torch.nn.Conv2d(in_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conditional_norm = ConditionalNorm2D(hidden_dim, timestep_dim, num_groups)
        self.layer_2 = torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2)

        self.need_attn = attn
        if self.need_attn:
            self.attn = MHAAttention2D(hidden_dim, num_groups=num_groups, num_heads=num_heads)

    def forward(self, input, time):
        h = self.layer_1(nonlinear(self.ln_1(input)))
        h = self.conditional_norm(h, time[:, :, None, None].expand(*time.shape[:2], *h.shape[2:]))
        skip = input if self.in_dim == self.hidden_dim else self.res_mapper(input)
        h = (self.layer_2(nonlinear(h)) + skip) / 2 ** (1 / 2)
        if self.need_attn:
            h = self.attn(h)
        return h


class ConditionalNorm2D(torch.nn.Module):

    def __init__(self, dim, emb_dim, num_groups=32):
        super(ConditionalNorm2D, self).__init__()
        self.norm = norm(dim, num_groups)
        self.layer = torch.nn.Conv2d(emb_dim, dim * 2, kernel_size=1)

    def forward(self, h, emb):
        emb = self.layer(nonlinear(emb))
        gamma, beta = torch.chunk(emb, 2, dim=1)  # split in channel dimension (b c h w)
        return self.norm(h) * (1.0 + gamma) + beta


class ConditionalNormResBlock2D(torch.nn.Module):
    def __init__(self, hidden_dim, embed_dim, kernel_size=3, num_groups=32, in_dim=-1, attn=False, local_attn=False,
                 local_attn_kernel=8, dropout=0.0, num_heads=4):
        super(ConditionalNormResBlock2D, self).__init__()

        if in_dim == -1:
            in_dim = hidden_dim
        else:
            self.res_mapper = torch.nn.Conv2d(in_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.ln_1 = norm(in_dim, num_groups)
        self.layer_1 = torch.nn.Conv2d(in_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conditional_norm = ConditionalNorm2D(hidden_dim, embed_dim, num_groups)
        self.dropout = torch.nn.Dropout2d(p=dropout)
        self.layer_2 = torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2)

        self.need_attn = attn or local_attn
        if attn:
            self.attn = MHAAttention2D(hidden_dim, num_groups=num_groups, num_heads=num_heads)
        elif local_attn:
            self.attn = LocalMHAAttention2D(hidden_dim, num_groups=num_groups, num_heads=num_heads,
                                            kernel_size=local_attn_kernel)

    def forward(self, x, emb):
        h = self.layer_1(nonlinear(self.ln_1(x)))
        h = self.conditional_norm(h, emb)
        h = self.layer_2(self.dropout(nonlinear(h)))
        skip = x if self.in_dim == self.hidden_dim else self.res_mapper(x)
        h = (h + skip) / 2 ** (1 / 2)
        if self.need_attn:
            h = self.attn(h)
        return h


class DownSample(torch.nn.Module):

    def __init__(self, input_dim, dim):
        super(DownSample, self).__init__()
        self.downsampler = torch.nn.Sequential(
            torch.nn.Linear(input_dim, dim),
            torch.nn.GELU()
        )

    def forward(self, x):
        return self.downsampler(x)


class ResBlock(torch.nn.Module):

    def __init__(self, dim):
        super(ResBlock, self).__init__()

        self.block = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.GELU(),
            torch.nn.Linear(dim, dim)
        )

    def forward(self, x):
        return torch.nn.functional.gelu((self.block(x) + x) / 2 ** (1 / 2))


class DownSampleConv1D(torch.nn.Module):

    def __init__(self, input_dim, dim, kernel_size=32, reverse=False):
        super(DownSampleConv1D, self).__init__()
        if not reverse:
            self.downsampler = torch.nn.Sequential(
                torch.nn.Conv1d(input_dim, dim, kernel_size=kernel_size + 2, padding=kernel_size // 2, stride=2),
                torch.nn.GELU()
            )
        else:
            self.downsampler = torch.nn.Sequential(
                torch.nn.ConvTranspose1d(input_dim, dim, kernel_size=kernel_size, padding=kernel_size // 2 - 1,
                                         stride=2),
                torch.nn.GELU()
            )

    def forward(self, x):
        return self.downsampler(x)


class ResBlockConv1D(torch.nn.Module):

    def __init__(self, dim, kernel_size=32):
        super(ResBlockConv1D, self).__init__()

        self.block = torch.nn.Sequential(
            torch.nn.Conv1d(dim, dim, kernel_size=kernel_size + 1, padding=kernel_size // 2),
            torch.nn.GELU(),
            torch.nn.Conv1d(dim, dim, kernel_size=kernel_size + 1, padding=kernel_size // 2)
        )

    def forward(self, x):
        return torch.nn.functional.gelu((self.block(x) + x) / 2 ** (1 / 2))


class GraphConv(torch.nn.Module):

    def __init__(self, in_features, out_features, use_sparse=False):
        super(GraphConv, self).__init__()
        self.layer = torch.nn.Linear(in_features, out_features)
        self.use_sparse = use_sparse

    def forward(self, vertices, edges, self_loop=True, normalize=True):
        # vertices - (n, features)
        # edges (2, edges)
        n_vertices, features = vertices.shape
        _, n_edges = edges.shape

        # add self loops
        if self_loop:
            idx = torch.arange(n_vertices).type_as(edges)
            self_edges = torch.stack([idx, idx], dim=0)
            edges = torch.cat([edges, self_edges], dim=1)
            n_edges += n_vertices

        if self.use_sparse:
            # create adjacent matrix of graph
            adj = torch.sparse_coo_tensor(edges, torch.ones(n_edges).type_as(vertices),
                                          (n_vertices, n_vertices), dtype=vertices.dtype, device=vertices.device)
            # aggregate features for each vertex based on it's neighbourhood
            agg = torch.sparse.mm(adj, vertices)

            # normalize calculated sum by inverse sqrt of vertex degree
            if normalize:
                D = torch.sparse.sum(adj, dim=-1).to_dense() ** (-1 / 2)
                agg = agg * D.unsqueeze(-1)
        else:
            agg = torch.zeros_like(vertices)
            neighbours = vertices[edges[1]]
            agg = agg.index_put_((edges[0],), neighbours, accumulate=True)

            if normalize:
                d = torch.zeros_like(vertices)
                d = d.index_put_((edges[0],), torch.ones_like(neighbours), accumulate=True)
                agg = agg * (d.clamp(min=1.0) ** (-1 / 2))

        # apply linear transformation on output
        out = self.layer(agg)
        return out
