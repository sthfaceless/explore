import torch.nn

from modules.common.model import *


class TemporalAttention2d(torch.nn.Module):
    def __init__(self, dim, q_dim=-1, k_dim=-1, num_heads=None, head_channel=32, dropout=0.0, num_groups=32):
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

        if q_dim == -1:
            q_dim = dim
        self.q_dim = q_dim
        if q_dim != dim:
            self.q_skip = torch.nn.Conv2d(q_dim, dim, kernel_size=1)

        if k_dim == -1:
            k_dim = dim
        self.k_dim = k_dim

        self.q_norm = norm(q_dim, num_groups)
        self.v_norm = norm(k_dim, num_groups)
        self.q = torch.nn.Conv2d(q_dim, dim, kernel_size=1)
        self.k = torch.nn.Conv2d(k_dim, dim, kernel_size=1)
        self.v = torch.nn.Conv2d(k_dim, dim, kernel_size=1)
        self.out = torch.nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, q, t, v=None):
        q_in = q
        q = self.q_norm(q)
        if v is None:
            v = q
        else:
            v = self.v_norm(v)
        q, k, v = self.q(q), self.k(v), self.v(v)

        # compute attention
        bt, c, h, w = q.shape
        # b*t, c, h, w -> b, t, n, d, hw -> b, t, n, hw, d -> b, hw, n, t, d
        q, k, v = map(lambda e: e.reshape(bt // t, t, self.num_heads, self.head_dim, h * w) \
                      .transpose(-1, -2).transpose(-2, -4), [q, k, v])
        # b, hw, n, t, t
        attn_weights = torch.nn.functional.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.scale, dim=-1)

        # attend to values (b hw n t d)
        out = torch.matmul(attn_weights, v)
        # b hw n t d -> b t n hw d -> b t n d hw -> b*t n*d, h, w
        out = out.transpose(-2, -4).transpose(-1, -2).reshape(bt, self.dim, h, w)
        out = self.out(out)

        if self.q_dim != self.dim:
            q_in = self.q_skip(q_in)
        return (out + q_in) / 2 ** (1 / 2)


class TemporalCondResBlock2d(torch.nn.Module):
    def __init__(self, hidden_dim, embed_dim, kernel_size=3, num_groups=32, in_dim=-1, attn=False, cross_cond=False,
                 dropout=0.0, num_heads=4):
        super(TemporalCondResBlock2d, self).__init__()

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

        self.need_attn = attn
        if attn:
            self.sc_attn = MHAAttention2D(hidden_dim, k_dim=hidden_dim * 2, num_groups=num_groups, num_heads=num_heads)
            self.cross_attn = MHAAttention2D(hidden_dim, num_groups=num_groups, num_heads=num_heads)
            self.temp_attn = TemporalAttention2d(hidden_dim, num_groups=num_groups, num_heads=num_heads)

    def forward(self, x, t, emb, cond=None):
        h = self.layer_1(nonlinear(self.ln_1(x)))
        h = self.conditional_norm(h, emb)
        h = self.layer_2(self.dropout(nonlinear(h)))
        skip = x if self.in_dim == self.hidden_dim else self.res_mapper(x)
        h = (h + skip) / 2 ** (1 / 2)
        if self.need_attn:
            bt, c, hh, ww = h.shape
            ht = h.reshape(bt // t, t, c, hh, ww)
            kv1 = repeat_dim(ht[:, :1], 1, t)  # first frame of each batch
            kv2 = torch.cat([ht[:, :1], ht[:, :-1]], dim=1)  # previous frame for each one
            kv = torch.cat([kv1, kv2], dim=2).reshape(bt, 2*c, hh, ww)
            h = self.sc_attn(h, v=kv)
            if cond is not None:
                h = self.cross_attn(q=h, v=cond)
            h = self.temp_attn(h, t)
        return h


class TemporalUNetDenoiser(torch.nn.Module):

    def __init__(self, shape, steps, kernel_size=3, hidden_dims=(128, 256, 256, 512), attention_dim=32,
                 num_groups=32, dropout=0.0, num_heads=4, embed_features=512, cond='cross',
                 extra_upsample_blocks=0):
        super(TemporalUNetDenoiser, self).__init__()
        features, h, w = shape
        self.features = features
        self.steps = steps
        self.cond = cond
        # Input mapping
        self.input_mapper = torch.nn.Conv2d(features, hidden_dims[0], kernel_size=kernel_size,
                                            padding=kernel_size // 2)

        # Time embeddings
        self.emb_features = embed_features
        self.time_layers = torch.nn.ModuleList([
            torch.nn.Linear(embed_features // 2, embed_features),
            torch.nn.Linear(embed_features, embed_features),
        ])

        # Conditional image embeddings
        self.cond_input = torch.nn.Conv2d(features, hidden_dims[0], kernel_size=kernel_size,
                                          padding=kernel_size // 2)
        self.cond_layers = torch.nn.ModuleList([torch.nn.ModuleList([])])
        self.cond_downsample = torch.nn.ModuleList([])
        current_resolution = h
        for prev_dim, dim in zip([hidden_dims[0]] + list(hidden_dims), hidden_dims):
            if prev_dim != dim:
                current_resolution //= 2
                self.cond_downsample.append(DownSample2d(prev_dim, dim, kernel_size=kernel_size))
                self.cond_layers.append(torch.nn.ModuleList([]))
            need_attn = current_resolution <= attention_dim
            block = ConditionalNormResBlock2D(hidden_dim=dim, embed_dim=self.emb_features, num_groups=num_groups,
                                              kernel_size=kernel_size, attn=need_attn, dropout=dropout,
                                              num_heads=num_heads)
            self.cond_layers[-1].append(block)
        self.cond_downsample.append(torch.nn.Identity())

        # Main encoder blocks
        self.encoder_layers = torch.nn.ModuleList([torch.nn.ModuleList([])])
        self.downsample_blocks = torch.nn.ModuleList([])
        current_resolution = h
        for prev_dim, dim in zip([hidden_dims[0]] + list(hidden_dims), hidden_dims):
            if prev_dim != dim:
                current_resolution //= 2
                self.downsample_blocks.append(DownSample2d(prev_dim, dim, kernel_size=kernel_size))
                self.encoder_layers.append(torch.nn.ModuleList([]))
            need_attn = current_resolution <= attention_dim
            block = TemporalCondResBlock2d(dim, embed_dim=self.emb_features, kernel_size=kernel_size,
                                           num_heads=num_heads, num_groups=num_groups, attn=need_attn, dropout=dropout,
                                           cross_cond=cond is not None)
            self.encoder_layers[-1].append(block)
        self.downsample_blocks.append(torch.nn.Identity())

        self.mid_layers = torch.nn.Module()
        need_attn = current_resolution <= attention_dim
        self.mid_layers.block_1 = TemporalCondResBlock2d(hidden_dim=hidden_dims[-1], embed_dim=self.emb_features,
                                                         num_groups=num_groups, kernel_size=kernel_size,
                                                         cross_cond=cond is not None,
                                                         attn=need_attn, dropout=dropout, num_heads=num_heads)

        # add extra res blocks for upsampling as it harder than downsampling
        _inverse_dims = hidden_dims[::-1]
        inverse_dims = []
        for dim_id in range(len(_inverse_dims)):
            inverse_dims.append(_inverse_dims[dim_id])
            if dim_id == 0 or _inverse_dims[dim_id] != _inverse_dims[dim_id - 1]:
                inverse_dims += [_inverse_dims[dim_id] for _ in range(extra_upsample_blocks)]
        self.decoder_layers = torch.nn.ModuleList([torch.nn.ModuleList([])])
        self.upsample_blocks = torch.nn.ModuleList([])
        for prev_dim, dim in zip([inverse_dims[0]] + list(inverse_dims), inverse_dims):
            if prev_dim != dim:
                current_resolution *= 2
                self.upsample_blocks.append(UpSample2d(prev_dim, dim, kernel_size=kernel_size))
                self.decoder_layers.append(torch.nn.ModuleList([]))
            need_attn = current_resolution <= attention_dim
            block = TemporalCondResBlock2d(dim, in_dim=2 * dim, embed_dim=self.emb_features, kernel_size=kernel_size,
                                           num_heads=num_heads, num_groups=num_groups, attn=need_attn, dropout=dropout,
                                           cross_cond=cond is not None)
            self.decoder_layers[-1].append(block)
        self.upsample_blocks.append(torch.nn.Identity())

        # Out latent prediction
        self.out_norm = norm(hidden_dims[0], num_groups=num_groups)
        self.out_mapper = torch.nn.Conv2d(hidden_dims[0], features, kernel_size=kernel_size,
                                          padding=kernel_size // 2)

    def forward(self, x, time, cond=None):

        b, temporal_dim, in_channels, height, width = x.shape
        x = x.view(b * temporal_dim, in_channels, height, width)

        h_time = get_timestep_encoding(time, self.emb_features // 4, self.steps)
        h_temporal = get_timestep_encoding(torch.arange(temporal_dim).type_as(time),
                                           self.emb_features // 4, temporal_dim)
        h_time = torch.cat([repeat_dim(h_time[:, None, :], 1, temporal_dim),
                            repeat_dim(h_temporal[None, :, :], 0, b)], dim=2)
        h_time = self.time_layers[1](nonlinear(self.time_layers[0](h_time))).view(b * temporal_dim, self.emb_features)

        # Map condition image to latents
        h_cond = self.cond_input(cond)
        conds = []
        for blocks, downsample in zip(self.cond_layers, self.cond_downsample):
            emb = h_time[:1, :, None, None]
            for block in blocks:
                h_cond = block(h_cond, emb)
            conds.append(h_cond)
            h_cond = downsample(h_cond)

        # Prepare input for mapping
        h = self.input_mapper(x)
        outs = []
        for blocks, downsample, h_cond in zip(self.encoder_layers, self.downsample_blocks, conds):
            emb = h_time[:, :, None, None]
            h_cond = repeat_dim(h_cond.unsqueeze(1), 1, temporal_dim)
            h_cond = h_cond.view(b * temporal_dim, *h_cond.shape[2:])
            for block in blocks:
                h = block(h, temporal_dim, emb, cond=h_cond)
                outs.append(h)
            h = downsample(h)

        # Mid mapping
        emb = h_time[:, :, None, None]
        h_cond = repeat_dim(conds[-1].unsqueeze(1), 1, temporal_dim)
        h_cond = h_cond.view(b * temporal_dim, *h_cond.shape[2:])
        h = self.mid_layers.block_1(h, temporal_dim, emb, cond=h_cond)

        # Decode latent
        for blocks, upsample, h_cond in zip(self.decoder_layers, self.upsample_blocks, reversed(conds)):
            emb = h_time[:, :, None, None]
            h_cond = repeat_dim(h_cond.unsqueeze(1), 1, temporal_dim)
            h_cond = h_cond.view(b * temporal_dim, *h_cond.shape[2:])
            for block in blocks:
                h = block(torch.cat([h, outs.pop()], dim=1), temporal_dim, emb, cond=h_cond)
            h = upsample(h)

        h = nonlinear(self.out_norm(h))
        out = self.out_mapper(h)

        return out.view(b, temporal_dim, in_channels, height, width)
