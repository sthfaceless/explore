import torch.nn

from modules.common.model import *
class TemporalAttention2d(MHAAttention2D):
    def forward(self, q, t=1, v=None):
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


class LocalSparseCausalAttention2d(LocalMHAAttention2D):
    def forward(self, h, v=None, t=1):
        bt, c, hh, ww = h.shape
        ht = h.reshape(bt // t, t, c, hh, ww)
        kv1 = repeat_dim(ht[:, :1], 1, t)  # first frame of each batch
        kv2 = torch.cat([ht[:, :1], ht[:, :-1]], dim=1)  # previous frame for each one
        kv = torch.cat([kv1, kv2], dim=2).reshape(bt, 2 * c, hh, ww)
        return super().forward(q=h, v=kv)


class SparseCausalAttention2d(MHAAttention2D):
    def forward(self, h, v=None, t=1):
        bt, c, hh, ww = h.shape
        ht = h.reshape(bt // t, t, c, hh, ww)
        kv1 = repeat_dim(ht[:, :1], 1, t)  # first frame of each batch
        kv2 = torch.cat([ht[:, :1], ht[:, :-1]], dim=1)  # previous frame for each one
        kv = torch.cat([kv1, kv2], dim=2).reshape(bt, 2 * c, hh, ww)
        return super().forward(q=h, v=kv)


class PseudoConv3d(torch.nn.Conv1d):
    def forward(self, x, t=1):
        bt, c, h, w = x.shape
        # bt c h w -> b t c hw -> b hw c t -> bhw c t
        x = x.reshape(bt // t, t, c, h * w).transpose(-1, -3).reshape(bt // t * h * w, c, t)
        x = super().forward(x)
        # bhw c t -> b hw c t -> b t c hw -> bt c h w
        x = x.reshape(bt // t, h * w, c, t).transpose(-1, -3).reshape(bt, c, h, w)
        return x


class TemporalCondResBlock2d(torch.nn.Module):
    def __init__(self, hidden_dim, embed_dim, kernel_size=3, num_groups=32, in_dim=-1, attn=False, local_attn=False,
                 local_attn_patch=8, dropout=0.0, num_heads=4):
        super(TemporalCondResBlock2d, self).__init__()

        if in_dim == -1:
            in_dim = hidden_dim
        else:
            self.res_mapper = torch.nn.Conv2d(in_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.ln_1 = norm(in_dim, num_groups)
        self.layer_1 = torch.nn.Conv2d(in_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.tlayer_1 = PseudoConv3d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conditional_norm = ConditionalNorm2D(hidden_dim, embed_dim, num_groups)
        self.dropout = torch.nn.Dropout2d(p=dropout)
        self.layer_2 = torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.tlayer_2 = PseudoConv3d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2)

        self.temp_attn = TemporalAttention2d(hidden_dim, num_groups=num_groups, num_heads=num_heads)
        self.need_attn = attn or local_attn
        if attn:
            self.sc_attn = SparseCausalAttention2d(hidden_dim, k_dim=hidden_dim * 2, num_groups=num_groups, num_heads=num_heads)
            self.cross_attn = MHAAttention2D(hidden_dim, num_groups=num_groups, num_heads=num_heads)
        elif local_attn:
            self.sc_attn = LocalSparseCausalAttention2d(hidden_dim, k_dim=hidden_dim * 2, num_groups=num_groups,
                                                        num_heads=num_heads)
            self.cross_attn = LocalMHAAttention2D(hidden_dim, num_groups=num_groups, num_heads=num_heads,
                                                  patch_size=local_attn_patch)

    def forward(self, x, t, emb, cond=None):
        h = self.layer_1(nonlinear(self.ln_1(x)))
        h = self.tlayer_1(h)
        h = self.conditional_norm(h, emb)
        h = self.layer_2(self.dropout(nonlinear(h)))
        h = self.tlayer_2(h)
        skip = x if self.in_dim == self.hidden_dim else self.res_mapper(x)
        h = (h + skip) / 2 ** (1 / 2)
        if self.need_attn:
            h = self.sc_attn(h, t=t)
            if cond is not None:
                h = self.cross_attn(q=h, v=cond)
            h = self.temp_attn(h, t=t)
        return h


class TemporalUNetDenoiser(torch.nn.Module):

    def __init__(self, shape, steps, kernel_size=3, hidden_dims=(128, 256, 256, 512), attention_dim=32,
                 num_groups=32, dropout=0.0, num_heads=4, embed_features=512, cond='cross',
                 local_attn_dim=64, local_attn_patch=8, extra_upsample_blocks=1):
        super(TemporalUNetDenoiser, self).__init__()
        features, h, w = shape
        self.features = features
        self.steps = steps
        self.cond = cond
        self.extra_upsample_blocks = extra_upsample_blocks
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
        if cond == 'cross':
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
                block = ConditionalNormResBlock2D(hidden_dim=dim, embed_dim=self.emb_features, num_groups=num_groups,
                                                  kernel_size=kernel_size, attn=False, dropout=dropout,
                                                  num_heads=num_heads)
                self.cond_layers[-1].append(block)
            self.cond_downsample.append(torch.nn.Identity())

        # Main encoder blocks
        self.encoder_layers = torch.nn.ModuleList([torch.nn.ModuleList([])])
        self.downsample_blocks = torch.nn.ModuleList([])
        current_resolution = h
        encoder_block_count = {hidden_dims[0]: 0}
        for prev_dim, dim in zip([hidden_dims[0]] + list(hidden_dims), hidden_dims):
            if prev_dim != dim:
                current_resolution //= 2
                encoder_block_count[dim] = 0
                self.downsample_blocks.append(DownSample2d(prev_dim, dim, kernel_size=kernel_size))
                self.encoder_layers.append(torch.nn.ModuleList([]))
            block = TemporalCondResBlock2d(dim, embed_dim=self.emb_features, kernel_size=kernel_size,
                                           num_heads=num_heads, num_groups=num_groups,
                                           attn=current_resolution <= attention_dim, dropout=dropout,
                                           local_attn=current_resolution <= local_attn_dim,
                                           local_attn_patch=local_attn_patch)
            self.encoder_layers[-1].append(block)
            encoder_block_count[dim] += 1
        self.downsample_blocks.append(torch.nn.Identity())

        self.mid_layers = torch.nn.Module()
        self.mid_layers.block_1 = TemporalCondResBlock2d(hidden_dim=hidden_dims[-1], embed_dim=self.emb_features,
                                                         num_groups=num_groups, kernel_size=kernel_size,
                                                         attn=current_resolution <= attention_dim,
                                                         local_attn=current_resolution <= local_attn_dim,
                                                         local_attn_patch=local_attn_patch,
                                                         dropout=dropout, num_heads=num_heads)

        # add extra res blocks for upsampling as it harder than downsampling
        _inverse_dims = hidden_dims[::-1]
        inverse_dims = []
        for dim_id in range(len(_inverse_dims)):
            inverse_dims.append(_inverse_dims[dim_id])
            if dim_id == 0 or _inverse_dims[dim_id] != _inverse_dims[dim_id - 1]:
                inverse_dims += [_inverse_dims[dim_id] for _ in range(extra_upsample_blocks)]
        self.decoder_layers = torch.nn.ModuleList([torch.nn.ModuleList([])])
        self.upsample_blocks = torch.nn.ModuleList([])
        decoder_block_count = {inverse_dims[0]: 0}
        for prev_dim, dim in zip([inverse_dims[0]] + list(inverse_dims), inverse_dims):
            if prev_dim != dim:
                current_resolution *= 2
                decoder_block_count[dim] = 0
                self.upsample_blocks.append(UpSample2d(prev_dim, dim, kernel_size=kernel_size))
                self.decoder_layers.append(torch.nn.ModuleList([]))
            in_dim = 2 * dim if decoder_block_count[dim] < encoder_block_count[dim] else dim
            block = TemporalCondResBlock2d(dim, in_dim=in_dim, embed_dim=self.emb_features, kernel_size=kernel_size,
                                           num_heads=num_heads, num_groups=num_groups,
                                           local_attn=current_resolution <= local_attn_dim,
                                           local_attn_patch=local_attn_patch,
                                           attn=current_resolution <= attention_dim, dropout=dropout)
            self.decoder_layers[-1].append(block)
            decoder_block_count[dim] += 1
        self.upsample_blocks.append(torch.nn.Identity())

        # Out latent prediction
        self.out_norm = norm(hidden_dims[0], num_groups=num_groups)
        self.out_mapper = torch.nn.Conv2d(hidden_dims[0], features, kernel_size=kernel_size,
                                          padding=kernel_size // 2)

    def forward(self, x, time, cond=None):

        if self.cond == 'concat':
            x = torch.cat([cond.unsqueeze(1), x], dim=1)

        b, temporal_dim, in_channels, height, width = x.shape
        x = x.view(b * temporal_dim, in_channels, height, width)

        h_time = get_timestep_encoding(time, self.emb_features // 4, self.steps)
        h_temporal = get_timestep_encoding(torch.arange(temporal_dim).type_as(time),
                                           self.emb_features // 4, temporal_dim)
        h_time = torch.cat([repeat_dim(h_time[:, None, :], 1, temporal_dim),
                            repeat_dim(h_temporal[None, :, :], 0, b)], dim=2)
        h_time = self.time_layers[1](nonlinear(self.time_layers[0](h_time))).view(b * temporal_dim, self.emb_features)

        conds = []
        if self.cond == 'cross':
            # Map condition image to latents
            h_cond = self.cond_input(cond)
            for blocks, downsample in zip(self.cond_layers, self.cond_downsample):
                emb = h_time[:1, :, None, None]
                for block in blocks:
                    h_cond = block(h_cond, emb)
                conds.append(h_cond)
                h_cond = downsample(h_cond)

        # Prepare input for mapping
        h = self.input_mapper(x)
        outs = []
        for layer_id, (blocks, downsample) in enumerate(zip(self.encoder_layers, self.downsample_blocks)):
            if self.cond == 'cross':
                h_cond = repeat_dim(conds[layer_id].unsqueeze(1), 1, temporal_dim)
                h_cond = h_cond.view(b * temporal_dim, *h_cond.shape[2:])
            else:
                h_cond = None
            emb = h_time[:, :, None, None]
            for block in blocks:
                h = block(h, temporal_dim, emb, cond=h_cond)
                outs.append(h)
            h = downsample(h)

        # Mid mapping
        if self.cond == 'cross':
            h_cond = repeat_dim(conds[-1].unsqueeze(1), 1, temporal_dim)
            h_cond = h_cond.view(b * temporal_dim, *h_cond.shape[2:])
        else:
            h_cond = None
        emb = h_time[:, :, None, None]
        h = self.mid_layers.block_1(h, temporal_dim, emb, cond=h_cond)

        # Decode latent
        for layer_id, (blocks, upsample) in enumerate(zip(self.decoder_layers, self.upsample_blocks)):
            emb = h_time[:, :, None, None]
            if self.cond == 'cross':
                h_cond = repeat_dim(conds[-(layer_id + 1)].unsqueeze(1), 1, temporal_dim)
                h_cond = h_cond.view(b * temporal_dim, *h_cond.shape[2:])
            else:
                h_cond = None
            for block_id, block in enumerate(blocks):
                if len(blocks) - block_id > self.extra_upsample_blocks:
                    h = torch.cat([h, outs.pop()], dim=1)
                h = block(h, temporal_dim, emb, cond=h_cond)
            h = upsample(h)

        h = nonlinear(self.out_norm(h))
        out = self.out_mapper(h)

        out = out.view(b, temporal_dim, in_channels, height, width)
        if self.cond == 'concat':
            out = out[:, 1:]

        return out