import math

import torch.nn

from modules.common.model import *
from modules.common.util import *
from modules.gen.model import *
from modules.nerf.util import render_gallery, positional_points_encoding


class DensityGrid:

    def __init__(self, res, decay_rate, dens_threshold, spp, frac=1.0, scene_scale=1.0, device=torch.device('cpu')):
        self.res = res
        self.dens_threshold = dens_threshold * spp
        self.decay_rate = decay_rate
        self.frac = frac
        self.scene_scale = scene_scale
        self.dgrid = torch.ones(res, res, res, device=device, dtype=torch.float32) * self.dens_threshold * 2
        self.dmask = torch.ones(res, res, res, device=device, dtype=torch.bool)

    def update_density_grid(self, model, pe_powers, stratified=False):
        self.dgrid *= self.decay_rate
        samples = int(self.res ** 3 * self.frac)

        points = torch.rand(samples, 3, device=self.dgrid.device)

        indexes = torch.tensor_split(torch.floor(points * self.res).long(), 3, dim=1)

        points = (points - 0.5) * self.scene_scale
        x = positional_points_encoding(points, pe_powers)  # (samples, 3*pe_features)
        with torch.no_grad():
            _, dens = model.forward(x)  # (samples, 1)

        self.dgrid[indexes] = torch.maximum(self.dgrid[indexes], dens)
        self.dmask = (self.dgrid >= self.dens_threshold)

    def important_intervals(self, dists, ray_o, ray_d):
        points = ray_o.unsqueeze(1) + ray_d.unsqueeze(1) * dists.unsqueeze(-1)

        normalized_points = torch.clip(points, - self.scene_scale / 2 + 1e-5,
                                       self.scene_scale / 2 - 1e-5) / self.scene_scale + 0.5
        int_indexes = torch.floor(normalized_points * self.res).long()
        msk_same = torch.all(torch.eq(int_indexes[:, :-1], int_indexes[:, 1:]), dim=-1).view(-1)  # (b * spp, )

        points_inside = (torch.max(torch.abs(points), dim=-1)[0] < self.scene_scale / 2 - 1e-5)
        msk_inside = torch.all(torch.stack([points_inside[:, :-1], points_inside[:, 1:]], dim=-1), dim=-1).view(-1)

        grid_indexes = torch.tensor_split(int_indexes[:, :-1].reshape(-1, 3), 3, dim=1)
        msk = torch.where(torch.logical_and(msk_same, msk_inside), self.dmask[grid_indexes].squeeze(-1), False)
        nonzero_indexes = torch.nonzero(msk, as_tuple=True)[0]

        return nonzero_indexes


class Nerf(torch.nn.Module):

    def __init__(self, hidden_dim=128, num_blocks=3, pe_powers=16,
                 transmittance_weight=0.1, transmittance_threshold=0.88):
        super(Nerf, self).__init__()
        self.pe_powers = pe_powers
        self.transmittance_weight = transmittance_weight
        self.transmittance_threshold = transmittance_threshold
        input_dim = pe_powers * 6

        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.LeakyReLU()
        )

        self.blocks = torch.nn.ModuleList([])
        for block_id in range(num_blocks):
            self.blocks.append(torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.LeakyReLU()
                ),
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim, hidden_dim)
                )
            ]))

        self.rgb_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 3),
            torch.nn.Sigmoid()
        )
        self.density_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, points):
        n_points, features = points.shape
        x = self.input_layer(points)
        for first_block, second_block in self.blocks:
            out = first_block(x)
            out = second_block(out)
            x = torch.nn.functional.leaky_relu((out + x) / 2 ** (1 / 2))
        rgb = self.rgb_layer(x)
        raw_dens = self.density_layer(x)
        density = torch.nn.functional.softplus(raw_dens - 1)  # shifted softplus
        return rgb, density

    def loss(self, coarse_pixels, fine_pixels, gt_pixels, coarse_weight, transmittance):
        out = {
            'coarse_loss': torch.nn.functional.mse_loss(coarse_pixels, gt_pixels),
            'fine_loss': torch.nn.functional.mse_loss(fine_pixels, gt_pixels),
            'trans_loss': - torch.minimum(torch.tensor(self.transmittance_threshold, dtype=torch.float32,
                                                       device=transmittance.device), torch.mean(transmittance))
        }
        out['loss'] = out['coarse_loss'] * coarse_weight + out['fine_loss'] + \
                      out['trans_loss'] * self.transmittance_weight
        return out

    def render_views(self, spp, batch_size, w=128, h=128, focal=1.5, camera_distance=4.0, dgrid=None,
                     device=torch.device('cpu')):
        return render_gallery(model=self, spp=spp, pe_powers=self.pe_powers,
                              batch_size=batch_size, w=w, h=h, focal=focal, camera_distance=camera_distance,
                              dgrid=dgrid, device=device)


class ConditionalNeRF(torch.nn.Module):

    def __init__(self, latent_dim, hidden_dim=32, num_blocks=4, pe_powers=12):
        super(ConditionalNeRF, self).__init__()
        self.input_dim = pe_powers * 6 + latent_dim
        self.blocks = torch.nn.ModuleList([])
        self.input_layer = torch.nn.Linear(self.input_dim, hidden_dim)
        for block_id in range(num_blocks):
            self.blocks.append(torch.nn.ModuleList([
                torch.nn.Linear(hidden_dim + latent_dim, hidden_dim),
                torch.nn.Linear(hidden_dim + latent_dim, hidden_dim),
                torch.nn.Linear(hidden_dim, hidden_dim)
            ]))
        self.density_layer = torch.nn.Linear(hidden_dim, 1)
        self.rgb_layer = torch.nn.Linear(hidden_dim, 3)

    def forward(self, input, latent):
        x = torch.cat([input, latent], dim=-1)
        x = self.input_layer(x)
        for mapper, block_1, block_2 in self.blocks:
            x = torch.cat([x, latent], dim=-1)
            h = torch.sin(x)
            h = block_1(h)
            h = torch.sin(h)
            h = block_2(h)
            x = (h + mapper(x)) / 2 ** (1 / 2)
        rgb = torch.sigmoid(self.rgb_layer(x))
        density = torch.nn.functional.softplus(self.density_layer(x) - 1)
        return rgb, density


class HiddenConditionalNeRF(torch.nn.Module):

    def __init__(self, latent_dim, hidden_dim=32, num_blocks=4, pe_powers=12):
        super(HiddenConditionalNeRF, self).__init__()
        self.input_dim = pe_powers * 6 + latent_dim
        self.blocks = torch.nn.ModuleList([])
        self.input_layer = torch.nn.Linear(self.input_dim, hidden_dim)
        for block_id in range(num_blocks):
            self.blocks.append(torch.nn.ModuleList([
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.Linear(hidden_dim + latent_dim, hidden_dim)
            ]))
        self.density_layer = torch.nn.Linear(hidden_dim, 1)
        self.rgb_layer = torch.nn.Linear(hidden_dim, 3)

    def forward(self, input, latent):
        x = torch.cat([input, latent], dim=-1)
        x = self.input_layer(x)
        for block_1, block_2 in self.blocks:
            h = torch.sin(x)
            h = torch.cat([block_1(h), latent], dim=-1)
            h = torch.sin(h)
            h = block_2(h)
            x = (h + x) / 2 ** (1 / 2)
        rgb = torch.sigmoid(self.rgb_layer(x))
        density = torch.nn.functional.softplus(self.density_layer(x) - 1)
        return rgb, density


class NerfLatentDecoder(torch.nn.Module):

    def __init__(self, latent_shape, out_dim, kernel_size=3, hidden_dims=(32, 64, 128, 256), attention_dim=16,
                 num_groups=32):
        super(NerfLatentDecoder, self).__init__()
        self.latent_shape = latent_shape
        self.hidden_dims = hidden_dims

        self.input_layer = torch.nn.Sequential(
            torch.nn.Conv2d(latent_shape[0], hidden_dims[0], kernel_size=kernel_size, padding=kernel_size // 2),
            ResBlock2d(hidden_dims[0], kernel_size=kernel_size, num_groups=num_groups),
            Attention2D(hidden_dims[0], num_groups=num_groups),
            ResBlock2d(hidden_dims[0], kernel_size=kernel_size, num_groups=num_groups)
        )
        current_resolution = latent_shape[1]
        blocks = []
        for prev_dim, dim in zip(hidden_dims, hidden_dims[1:]):
            if prev_dim != dim:
                blocks.append(UpSample2d(prev_dim, dim))
                current_resolution *= 2
            blocks.append(ResBlock2d(dim, kernel_size, num_groups=num_groups))
            if current_resolution <= attention_dim:
                blocks.append(Attention2D(dim, num_groups=num_groups))
        self.model = torch.nn.Sequential(*blocks,
                                         norm(num_groups=num_groups, dims=hidden_dims[-1]),
                                         torch.nn.SiLU(),
                                         torch.nn.Conv2d(hidden_dims[-1], out_dim, kernel_size=kernel_size,
                                                         padding=kernel_size // 2))

    def forward(self, x):
        out = self.input_layer(x)
        out = self.model(out)
        return out


class NerfVAE(VAE):

    def __init__(self, layers, shapes, latent_dim=256, kl_weight=5 * 1e-4, l1=1e-6, disc_weight=1e-2,
                 latent_noise=0.0):
        super(NerfVAE, self).__init__(latent_dim, latent_noise)

        # set loss parameters
        self.kl_weight = kl_weight
        self.disc_weight = disc_weight
        self.l1 = l1

        # nerf weights layout
        self.layers = [layer.replace(".", "_") for layer in layers]
        self.shapes = shapes
        self.offsets = [0]
        self.sizes = []
        for shape in shapes:
            size = 1
            for dim in shape:
                size *= dim
            self.sizes.append(size)
            self.offsets.append(size + self.offsets[-1])

    def build_encoder(self, input_dim, hidden_dims, reverse=False):
        __hidden_dims = hidden_dims if not reverse else hidden_dims[::-1]
        blocks = []
        for prev_dim, dim in zip([input_dim] + list(__hidden_dims), __hidden_dims):
            if prev_dim != dim:
                blocks.append(DownSample(prev_dim, dim))
            blocks.append(ResBlock(dim))

        return torch.nn.Sequential(*blocks)

    def loss(self, data, disc_loss=None):
        pred, x, mu, logsigma, latent = data

        out = {
            'rec_loss': torch.nn.functional.mse_loss(pred, x['weights']),
            'kl_loss': self.kl_loss(mu, logsigma),
            'latent_reg': torch.norm(latent, p=1)
        }
        out['loss'] = out['rec_loss']
        out['loss'] += out['kl_loss'] * self.kl_weight
        out['loss'] += out['latent_reg'] * self.l1
        if disc_loss is not None:
            out['disc_loss'] = - disc_loss * self.disc_weight
            out['loss'] += out['disc_loss']

        return out


class NerfVAEBlocks(NerfVAE):

    def __init__(self, *args, hidden_dims=(1024, 512, 256), **kwargs):
        super(NerfVAEBlocks, self).__init__(*args, **kwargs)

        # creating model
        out_dims = []
        self.encoders = torch.nn.ModuleDict({})
        self.decoders = torch.nn.ModuleDict({})
        for layer_id, layer in enumerate(self.layers):
            self.encoders[layer] = self.build_encoder(self.sizes[layer_id], hidden_dims)
            out_dims.append(hidden_dims[-1])
            self.decoders[layer] = torch.nn.Sequential(
                self.build_encoder(self.latent_dim, hidden_dims, reverse=True),
                torch.nn.Linear(hidden_dims[0], self.sizes[layer_id]),
                torch.nn.Tanh()
            )

        out_dim = sum(out_dims)
        self.mu_network = torch.nn.Sequential(
            torch.nn.Linear(out_dim, self.latent_dim)
        )
        self.sigma_network = torch.nn.Sequential(
            torch.nn.Linear(out_dim, self.latent_dim)
        )

    def encode(self, x):
        embeds = []
        for layer, start, end in zip(self.layers, self.offsets[:-1], self.offsets[1:]):
            embeds.append(self.encoders[layer](x['weights'][:, start:end]))
        embed = torch.cat(embeds, dim=1)
        mu = self.mu_network(embed)
        logsigma = self.sigma_network(embed)
        return mu, logsigma

    def decode(self, latent):
        weights = []
        for layer in self.layers:
            weights.append(self.decoders[layer](latent))
        pred = torch.cat(weights, dim=1)
        return pred


class NerfVAELinear(NerfVAE):

    def __init__(self, *args, hidden_dims=(1024, 512, 256), **kwargs):
        super(NerfVAELinear, self).__init__(*args, **kwargs)

        # creating model
        self.encoder = self.build_encoder(self.offsets[-1], hidden_dims)
        self.decoder = torch.nn.Sequential(
            self.build_encoder(self.latent_dim, hidden_dims, reverse=True),
            torch.nn.Linear(hidden_dims[0], self.offsets[-1]),
            torch.nn.Tanh()
        )

        self.mu_network = torch.nn.Sequential(
            torch.nn.Linear(hidden_dims[-1], self.latent_dim)
        )
        self.sigma_network = torch.nn.Sequential(
            torch.nn.Linear(hidden_dims[-1], self.latent_dim)
        )

    def encode(self, x):
        x = x['weights']
        embed = self.encoder(x)
        mu = self.mu_network(embed)
        logsigma = self.sigma_network(embed)
        return mu, logsigma

    def decode(self, latent):
        return self.decoder(latent)


class NerfVAEConv1D(NerfVAE):

    def __init__(self, *args, hidden_dims=(1024, 512, 256), kernel_size=32, **kwargs):
        super(NerfVAEConv1D, self).__init__(*args, **kwargs)
        self.hidden_dims = hidden_dims

        # build model
        self.encoder, self.down_length = self.build_conv_encoder(self.offsets[-1], 1, hidden_dims,
                                                                 kernel_size=kernel_size)
        self.mu_network = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dims[-1] * self.down_length[-1], self.latent_dim)
        )
        self.sigma_network = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dims[-1] * self.down_length[-1], self.latent_dim)
        )
        self.decoder, self.up_length = self.build_conv_encoder(self.latent_dim, 1, hidden_dims,
                                                               kernel_size=kernel_size, reverse=True)
        self.out_layer = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dims[0] * self.up_length[-1], self.offsets[-1]),
            torch.nn.Tanh()
        )

    def build_conv_encoder(self, length, input_dim, hidden_dims, kernel_size=32, reverse=False):
        __hidden_dims = hidden_dims if not reverse else hidden_dims[::-1]
        blocks, lens = [], [length]
        for prev_dim, dim in zip([input_dim] + list(__hidden_dims), __hidden_dims):
            if prev_dim != dim:
                blocks.append(DownSampleConv1D(prev_dim, dim, reverse=reverse, kernel_size=kernel_size))
                if not reverse:
                    lens.append(int(math.floor(lens[-1] / 2)))
                else:
                    lens.append(lens[-1] * 2)
            blocks.append(ResBlockConv1D(dim, kernel_size=kernel_size))

        return torch.nn.Sequential(*blocks), lens

    def encode(self, x):
        x = x['weights']
        embed = self.encoder(x.unsqueeze(1)).view(-1, self.hidden_dims[-1] * self.down_length[-1])  # (b features)
        mu = self.mu_network(embed)
        logsigma = self.sigma_network(embed)
        return mu, logsigma

    def decode(self, latent):
        emb = self.decoder(latent.unsqueeze(1)).view(-1, self.hidden_dims[0] * self.up_length[-1])
        return self.out_layer(emb)


class NerfWeightsDiscriminator(torch.nn.Module):

    def __init__(self, layers, shapes):
        super(NerfWeightsDiscriminator, self).__init__()
        self.layers = [layer.replace(".", "_") for layer in layers]
        self.shapes = shapes
        self.offsets = [0]
        self.sizes = []
        for shape in shapes:
            size = 1
            for dim in shape:
                size *= dim
            self.sizes.append(size)
            self.offsets.append(size + self.offsets[-1])

    def build_encoder(self, input_dim, hidden_dims):
        blocks = []
        for prev_dim, dim in zip([input_dim] + list(hidden_dims), hidden_dims):
            if prev_dim != dim:
                blocks.append(DownSample(prev_dim, dim))
            blocks.append(ResBlock(dim))

        return torch.nn.Sequential(*blocks)

    def forward(self, x):
        raise NotImplementedError

    def loss(self, pred, label):
        loss = - torch.mean(label * torch.log(pred + 1e-6) + (1 - label) * torch.log(1 - pred + 1e-6))
        return loss


class NerfWeightsBlockDiscriminator(NerfWeightsDiscriminator):

    def __init__(self, *args, hidden_dims=(256, 128, 64), **kwargs):
        super(NerfWeightsBlockDiscriminator, self).__init__(*args, **kwargs)

        # build model
        self.encoders = torch.nn.ModuleDict({})
        out_dims = []
        for layer_id, layer in enumerate(self.layers):
            self.encoders[layer] = self.build_encoder(self.sizes[layer_id], hidden_dims)
            out_dims.append(hidden_dims[-1])
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(sum(out_dims), 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        embeds = []
        for layer, start, end in zip(self.layers, self.offsets[:-1], self.offsets[1:]):
            embeds.append(self.encoders[layer](x[:, start:end]))
        embed = torch.cat(embeds, dim=1)
        out = self.predictor(embed)
        return out


class NerfWeightsLinearDiscriminator(NerfWeightsDiscriminator):

    def __init__(self, *args, hidden_dims=(256, 128, 64), **kwargs):
        super(NerfWeightsLinearDiscriminator, self).__init__(*args, **kwargs)

        # build model
        self.encoder = self.build_encoder(self.offsets[-1], hidden_dims)
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dims[-1], 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        embed = self.encoder(x)
        out = self.predictor(embed)
        return out


class XUnetBlock(torch.nn.Module):

    def __init__(self, block, dim, num_groups=32, need_attn=False, cond='xunet', num_heads=4):
        super(XUnetBlock, self).__init__()
        self.block = block
        self.need_attn = need_attn
        if self.need_attn and cond == 'xunet':
            self.cross_attn = MHAAttention2D(dim, num_groups=num_groups, num_heads=num_heads, cross=True)

    def forward(self, x, emb):
        h = self.block(x, emb)
        if self.need_attn:
            h_orig, h_cond = torch.chunk(h, 2, dim=0)
            h_orig = self.cross_attn(q=h_orig, v=h_cond)
            h_cond = self.cross_attn(q=h_cond, v=h_orig)
            h = torch.cat([h_orig, h_cond], dim=0)
        return h


class XUNetDenoiser(torch.nn.Module):

    def __init__(self, shape, steps, kernel_size=3, hidden_dims=(128, 256, 256, 512), attention_dim=32,
                 num_groups=32, dropout=0.0, num_heads=4, time_features=1024, pos_enc=16,
                 extra_upsample_blocks=1, cond='xunet'):
        super(XUNetDenoiser, self).__init__()
        self.cond = cond
        features, h, w = shape
        self.features = features
        self.steps = steps
        # Input mapping
        input_features = 2 * features if cond == 'concat' else features
        self.input_mapper = torch.nn.Conv2d(input_features, hidden_dims[0], kernel_size=kernel_size,
                                            padding=kernel_size // 2)

        # Time embeddings
        self.time_features = time_features
        self.time_layers = torch.nn.ModuleList([
            torch.nn.Linear(time_features, time_features),
            torch.nn.Linear(time_features, time_features),
        ])
        # Positional embeddings
        self.pos_features = pos_enc * 3 * (2 if cond == 'concat' else 1)
        if cond == 'xunet':
            self.embed_cond = torch.nn.Embedding(num_embeddings=2, embedding_dim=self.pos_features)
        elif cond == 'concat':
            self.embed_cond = torch.nn.Parameter(torch.zeros(self.pos_features, requires_grad=True))
            torch.nn.init.xavier_uniform_(self.embed_cond.data)
        else:
            raise NotImplementedError(f'Unknown conditional embedding {cond}')
        self.embed_pixel = torch.nn.Parameter(torch.zeros(self.pos_features, shape[1], shape[2], requires_grad=True))
        torch.nn.init.xavier_uniform_(self.embed_pixel.data)
        self.emb_maps = torch.nn.ModuleList([torch.nn.Conv2d(self.pos_features, self.emb_features,
                                                             kernel_size=kernel_size, stride=kernel_size // 2)])

        self.encoder_layers = torch.nn.ModuleList([torch.nn.ModuleList([])])
        self.downsample_blocks = torch.nn.ModuleList([])
        current_resolution = h
        for prev_dim, dim in zip([hidden_dims[0]] + list(hidden_dims), hidden_dims):
            if prev_dim != dim:
                current_resolution //= 2
                self.emb_maps.append(DownSample2d(self.pos_features, self.emb_features,
                                                  kernel_size=kernel_size,
                                                  scale_factor=current_resolution / h))
                self.downsample_blocks.append(DownSample2d(prev_dim, dim, kernel_size=kernel_size))
                self.encoder_layers.append(torch.nn.ModuleList([]))
            need_attn = current_resolution <= attention_dim
            block = ConditionalNormResBlock2D(hidden_dim=dim, embed_dim=self.emb_features, num_groups=num_groups,
                                              kernel_size=kernel_size, attn=need_attn, dropout=dropout,
                                              num_heads=num_heads)
            block = XUnetBlock(block, dim=dim, num_groups=num_groups, need_attn=need_attn, num_heads=num_heads,
                               cond=cond)
            self.encoder_layers[-1].append(block)
        self.downsample_blocks.append(torch.nn.Identity())

        self.mid_layers = torch.nn.Module()
        need_attn = current_resolution <= attention_dim
        block1 = ConditionalNormResBlock2D(hidden_dim=hidden_dims[-1], embed_dim=self.emb_features,
                                           num_groups=num_groups, kernel_size=kernel_size,
                                           attn=need_attn, dropout=dropout, num_heads=num_heads)
        self.mid_layers.block_1 = XUnetBlock(block1, dim=hidden_dims[-1], num_groups=num_groups, need_attn=need_attn,
                                             num_heads=num_heads, cond=cond)

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
            block = ConditionalNormResBlock2D(hidden_dim=dim, in_dim=2 * dim, embed_dim=self.emb_features,
                                              num_groups=num_groups,
                                              kernel_size=kernel_size, attn=need_attn, dropout=dropout,
                                              num_heads=num_heads)
            block = XUnetBlock(block, dim=dim, num_groups=num_groups, need_attn=need_attn, num_heads=num_heads,
                               cond=cond)
            self.decoder_layers[-1].append(block)
        self.upsample_blocks.append(torch.nn.Identity())

        # Out latent prediction
        self.out_norm = norm(hidden_dims[0], num_groups=num_groups)
        self.out_mapper = torch.nn.Conv2d(hidden_dims[0], features * 2, kernel_size=kernel_size,
                                          padding=kernel_size // 2)

    def forward(self, x, time, x_cond, time_cond, ray_o, ray_d, ray_o_cond, ray_d_cond):
        if self.cond == 'xunet':
            x = torch.cat([x, x_cond], dim=0)
        else:
            x = torch.cat([x, x_cond], dim=1)

        b, in_channels, height, width = x.shape
        # Encode time embeddings
        if self.cond == 'xunet':
            time = torch.cat([time, time_cond], dim=0)
            h_time = get_timestep_encoding(time, self.emb_features, self.steps)[:, :, None, None] \
                .expand(b, self.emb_features, height, width)
        elif self.cond == 'concat':
            h_time = get_timestep_encoding(time, self.emb_features, self.steps)[:, :, None, None] \
                .expand(b, self.emb_features, height, width)
        else:
            raise NotImplementedError(f'Unknown conditional type for time {self.cond}')
        h_time = self.time_layers[1](nonlinear(self.time_layers[0](h_time)))

        # Encode pos embeddings
        if self.cond == 'xunet':
            ray_d = torch.cat([ray_d, ray_d_cond], dim=0)
            ray_o = torch.cat([ray_o, ray_o_cond], dim=0)
        elif self.cond == 'concat':
            ray_d = torch.cat([ray_d, ray_d_cond], dim=1)
            ray_o = torch.cat([ray_o, ray_o_cond], dim=1)
        else:
            raise NotImplementedError(f'Unknown conditional type for rays positional encoding {self.cond}')
        h_d = get_positional_encoding(ray_d.movedim(1, -1), self.pos_features // 2).movedim(-1, 1)
        h_o = get_positional_encoding(ray_o, self.pos_features // 2)[:, :, None, None] \
            .expand(b, self.pos_features // 2, height, width)
        h_pos = torch.cat([h_d, h_o], dim=1)
        if self.cond == 'xunet':
            orig_id, cond_id = torch.zeros(b // 2).type_as(x).long(), torch.ones(b // 2).type_as(x).long()
            h_cond = self.embed_cond(torch.cat([orig_id, cond_id], dim=0))[:, :, None, None] \
                .expand(b, self.pos_features, height, width)
        elif self.cond == 'concat':
            h_cond = self.embed_cond[:, :, None, None].expand(b, self.pos_features, height, width)
        else:
            raise NotImplementedError(f'Unknown conditional type for cond embedding {self.cond}')
        h_pos += h_cond
        h_pos += self.embed_pixel[None, :, :, :].expand(b, self.pos_features, height, width)

        # Prepare embeddings for each layer
        pose_embs = [map_emb(h_pos) for map_emb in self.emb_maps]
        # Prepare input for mapping
        h = self.input_mapper(x)
        outs = []
        for blocks, downsample, pose_emb in zip(self.encoder_layers, self.downsample_blocks, pose_embs):
            emb = h_time[:, :, None, None] + pose_emb
            for block in blocks:
                h = block(h, emb)
                outs.append(h)
            h = downsample(h)
        # Mid mapping
        emb = h_time[:, :, None, None] + pose_embs[-1]
        h = self.mid_layers.block_1(h, emb)
        # Decode latent
        for blocks, upsample, pose_emb in zip(self.decoder_layers, self.upsample_blocks, pose_embs[::-1]):
            emb = h_time[:, :, None, None] + pose_emb
            for block in blocks:
                h = block(torch.cat([h, outs.pop()], dim=1), emb)
            h = upsample(h)

        h = nonlinear(self.out_norm(h))
        out = self.out_mapper(h)

        # take only x
        if self.cond == 'xunet':
            out = torch.chunk(out, 2, dim=0)[0]
        eps, weight = torch.chunk(out, 2, dim=1)
        weight = (torch.tanh(weight) + 1) / 2

        return eps, weight
