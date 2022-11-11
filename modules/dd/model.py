from modules.common.model import *
from modules.gen.model import VAE


class UNetDenoiser(torch.nn.Module):

    def __init__(self, shape, steps, kernel_size=3, hidden_dims=(16, 32, 64), attention_dim=32, num_heads=4,
                 residual_scaling=1 / 2 ** (1 / 2), num_groups=32):
        super(UNetDenoiser, self).__init__()
        features, h, w = shape
        self.features = features
        self.steps = steps
        # Input mapping
        self.input_mapper = torch.nn.Conv2d(features, hidden_dims[0], kernel_size=kernel_size, padding=kernel_size // 2)

        # Timestep mapping
        self.timestep_features = hidden_dims[0] * 4
        self.timestep_layers = torch.nn.ModuleList([
            torch.nn.Linear(features, self.timestep_features),
            torch.nn.Linear(self.timestep_features, self.timestep_features),
        ])

        self.encoder_layers = torch.nn.ModuleList([torch.nn.ModuleList([])])
        self.downsample_blocks = torch.nn.ModuleList([])
        current_resolution = w
        for prev_dim, dim in zip([hidden_dims[0]] + list(hidden_dims), hidden_dims):
            if prev_dim != dim:
                self.downsample_blocks.append(DownSample2d(prev_dim, dim, kernel_size=kernel_size))
                self.encoder_layers.append(torch.nn.ModuleList([]))
                current_resolution /= 2
            block = TimestepResBlock2D(hidden_dim=dim, timestep_dim=self.timestep_features, num_groups=num_groups,
                                       attn=current_resolution <= attention_dim, num_heads=num_heads)
            self.encoder_layers[-1].append(block)
        self.downsample_blocks.append(torch.nn.Identity())

        self.mid_layers = torch.nn.Module()
        self.mid_layers.block_1 = TimestepResBlock2D(hidden_dim=hidden_dims[-1],
                                                     timestep_dim=self.timestep_features, num_groups=num_groups,
                                                     num_heads=num_heads)
        self.mid_layers.block_2 = TimestepResBlock2D(hidden_dim=hidden_dims[-1],
                                                     timestep_dim=self.timestep_features, num_groups=num_groups,
                                                     num_heads=num_heads)

        inverse_dims = hidden_dims[::-1]
        self.decoder_layers = torch.nn.ModuleList([torch.nn.ModuleList([])])
        self.upsample_blocks = torch.nn.ModuleList([])
        for prev_dim, dim in zip([inverse_dims[0]] + list(inverse_dims), inverse_dims):
            if prev_dim != dim:
                self.upsample_blocks.append(UpSample2d(prev_dim, dim, kernel_size=kernel_size))
                self.decoder_layers.append(torch.nn.ModuleList([]))
                current_resolution *= 2
            block = TimestepResBlock2D(hidden_dim=dim, in_dim=2 * dim, attn=current_resolution <= attention_dim,
                                       timestep_dim=self.timestep_features, num_groups=num_groups, num_heads=num_heads)
            self.decoder_layers[-1].append(block)
        self.upsample_blocks.append(torch.nn.Identity())

        # Out latent prediction
        self.out_norm = norm(hidden_dims[0], num_groups=num_groups)
        self.out_mapper = torch.nn.Conv2d(hidden_dims[0], features * 2, kernel_size=kernel_size,
                                          padding=kernel_size // 2)

    def forward(self, input, time):
        # Prepare input for mapping
        h = self.input_mapper(input)
        # Encode time
        h_time = get_timestep_encoding(time, self.features, self.steps)
        h_time = nonlinear(self.timestep_layers[0](h_time))
        h_time = self.timestep_layers[1](h_time)
        # Encode latent
        outs = []
        for blocks, downsample in zip(self.encoder_layers, self.downsample_blocks):
            for block in blocks:
                h = block(h, h_time)
                outs.append(h)
            h = downsample(h)
        # Mid mapping
        h = self.mid_layers.block_1(h, h_time)
        h = self.mid_layers.block_2(h, h_time)
        # Decode latent
        for blocks, upsample in zip(self.decoder_layers, self.upsample_blocks):
            for block in blocks:
                h = block(torch.cat([h, outs.pop()], dim=1), h_time)
            h = upsample(h)

        h = nonlinear(self.out_norm(h))
        out = self.out_mapper(h)

        eps, weight = torch.chunk(out, 2, dim=1)
        weight = (torch.tanh(weight) + 1) / 2

        return eps, weight


class VAEEncoder2D(torch.nn.Module):

    def __init__(self, latent_dim, shape, kernel_size=3,
                 attention_dim=32, hidden_dims=(8, 8, 16, 16, 32, 32, 64, 64), num_groups=8):
        super(VAEEncoder2D, self).__init__()
        self.latent_dim = latent_dim
        self.shape = shape
        # map inpit image to hidden dims
        self.input_mapper = torch.nn.Conv2d(in_channels=shape[0], out_channels=hidden_dims[0],
                                            kernel_size=kernel_size, padding=kernel_size // 2, stride=1)

        # encode image to latents
        self.encoder_blocks = torch.nn.ModuleList([])
        current_res = shape[1]
        for prev_dim, dim in zip(hidden_dims, hidden_dims[1:]):
            if prev_dim != dim:
                self.encoder_blocks.append(DownSample2d(in_dim=prev_dim, out_dim=dim, kernel_size=kernel_size))
                current_res //= 2
            self.encoder_blocks.append(ResBlock2d(in_dim=dim, hidden_dim=dim, num_groups=num_groups,
                                                  kernel_size=kernel_size))
        # out norm of encoder
        self.encoder_norm = torch.nn.GroupNorm(num_channels=self.latent_shape[0], num_groups=num_groups)

        self.latent_shape = (latent_dim // (current_res * current_res), current_res, current_res)
        self.mu_layer = torch.nn.Conv2d(in_channels=hidden_dims[-1], out_channels=self.latent_shape[0],
                                        kernel_size=kernel_size, padding=kernel_size // 2, stride=1)

        self.sigma_layer = torch.nn.Conv2d(in_channels=hidden_dims[-1], out_channels=self.latent_shape[0],
                                           kernel_size=kernel_size, padding=kernel_size // 2, stride=1)

    def get_latent_shape(self):
        return self.latent_shape

    def kl_loss(self, mu, logsigma):
        return torch.mean(torch.sum(-logsigma + (mu ** 2 + torch.exp(2 * logsigma) - 1) / 2, dim=1), dim=0)

    def reparametrize(self, mu, logsigma):
        sigma = torch.exp(logsigma)
        return mu + sigma * torch.randn_like(sigma, device=sigma.device)

    def encode(self, x):
        h = self.input_mapper(x)
        for block in self.encoder_blocks:
            h = block(x)
        h = torch.nn.functional.silu(self.encoder_norm(h))
        mu = self.mu_layer(h)
        logsigma = self.sigma_layer(h)
        return mu.view(-1, self.latent_dim), logsigma.view(-1, self.latent_dim)

    def forward(self, x):
        mu, logsigma = self.encode(x)
        h = self.reparametrize(mu, logsigma)
        return h, mu, logsigma


class Decoder2D(torch.nn.Module):

    def __init__(self, shape, hidden_dims=(64, 64, 32, 32, 16, 16, 8, 8), attention_dim=32, num_groups=8,
                 kernel_size=3):
        super(Decoder2D, self).__init__()
        self.shape = shape
        # input norm of decoder and mapping to hiddens
        self.decoder_norm = torch.nn.GroupNorm(num_channels=shape[0], num_groups=num_groups)
        self.decoder_layer = torch.nn.Conv2d(in_channels=shape[0], out_channels=hidden_dims[-1],
                                             kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
        # decode latent with res blocks
        self.decoder_blocks = torch.nn.ModuleList([])
        invert_hiddens = hidden_dims[::-1]
        for prev_dim, dim in zip(invert_hiddens, invert_hiddens[1:]):
            if prev_dim != dim:
                self.decoder_blocks.append(UpSample2d(in_dim=prev_dim, out_dim=dim, kernel_size=kernel_size))
            self.decoder_blocks.append(ResBlock2d(in_dim=dim, hidden_dim=dim, num_groups=num_groups,
                                                  kernel_size=kernel_size))
        # out of decoder mapping
        self.out_norm = torch.nn.GroupNorm(num_channels=hidden_dims[0], num_groups=num_groups)
        self.out_layer = torch.nn.Conv2d(in_channels=hidden_dims[0], out_channels=3,
                                         kernel_size=kernel_size, padding=kernel_size // 2, stride=1)

    def forward(self, x):
        h = x.view(-1, *self.shape)
        h = torch.nn.functional.silu(self.decoder_norm(h))
        h = self.decoder_layer(h)
        for block in self.decoder_blocks:
            h = block(h)
        h = self.out_norm(h)
        out = torch.nn.functional.tanh(self.out_layer(h))
        return out


class ImageVAE(VAE):

    def __init__(self, latent_dim, shape, hidden_dims=(8, 8, 16, 16, 32, 32, 64, 64), latent_noise=0.0,
                 kernel_size=3, num_groups=8):
        super(ImageVAE, self).__init__(latent_dim, latent_noise)
        # map inpit image to hidden dims
        self.input_mapper = torch.nn.Conv2d(in_channels=shape[0], out_channels=hidden_dims[0],
                                            kernel_size=kernel_size, padding=kernel_size // 2, stride=1)

        # encode image to latents
        self.encoder_blocks = torch.nn.ModuleList([])
        current_res = shape[1]
        for prev_dim, dim in zip(hidden_dims, hidden_dims[1:]):
            if prev_dim != dim:
                self.encoder_blocks.append(DownSample2d(in_dim=prev_dim, out_dim=dim, kernel_size=kernel_size))
                current_res /= 2
            self.encoder_blocks.append(ResBlock2d(in_dim=dim, hidden_dim=dim, num_groups=num_groups,
                                                  kernel_size=kernel_size))
        # out norm of encoder
        self.encoder_norm = torch.nn.GroupNorm(num_channels=self.latent_shape[0], num_groups=num_groups)

        # mu and sigma from latents
        self.latent_shape = (latent_dim // (current_res * current_res), current_res, current_res)
        self.mu_layer = torch.nn.Conv2d(in_channels=hidden_dims[-1], out_channels=self.latent_shape[0],
                                        kernel_size=kernel_size, padding=kernel_size // 2, stride=1)

        self.sigma_layer = torch.nn.Conv2d(in_channels=hidden_dims[-1], out_channels=self.latent_shape[0],
                                           kernel_size=kernel_size, padding=kernel_size // 2, stride=1)

        # input norm of decoder and mapping to hiddens
        self.decoder_norm = torch.nn.GroupNorm(num_channels=self.latent_shape[0], num_groups=num_groups)
        self.decoder_layer = torch.nn.Conv2d(in_channels=self.latent_shape[0], out_channels=hidden_dims[-1],
                                             kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
        # decode latent with res blocks
        self.decoder_blocks = torch.nn.ModuleList([])
        invert_hiddens = hidden_dims[::-1]
        for prev_dim, dim in zip(invert_hiddens, invert_hiddens[1:]):
            if prev_dim != dim:
                self.decoder_blocks.append(UpSample2d(in_dim=prev_dim, out_dim=dim, kernel_size=kernel_size))
                current_res *= 2
            self.decoder_blocks.append(ResBlock2d(in_dim=dim, hidden_dim=dim, num_groups=num_groups,
                                                  kernel_size=kernel_size))
        # out of decoder mapping
        self.out_norm = torch.nn.GroupNorm(num_channels=hidden_dims[0], num_groups=num_groups)
        self.out_layer = torch.nn.Conv2d(in_channels=hidden_dims[0], out_channels=3,
                                         kernel_size=kernel_size, padding=kernel_size // 2, stride=1)

    def encode(self, x):
        h = self.input_mapper(x)
        for block in self.encoder_blocks:
            h = block(x)
        h = torch.nn.functional.silu(self.encoder_norm(h))
        mu = self.mu_layer(h)
        logsigma = self.sigma_layer(h)
        return mu.view(-1, self.latent_dim), logsigma.view(-1, self.latent_dim)

    def decode(self, latent):
        h = latent.view(-1, *self.latent_shape)
        h = torch.nn.functional.silu(self.decoder_norm(h))
        h = self.decoder_layer(h)
        for block in self.decoder_blocks:
            h = block(h)
        h = self.out_norm(h)
        out = torch.nn.functional.tanh(self.out_layer(h))
        return out
