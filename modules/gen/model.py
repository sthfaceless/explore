import torch


class VAE(torch.nn.Module):

    def __init__(self, latent_dim, latent_noise=0.0):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.latent_noise = latent_noise

    def decode(self, latent):
        raise NotImplementedError

    def encode(self, x):
        raise NotImplementedError

    def kl_loss(self, mu, logsigma):
        return torch.mean(torch.sum(-logsigma + (mu ** 2 + torch.exp(2 * logsigma)) / 2 - 0.5, dim=1), dim=0)

    def reparametrize(self, mu, logsigma):
        sigma = torch.exp(logsigma)
        return mu + sigma * torch.randn_like(sigma, device=sigma.device)

    def forward(self, x):
        mu, logsigma = self.encode(x)
        latent = self.reparametrize(mu, logsigma)
        latent = latent + torch.randn_like(latent, device=latent.device) * self.latent_noise
        pred = self.decode(latent)
        return pred, x, mu, logsigma, latent

    def sample(self, n, device):
        points = torch.randn(n, self.latent_dim, device=device)
        samples = self.decode(points)
        return samples


class VAEWrapper(VAE):

    def __init__(self, encoder, decoder, latent_dim, latent_noise=0.0):
        super(VAEWrapper, self).__init__(latent_dim, latent_noise)
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        return self.encoder(x)

    def decode(self, latent):
        return self.decoder(latent)
