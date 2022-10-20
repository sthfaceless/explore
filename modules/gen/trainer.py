import pytorch_lightning as pl
import torch


class Diffusion(pl.LightningModule):

    def __init__(self, dataset=None, model=None, learning_rate=1e-4, batch_size=128, min_lr_rate=0.1,
                 diffusion_steps=1000, min_beta=1e-2, max_beta=1e-4, beta_schedule='cos'):
        super(Diffusion, self).__init__()

        self.dataset = dataset
        self.model = model

        self.learning_rate = learning_rate
        self.min_lr_rate = min_lr_rate
        self.batch_size = batch_size

        self.min_beta = min_beta
        self.max_beta = max_beta
        self.diffusion_steps = diffusion_steps
        self.beta_schedule = beta_schedule

        self.register_buffer('betas', self.get_beta_schedule())
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('head_alphas', torch.cumprod(self.alphas, dim=-1))

    def get_beta_schedule(self):
        if self.beta_schedule == 'linear':
            return torch.linspace(start=self.min_beta, end=self.max_beta, steps=self.diffusion_steps)
        elif self.beta_schedule == 'cos':
            s = 1e-6
            f = torch.cos(
                (torch.linspace(start=0, end=1, steps=self.diffusion_steps + 1) + s) / (1 + s) * 3.1415926 / 2) ** 2
            head_alphas = f / f[0]
            betas = torch.clip(1 - head_alphas[1:] / head_alphas[:-1], min=1e-6, max=0.999)
            return betas
        else:
            raise NotImplementedError

    def q_sample(self, x, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x)
        b, shape = len(t), [1 for _ in x.shape[1:]]
        t = t.view(b, *shape)
        return x * torch.sqrt(self.head_alphas[t]) + noise * torch.sqrt(1 - self.head_alphas[t])

    def p_sample(self, x, t, **kwargs):
        z = torch.randn_like(x)
        z[t == 0] = 0
        b, shape = len(t), [1 for _ in x.shape[1:]]
        t = t.view(b, *shape)
        x = (x - self.forward(x, t.view(-1), **kwargs) * self.betas[t] / torch.sqrt(1 - self.head_alphas[t])) \
            / torch.sqrt(self.alphas[t]) + self.betas[t] * z
        return x

    def step(self, batch):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('val_loss', loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(lr=self.learning_rate, params=self.model.parameters(), betas=(0.9, 0.99))
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                            T_0=10, T_mult=1, last_epoch=-1,
                                                                            eta_min=self.min_lr_rate
                                                                                    * self.learning_rate)
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False,
                                           num_workers=2 * torch.cuda.device_count(),
                                           pin_memory=True, prefetch_factor=2)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False,
                                           num_workers=2 * torch.cuda.device_count(),
                                           pin_memory=True, prefetch_factor=2)
