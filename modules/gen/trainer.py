import pytorch_lightning as pl

from modules.common.trainer import *


class Diffusion(pl.LightningModule):

    def __init__(self, dataset=None, model=None, use_ema=False, learning_rate=1e-4, batch_size=128, min_lr_rate=0.1,
                 diffusion_steps=1000, sample_steps=64, steps=10000, epochs=100, clip_denoised=True,
                 min_beta=1e-4, max_beta=0.02, beta_schedule='cos', kl_weight=0.0):
        super(Diffusion, self).__init__()

        self.dataset = dataset
        self.model = model
        self.use_ema = use_ema
        if use_ema:
            self.ema_model = EMA(self.model)

        self.learning_rate = learning_rate
        self.min_lr_rate = min_lr_rate
        self.batch_size = batch_size
        self.kl_weight = kl_weight

        self.min_beta = min_beta
        self.max_beta = max_beta
        self.diffusion_steps = diffusion_steps
        self.sample_steps = sample_steps
        self.beta_schedule = beta_schedule
        self.clip_denoised = clip_denoised
        self.steps = steps
        self.epochs = epochs

        self.register_buffer('betas', self.get_beta_schedule())
        self.register_buffer('log_betas', torch.log(self.betas.clamp(min=1e-8)))
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('sqrt_alphas', torch.sqrt(self.alphas))
        self.register_buffer('head_alphas', torch.cumprod(self.alphas, dim=-1))
        self.register_buffer('head_alphas_pred', torch.cat([torch.ones(1, dtype=torch.float32), self.head_alphas[:-1]]))
        self.register_buffer('betas_tilde', self.betas * (1 - self.head_alphas_pred) / (1 - self.head_alphas))
        self.register_buffer('betas_tilde_aligned', torch.cat([self.betas_tilde[1:2], self.betas_tilde[1:]]))
        self.register_buffer('log_betas_tilde_aligned', torch.log(self.betas_tilde_aligned.clamp(min=1e-8)))
        # precompute coefs for speed up training
        self.register_buffer('q_posterior_x0_coef',
                             torch.sqrt(self.head_alphas_pred) * self.betas / (1 - self.head_alphas))
        self.register_buffer('q_posterior_xt_coef',
                             torch.sqrt(self.alphas) * (1 - self.head_alphas_pred) / (1 - self.head_alphas))
        self.register_buffer('p_posterior_eps_coef', self.betas / torch.sqrt(1 - self.head_alphas))

        self.var_weight = torch.nn.Parameter(torch.zeros((diffusion_steps,), dtype=torch.float32), requires_grad=True)

    def get_sample_steps(self, steps=None):
        if steps is None:
            steps = self.sample_steps
        sample_steps = [-1, 0]
        for t in range(steps - 1):
            sample_steps.append(int((t + 1) / (steps - 1) * (self.diffusion_steps - 1)))
        return list(reversed(sample_steps))

    def kl_loss(self, mean1, logvar1, mean2, logvar2):
        kl_tensor = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + (mean1 - mean2) ** 2
                           * torch.exp(-logvar2))
        batch_losses = torch.mean(kl_tensor, dim=list(range(1, len(kl_tensor.shape))))
        return batch_losses

    def nll_loss(self, x, mean, logvar):
        return 0.5 * torch.mean(np.log(2.0 * torch.pi) + logvar + (x - mean) ** 2 / torch.exp(logvar),
                                dim=list(range(1, len(x.shape))))

    def get_beta_schedule(self):
        if self.beta_schedule == 'linear':
            return torch.linspace(start=self.min_beta, end=self.max_beta, steps=self.diffusion_steps)
        elif self.beta_schedule == 'cos':
            s = 0.008
            f = torch.cos(
                (torch.linspace(start=0, end=1 - 1 / 32, steps=self.diffusion_steps + 1) + s) / (
                        1 + s) * torch.pi / 2) ** 2
            head_alphas = f / f[0]
            betas = torch.clip(1 - head_alphas[1:] / head_alphas[:-1], min=0, max=0.999)
            return betas
        else:
            raise NotImplementedError

    def predict_x0(self, xt, t, eps, clip_denoised=None):
        b, shape = len(t), [1 for _ in xt.shape[1:]]
        t = t.view(b, *shape)
        x0 = xt / torch.sqrt(self.head_alphas[t]) - eps * torch.sqrt(1 / self.head_alphas[t] - 1)
        clip_denoised = self.clip_denoised if clip_denoised is None else clip_denoised
        if clip_denoised:
            x0 = torch.clamp(x0, min=-1.0, max=1.0)
        return x0

    def q_sample(self, x, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x)
        b, shape = len(t), [1 for _ in x.shape[1:]]
        t = t.view(b, *shape)
        return x * torch.sqrt(self.head_alphas[t]) + noise * torch.sqrt(1 - self.head_alphas[t])

    def q_posterior_mean_variance(self, x0, xt, t, learned_var=False):
        b, shape = len(x0), [1 for _ in x0.shape[1:]]
        t = t.view(b, *shape)
        mean = x0 * self.q_posterior_x0_coef[t] + xt * self.q_posterior_xt_coef[t]
        if learned_var:
            var_weight = self.var_weight[t]
        else:
            var_weight = torch.zeros_like(self.var_weight[t])
        logvar = self.log_betas[t] * var_weight + self.log_betas_tilde_aligned[t] * (1 - var_weight)
        return mean, logvar

    def p_posterior_mean_variance(self, x_noised, t, eps, simplified=False, clip_denoised=None):
        b, shape = len(x_noised), [1 for _ in x_noised.shape[1:]]
        t = t.view(b, *shape)
        if simplified:
            mean = (x_noised - eps * self.p_posterior_eps_coef[t]) / self.sqrt_alphas[t]
            logvar = self.log_betas[t] * self.var_weight[t] \
                     + self.log_betas_tilde_aligned[t] * (1 - self.var_weight[t])
        else:
            x0 = self.predict_x0(x_noised, t, eps, clip_denoised=clip_denoised)
            mean, logvar = self.q_posterior_mean_variance(x0, x_noised, t, learned_var=True)
        return mean, logvar

    def p_sample_stride(self, xt, prev_t, curr_t, clip_denoised=None, simplified=False, eps=None, **kwargs):

        # reshape for x shape
        b, shape = len(curr_t), [1 for _ in xt.shape[1:]]
        prev_t, curr_t = prev_t.view(b, *shape), curr_t.view(b, *shape)

        # make noise for all except last step
        z = torch.randn_like(xt)
        z[curr_t.expand(z.shape) == 0] = 0

        # predict epsilon and variance (b 2*c h w)
        if eps is None:
            eps = self.forward(xt, curr_t.view(-1), **kwargs)

        # recalculate beta schedule with stride
        head_alphas = self.head_alphas[curr_t]
        prev_head_alphas = torch.where(prev_t >= 0, self.head_alphas[prev_t], torch.ones_like(self.head_alphas[prev_t]))
        betas = torch.clamp(1 - head_alphas / prev_head_alphas, min=1e-8, max=0.999)
        alphas = 1 - betas

        # calculate mean either with simplified formula from paper or with full formula
        if simplified:
            mean = (xt - eps * betas / torch.sqrt(1 - head_alphas)) / torch.sqrt(alphas)
        else:
            x0 = xt / torch.sqrt(head_alphas) - eps * torch.sqrt(1 / head_alphas - 1)
            clip_denoised = self.clip_denoised if clip_denoised is None else clip_denoised
            if clip_denoised:
                x0 = torch.clamp(x0, min=-1.0, max=1.0)
            mean = (x0 * torch.sqrt(prev_head_alphas) * betas / (1 - head_alphas) +
                    xt * torch.sqrt(alphas) * (1 - prev_head_alphas) / (1 - head_alphas))

        # fix the problem when betas_tilde with t = 0 is zero
        prev_head_alphas_aligned = torch.where(prev_t >= 0, self.head_alphas[prev_t],
                                               self.head_alphas[torch.zeros_like(prev_t)])
        head_alphas_aligned = torch.where(prev_t >= 0, self.head_alphas[curr_t],
                                          self.head_alphas[torch.ones_like(curr_t)])
        betas_aligned = torch.clamp(1 - head_alphas_aligned / prev_head_alphas_aligned, min=1e-8, max=0.999)
        betas_tilde = torch.clamp((1 - prev_head_alphas_aligned) / (1 - head_alphas_aligned) * betas_aligned,
                                  min=1e-8, max=0.999)
        logvar = torch.log(betas) * self.var_weight[curr_t] + torch.log(betas_tilde) * (1 - self.var_weight[curr_t])

        x = mean + torch.exp(logvar / 2) * z
        return x

    def p_sample(self, xt, t, eps=None, clip_denoised=None, **kwargs):
        z = torch.randn_like(xt)
        z[t == 0] = 0
        b, shape = len(t), [1 for _ in xt.shape[1:]]
        t = t.view(b, *shape)
        if eps is None:
            eps = self.forward(xt, t.view(-1), **kwargs)

        mean, logvar = self.p_posterior_mean_variance(xt, t, eps, clip_denoised=clip_denoised)
        x = mean + torch.exp(logvar / 2) * z
        return x

    def get_losses(self, x, x_noised, t, noise, eps):

        mse_loss = torch.nn.functional.mse_loss(noise, eps)

        true_mean, true_logvar = self.q_posterior_mean_variance(x, x_noised, t)
        pred_mean, pred_logvar = self.p_posterior_mean_variance(x_noised, t, eps, clip_denoised=False)

        b, shape = len(t), [1 for _ in x.shape[1:]]
        t = t.view(b, *shape)
        kl_loss = torch.mean(torch.where(t == 0, self.nll_loss(x, pred_mean.detach(), pred_logvar),
                                         self.kl_loss(true_mean, true_logvar, pred_mean.detach(), pred_logvar)))

        return {
            'mse_loss': mse_loss,
            'kl_loss': kl_loss,
            'loss': mse_loss + self.kl_weight * kl_loss
        }

    def forward(self, x, t, train=True, **kwargs):
        if (not self.training or not train) and self.use_ema:
            model = self.ema_model.module
        else:
            model = self.model
        return model(x, t, **kwargs)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.use_ema:
            self.ema_model.update(self.model)

    def step(self, batch):
        ####
        # must return dict with 'loss' key
        ####
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        for k, v in loss.items():
            self.log(f'train_{k}', v, prog_bar=True, sync_dist=True)

        return loss['loss']

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        for k, v in loss.items():
            self.log(f'val_{k}', v, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(lr=self.learning_rate, params=self.model.parameters(), betas=(0.9, 0.99))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.learning_rate,
                                                        pct_start=1 / self.epochs, div_factor=2.0,
                                                        final_div_factor=1 / (2.0 * self.min_lr_rate),
                                                        epochs=self.epochs, steps_per_epoch=self.steps)
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step'
        }
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False,
                                           num_workers=2 * torch.cuda.device_count(),
                                           pin_memory=True, prefetch_factor=2)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False,
                                           num_workers=2 * torch.cuda.device_count(),
                                           pin_memory=True, prefetch_factor=2)
