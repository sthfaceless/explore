import pytorch_lightning as pl
import torch

from modules.common.util import approx_standard_normal_cdf


class Diffusion(pl.LightningModule):

    def __init__(self, dataset=None, model=None, learning_rate=1e-4, batch_size=128, min_lr_rate=0.1,
                 diffusion_steps=1000, sample_steps=128,
                 min_beta=1e-2, max_beta=1e-4, beta_schedule='cos', kl_weight=1e-3, ll_delta=1/255):
        super(Diffusion, self).__init__()

        self.dataset = dataset
        self.model = model

        self.learning_rate = learning_rate
        self.min_lr_rate = min_lr_rate
        self.batch_size = batch_size
        self.kl_weight = kl_weight
        self.ll_delta = ll_delta

        self.min_beta = min_beta
        self.max_beta = max_beta
        self.diffusion_steps = diffusion_steps
        self.sample_steps = sample_steps
        self.beta_schedule = beta_schedule

        self.register_buffer('betas', self.get_beta_schedule())
        self.register_buffer('log_betas', torch.log(self.betas.clamp(min=1e-8)))
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('sqrt_alphas', torch.sqrt(self.alphas))
        self.register_buffer('head_alphas', torch.cumprod(self.alphas, dim=-1))
        self.register_buffer('head_alphas_pred', torch.cat([torch.ones(1, dtype=torch.float32), self.head_alphas[:-1]]))
        self.register_buffer('betas_tilde', self.betas *
                             (1 - torch.cat([torch.ones(1, dtype=torch.float32), self.head_alphas[:-1]])) /
                             (1 - self.head_alphas))
        self.register_buffer('betas_tilde_aligned', torch.cat([self.betas_tilde[1:2], self.betas_tilde[1:]]))
        self.register_buffer('log_betas_tilde_aligned', torch.log(self.betas_tilde_aligned.clamp(min=1e-8)))
        # precompute coefs for speed up training
        self.register_buffer('q_posterior_x0_coef',
                             torch.sqrt(self.head_alphas_pred) * self.betas / (1 - self.head_alphas))
        self.register_buffer('q_posterior_xt_coef',
                             torch.sqrt(self.alphas) * (1 - self.head_alphas_pred) / (1 - self.head_alphas))
        self.register_buffer('p_posterior_eps_coef', self.betas / torch.sqrt(1 - self.head_alphas))

    def kl_loss(self, mean1, logvar1, mean2, logvar2):
        kl_tensor = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + (mean1 - mean2) ** 2
                           * torch.exp(-logvar2))
        batch_losses = torch.mean(kl_tensor, dim=list(range(1, len(kl_tensor.shape))))
        return batch_losses

    def ll_loss(self, x, mean, logvar):
        x_std = (x - mean) * torch.exp(-logvar / 2)
        delta = self.ll_delta
        upper_bound = approx_standard_normal_cdf(x_std + delta)
        lower_bound = approx_standard_normal_cdf(x_std - delta)
        log_probs_tensor = torch.where(x < -0.999, torch.log(upper_bound.clamp(min=1e-8)),
                                       torch.where(x > 0.999, torch.log((1 - lower_bound).clamp(min=1e-8)),
                                                   torch.log((upper_bound - lower_bound).clamp(min=1e-8))))
        return - torch.mean(log_probs_tensor, dim=list(range(1, len(log_probs_tensor.shape))))

    def get_beta_schedule(self):
        if self.beta_schedule == 'linear':
            return torch.linspace(start=self.min_beta, end=self.max_beta, steps=self.diffusion_steps)
        elif self.beta_schedule == 'cos':
            s = 0.008
            f = torch.cos(
                (torch.linspace(start=0, end=1, steps=self.diffusion_steps + 1) + s) / (1 + s) * torch.pi / 2) ** 2
            head_alphas = f / f[0]
            betas = torch.clip(1 - head_alphas[1:] / head_alphas[:-1], min=0, max=0.999)
            return betas
        else:
            raise NotImplementedError

    def q_sample(self, x, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x)
        b, shape = len(t), [1 for _ in x.shape[1:]]
        t = t.view(b, *shape)
        return x * torch.sqrt(self.head_alphas[t]) + noise * torch.sqrt(1 - self.head_alphas[t])

    def get_sample_steps(self):
        sample_steps = [-1, 0]
        for t in range(self.sample_steps - 1):
            sample_steps.append(int((t + 1) / (self.sample_steps - 1) * (self.diffusion_steps - 1)))
        return list(reversed(sample_steps))

    def p_sample_stride(self, x, prev_t, curr_t, **kwargs):

        # reshape for x shape
        b, shape = len(curr_t), [1 for _ in x.shape[1:]]
        prev_t, curr_t = prev_t.view(b, *shape), curr_t.view(b, *shape)

        # make noise for all except last step
        z = torch.randn_like(x)
        z[curr_t.expand(z.shape) == 0] = 0

        # predict epsilon and variance (b 2*c h w)
        eps, var_weight = self.forward(x, curr_t.view(-1), **kwargs)

        # recalculate beta schedule with stride
        prev_head_alphas = torch.where(prev_t >= 0, self.head_alphas[prev_t], torch.ones_like(self.head_alphas[prev_t]))
        head_alphas = self.head_alphas[curr_t]
        betas = torch.clamp(1 - head_alphas / prev_head_alphas, min=1e-8, max=0.999)
        alphas = 1 - betas
        # fix the problem when betas_tilde with t = 0 is zero
        prev_head_alphas_aligned = torch.where(prev_t >= 0, self.head_alphas[prev_t],
                                               self.head_alphas[torch.zeros_like(prev_t)])
        head_alphas_aligned = torch.where(prev_t >= 0, self.head_alphas[curr_t],
                                          self.head_alphas[torch.ones_like(curr_t)])
        betas_aligned = torch.clamp(1 - head_alphas_aligned / prev_head_alphas_aligned, min=1e-8, max=0.999)
        betas_tilde = torch.clamp((1 - prev_head_alphas_aligned) / (1 - head_alphas_aligned) * betas_aligned,
                                  min=1e-8, max=0.999)

        mean = (x - eps * betas / torch.sqrt(1 - head_alphas)) / torch.sqrt(alphas)
        logvar = torch.log(betas) * var_weight + torch.log(betas_tilde) * (1 - var_weight)

        x = mean + torch.exp(logvar / 2) * z
        return x

    def p_sample(self, x, t, **kwargs):
        z = torch.randn_like(x)
        z[t == 0] = 0
        b, shape = len(t), [1 for _ in x.shape[1:]]
        t = t.view(b, *shape)
        eps, var_weight = self.forward(x, t.view(-1), **kwargs)

        mean = (x - eps * self.p_posterior_eps_coef[t]) / self.sqrt_alphas[t]
        logvar = self.log_betas[t] * var_weight + self.log_betas_tilde_aligned[t] * (1 - var_weight)
        x = mean + torch.exp(logvar / 2) * z
        return x

    def q_posterior_mean_variance(self, x, x_noised, t):
        b, shape = len(x), [1 for _ in x.shape[1:]]
        t = t.view(b, *shape)
        mean = x * self.q_posterior_x0_coef[t] + x_noised * self.q_posterior_xt_coef[t]
        logvar = self.log_betas_tilde_aligned[t]
        return mean, logvar

    def p_posterior_mean_variance(self, x_noised, t, eps, var_weight):
        b, shape = len(x_noised), [1 for _ in x_noised.shape[1:]]
        t = t.view(b, *shape)
        mean = (x_noised - eps * self.p_posterior_eps_coef[t]) / self.sqrt_alphas[t]
        logvar = self.log_betas[t] * var_weight + self.log_betas_tilde_aligned[t] * (1 - var_weight)
        return mean, logvar

    def get_losses(self, x, x_noised, t, noise, eps, var_weight):

        mse_loss = torch.nn.functional.mse_loss(noise, eps)

        true_mean, true_logvar = self.q_posterior_mean_variance(x, x_noised, t)
        pred_mean, pred_logvar = self.p_posterior_mean_variance(x_noised, t, eps, var_weight)

        b, shape = len(t), [1 for _ in x.shape[1:]]
        t = t.view(b, *shape)
        kl_loss = torch.mean(torch.where(t == 0, self.ll_loss(x, pred_mean.detach(), pred_logvar),
                                         self.kl_loss(true_mean, true_logvar, pred_mean.detach(), pred_logvar)))

        return {
            'mse_loss': mse_loss,
            'kl_loss': kl_loss,
            'loss': mse_loss + self.kl_weight * kl_loss
        }

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
