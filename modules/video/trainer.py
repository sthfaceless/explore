import pytorch_lightning as pl

from modules.common.trainer import *
from modules.gen.trainer import Diffusion
from modules.video.model import *


class LandscapeDiffusion(Diffusion):

    def __init__(self, unet_hiddens=(64, 64, 64, 128, 128, 128, 256, 256, 256, 384, 384, 384, 512, 512),
                 dataset=None, shape=(3, 128, 256),
                 features_dim=0, steps=10000, learning_rate=1e-4, batch_size=1, dropout=0.1,
                 min_lr_rate=0.01, attention_dim=32, epochs=30, diffusion_steps=4000, sample_steps=128,
                 kl_weight=1e-3, beta_schedule='cos', debug=True, num_heads=4, use_ema=False,
                 tempdir=None, gap=300, frames=8, classifier_free=0.1, clf_weight=12.5,
                 local_attn_dim=64, local_attn_patch=8, cond='cross', extra_upsample_blocks=1,
                 min_beta=1e-4, max_beta=2e-2, clearml=None, log_samples=5, log_every=32):
        self.save_hyperparameters(ignore=['clearml', 'dataset'])
        model = TemporalUNetDenoiser(shape=shape, steps=diffusion_steps, hidden_dims=unet_hiddens,
                                     attention_dim=attention_dim, num_heads=num_heads, dropout=dropout,
                                     extra_upsample_blocks=extra_upsample_blocks,
                                     local_attn_dim=local_attn_dim, local_attn_patch=local_attn_patch, cond=cond)
        super(LandscapeDiffusion, self).__init__(dataset=dataset, model=model, diffusion_steps=diffusion_steps,
                                                 learning_rate=learning_rate, min_lr_rate=min_lr_rate,
                                                 sample_steps=sample_steps, batch_size=batch_size, min_beta=min_beta,
                                                 max_beta=max_beta, kl_weight=kl_weight, use_ema=use_ema,
                                                 beta_schedule=beta_schedule, steps=steps, epochs=epochs)

        self.shape = shape
        self.features = self.shape[features_dim]
        self.classifier_free = classifier_free
        self.clf_weight = clf_weight

        self.w = shape[2]
        self.h = shape[1]
        self.frames = frames
        self.gap = gap

        self.custom_logger = SimpleLogger(clearml)
        self.log_samples = log_samples
        self.tempdir = tempdir
        self.debug = debug
        self.log_every = log_every

    def sample(self, n_samples, cond):
        batched_noise = torch.randn((n_samples, *self.shape), device=self.device)
        noise = repeat_dim(batched_noise.unsqueeze(1), 1, self.frames)
        x = noise
        sample_steps = self.get_sample_steps()
        for sample_iter, (t_curr, t_prev) in enumerate(zip(sample_steps[:-1], sample_steps[1:])):
            ones = torch.ones(n_samples).type_as(x).long()
            eps_cond = self.forward(x, ones * t_curr, cond=cond)
            eps_uncond = self.forward(x, ones * t_curr, cond=self.q_sample(
                cond, ones * (self.diffusion_steps - 1), batched_noise))
            eps = (1 + self.clf_weight) * eps_cond - self.clf_weight * eps_uncond
            x = self.p_sample_stride(x, ones * t_prev, ones * t_curr, eps=eps)
            # if self.debug and sample_iter % self.log_every == 0:
            #     videos_frames = prepare_torch_images(torch.cat([cond.unsqueeze(1), x], dim=1))
            #     for video_id in range(len(videos_frames)):
            #         self.custom_logger.log_gif(tensor2list(videos_frames[video_id]), self.gap,
            #                                    f'step_{sample_iter}_{video_id}',
            #                                    epoch=self.current_epoch)

        return x

    def step(self, batch):
        x, cond = batch['frames'][:, 1:], batch['frames'][:, 0]
        batched_noise = torch.randn_like(cond)
        noise = repeat_dim(batched_noise.unsqueeze(1), 1, self.frames)

        t = torch.randint(low=0, high=self.diffusion_steps - 1, size=(len(x),)).type_as(x).long()
        clf_free_msk = torch.rand(len(x)).type_as(x) > self.classifier_free

        x_noised = self.q_sample(x, t, noise)
        eps = self.forward(x_noised, t, cond=torch.where(clf_free_msk[:, None, None, None], cond,
                                                         self.q_sample(
                                                             cond, torch.ones_like(t) * (self.diffusion_steps - 1),
                                                             batched_noise)))

        return self.get_losses(x, x_noised, t, noise, eps)

    def on_validation_epoch_end(self):
        data_iter = iter(self.trainer.val_dataloaders[0])
        generated = 0
        while generated < self.log_samples:
            batch = next(data_iter)
            with torch.no_grad():
                cond = batch['frames'][:, 0].to(self.device)
                x = self.sample(self.batch_size, cond=cond)
            videos_frames = prepare_torch_images(torch.cat([cond.unsqueeze(1), x], dim=1))
            for video_id in range(len(videos_frames)):
                self.custom_logger.log_gif(tensor2list(videos_frames[video_id]), self.gap,
                                           f'sample_{generated + video_id}', epoch=self.current_epoch)
            generated += self.batch_size

        # log current lr
        if self.debug:
            self.custom_logger.log_gif(tensor2list(prepare_torch_images(batch['frames'][0])), self.gap,
                                       f'train_example', epoch=self.current_epoch)
            self.log('learning_rate', self.lr_schedulers().get_last_lr()[0], prog_bar=True)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.use_ema:
            self.ema_model.update(self.model)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False,
                                           num_workers=torch.cuda.device_count() * 2,
                                           pin_memory=True, drop_last=False, prefetch_factor=2)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False,
                                           num_workers=torch.cuda.device_count() * 2,
                                           pin_memory=True, drop_last=False, prefetch_factor=2)


class FrameInterpolation(pl.LightningModule):

    def __init__(self, dataset=None, clearml=None):
        super(FrameInterpolation, self).__init__()
        self.save_hyperparameters(ignore=['dataset', 'clearml'])

    def forward(self, x, ids):
        return {}

    def loss(self, out, gts):
        return {}

    def training_step(self, batch):
        return {}

    def validation_step(self, batch):
        return {}

    def on_validation_epoch_end(self):
        pass

    def configure_optimizers(self):
        opt = torch.optim.Adam(params=self.parameters(), lr=self.learning_rate, betas=(0.5, 0.9))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.learning_rate,
                                                        pct_start=self.pct_start, div_factor=self.div_factor,
                                                        final_div_factor=self.final_div_factor,
                                                        epochs=self.epochs, steps_per_epoch=self.steps)
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step'
        }
        return [opt], [scheduler]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False,
                                           num_workers=torch.cuda.device_count() * 2,
                                           pin_memory=True, drop_last=False, prefetch_factor=2)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False,
                                           num_workers=torch.cuda.device_count() * 2,
                                           pin_memory=True, drop_last=False, prefetch_factor=2)
