import gc
import os.path
from random import shuffle
from time import time

import pytorch_lightning as pl
import torch.cuda
from PIL import Image

from modules.dd.model import *
from modules.gen.trainer import Diffusion
from modules.nerf.dataset import *
from modules.nerf.model import *
from modules.nerf.util import *


class SimpleLogger:

    def __init__(self, clearml=None):
        self.clearml = clearml

    def log_image(self, image, name, epoch=0):
        if self.clearml:
            self.clearml.report_image('valid', f"{name}", iteration=epoch, image=image)
        else:
            Image.fromarray(image).save(f"{name}.png")

    def log_images(self, images, prefix, epoch=0):
        for image_id, image in enumerate(images):
            self.log_image(image, f'{prefix}_{image_id}', epoch)


class NerfTrainer:

    def __init__(self, model, spp=128, coarse_weight=0.1, pe_powers=16):
        self.model = model
        self.spp = spp
        self.coarse_weight = coarse_weight
        self.pe_powers = pe_powers

    def render_views(self, scene):
        images = scene.imgs[np.random.randint(low=0, high=len(scene.imgs), size=8)]
        images = (images * 255.0).astype(np.uint8)
        h, w, _ = scene.imgs[0].shape
        gallery = np.array(images).reshape((2, 4, h, w, 3)).transpose(0, 2, 1, 3, 4).reshape((2 * h, 4 * w, 3))
        return gallery

    def train(self, scene_root, batch_rays, steps, log_every, logger, iteration_id, device=torch.device('cpu'),
              learning_rate=1e-3, max_lr=1e-3, steps_epoch=1000, pct_start=0.3, model_name='model',
              dgrid_res=64, dgrid_steps=16, dgrid_decay=0.95):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr,
                                                           steps_per_epoch=steps_epoch, epochs=steps // steps_epoch,
                                                           pct_start=pct_start)

        scene = NerfScene(scene_root, batch_rays)
        loader = torch.utils.data.DataLoader(scene, batch_size=None, prefetch_factor=2)
        loader_iter = iter(loader)
        Image.fromarray(self.render_views(scene)).save(f'orig.png')

        train_start_time = time()
        for step in range(steps):

            optimizer.zero_grad()

            batch = next(loader_iter)
            batch = {k: v.to(device) for k, v in batch.items()}

            coarse_pixels, fine_pixels, trans = render_batch(batch, self.model, self.pe_powers, self.spp)
            gt_pixels = batch['pixels']
            out = self.model.loss(coarse_pixels, fine_pixels, gt_pixels, self.coarse_weight, trans)
            out['loss'].backward()
            optimizer.step()

            lr_scheduler.step()

            if (step + 1) % log_every == 0:
                # render nerf from several poses and measure time to render one image
                start = time()
                img = render_gallery(model=self.model, batch_size=batch_rays, spp=self.spp, pe_powers=self.pe_powers,
                                     w=128, h=128, focal=scene.get_focal(), camera_distance=scene.get_mean_distance(),
                                     device=device)
                end = time()
                per_image_time = (end - start) / 8

                Image.fromarray(img).save(f'gallery.png')
                print(f"--- Finished {step + 1} steps, loss - {out['loss']}, per-image time - {per_image_time}s")

                torch.save(self.model.state_dict(), model_name)
        train_end_time = time()
        print(f"Time consumed for training {train_end_time - train_start_time:.2f}s")


class NerfGenTrainer:

    def __init__(self, gen_model, gen_disc, nerf_model):
        self.gen_model = gen_model
        self.gen_disc = gen_disc
        self.nerf_model = nerf_model

    def train(self, class_path, epochs, batch_size, train_rate=0.7, stop_patience=5, stop_eps=1e-4,
              log_samples=5, logger=None, nerf_spp=128, nerf_batch=4096, model_name='nerf_gen',
              device=torch.device('cpu'), weights_name='model', disc_warmup=5, debug_samples=5,
              learning_rate=1e-3, max_lr=1e-3, pct_start=0.3):

        nerf_paths = [os.path.join(class_path, obj, weights_name) for obj in os.listdir(class_path)]
        train_size = int(len(nerf_paths) * train_rate)
        test_size = len(nerf_paths) - train_size

        optimizer = torch.optim.Adam(self.gen_model.parameters(), lr=learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
        # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr,
        #                                                    steps_per_epoch=train_size // batch_size, epochs=epochs,
        #                                                    pct_start=pct_start)

        disc_optimizer = torch.optim.Adam(self.gen_disc.parameters(), lr=learning_rate)
        disc_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(disc_optimizer, gamma=0.97)

        print("Begin NeRF generator training")
        history = {}
        for epoch in range(epochs):
            print(f"Started epoch {epoch + 1}")
            epoch_start_time = time()
            # new epoch begins
            shuffle(nerf_paths)
            train_paths, test_paths = nerf_paths[:train_size], nerf_paths[train_size:]
            train_dataset = NerfWeights(paths=train_paths, batch_size=batch_size)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, prefetch_factor=2,
                                                       shuffle=True, num_workers=2, drop_last=True)
            print(f"Loaded data in memory, train items: {len(train_paths)} test items: {len(test_paths)}")

            print("Start train loop")
            train_out = []
            self.gen_model.train()
            for batch in tqdm(train_loader):

                batch['weights'] = batch['weights'].to(device)

                optimizer.zero_grad()
                pred = self.gen_model.forward(batch)
                if epoch >= disc_warmup:
                    disc_pred = self.gen_disc.forward(pred[0])
                    disc_loss = self.gen_disc.loss(disc_pred, torch.zeros(disc_pred.shape[0], device=device,
                                                                          dtype=torch.float32))
                else:
                    disc_loss = None

                out = self.gen_model.loss(pred, disc_loss)
                out['loss'].backward()
                optimizer.step()

                if epoch >= disc_warmup:
                    disc_optimizer.zero_grad()
                    pred = self.gen_model.forward(batch)[0]
                    inp = torch.cat([batch['weights'], pred], dim=0)
                    labels = torch.cat([torch.ones(batch['weights'].shape[0], device=device, dtype=torch.float32),
                                        torch.zeros(pred.shape[0], device=device, dtype=torch.float32)], dim=0)
                    disc_out = self.gen_disc.forward(inp)
                    disc_loss = self.gen_disc.loss(disc_out, labels)
                    disc_loss.backward()
                    disc_optimizer.step()

                train_out.append({k: val.detach().cpu().numpy() for k, val in out.items()})

            lr_scheduler.step()
            if epoch >= disc_warmup:
                disc_lr_scheduler.step()

            test_dataset = NerfWeights(paths=test_paths, batch_size=batch_size)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, prefetch_factor=2,
                                                      shuffle=True, num_workers=2, drop_last=True)
            test_out = []
            print("Start test loop")
            self.gen_model.eval()
            for batch in tqdm(test_loader):
                batch['weights'] = batch['weights'].to(device)
                with torch.no_grad():
                    pred = self.gen_model.forward(batch)
                    if epoch >= disc_warmup:
                        disc_pred = self.gen_disc.forward(pred[0])
                        disc_loss = self.gen_disc.loss(disc_pred, torch.zeros(disc_pred.shape[0], device=device,
                                                                              dtype=torch.float32))
                    else:
                        disc_loss = None
                    out = self.gen_model.loss(pred, disc_loss)

                test_out.append({k: val.detach().cpu().numpy() for k, val in out.items()})

            print(f"Finished {epoch + 1} epochs, elapsed time - {time() - epoch_start_time:.2}s")

            merge_history(history, train_out, key='train')
            for metric, vals in history['train'].items():
                print(f'--- train {metric} - {float(vals[-1]):.6f}')
                if logger is not None:
                    logger.report_scalar(title=metric, series='train', iteration=epoch, value=float(vals[-1]))

            merge_history(history, test_out, key='test')
            for metric, vals in history['test'].items():
                print(f'--- test {metric} - {float(vals[-1]):.6f}')
                if logger is not None:
                    logger.report_scalar(title=metric, series='test', iteration=epoch, value=float(vals[-1]))

            # log samples from model
            with torch.no_grad():
                weights_samples = self.gen_model.sample(n=log_samples, device=device)
                for log_sample in range(log_samples):
                    weights = build_nerf_weights(layers=test_dataset.get_layers(), shapes=test_dataset.get_shapes(),
                                                 raw=weights_samples[log_sample])
                    self.nerf_model.load_state_dict(weights)
                    gallery = self.nerf_model.render_views(spp=nerf_spp, batch_size=nerf_batch, device=device)
                    if logger is None:
                        Image.fromarray(gallery).save(f"sample_{log_sample}.png")
                    else:
                        logger.report_image('valid', f"sample_{log_sample}.png", iteration=epoch,
                                            image=gallery)

            # log model reconstruction capability
            with torch.no_grad():
                weights_origs = torch.stack(
                    [item['weights'].to(device) for item in choices(test_dataset, k=debug_samples)], dim=0)
                weights_recs = self.gen_model.forward({'weights': weights_origs})[0]
                for log_rec in range(len(weights_recs)):
                    weights_rec = build_nerf_weights(layers=test_dataset.get_layers(), shapes=test_dataset.get_shapes(),
                                                     raw=weights_recs[log_rec])
                    self.nerf_model.load_state_dict(weights_rec)
                    gallery_rec = self.nerf_model.render_views(spp=nerf_spp, batch_size=nerf_batch, device=device)

                    weights_orig = build_nerf_weights(layers=test_dataset.get_layers(),
                                                      shapes=test_dataset.get_shapes(),
                                                      raw=weights_origs[log_rec])
                    self.nerf_model.load_state_dict(weights_orig)
                    gallery_orig = self.nerf_model.render_views(spp=nerf_spp, batch_size=nerf_batch, device=device)

                    gallery = np.concatenate([gallery_orig, gallery_rec], axis=0)
                    if logger is None:
                        Image.fromarray(gallery).save(f"rec_{log_rec}.png")
                    else:
                        logger.report_image('valid', f"rec_{log_rec}.png", iteration=epoch,
                                            image=gallery)

            save_best(history, self.gen_model, model_name)

            # if early_stop(history, stop_patience, stop_eps):
            #     break


class NerfClassTrainer(pl.LightningModule):

    def __init__(self, dataset, n_objects, epochs=100, steps=1000, batch_rays=8192,
                 batch_objects=16, embed_shape=(16, 8, 8), embed_noise=0.1,
                 decoder_hiddens=(32, 32, 64, 64, 128, 128), positional_dim=256, attention_dim=32, lr_embed=1e-4,
                 nerf_blocks=4, nerf_hidden=32, nerf_pe=12, nerf_spp=128, coarse_weight=0.1, image_size=128,
                 accumulate_gradients=1, learning_rate=5 * 1e-4, clearml=None, val_samples=5,
                 num_groups=32, density_reg=0.0, model_out='model_last.ckpt', near=4 - 3 ** (1 / 2),
                 far=4 + 3 ** (1 / 2),
                 transmittance_reg=0.75, transmittance_warmup=5, transmittance_min=0.3, transmittance_weight=0.0,
                 warmup_epochs=3, initial_lr_ratio=0.1, min_lr_ratio=0.01):
        super(NerfClassTrainer, self).__init__()
        self.save_hyperparameters(ignore=['dataset', 'clearml'])
        self.n_objects = n_objects
        self.model_out = model_out

        self.near = near
        self.far = far
        self.nerf_spp = nerf_spp
        self.nerf_pe = nerf_pe
        self.coarse_weight = coarse_weight
        self.image_size = image_size

        self.learning_rate = learning_rate
        self.lr_embed = lr_embed
        self.batch_rays = batch_rays
        self.batch_objects = batch_objects
        self.batch_size = self.batch_rays * self.batch_objects
        self.pct_start = warmup_epochs / epochs
        self.div_factor = 1 / initial_lr_ratio
        self.final_div_factor = initial_lr_ratio / min_lr_ratio
        self.accumulate_gradients = accumulate_gradients
        self.steps = steps
        self.epochs = epochs
        self.embed_noise = embed_noise
        self.embed_shape = embed_shape
        self.embed_dim = 1
        self.val_samples = val_samples
        for shp in self.embed_shape:
            self.embed_dim *= shp

        self.density_reg = density_reg
        self.transmittance_reg = transmittance_min
        self.transmittance_min = transmittance_min
        self.transmittance_max = transmittance_reg
        self.transmittance_warmup = transmittance_warmup
        self.transmittance_weight = transmittance_weight

        self.dataset = dataset
        self.custom_logger = SimpleLogger(clearml)

        self.latents = torch.nn.Embedding(num_embeddings=self.n_objects, embedding_dim=self.embed_dim)
        torch.nn.init.normal_(self.latents.weight, 0, 1 / self.latents.embedding_dim ** (1 / 2))
        self.register_buffer('latent_std', torch.ones(self.embed_dim) / self.latents.embedding_dim ** (1 / 2))

        self.decoder = NerfLatentDecoder(latent_shape=self.embed_shape, out_dim=positional_dim * 3,
                                         attention_dim=attention_dim, hidden_dims=decoder_hiddens,
                                         num_groups=num_groups)
        self.nerf = ConditionalNeRF(latent_dim=positional_dim * 3, num_blocks=nerf_blocks, hidden_dim=nerf_hidden,
                                    pe_powers=nerf_pe)

        self.automatic_optimization = False

    def render_images(self, n_images):

        idxs = torch.tensor([sample(self.dataset.cache_keys, k=n_images)], device=self.device).long()
        latents = self.latents(idxs).view(n_images, *self.embed_shape)
        galleries = render_latent_nerf(latents=latents, model=self,
                                       w=self.image_size, h=self.image_size, focal=self.dataset.get_focal(),
                                       camera_distance=self.dataset.get_camera_distance(),
                                       near_val=self.near, far_val=self.far,
                                       batch_rays=self.batch_rays * self.batch_objects)
        return galleries

    def forward(self, latents, near, far, base_radius, poses, pixel_coords):

        n_objects = len(latents)
        encodings = self.decoder(latents)  # (b, features, dim, dim)
        # make points from batch and encode them with positional encoding and conditional encoding
        ray_o, ray_d = get_rays(poses, pixel_coords)

        dists = sample_dists(near=near, far=far, spp=self.nerf_spp)
        mu, sigma = conical_gaussians(ray_o, ray_d, dists, radius=base_radius)
        positional_features = encode_gaussians(mu, sigma, pe_powers=self.nerf_pe)
        direction_features = get_positional_encoding(normalize_vector(-ray_d), self.nerf_pe * 3).unsqueeze(1)
        positional_features = torch.cat([positional_features, direction_features.expand(positional_features.shape)],
                                        dim=-1)
        total, spp, n_features = positional_features.shape

        # encode points to latent
        nums = torch.arange(start=0, end=n_objects).type_as(latents).long().view(n_objects, 1, 1) \
            .repeat(1, total // n_objects, spp).view(total * spp)
        latents = latent_encoding(mu.view(total * spp, 3), encodings, nums)

        # render coarse pixels
        coarse_rgb, coarse_density = self.nerf.forward(positional_features.view(total * spp, n_features), latents)
        coarse_rgb, coarse_density = coarse_rgb.view(total, spp, 3), coarse_density.view(total, spp)
        coarse_pixels, coarse_weights, coarse_transmittance = render_pixels(coarse_rgb, coarse_density, dists)

        # adaptively find optimal distances based on coarse sampling
        dists = adaptive_sample_dists(near=near, far=far, spp=self.nerf_spp, coarse_dists=dists,
                                      weights=coarse_weights)
        mu, sigma = conical_gaussians(ray_o, ray_d, dists, radius=base_radius)
        positional_features = encode_gaussians(mu, sigma, pe_powers=self.nerf_pe)
        positional_features = torch.cat([positional_features, direction_features.expand(positional_features.shape)],
                                        dim=-1)
        total, spp, n_features = positional_features.shape
        nums = torch.arange(start=0, end=n_objects).type_as(latents).long().view(n_objects, 1, 1) \
            .repeat(1, total // n_objects, spp).view(total * spp)
        latents = latent_encoding(mu.view(total * spp, 3), encodings, nums)

        # render pixels
        fine_rgb, fine_density = self.nerf.forward(positional_features.view(total * spp, n_features), latents)
        fine_rgb, fine_density = fine_rgb.view(total, spp, 3), fine_density.view(total, spp)
        fine_pixels, fine_weights, fine_transmittance = render_pixels(fine_rgb, fine_density, dists)

        return {
            'coarse_pixels': coarse_pixels,
            'fine_pixels': fine_pixels,
            'transmittance': torch.cat([coarse_transmittance, fine_transmittance], dim=-1),
            # 'transmittance': coarse_transmittance,
            'density': torch.cat([coarse_density, fine_density], dim=-1)
            # 'density': coarse_density
        }

    def render(self, batch, train=True):

        # take embedding for scene and decode it to positional features
        idxs = batch['id'].view(-1)
        n_objects = len(idxs)
        latent = self.latents.forward(idxs)
        if train:
            latent = latent + torch.randn_like(latent) * self.latent_std.unsqueeze(0) * self.embed_noise
        latent = latent.view(n_objects, *self.embed_shape)

        near, far, base_radius = batch['near'].view(-1), batch['far'].view(-1), batch['base_radius'].view(-1)
        poses = batch['poses'].view(-1, 3, 4)
        pixel_coords = batch['pixel_coords'].view(-1, 3)
        return self(latents=latent, near=near, far=far, base_radius=base_radius, poses=poses,
                    pixel_coords=pixel_coords)

    def loss(self, out, gt_pixels):
        loss = {
            'mse_loss': torch.nn.functional.mse_loss(out['fine_pixels'], gt_pixels)
                        + torch.nn.functional.mse_loss(out['coarse_pixels'], gt_pixels) * self.coarse_weight,
            'mean_density': torch.mean(out['density']),
            'mean_transmittance': torch.mean(out['transmittance'])
        }
        loss['loss'] = loss['mse_loss']
        loss['loss'] += loss['mean_density'] * self.density_reg
        loss['loss'] += -torch.clamp(loss['mean_transmittance'], max=self.transmittance_reg) * self.transmittance_weight
        return loss

    def training_step(self, batch, batch_idx):
        # batch = {k: t.to(self.device) for k, t in batch.items()}
        gt_pixels = batch['pixels'].view(-1, 3)

        opt_model, opt_embed = self.optimizers()

        out = self.render(batch)

        loss = self.loss(out, gt_pixels)
        self.manual_backward(loss['loss'])

        if batch_idx % self.accumulate_gradients == 0:
            opt_model.step()
            opt_embed.step()
            opt_model.zero_grad(set_to_none=True)
            opt_embed.zero_grad(set_to_none=True)

        sch_model, sch_embed = self.lr_schedulers()

        # step for OneCycleLr
        sch_model.step()
        sch_embed.step()

        self.log('train_loss', loss['loss'], prog_bar=True, sync_dist=True)
        self.log('train_mse', loss['mse_loss'], prog_bar=True, sync_dist=True)
        self.log('train_density', loss['mean_density'], prog_bar=True, sync_dist=True)
        self.log('train_transmittance', loss['mean_transmittance'], prog_bar=True, sync_dist=True)
        self.log('model_learning_rate', sch_model.get_last_lr()[0], prog_bar=True, sync_dist=True)
        self.log('embed_learning_rate', sch_embed.get_last_lr()[0], prog_bar=True, sync_dist=True)

    def on_train_epoch_start(self):
        if self.current_epoch <= self.transmittance_warmup:
            current_std = torch.std(self.latents.weight.detach(), dim=0)
            self.latent_std = torch.maximum(current_std, torch.ones_like(current_std) / self.embed_dim ** (1 / 2))
            self.transmittance_reg = self.current_epoch / self.transmittance_warmup \
                                     * (self.transmittance_max - self.transmittance_min) + self.transmittance_min

    def validation_step(self, batch, batch_idx):
        # batch = {k: t.to(self.device) for k, t in batch.items()}
        gt_pixels = batch['pixels'].view(-1, 3)
        out = self.render(batch, train=False)
        loss = self.loss(out, gt_pixels)
        self.log('val_loss', loss['loss'], prog_bar=True, sync_dist=True)
        self.log('val_mse', loss['mse_loss'], prog_bar=True, sync_dist=True)
        self.log('val_density', loss['mean_density'], prog_bar=True, sync_dist=True)
        self.log('val_transmittance', loss['mean_transmittance'], prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self):
        images = self.render_images(self.val_samples)
        self.custom_logger.log_images(images, 'sample', self.current_epoch)

        model = {
            'latents': self.latents,
            'decoder': self.decoder,
            'nerf': self.nerf
        }
        torch.save(model, self.model_out)

    def configure_optimizers(self):
        opt_model = torch.optim.Adam(params=list(self.decoder.parameters()) + list(self.nerf.parameters()),
                                     lr=self.learning_rate, betas=(0.5, 0.9))
        opt_embed = torch.optim.Adam(params=self.latents.parameters(), lr=self.lr_embed, betas=(0.5, 0.9))
        scheduler_model = torch.optim.lr_scheduler.OneCycleLR(opt_model, max_lr=self.learning_rate,
                                                              pct_start=self.pct_start, div_factor=self.div_factor,
                                                              final_div_factor=self.final_div_factor,
                                                              epochs=self.epochs, steps_per_epoch=self.steps)
        scheduler_embed = torch.optim.lr_scheduler.OneCycleLR(opt_embed, max_lr=self.lr_embed, pct_start=self.pct_start,
                                                              div_factor=self.div_factor,
                                                              final_div_factor=self.final_div_factor,
                                                              epochs=self.epochs, steps_per_epoch=self.steps)
        return [opt_model, opt_embed], [scheduler_model, scheduler_embed]

    def train_dataloader(self):
        self.dataset.reset_cache()
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_objects, shuffle=False,
                                           num_workers=torch.cuda.device_count() * 2,
                                           pin_memory=False, drop_last=False, prefetch_factor=1)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_objects, shuffle=False,
                                           num_workers=torch.cuda.device_count() * 2,
                                           pin_memory=False, drop_last=False, prefetch_factor=1)


class LatentDiffusion(Diffusion):

    def __init__(self, shape, unet_hiddens, dataset=None, decoder_path=None,
                 features_dim=0, steps=10000, batch_size=32, learning_rate=1e-4,
                 min_lr_rate=0.01, attention_dim=16, epochs=100, diffusion_steps=1000, sample_steps=128,
                 is_latent=True, kl_weight=1e-3,
                 min_beta=1e-4, max_beta=1e-2, clearml=None, log_samples=5, img_size=128, focal=1.5):
        self.save_hyperparameters(ignore=['clearml', 'dataset'])
        model = UNetDenoiser(shape=shape, steps=diffusion_steps, hidden_dims=unet_hiddens,
                             attention_dim=attention_dim)
        super(LatentDiffusion, self).__init__(dataset=dataset, model=model, diffusion_steps=diffusion_steps,
                                              learning_rate=learning_rate, min_lr_rate=min_lr_rate,
                                              sample_steps=sample_steps, batch_size=batch_size, min_beta=min_beta,
                                              max_beta=max_beta, kl_weight=kl_weight)

        self.decoder_path = decoder_path
        self.is_latent = is_latent
        self.shape = shape
        self.features_dim = features_dim
        self.features = self.shape[features_dim]

        self.log_samples = log_samples
        self.img_size = img_size
        self.focal = focal

        self.custom_logger = SimpleLogger(clearml)

    def sample(self, n_samples):
        x = torch.randn((n_samples, *self.shape), device=self.device)
        sample_steps = self.get_sample_steps()
        for t_curr, t_prev in zip(sample_steps[:-1], sample_steps[1:]):
            ones = torch.ones(n_samples).type_as(x).long()
            x = self.p_sample_stride(x, ones * t_prev, ones * t_curr)
        return x

    def forward(self, x, t):
        return self.model(x, t)

    def step(self, batch):
        noise = torch.randn_like(batch)
        t = torch.randint(low=0, high=self.diffusion_steps - 1, size=[len(batch)]).type_as(batch).long()

        latent_noised = self.q_sample(batch, t, noise)
        eps, var_weight = self.forward(latent_noised, t)

        return self.get_losses(batch, latent_noised, t, noise, eps, var_weight)

    def on_validation_epoch_end(self):
        with torch.no_grad():
            h = self.sample(self.log_samples).detach()
        if not self.is_latent:
            # log sampled images
            images = h.cpu().numpy()
            images = np.clip(np.moveaxis(images, 1, -1), -1.0, 1.0)
            images = ((images + 1.0) * (255 / 2)).astype(np.uint8)
            images = [images[image_id] for image_id in range(len(images))]
            self.custom_logger.log_images(images, 'sample', epoch=self.current_epoch)
        elif self.decoder_path:
            model = NerfClassTrainer.load_from_checkpoint(self.decoder_path, map_location=self.device, dataset=None) \
                .to(self.device)
            galleries = render_latent_nerf(h, model, w=self.img_size, h=self.img_size, focal=self.focal)
            self.custom_logger.log_images(galleries, 'samples', epoch=self.current_epoch)

            del model, galleries
            gc.collect()

        # log current lr
        self.log('learning_rate', self.lr_schedulers().get_last_lr()[0], prog_bar=True)


class MultiVAETrainer(pl.LightningModule):

    def __init__(self, path, epochs=100, steps=300, views=64, latent_dim=256,
                 hidden_dims=(8, 8, 16, 16, 32, 32, 64, 64), kl_weight=5 * 1e-4, attention_dim=16,
                 learning_rate=1e-3, batch_size=32, min_lr_ratio=0.01, initial_lr_ratio=0.1, warmup_epochs=3,
                 img_size=64, n_views=64, cache_size=-1, sample_views=10, clearml=None, val_samples=5):
        super(MultiVAETrainer, self).__init__()
        # data loading
        self.n_views = n_views
        self.dataset = NerfViews(path=path, batch_size=batch_size, image_size=img_size, n_views=n_views,
                                 sample=sample_views, cache_size=cache_size)
        # training parameters
        self.epochs = epochs
        self.steps = steps
        self.batch_size = batch_size
        self.kl_weight = kl_weight
        self.learning_rate = learning_rate
        self.min_lr_ratio = min_lr_ratio
        self.initial_lr_ratio = initial_lr_ratio
        self.pct_start = warmup_epochs / epochs
        self.div_factor = 1 / initial_lr_ratio
        self.final_div_factor = 1 / min_lr_ratio * initial_lr_ratio
        self.latent_dim = latent_dim
        # logging parameters
        self.custom_logger = SimpleLogger(clearml)
        self.val_samples = val_samples
        # models
        self.encoder = VAEEncoder2D(latent_dim=latent_dim, hidden_dims=hidden_dims, attention_dim=attention_dim,
                                    shape=(3, img_size, img_size))
        self.decoders = [Decoder2D(shape=self.encoder.latent_shape, hidden_dims=hidden_dims[::-1],
                                   attention_dim=attention_dim) for _ in range(views)]

    def forward(self, x, ids):
        latents, mu, logsigma = self.encoder.forward(x)
        preds = [self.decoders[idx].forward(latents) for idx in ids]

        return {
            'preds': preds,
            'latents': latents,
            'mu': mu,
            'logsigma': logsigma
        }

    def loss(self, out, gts):
        loss = {
            'kl_loss': self.encoder.kl_loss(mu=out['mu'], logsigma=out['logsigma']),
            'rec_loss': torch.sum(torch.stack([torch.nn.functional.mse_loss(pred, gt)
                                               for pred, gt in zip(out['preds'], gts)]))
        }
        loss['loss'] = loss['rec_loss'] + self.kl_weight * loss['kl_loss']
        return loss

    def training_step(self, batch):
        ids = batch['ids']
        data = [d.to(self.device) for d in batch['data']]
        out = self.forward(data[0], ids)
        loss = self.loss(out, data[1:])
        self.log('train_loss', loss['loss'], prog_bar=True)
        self.log('train_kl_loss', loss['kl_loss'], prog_bar=True)
        self.log('train_rec_loss', loss['rec_loss'], prog_bar=True)
        return loss['loss']

    def validation_step(self, batch):
        ids = batch['ids']
        data = [d.to(self.device) for d in batch['data']]
        out = self.forward(data[0], ids)
        loss = self.loss(out, data[1:])
        self.log('val_loss', loss['loss'], prog_bar=True)
        self.log('val_kl_loss', loss['kl_loss'], prog_bar=True)
        self.log('val_rec_loss', loss['rec_loss'], prog_bar=True)
        return loss['loss']

    def sample(self, n_samples, view_id):
        latent = torch.randn(n_samples, self.latent_dim, device=self.device)
        with torch.no_grad():
            views = self.decoders[view_id].forward(latent)
        return views

    def on_validation_epoch_end(self):

        scene_indexes = self.dataset.get_cache_indexes(self.val_samples)
        inputs = self.dataset.get_cache_items(scene_indexes,
                                              np.random.randint(low=0, high=self.n_views, size=self.val_samples))
        ids = choices(range(self.n_views), k=self.val_samples)
        with torch.no_grad():
            views = self.forward(torch.from_numpy(inputs).to(self.device), ids)['preds']

        inputs = denormalize_image(inputs)
        for view_id, view in enumerate(views):
            view = denormalize_image(view.detach().cpu().numpy())
            orig = denormalize_image(self.dataset.get_cache_items(scene_indexes,
                                                                  np.full(self.val_samples, fill_value=ids[view_id],
                                                                          dtype=np.int32)))
            # n h w 3 -> h w*n 3
            n, h, w, _ = inputs.shape
            gallery = np.concatenate(map(lambda x: x.moveaxis(0, 1).reshape(h, n * w, 3), [inputs, orig, view]), axis=0)
            self.custom_logger.log_image(gallery, f'view_rec_{ids[view_id]}', self.current_epoch)

        for idx in ids:
            images = self.sample(self.val_samples, idx)
            images = denormalize_image(images.detach().cpu().numpy())
            n, h, w, _ = images.shape
            gallery = images.moveaxis(0, 1).reshape(h, n * w, 3)
            self.custom_logger.log_image(gallery, f'sample_view_{idx}', self.current_epoch)

    def configure_optimizers(self):
        params = self.encoder.parameters()
        for decoder in self.decoders:
            params += decoder.parameters()
        opt = torch.optim.Adam(params=params, lr=self.learning_rate, betas=(0.5, 0.9))
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
        self.dataset.reset_cache()
        return torch.utils.data.DataLoader(self.dataset, batch_size=None, shuffle=False,
                                           num_workers=torch.cuda.device_count() * 2,
                                           pin_memory=True, drop_last=False, prefetch_factor=2)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=None, shuffle=False,
                                           num_workers=torch.cuda.device_count() * 2,
                                           pin_memory=True, drop_last=False, prefetch_factor=2)


class NVSDiffusion(Diffusion):

    def __init__(self, shape, xunet_hiddens, dataset=None, dropout=0.0, classifier_free=0.1,
                 batch_size=128, learning_rate=1e-4, min_lr_rate=0.1, attention_dim=16, diffusion_steps=256,
                 sample_steps=128, kl_weight=1e-3,
                 min_beta=1e-4, max_beta=1e-2, clearml=None, log_samples=5, log_length=16, focal=1.5):
        self.save_hyperparameters(ignore=['clearml', 'dataset'])
        model = XUNetDenoiser(shape=shape, steps=diffusion_steps, hidden_dims=xunet_hiddens,
                              attention_dim=attention_dim, dropout=dropout)
        super(NVSDiffusion, self).__init__(dataset=dataset, model=model, diffusion_steps=diffusion_steps,
                                           learning_rate=learning_rate, batch_size=batch_size, min_lr_rate=min_lr_rate,
                                           min_beta=min_beta, max_beta=max_beta, sample_steps=sample_steps,
                                           kl_weight=kl_weight)

        self.shape = shape
        self.in_features, self.h, self.w = shape
        self.focal = focal

        self.classifier_free = classifier_free

        self.log_samples = log_samples
        self.log_length = log_length
        self.custom_logger = SimpleLogger(clearml)

    def sample(self, n_seq, cond, ray_o, ray_d, dist=4.0):
        n_samples = len(cond)
        cond_sequence, ray_o_sequence, ray_d_sequence = [cond], [ray_o], [ray_d]
        for item_id in range(n_seq):
            x = torch.randn((n_samples, *self.shape)).type_as(cond)
            poses = get_random_poses(torch.ones(n_samples).type_as(cond) * dist)
            ray_o, ray_d = get_images_rays(h=self.h, w=self.w,
                                           focal=torch.ones(n_samples).type_as(cond) * self.focal, poses=poses)
            # b h w 3 -> b 3 h w
            ray_d = torch.movedim(ray_d, -1, 1)
            for t in reversed(range(self.diffusion_steps)):
                idx = randint(0, len(cond_sequence) - 1)
                x = torch.cat([x] + [cond_sequence[idx]], dim=0)
                t = torch.cat([torch.ones(n_samples).type_as(x) * t, -torch.ones(n_samples).type_as(x)], dim=0).long()
                _ray_o = torch.cat([ray_o] + [ray_o_sequence[idx]], dim=0)
                _ray_d = torch.cat([ray_d] + [ray_d_sequence[idx]], dim=0)
                x = torch.chunk(self.p_sample(x, t, ray_o=_ray_o, ray_d=_ray_d), 2, dim=0)[0]
            cond_sequence.append(x)
            ray_o_sequence.append(ray_o)
            ray_d_sequence.append(ray_d)
        return cond_sequence

    def forward(self, x, t, ray_o, ray_d):
        return self.model(x, t, ray_o, ray_d)

    def step(self, batch):
        # extract origin view
        x = batch['view']
        noise = torch.randn_like(x)
        t = torch.randint(low=0, high=self.diffusion_steps - 1, size=(len(x),)).type_as(batch['focal']).long()
        x_noised = self.q_sample(x, t, noise)
        # extract conditional view with classifier free guidance
        clf_free_msk = torch.rand(len(x)).type_as(x) > self.classifier_free
        x_cond = torch.where(clf_free_msk[:, None, None, None], batch['cond'], torch.randn_like(batch['cond']))
        # extract positional information
        origin_ray_o, origin_ray_d = get_images_rays(self.h, self.w, batch['focal'], batch['view_poses'])
        cond_ray_o, cond_ray_d = get_images_rays(self.h, self.w, batch['focal'], batch['cond_poses'])
        # concat origin with conditional to run them simultaneously
        ray_o = torch.cat([origin_ray_o, cond_ray_o], dim=0)
        ray_d = torch.cat([origin_ray_d, cond_ray_d], dim=0).movedim(-1, 1)  # b h w 3 -> b 3 h w
        x_noised_pair = torch.cat([x_noised, x_cond], dim=0)
        t_pair = torch.cat([t, -torch.ones_like(t)], dim=0)
        eps, var_weight = self.forward(x_noised_pair, t_pair, ray_o, ray_d)
        eps, var_weight = torch.chunk(eps, 2, dim=0)[0], torch.chunk(var_weight, 2, dim=0)
        return self.get_losses(x, x_noised, t, noise, eps, var_weight)

    def on_validation_epoch_end(self):
        data_iter = iter(self.trainer.val_dataloaders[0])
        batch = next(data_iter)
        cond = batch['cond'][:self.log_samples].to(self.device)
        poses = batch['cond_poses'][:self.log_samples].to(self.device)
        focal = batch['focal'][:self.log_samples].to(self.device)
        ray_o, ray_d = get_images_rays(self.h, self.w, focal, poses)
        with torch.no_grad():
            images = self.sample(self.log_length - 1, cond, ray_o, ray_d.movedim(-1, 1))
        # seq b 3 h w -> b seq 3 h w -> b seq h w 3
        images = torch.stack(images, dim=0).transpose(0, 1).movedim(2, -1)
        b, seq, h, w, d = images.shape
        images = images.reshape(b, 4, seq // 4, h, w, d).transpose(2, 3).reshape(b, 4 * h, seq // 4 * w,
                                                                                 d).cpu().numpy()
        galleries = [denormalize_image(images[idx]) for idx in range(b)]
        self.custom_logger.log_images(galleries, 'sample', self.current_epoch)

        del images, batch, galleries
        gc.collect()
        torch.cuda.empty_cache()

    def train_dataloader(self):
        self.dataset.reset_cache()
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False,
                                           num_workers=2 * torch.cuda.device_count(),
                                           pin_memory=True, drop_last=False, prefetch_factor=2)
