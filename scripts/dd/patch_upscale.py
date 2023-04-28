from modules.common.util import *
from modules.common.model import *
from modules.common.trainer import *
from scripts.dd.swin2sr import Swin2SR

import numpy as np
import os
import cv2
from random import choice, randint, shuffle, choices, random
import gc

import torch
import torchvision
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
from PIL import Image
import matplotlib.pyplot as plt
from einops import rearrange, reduce, repeat
import requests
from concurrent.futures import ProcessPoolExecutor

from argparse import ArgumentParser
import clearml


def nonlinear(x):
    return torch.nn.functional.silu(x)


def get_YUV(R, G, B):
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = 0.5 - 0.168736 * R - 0.331264 * G + 0.5 * B
    Cr = 0.5 + 0.5 * R - 0.418688 * G - 0.081312 * B
    return Y, Cb, Cr


def get_RGB(Y, Cb, Cr):
    R = Y + 1.402 * (Cr - 0.5)
    G = Y - 0.344136 * (Cb - 0.5) - 0.714136 * (Cr - 0.5)
    B = Y + 1.772 * (Cb - 0.5)
    return R, G, B


def convert_RGB2YUV(image):
    R = image[..., 0]
    G = image[..., 1]
    B = image[..., 2]

    return np.stack(get_YUV(R, G, B), axis=-1)


def convert_YUV2RGB(image):
    Y = image[..., 0]
    Cb = image[..., 1]
    Cr = image[..., 2]

    return np.stack(get_RGB(Y, Cb, Cr), axis=-1)


def torch_convert_RGB2YUV(tensor):
    R = tensor[:, 0]
    G = tensor[:, 1]
    B = tensor[:, 2]
    return torch.stack(get_YUV(R, G, B), dim=1)


def torch_convert_YUV2RGB(tensor):
    Y = tensor[:, 0]
    Cb = tensor[:, 1]
    Cr = tensor[:, 2]
    return torch.stack(get_RGB(Y, Cb, Cr), dim=1)


def to_tensor(image, yuv=True):
    image = image.astype(np.float32) / 255.0
    if yuv:
        image = convert_RGB2YUV(image)
    image = image * 2 - 1.0
    return torch.tensor(image).movedim(-1, -3)


def to_image(tensor, yuv=True):
    image = (tensor.movedim(-3, -1) * 0.5 + 0.5).detach().cpu().numpy()
    if yuv:
        image = convert_YUV2RGB(image)
    image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    return image


def multiscale_loss(x1, x2, scales=(1, 2), eps=1e-6):
    loss = 0
    for scale in scales:
        __x1 = torch.nn.functional.interpolate(x1, scale_factor=1 / scale, mode='bicubic')
        __x2 = torch.nn.functional.interpolate(x2, scale_factor=1 / scale, mode='bicubic')
        loss += torch.sqrt((__x1 - __x2) ** 2 + eps ** 2).mean()
    return loss / len(scales)


def gram_loss(features1, features2):
    features1, features2 = map(lambda t: rearrange(t, 'b c h w -> b c (h w)'), [features1, features2])
    gram1, gram2 = map(lambda t: torch.bmm(t, t.transpose(-1, -2)), [features1, features2])
    return torch.nn.functional.mse_loss(gram1, gram2)


class RandomBinaryFilter(torch.nn.Module):

    def __init__(self, in_channels=1, filters=48, kernel_size=5):
        super(RandomBinaryFilter, self).__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=in_channels,
            bias=False)

        filters = torch.bernoulli(torch.torch.full(self.conv.weight.data.shape, 0.5)) * 2 - 1
        filters[torch.rand(filters.shape) > 0.75] = 0

        self.conv.weight.data.copy_(filters)
        self.conv.weight.requires_grad_(False)

    def forward(self, x):
        return self.conv(x)


class FeatureBlock(torch.nn.Module):

    def __init__(self, dim, in_dim=-1, kernel_size=3, group=4, fft=False):
        super(FeatureBlock, self).__init__()
        self.fft = fft
        self.in_dim = in_dim if in_dim > 0 else dim

        self.conv1 = torch.nn.Conv2d(in_dim, dim, kernel_size=kernel_size, padding=1, groups=self.in_dim // group)
        self.conv1_linear = torch.nn.Conv2d(dim, dim, kernel_size=1)

        self.conv2 = torch.nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=1, groups=dim // group)
        self.conv2_linear = torch.nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        h = self.conv1_linear(self.conv1(nonlinear(x)))
        if self.fft:
            h = torch.fft.fft2(x, norm='ortho').real
        h = self.conv2_linear(self.conv2(nonlinear(h)))
        if self.fft:
            h = torch.fft.fft2(x, norm='ortho').real

        return (h + x) / 2 ** 0.5


class ResBlock(torch.nn.Module):

    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.norm1 = norm(dim)
        self.conv1 = torch.nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.norm2 = norm(dim)
        self.conv2 = torch.nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        h = self.conv1(nonlinear(self.norm1(x)))
        h = self.conv2(nonlinear(self.norm2(h)))
        return (x + h) / 2 ** 0.5


class PatchDiscriminator(torch.nn.Module):

    def __init__(self, in_channels=1, dim=128, n_blocks=2):
        super(PatchDiscriminator, self).__init__()
        self.first_conv = torch.nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)
        self.feature_extractor = torch.nn.Sequential(*[ResBlock(dim) for _ in range(n_blocks)])
        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.utils.spectral_norm(torch.nn.Linear(dim, 1)),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        h = self.first_conv(x)
        features = self.feature_extractor(h)
        probs = self.classifier(features)
        return probs.view(-1)


class PatchEnhancer(torch.nn.Module):

    def __init__(self, in_channels=1, dim=32, tile_pad=2, n_blocks=4, scale=2, fft=False):
        super(PatchEnhancer, self).__init__()
        self.in_channels = in_channels
        self.tile_pad = tile_pad
        self.scale = scale
        self.input_conv = torch.nn.Conv2d(in_channels, dim, kernel_size=1 + self.tile_pad * 2, padding=0, stride=1)
        self.blocks = torch.nn.ModuleList([FeatureBlock(dim, dim, fft=fft) for _ in range(n_blocks)])
        self.out = torch.nn.Conv2d(dim // self.scale ** 2, in_channels, kernel_size=3, padding=1)

    def forward(self, x, return_features=False):
        upscaled = torch.nn.functional.interpolate(x, scale_factor=self.scale, mode='bilinear')

        # extract channels
        h = x[:, :self.in_channels]

        # first conv with no padding to use tile_pad
        h = self.input_conv(h)

        # deep feature extraction
        for block in self.blocks:
            h = block(h)

        features = h

        h = torch.nn.functional.pixel_shuffle(h, upscale_factor=self.scale)
        # output conv
        h = self.out(nonlinear(h))

        # get updated and remained part
        updated = upscaled[:, :self.in_channels, self.tile_pad * self.scale: -self.tile_pad * self.scale,
                  self.tile_pad * self.scale: -self.tile_pad * self.scale]
        remained = upscaled[:, self.in_channels:, self.tile_pad * self.scale: -self.tile_pad * self.scale,
                   self.tile_pad * self.scale: -self.tile_pad * self.scale]

        result = torch.cat((updated + h, remained), dim=-3)
        if return_features:
            return result, features

        return result


class SRImagesBase:

    def __init__(self, folder, tile=16, tile_pad=2, scale=2, cache_size=512, w=2560, h=1440):
        super(SRImagesBase, self).__init__()
        self.w = w
        self.h = h
        self.tile = tile
        self.tile_pad = tile_pad
        self.scale = scale
        if type(folder) is not list:
            folder = [folder]
        self.folder = folder
        self.files = self.find_files()

        self.cache_size = cache_size
        self.cache = []

    def find_files(self):
        self.files = [os.path.join(f, file) for f in self.folder for file in os.listdir(f) if
                      os.path.splitext(file)[1] in ('.png', '.jpg', '.jpeg')]
        return self.files

    def load_file(self, file, resize=False):
        frame = cv2.imread(file)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize:
            h, w = frame.shape[:2]
            frame_ratio = w / h
            if frame_ratio > self.w / self.h:
                # width is bigger so let's crop it
                true_w = int(h * self.w / self.h)
                start_w = int((w - true_w) / 2)
                frame = frame[:, start_w: start_w + true_w]
            else:
                # height is bigger
                true_h = int(w / self.w * self.h)
                start_h = int((h - true_h) / 2)
                frame = frame[start_h: start_h + true_h]
            frame = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_AREA)
        return to_tensor(frame)

    def load_files(self, resize=True):
        return [self.load_file(file, resize=resize) for file in self.files]

    def reset_cache(self):

        if self.cache_size <= 0:
            raise RuntimeError('Caching is not enabled, use cache_size>0 when initializing dataset')

        cache_files = []
        for folder in self.folder:
            folder_files = [os.path.join(folder, file) for file in os.listdir(folder) if
                            os.path.splitext(file)[1] in ('.png', '.jpg', '.jpeg')]
            shuffle(folder_files)
            cache_files.extend(folder_files[:self.cache_size // len(self.folder)])

        self.cache = [self.load_file(file) for file in cache_files]
        gc.collect()

    def load_patch(self, tensor):

        c, h, w = tensor.shape

        tile = (self.tile + self.tile_pad * 2) * self.scale
        i_start = randint(0, h - tile - 1)
        j_start = randint(0, w - tile - 1)
        sr_patch = tensor[:, i_start:i_start + tile, j_start:j_start + tile]

        lr_patch = sr_patch
        # random noise
        if random() < 0.5:
            lr_patch += torch.randn_like(lr_patch) * 0.05
        # random blurring
        if random() < 0.5:
            ks = choice([1, 3, 5])
            lr_patch = torchvision.transforms.functional.gaussian_blur(lr_patch, (ks, ks))
        # random method down sampling
        lr_patch = torch.nn.functional.interpolate(lr_patch.unsqueeze(0), scale_factor=1 / self.scale,
                                                   mode=choice(['bilinear', 'bicubic', 'nearest-exact', 'area']))[0]

        return {
            'lr': lr_patch,
            'sr': sr_patch
        }


class PatchSRImages(SRImagesBase, torch.utils.data.IterableDataset):
    def __iter__(self):
        return self

    def __next__(self):
        if self.cache_size < 0:
            # update list in case of new files
            file = choice(self.find_files())
            return self.load_patch(self.load_file(file))
        else:
            return self.load_patch(choice(self.cache))


class FullSRImages(SRImagesBase, torch.utils.data.Dataset):

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.load_file(self.find_files()[idx], resize=True)


class ImageEnhancer(pl.LightningModule):

    def __init__(self,
                 model=None,
                 disc=None,
                 dataset=None,
                 test_dataset=None,
                 clearml=None,
                 in_channels=1,
                 dim=16,
                 teacher=None,
                 teacher_rate=1.0,
                 teacher_dim=60,
                 teacher_w=1e-4,
                 ema_weight=0.995,
                 learning_rate=1e-3,
                 disc_lr=1e-4,
                 disc_w=1e-3,
                 disc_warmup=3,
                 initial_lr_rate=0.1,
                 min_lr_rate=0.01,
                 batch_size=1024,
                 full_batch_size=1,
                 epochs=300,
                 steps=1000,
                 samples_epoch=5,
                 scale=2,
                 tile=16,
                 tile_pad=2,
                 debug=True,
                 random_filters=64,
                 random_filters_kernel=5,
                 ):
        super(ImageEnhancer, self).__init__()

        self.save_hyperparameters(
            ignore=['model', 'disc', 'dataset' 'test_dataset', 'test_images', 'clearml', 'teacher'])
        self.automatic_optimization = False

        self.scale = scale
        self.tile = tile
        self.tile_pad = tile_pad
        self.in_channels = in_channels

        self.custom_logger = SimpleLogger(clearml=clearml)
        self.debug = debug

        self.dataset = dataset
        self.test_dataset = test_dataset

        self.teacher = teacher if exists(teacher) else None
        self.teacher_rate = teacher_rate
        self.teacher_w = teacher_w
        if self.teacher:
            self.teacher_projection = torch.nn.Conv2d(dim, teacher_dim, kernel_size=1)

        self.random_filter = RandomBinaryFilter(in_channels=in_channels, filters=random_filters,
                                                kernel_size=random_filters_kernel)

        self.model = model
        self.disc = disc
        self.use_ema = ema_weight is not None
        if self.use_ema:
            self.ema_model = EMA(self.model, decay=ema_weight)

        self.learning_rate = learning_rate
        self.disc_lr = disc_lr
        self.disc_w = disc_w
        self.disc_warmup = disc_warmup
        self.min_lr_rate = min_lr_rate
        self.initial_lr_rate = initial_lr_rate
        self.batch_size = batch_size
        self.full_batch_size = full_batch_size
        self.steps = steps
        self.epochs = epochs
        self.samples_epoch = samples_epoch

    def forward(self, x, train=True, **kwargs):

        model = cases([
            (self.use_ema and (not train or not self.training), self.ema_model.module),
            self.model
        ])

        return model(x, return_features=train, **kwargs)

    def get_sr(self, batch):

        # remove padding from orig batch
        orig = batch['sr']
        orig = orig[:, :,
               self.tile_pad * self.scale: -self.tile_pad * self.scale,
               self.tile_pad * self.scale: -self.tile_pad * self.scale]
        return orig

    def upscale_step(self, batch):

        # enhance it
        lr = batch['lr']
        upscaled, features = self.forward(lr, train=True)

        # return loss on real SR
        return self.get_losses(upscaled, lr, self.get_sr(batch), features)

    def disc_step(self, batch):

        with torch.no_grad():
            upscaled = self.forward(batch['lr'], train=True)[0]

        # discriminator must learn how fake the example looks like
        preds = self.disc(torch.cat([upscaled, self.get_sr(batch)], dim=0))
        target = torch.ones_like(preds)
        target[len(upscaled):] = 0

        values = target * torch.log(preds + 1e-6) + (1 - target) * torch.log(1 - preds + 1e-6)
        return - torch.mean(torch.nan_to_num(values, nan=0, posinf=1, neginf=-1))

    def base_metrics(self, x, sr):
        x_normalized = torch_convert_YUV2RGB(x * 0.5 + 0.5).clip(0, 1)
        sr_normalized = torch_convert_YUV2RGB(sr * 0.5 + 0.5).clip(0, 1)
        return {
            'psnr': torchmetrics.functional.peak_signal_noise_ratio(x_normalized, sr_normalized, data_range=1.0),
            'ssim': torchmetrics.functional.structural_similarity_index_measure(
                x_normalized, sr_normalized, data_range=1.0)
        }

    @torch.inference_mode()
    def call_teacher(self, lr, window_size=8):
        # upscale model with teacher
        patches = torch_convert_YUV2RGB(lr * 0.5 + 0.5)

        # pad for SwinSR
        tile_size = self.tile + self.tile_pad * 2
        pad_size = (tile_size // window_size + 1) * window_size - tile_size
        patches = torch.cat([patches, torch.flip(patches, [2])], 2)[:, :, :tile_size + pad_size, :]
        patches = torch.cat([patches, torch.flip(patches, [3])], 3)[:, :, :, :tile_size + pad_size]

        patches, features = self.teacher(patches, return_features=True)[:, :,
                            self.tile_pad * self.scale: -(self.tile_pad + pad_size) * self.scale,
                            self.tile_pad * self.scale: -(self.tile_pad + pad_size) * self.scale]

        patches = torch_convert_RGB2YUV(patches) * 2 - 1.0

        return patches, features

    def get_losses(self, x, lr, sr, features):

        teacher_loss = {}
        if self.teacher:
            sr, teacher_features = self.call_teacher(lr)
            features = self.teacher_projection(features)
            teacher_loss['teacher_loss'] = gram_loss(features, teacher_features)

        x = x[:, :self.in_channels]
        sr = sr[:, :self.in_channels]
        # loss on downsampled images to reduce artifacts
        aux_loss = multiscale_loss(torch.cat([x, self.random_filter(x)], dim=1),
                                   torch.cat([sr, self.random_filter(sr)], dim=1))

        # loss on high frequency details compared with gaussian blurred
        x_details = x - torchvision.transforms.functional.gaussian_blur(x, kernel_size=(5, 5))
        sr_details = sr - torchvision.transforms.functional.gaussian_blur(sr, kernel_size=(5, 5))
        high_freq_loss = torch.nn.functional.l1_loss(x_details, sr_details)

        # adversarial based loss
        adv_loss = {}
        if self.disc and self.current_epoch >= self.disc_warmup:
            # we want to minimize example falseness estimated by discriminator
            adv_loss['adv_loss'] = torch.mean(self.disc(x))

        return {
            'aux_loss': aux_loss,
            'high_freq': high_freq_loss,
            **adv_loss,
            **teacher_loss,
            'loss': aux_loss + high_freq_loss + (
                    adv_loss['adv_loss'] if 'adv_loss' in adv_loss else 0) * self.disc_w + (
                        teacher_loss['teacher_loss'] if 'teacher_loss' in teacher_loss else 0) * self.teacher_w
        }

    @torch.inference_mode()
    def upscale_images(self, images):

        b, c, h, w = images.shape

        # divide image to patches
        __tiles = rearrange(images, 'b c (n p1) (m p2) -> b c n m p1 p2', n=h // self.tile, m=w // self.tile)

        # add up and down padding in height
        up_pad = __tiles[:, :, 1:, :, :self.tile_pad, :]
        up_pad = torch.cat([up_pad, torch.zeros_like(up_pad[:, :, :1])], dim=2)
        down_pad = __tiles[:, :, :-1, :, -self.tile_pad:, :]
        down_pad = torch.cat([torch.zeros_like(down_pad[:, :, :1]), down_pad], dim=2)
        __tiles = torch.cat([down_pad, __tiles, up_pad], dim=-2)

        # add left and right paddings in width
        right_pad = __tiles[:, :, :, 1:, :, :self.tile_pad]
        right_pad = torch.cat([right_pad, torch.zeros_like(right_pad[:, :, :, :1])], dim=3)
        left_pad = __tiles[:, :, :, :-1, :, -self.tile_pad:]
        left_pad = torch.cat([torch.zeros_like(left_pad[:, :, :, :1]), left_pad], dim=3)
        __tiles = torch.cat([left_pad, __tiles, right_pad], dim=-1)

        # rearrange to batch tiles
        tiles = rearrange(__tiles, 'b c n m p1 p2 -> (b n m) c p1 p2')

        upscaled = []
        for __tiles in torch.split(tiles, self.batch_size, dim=0):
            upscaled.append(self.forward(__tiles.to(self.device), train=False).to(images.device))
            del __tiles
            gc.collect()
            torch.cuda.empty_cache()

        upscaled = torch.cat(upscaled, dim=0)
        upscaled = rearrange(upscaled, '(b n m) c p1 p2 -> b c (n p1) (m p2)', n=h // self.tile, m=w // self.tile)

        return upscaled

    def on_validation_epoch_end(self):

        # get original SR images and downsample them
        test_images = self.test_dataset.load_files()
        train_images = [self.dataset.load_file(file, resize=True)
                        for file in choices(self.dataset.find_files(), k=self.samples_epoch)]
        orig_images = torch.stack(test_images + train_images, dim=0)
        down_sampled = torch.nn.functional.interpolate(orig_images, scale_factor=1 / self.scale, mode='bilinear')

        # upscale images with model
        upscaled_images = self.upscale_images(down_sampled)

        # log images
        self.custom_logger.log_images(tensor2list(to_image(orig_images)), prefix='orig', epoch=self.current_epoch)
        self.custom_logger.log_images(tensor2list(to_image(upscaled_images)), prefix='upscaled',
                                      epoch=self.current_epoch)

        metrics = self.base_metrics(upscaled_images, orig_images)
        for k, v in metrics.items():
            self.log(f'train_val_{k}', v, prog_bar=True, sync_dist=True)

        if self.debug:
            self.log('learning_rate', self.lr_schedulers().get_last_lr()[0], prog_bar=True)

    def training_step(self, batch, batch_idx):

        # enable first optimizer for patch upscaling
        optimizers = self.optimizers()
        optimizer_upscale = optimizers[0] if isinstance(optimizers, list) else optimizers
        self.toggle_optimizer(optimizer_upscale, optimizer_idx=0)
        optimizer_upscale.zero_grad(set_to_none=True)

        # upscale patches and calculate loss
        loss = self.upscale_step(batch)
        for k, v in loss.items():
            self.log(f'train_{k}', v, prog_bar=True, sync_dist=True)

        # do backward for upscaler model
        self.manual_backward(loss['loss'])
        optimizer_upscale.step()
        self.untoggle_optimizer(optimizer_idx=0)

        # train discriminator
        if self.disc:
            # enable disc optimizer
            optimizer_disc = self.optimizers()[1]
            self.toggle_optimizer(optimizer_disc, optimizer_idx=1)
            optimizer_disc.zero_grad(set_to_none=True)

            # calculate disc loss and log it
            loss = self.disc_step(batch)
            self.log(f'disc_loss', loss, prog_bar=True, sync_dist=True)

            # do backward propagation over disc parameters
            self.manual_backward(loss)
            optimizer_disc.step()
            self.untoggle_optimizer(optimizer_idx=1)

        # run lr schedulers
        schedulers = self.lr_schedulers()
        schedulers = schedulers if isinstance(schedulers, list) else [schedulers]
        for sch in schedulers:
            if type(sch) is dict:
                interval = sch['interval']
                sch = sch['scheduler']
            else:
                interval = 'epoch'
            if interval == 'step':
                sch.step()
            elif interval == 'epoch' and self.trainer.is_last_batch:
                sch.step()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.use_ema:
            self.ema_model.update(self.model)

    def validation_step(self, sr, batch_idx):
        down_sampled = torch.nn.functional.interpolate(sr, scale_factor=1 / self.scale, mode='bilinear')
        upscaled = self.upscale_images(down_sampled)
        metrics = self.base_metrics(upscaled, sr)
        for k, v in metrics.items():
            self.log(f'val_{k}', v, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        params = list(self.model.parameters()) + ([self.teacher_projection.parameters()] if self.teacher else [])
        optimizer = torch.optim.Adam(lr=self.learning_rate, params=params, betas=(0.9, 0.99))
        optimizers = [optimizer]

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.learning_rate,
                                                             pct_start=3 / self.epochs,
                                                             div_factor=1 / self.initial_lr_rate,
                                                             final_div_factor=self.initial_lr_rate / self.min_lr_rate,
                                                             epochs=self.epochs, steps_per_epoch=self.steps),
            'interval': 'step'
        }
        schedulers = [scheduler]

        if self.disc:
            optimizers += [torch.optim.Adam(lr=self.disc_lr, params=self.disc.parameters(), betas=(0.9, 0.99))]

        return optimizers, schedulers

    def train_dataloader(self):
        self.dataset.reset_cache()
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False,
                                           num_workers=4 * torch.cuda.device_count(),
                                           pin_memory=True, prefetch_factor=2)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.full_batch_size, shuffle=False,
                                           num_workers=4 * torch.cuda.device_count(),
                                           pin_memory=True, prefetch_factor=2)


def get_parser():
    parser = ArgumentParser(description="Training patch upscaler model")
    # Input data settings
    parser.add_argument("--dataset", default=[], nargs='+', help="Path to folders with SR images")
    parser.add_argument("--test_dataset", default=[], nargs='+', help="Path to folders with test images")
    parser.add_argument("--sample", default=None, help="Path to checkpoint model")
    parser.add_argument("--tmp", default="tmp", help="temporary directory for logs etc")
    parser.add_argument("--teacher", default=None, help="name of SwinSR checkpoint (Swin2SR_Lightweight_X2_64)")
    parser.add_argument("--cache_size", default=512, type=int, help="Cache images for each epoch")

    # Training settings
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate for model")
    parser.add_argument("--disc_lr", default=1e-4, type=float, help="Learning rate for discriminator")
    parser.add_argument("--disc_w", default=1e-3, type=float, help="Weight for adversarial loss")
    parser.add_argument("--disc_warmup", default=3, type=int, help="Discriminator warmup epochs")
    parser.add_argument("--ema", default=0.995, type=float, help="Ema weight")
    parser.add_argument("--teacher_rate", default=1.0, type=float, help="How much of teacher image to use for training")
    parser.add_argument("--teacher_w", default=1e-4, type=float, help="Weight for teacher features similarity loss")
    parser.add_argument("--teacher_dim", default=60, type=int, help="Teacher feature dimension")
    parser.add_argument("--min_lr_rate", default=0.01, type=float, help="Minimal LR ratio to decay")
    parser.add_argument("--initial_lr_rate", default=0.1, type=float, help="Initial LR ratio")
    parser.add_argument("--epochs", default=300, type=int, help="Epochs in training")
    parser.add_argument("--steps", default=5000, type=int, help="Steps in training")
    parser.add_argument("--batch_size", default=512, type=int, help="Batch size in training")
    parser.add_argument("--full_batch_size", default=1, type=int, help="Batch size for full images in valid")
    parser.add_argument("--acc_grads", default=1, type=int,
                        help="Steps to accumulate gradients to emulate larger batch size")
    parser.add_argument("--samples_epoch", default=5, type=int, help="Samples of generator in one epoch")
    parser.add_argument("--w", default=2560, type=int, help="SR width")
    parser.add_argument("--h", default=1440, type=int, help="SR height")
    parser.add_argument("--tile", default=16, type=int, help="Tile size after downscale")
    parser.add_argument("--tile_pad", default=2, type=int, help="Tile overlap after downscale")
    parser.add_argument("--scale", default=2, type=int, help="How much down scale SR patches")

    # Model settings
    parser.add_argument("--dim", default=16, type=int, help="Base channel")
    parser.add_argument("--disc_dim", default=128, type=int, help="Discriminator base channel")
    parser.add_argument("--in_channels", default=3, type=int, choices=[1, 3], help="1 or 3 use only Luma or Cb Cr too")
    parser.add_argument("--n_blocks", default=4, type=int, help="Feature extraction blocks")
    parser.add_argument("--disc_blocks", default=2, type=int, help="Discriminator blocks")
    parser.add_argument("--random_filters", default=48, type=int, help="Binary filters for loss")
    parser.add_argument("--random_filters_kernel", default=5, type=int, help="Binary filters kernel size")
    parser.add_argument("--fft", action='store_true')
    parser.add_argument("--disc", action='store_true')
    parser.set_defaults(fft=False, disc=False)

    # Meta settings
    parser.add_argument("--out_model_name", default="patch_upscaler", help="Name of output model path")
    parser.add_argument("--profile", default=None, choices=['simple', 'advanced'], help="Whether to use profiling")
    parser.add_argument("--task_name", default="Patch upscaling", help="ClearML task name")
    parser.add_argument("--clearml", action='store_true')
    parser.set_defaults(clearml=False)

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    if args.clearml:
        print("Initializing ClearML")
        task = clearml.Task.init(project_name='upscaling', task_name=args.task_name, reuse_last_task_id=True,
                                 auto_connect_frameworks=False)
        task.connect(args, name='config')
        logger = task.get_logger()
    else:
        logger = None

    # load pretrained upscaler
    teacher = None
    if args.teacher:
        model_path = f'{args.tmp}/{args.teacher}.pth'

        if not os.path.exists(model_path):
            url = f'https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/{args.teacher}.pth'
            r = requests.get(url, allow_redirects=True, verify=False)
            os.makedirs(model_path, exist_ok=True)
            open(model_path, 'wb').write(r.content)

        model = Swin2SR(upscale=2, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                        mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
        sd = torch.load(model_path, map_location='cpu')
        model.load_state_dict(sd, strict=True)
        model.eval()
        teacher = model

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=os.path.dirname(args.out_model_name),
                                                       filename=os.path.basename(args.out_model_name))

    dataset = PatchSRImages(folder=args.dataset, tile=args.tile, tile_pad=args.tile_pad, scale=args.scale,
                            cache_size=args.cache_size, w=args.w, h=args.h)
    test_dataset = FullSRImages(folder=args.test_dataset if args.test_dataset else args.dataset,
                                tile=args.tile, tile_pad=args.tile_pad, scale=args.scale,
                                cache_size=args.cache_size, w=args.w, h=args.h)

    model = PatchEnhancer(in_channels=args.in_channels, dim=args.dim, n_blocks=args.n_blocks,
                          tile_pad=args.tile_pad, fft=args.fft)
    if args.disc:
        disc = PatchDiscriminator(in_channels=args.in_channels, dim=args.disc_dim, n_blocks=args.disc_blocks)
    else:
        disc = None

    devices = list(range(torch.cuda.device_count()))
    if args.sample:
        enhancer = ImageEnhancer.load_from_checkpoint(args.sample, model=model, dataset=dataset, disc=disc,
                                                      test_dataset=test_dataset, in_channels=args.in_channels,
                                                      clearml=logger, teacher=teacher, disc_lr=args.disc_lr,
                                                      teacher_rate=args.teacher_rate, ema_weight=args.ema,
                                                      learning_rate=args.lr, full_batch_size=args.full_batch_size,
                                                      initial_lr_rate=args.initial_lr_rate,
                                                      disc_warmup=args.disc_warmup, dim=args.dim,
                                                      teacher_dim=args.teacher_dim, teacher_w=args.teacher_w,
                                                      min_lr_rate=args.min_lr_rate, disc_w=args.disc_w,
                                                      random_filters_kernel=args.random_filters_kernel,
                                                      batch_size=args.batch_size, epochs=args.epochs, steps=args.steps,
                                                      samples_epoch=args.samples_epoch, scale=args.scale,
                                                      tile=args.tile, tile_pad=args.tile_pad)
        enhancer.to(f'cuda:{devices[0]}')
        orig_images = torch.stack(test_dataset.load_files(resize=True), dim=0)
        lr_images = torch.nn.functional.interpolate(orig_images, scale_factor=1 / args.scale, mode='bicubic')
        sr_images = to_image(enhancer.upscale_images(lr_images))

        enhancer.custom_logger.log_images(tensor2list(to_image(lr_images)), prefix='lr', epoch=0)
        enhancer.custom_logger.log_images(tensor2list(to_image(orig_images)), prefix='orig', epoch=0)
        enhancer.custom_logger.log_images(tensor2list(to_image(sr_images)), prefix='sr', epoch=0)
    else:
        enhancer = ImageEnhancer(model=model, dataset=dataset, test_dataset=test_dataset, clearml=logger, disc=disc,
                                 in_channels=args.in_channels, random_filters=args.random_filters, teacher=teacher,
                                 teacher_rate=args.teacher_rate, ema_weight=args.ema, learning_rate=args.lr,
                                 initial_lr_rate=args.initial_lr_rate, min_lr_rate=args.min_lr_rate,
                                 dim=args.dim, teacher_dim=args.teacher_dim, teacher_w=args.teacher_w,
                                 batch_size=args.batch_size, epochs=args.epochs, steps=args.steps,
                                 disc_lr=args.disc_lr, disc_w=args.disc_w, disc_warmup=args.disc_warmup,
                                 samples_epoch=args.samples_epoch, scale=args.scale,
                                 random_filters_kernel=args.random_filters_kernel,
                                 tile=args.tile, tile_pad=args.tile_pad, full_batch_size=args.full_batch_size)
        trainer = Trainer(max_epochs=args.epochs, limit_train_batches=args.steps,
                          enable_model_summary=True, enable_progress_bar=True, enable_checkpointing=True,
                          strategy=DDPStrategy(find_unused_parameters=True), precision=16,
                          profiler=args.profile,
                          accumulate_grad_batches=args.acc_grads,
                          accelerator='gpu', devices=devices, callbacks=[checkpoint_callback])
        trainer.fit(enhancer)
