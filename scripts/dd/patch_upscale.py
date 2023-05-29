from modules.common.util import *
from modules.common.model import *
from modules.common.trainer import *
from scripts.dd.swin2sr import Swin2SR

import numpy as np
import os
import cv2
from random import choice, randint, shuffle, choices, random
import gc
from itertools import chain

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
from tqdm import tqdm
from argparse import ArgumentParser
import clearml

from skvideo.io import FFmpegWriter


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


def center_crop(tensor, shape):
    # cut to shape
    h_tile = tensor.shape[-2] - shape[-2]
    w_tile = tensor.shape[-1] - shape[-1]
    h_slice = slice(h_tile // 2 + h_tile % 2, -h_tile // 2) if h_tile > 0 else slice(None)
    w_slice = slice(w_tile // 2 + w_tile % 2, -w_tile // 2) if w_tile > 0 else slice(None)
    tensor = tensor[:, :, h_slice, w_slice]
    return tensor


def charbonnier_loss(x, y, eps=1e-6):
    return torch.sqrt((x - y) ** 2 + eps).mean()


def multiscale_loss(x1, x2, scales=(1, 2)):
    loss = 0
    for scale in scales:
        __x1 = torch.nn.functional.interpolate(x1, scale_factor=1 / scale, mode='bilinear')
        __x2 = torch.nn.functional.interpolate(x2, scale_factor=1 / scale, mode='bilinear')
        loss += charbonnier_loss(__x1, __x2)
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


class MobileBlock(torch.nn.Module):

    def __init__(self, dim, in_dim=-1, kernel_size=3, group=4, fft=False):
        super(MobileBlock, self).__init__()
        self.fft = fft
        self.dim = dim
        self.in_dim = in_dim if in_dim > 0 else dim
        if self.in_dim != dim:
            self.skip_conv = torch.nn.Conv2d(in_dim, dim, kernel_size=1)

        self.conv1 = torch.nn.Conv2d(in_dim, dim, kernel_size=kernel_size, padding=kernel_size // 2,
                                     groups=self.in_dim // group)
        self.conv1_linear = torch.nn.Conv2d(dim, dim, kernel_size=1)

        self.conv2 = torch.nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim // group)
        self.conv2_linear = torch.nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        h = self.conv1_linear(self.conv1(nonlinear(x)))
        if self.fft:
            h = torch.fft.fft2(x, norm='ortho').real
        h = self.conv2_linear(self.conv2(nonlinear(h)))
        if self.fft:
            h = torch.fft.fft2(x, norm='ortho').real
        x = x if self.in_dim == self.dim else self.skip_conv(x)
        return (h + x) / 2 ** 0.5


class FeatureBlock(torch.nn.Module):

    def __init__(self, dim, in_dim=-1, kernel_size=3, fft=False):
        super(FeatureBlock, self).__init__()
        self.fft = fft
        self.dim = dim
        self.in_dim = in_dim if in_dim > 0 else self.dim

        self.conv1 = torch.nn.Conv2d(self.in_dim, self.dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = torch.nn.Conv2d(self.dim, self.dim, kernel_size=kernel_size, padding=kernel_size // 2)

        if self.in_dim != dim:
            self.skip_conv = torch.nn.Conv2d(self.in_dim, self.dim, kernel_size=1)

    def forward(self, x):
        h = self.conv1(nonlinear(x))
        if self.fft:
            h = torch.fft.fft2(x, norm='ortho').real
        h = self.conv2(nonlinear(h))
        if self.fft:
            h = torch.fft.fft2(x, norm='ortho').real
        x = x if self.dim == self.in_dim else self.skip_conv(x)
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


class SimpleDiscriminator(torch.nn.Module):

    def __init__(self, in_channels=1, dim=128, n_blocks=2):
        super(SimpleDiscriminator, self).__init__()
        self.in_channels = in_channels
        self.first_conv = torch.nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)
        self.feature_extractor = torch.nn.Sequential(*[ResBlock(dim) for _ in range(n_blocks)])
        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.utils.spectral_norm(torch.nn.Linear(dim, 1)),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        h = self.first_conv(x[:, :self.in_channels])
        features = self.feature_extractor(h)
        probs = self.classifier(features)
        return probs.view(-1)


class DownSample2d(torch.nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size=3, scale_factor=0.5, use_conv=True):
        super(DownSample2d, self).__init__()
        self.scale_factor = scale_factor
        self.use_conv = use_conv
        if self.use_conv:
            self.conv = torch.nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size,
                                        padding=kernel_size // 2, stride=1)

    def forward(self, x):
        h = self.conv(x) if self.use_conv else x
        h = torch.nn.functional.interpolate(h, scale_factor=self.scale_factor, mode='bilinear')
        return h


class UpSample2d(torch.nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size=3, scale_factor=2.0, use_conv=True):
        super(UpSample2d, self).__init__()
        self.scale_factor = scale_factor
        self.use_conv = use_conv
        if self.use_conv:
            self.conv = torch.nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size,
                                        stride=1, padding=kernel_size // 2)

    def forward(self, x):
        h = torch.nn.functional.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        h = self.conv(h) if self.use_conv else h
        return h


class MobileEnhancer(torch.nn.Module):

    def __init__(self, in_channels=1, dim=32, input_kernel=5, pad_input=False, n_blocks=4, scale=2, fft=False,
                 enhance=True, upscale_mode='nearest-exact', n_layers=1):
        super(MobileEnhancer, self).__init__()
        self.in_channels = in_channels
        self.scale = scale
        self.enhance = enhance
        self.upscale_mode = upscale_mode

        self.input_conv = torch.nn.Conv2d(self.in_channels, dim, kernel_size=input_kernel,
                                          padding=input_kernel // 2 if pad_input else 0, stride=1)
        self.blocks = torch.nn.ModuleList([])
        for block_id in range(n_blocks):
            if block_id > 0:
                self.blocks.append(torch.nn.Conv2d(dim * 2 ** (block_id - 1), dim * 2 ** block_id, kernel_size=1))
            for _ in range(n_layers):
                self.blocks.append(MobileBlock(dim * 2 ** block_id, dim * 2 ** block_id, fft=fft))
        self.out = torch.nn.Conv2d(dim * 2 ** (n_blocks - 1) // self.scale ** 2, in_channels, kernel_size=3, padding=1)

    def forward(self, x, return_features=False):
        upscaled = torch.nn.functional.interpolate(x, scale_factor=self.scale, mode=self.upscale_mode)

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
        upscaled = center_crop(upscaled, h.shape)
        if self.enhance:
            result = torch.cat((upscaled[:, :self.in_channels] + h, upscaled[:, self.in_channels:]), dim=-3)
        else:
            result = torch.cat((h, upscaled[:, self.in_channels:]), dim=-3)
        if return_features:
            return result, features

        return result


class ResNetEnhancer(torch.nn.Module):

    def __init__(self, in_channels=1, dim=32, input_kernel=5, pad_input=False, n_blocks=4, scale=2, fft=False,
                 enhance=False, upscale_mode='nearest-exact', n_layers=1):
        super(ResNetEnhancer, self).__init__()
        self.in_channels = in_channels
        self.scale = scale
        self.enhance = enhance
        self.upscale_mode = upscale_mode
        self.input_conv = torch.nn.Conv2d(self.in_channels, dim, kernel_size=input_kernel,
                                          padding=input_kernel // 2 if pad_input else 0, stride=1)
        self.blocks = torch.nn.ModuleList([])
        for block_id in range(n_blocks):
            if block_id > 0:
                self.blocks.append(torch.nn.Conv2d(dim * 2 ** (block_id - 1), dim * 2 ** block_id, kernel_size=1))
            for _ in range(n_layers):
                self.blocks.append(FeatureBlock(dim * 2 ** block_id, dim * 2 ** block_id, fft=fft))
        self.out = torch.nn.Conv2d(dim * 2 ** (n_blocks - 1) // self.scale ** 2, in_channels, kernel_size=3, padding=1)

    def forward(self, x, return_features=False):
        upscaled = torch.nn.functional.interpolate(x, scale_factor=self.scale, mode=self.upscale_mode)

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
        upscaled = center_crop(upscaled, h.shape)
        if self.enhance:
            result = torch.cat((upscaled[:, :self.in_channels] + h, upscaled[:, self.in_channels:]), dim=-3)
        else:
            result = torch.cat((h, upscaled[:, self.in_channels:]), dim=-3)
        if return_features:
            return result, features

        return result


class UNetEnhancer(torch.nn.Module):

    def __init__(self, in_channels=1, dim=32, n_blocks=3, scale=2, input_kernel=3, pad_input=True, fft=False,
                 n_layers=2, enhance=True, upscale_mode='nearest-exact'):
        super(UNetEnhancer, self).__init__()
        self.in_channels = in_channels
        self.scale = scale
        self.enhance = enhance
        self.upscale_mode = upscale_mode
        self.input_conv = torch.nn.Conv2d(self.in_channels, dim, kernel_size=input_kernel,
                                          padding=input_kernel // 2 if pad_input else 0, stride=1)
        self.down_blocks = torch.nn.ModuleList([torch.nn.ModuleList(
            [FeatureBlock(dim * 2 ** idx, fft=fft) for _ in range(n_layers)])
            for idx in range(n_blocks)])
        self.downs = torch.nn.ModuleList(
            [DownSample2d(dim * 2 ** (idx - 1), dim * 2 ** idx) for idx in range(1, n_blocks)])
        self.downs.append(torch.nn.Identity())

        self.up_blocks = torch.nn.ModuleList([torch.nn.ModuleList(
            [FeatureBlock(dim * 2 ** idx, in_dim=(int(block_id == 0 and idx != n_blocks - 1) + 1) * dim * 2 ** idx,
                          fft=fft) for block_id in range(n_layers)]) for idx in reversed(range(n_blocks))])
        self.ups = torch.nn.ModuleList([UpSample2d(dim * 2 ** idx, dim * 2 ** (idx - 1)) for idx in
                                        reversed(range(1, n_blocks))])
        self.ups.append(torch.nn.Identity())
        self.out = torch.nn.Conv2d(dim, in_channels, kernel_size=3, padding=1)

    def forward(self, x, return_features=False):
        # upscale initial patch for enhancing
        upscaled = torch.nn.functional.interpolate(x, scale_factor=self.scale, mode=self.upscale_mode)

        # extract channels
        h = self.input_conv(upscaled[:, :self.in_channels])

        # deep feature extraction
        outs = []
        for idx, (blocks, down) in enumerate(zip(self.down_blocks, self.downs)):
            for block in blocks:
                h = block(h)
            if idx != len(self.down_blocks) - 1:
                outs.append(h)
            h = down(h)

        for idx, (blocks, up) in enumerate(zip(self.up_blocks, self.ups)):
            if idx != 0:
                h = torch.cat([h, outs.pop()], dim=1)
            for block in blocks:
                h = block(h)
            h = up(h)

        # save features in case of teacher loss
        features = h

        # output conv
        h = self.out(nonlinear(h))

        # get updated and remained part
        upscaled = center_crop(upscaled, h.shape)
        if self.enhance:
            result = torch.cat((upscaled[:, :self.in_channels] + h, upscaled[:, self.in_channels:]), dim=-3)
        else:
            result = torch.cat((h, upscaled[:, self.in_channels:]), dim=-3)
        if return_features:
            return result, features

        return result


class SRImagesBase:

    def __init__(self, folder, tile=16, tile_pad=2, scale=2, cache_size=512, w=2560, h=1440, preprocessed=False,
                 augment=False, bitrate='b4', orig='hr'):
        super(SRImagesBase, self).__init__()
        self.w = w
        self.h = h
        self.tile = tile
        self.tile_pad = tile_pad
        self.scale = scale
        self.preprocessed = preprocessed
        self.bitrate = bitrate
        self.augment = augment
        self.orig = orig
        if type(folder) is not list:
            folder = [folder]
        self.folder = folder
        self.offsets = []
        self.sizes = []
        self.files = self.find_files()

        self.cache_size = cache_size
        self.cache = []

    def find_files(self):
        folders = [[os.path.join(folder, file) for file in os.listdir(folder) if
                    os.path.splitext(file)[1] in ('.png', '.jpg', '.jpeg')] for folder in self.folder]
        self.sizes = [len(folder) for folder in folders]
        self.offsets = [sum(self.sizes[:idx]) for idx in range(len(self.sizes))]
        self.files = list(chain(*folders))
        return self.files

    def __load_file(self, file, resize=False):
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

    def load_file(self, file, resize=False):
        if self.preprocessed:
            root = os.path.abspath(os.path.join(os.path.dirname(file), os.pardir))
            base = os.path.basename(file)
            return {
                mode: self.__load_file(os.path.join(root, mode, base), resize=False) for mode in os.listdir(root)
                if
                os.path.isdir(os.path.join(root, mode)) and (self.bitrate is None or mode in (self.bitrate, self.orig))
            }
        else:
            return {self.orig: self.__load_file(file, resize=resize)}

    def load_files(self, resize=True, files=None):
        if not files:
            files = self.files
        return [self.load_file(file, resize=resize) for file in files]

    def __reset_cache(self):
        cache_files = []
        for folder in self.folder:
            folder_files = [os.path.join(folder, file) for file in os.listdir(folder) if
                            os.path.splitext(file)[1] in ('.png', '.jpg', '.jpeg')]
            shuffle(folder_files)
            cache_files.extend(folder_files[:self.cache_size // len(self.folder)])

        self.cache = self.load_files(resize=False, files=cache_files)
        gc.collect()

    def reset_cache(self):

        if self.cache_size <= 0:
            raise RuntimeError('Caching is not enabled, use cache_size>0 when initializing dataset')

        # not block training
        if len(self.cache) > 0:
            run_async(self.__reset_cache)
        else:
            self.__reset_cache()

    def load_patch(self, obj):

        sr = obj[self.orig]

        c, h, w = sr.shape

        tile = (self.tile + self.tile_pad * 2) * self.scale
        i_start = randint(0, h - tile - 1)
        j_start = randint(0, w - tile - 1)
        sr_patch = sr[:, i_start:i_start + tile, j_start:j_start + tile]

        if self.preprocessed:
            lr = obj[choice([k for k in obj.keys() if k != self.orig])] if self.bitrate is None else obj[self.bitrate]
            lr_patch = lr[:, int(i_start / self.scale):int((i_start + tile) / self.scale),
                       int(j_start / self.scale):int((j_start + tile) / self.scale)]
        else:

            lr_patch = sr_patch
            if self.augment:
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
        if self.cache_size < 0 or len(self.cache) == 0:
            # update list in case of new files
            file = choice(self.find_files())
            return self.load_patch(self.load_file(file))
        else:
            return self.load_patch(choice(self.cache))


class TrainSRImages(SRImagesBase, torch.utils.data.Dataset):

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        group_id = idx % len(self.sizes)
        idx = self.offsets[group_id] + (idx // len(self.sizes)) % self.sizes[group_id]
        obj = self.load_file(self.files[idx], resize=True)
        return {
            'lr': obj[self.orig],
            'sr': obj[choice([k for k in obj.keys() if k != self.orig])] if self.bitrate is None else obj[self.bitrate]
        }


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
                 patched=True,
                 orig='hr',
                 bitrate='b4',
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
                 acc_grads=1,
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

        self.patched = patched
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.orig = orig
        self.bitrate = bitrate

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
        self.acc_grads = acc_grads
        self.samples_epoch = samples_epoch

    def forward(self, x, train=True, **kwargs):

        model = self.ema_model.module if self.use_ema and (not train or not self.training) else self.model

        return model(x, return_features=train, **kwargs)

    def fixpad(self, image):
        # remove padding from orig batch
        if self.patched:
            image = image[:, :,
                    self.tile_pad * self.scale: -self.tile_pad * self.scale,
                    self.tile_pad * self.scale: -self.tile_pad * self.scale]
        return image

    def upscale_step(self, batch):

        # enhance it
        lr = batch['lr']
        upscaled, features = self.forward(lr, train=True)

        # return loss on real SR
        return self.get_losses(upscaled, lr, self.fixpad(batch['sr']), features)

    def disc_step(self, batch):

        with torch.no_grad():
            upscaled = self.forward(batch['lr'], train=True)[0]

        # discriminator must learn how fake the example looks like
        preds = self.disc(torch.cat([upscaled, self.fixpad(batch['sr'])], dim=0))
        target = torch.ones_like(preds)
        target[len(upscaled):] = torch.rand_like(target[len(upscaled):]) * 0.1

        values = target * torch.log(preds + 1e-6) + (1 - target) * torch.log(1 - preds + 1e-6)
        bce_loss = - torch.mean(torch.nan_to_num(values, nan=0, posinf=1, neginf=-1))
        # hinge_loss = torch.mean(torch.maximum(torch.zeros_like(preds), 1 - target * (preds * 2 - 1.0)))

        return bce_loss

    def base_metrics(self, x, sr):
        x_normalized = torch_convert_YUV2RGB(x * 0.5 + 0.5).clip(0, 1)
        sr_normalized = torch_convert_YUV2RGB(sr * 0.5 + 0.5).clip(0, 1)
        return {
            'psnr': torchmetrics.functional.peak_signal_noise_ratio(x_normalized, sr_normalized, data_range=1.0),
            'ssim': torchmetrics.functional.structural_similarity_index_measure(
                x_normalized, sr_normalized, data_range=1.0)
        }

    @torch.no_grad()
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

        loss = {
            'loss': 0.0
        }

        x = x[:, :self.in_channels]
        sr = sr[:, :self.in_channels]
        # loss on down sampled images to reduce artifacts
        loss['aux_loss'] = multiscale_loss(torch.cat([x, self.random_filter(x)], dim=1),
                                           torch.cat([sr, self.random_filter(sr)], dim=1))
        loss['loss'] = loss['aux_loss'] + loss['loss']

        # loss on high frequency details compared with gaussian blurred
        x_details = x - torchvision.transforms.functional.gaussian_blur(x, kernel_size=(5, 5))
        sr_details = sr - torchvision.transforms.functional.gaussian_blur(sr, kernel_size=(5, 5))
        loss['high_freq'] = charbonnier_loss(x_details, sr_details)
        loss['loss'] += loss['high_freq']

        # adversarial based loss
        if self.disc and self.current_epoch >= self.disc_warmup:
            # we want to minimize example falseness estimated by discriminator
            loss['adv_loss'] = - torch.mean(torch.log(1 - self.disc(x) + 1e-6))
            loss['loss'] += loss['adv_loss'] * self.disc_w

        if self.teacher:
            sr, teacher_features = self.call_teacher(lr)
            loss['teacher_loss'] = gram_loss(self.teacher_projection(features), teacher_features)
            loss['loss'] += loss['teacher_loss'] * self.teacher_w

        return loss

    def patch_upscale(self, images):

        b, c, h_orig, w_orig = images.shape
        h_add = self.tile - (h_orig - h_orig // self.tile * self.tile)
        w_add = self.tile - (w_orig - w_orig // self.tile * self.tile)
        images = torch.cat([torch.zeros_like(images[:, :, :h_add // 2 + h_add % 2]),
                            images,
                            torch.zeros_like(images[:, :, :h_add // 2])], dim=2)
        images = torch.cat([torch.zeros_like(images[:, :, :, :w_add // 2 + w_add % 2]),
                            images,
                            torch.zeros_like(images[:, :, :, :w_add // 2])], dim=3)
        h, w = images.shape[-2:]
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

        upscaled = upscaled[:, :, (h_add // 2 + h_add % 2) * self.scale: -h_add // 2 * self.scale,
                   (w_add // 2 + w_add % 2) * self.scale: -w_add // 2 * self.scale]

        return upscaled

    @torch.no_grad()
    def upscale_images(self, images):
        if self.patched:
            return self.patch_upscale(images)
        images_device = images.device
        return self.forward(images.to(self.device), train=False).to(images_device)

    def __load_dataset_items(self, dataset, k):
        return [dataset.load_file(file, resize=True)
                for file in choices(dataset.find_files(), k=k)]

    def on_train_epoch_end(self):
        # log model weights and grads
        data_iter = iter(self.trainer.train_dataloader)
        batch = {k: v.to(self.device) for k, v in next(data_iter).items()}
        self.model.zero_grad(set_to_none=True)
        loss = self.upscale_step(batch)
        loss['loss'].backward()

        for name, param in self.model.named_parameters():
            if (not "weight" in name) or (param.grad is None):
                continue
            self.custom_logger.log_distribution(param.grad.view(-1).cpu().numpy(), f'{name}_grad', self.current_epoch)
            self.custom_logger.log_distribution(param.detach().view(-1).cpu().numpy(), name, self.current_epoch)
        self.model.zero_grad(set_to_none=True)

    def on_validation_epoch_end(self):

        metrics = self.trainer.logged_metrics
        for name, value in metrics.items():
            self.custom_logger.log_value(value, name, epoch=self.current_epoch)

        # get original SR images and down sample them
        test_items = self.__load_dataset_items(self.test_dataset, self.samples_epoch)
        orig_images = torch.stack([item[self.orig] for item in test_items], dim=0)
        if self.bitrate:
            down_sampled = torch.stack([item[self.bitrate] for item in test_items], dim=0)
        else:
            down_sampled = torch.nn.functional.interpolate(orig_images, scale_factor=1 / self.scale, mode='bicubic')

        # upscale images with model
        upscaled_images = self.upscale_images(down_sampled)

        # log images
        self.custom_logger.log_images(tensor2list(to_image(orig_images)), prefix='orig', epoch=self.current_epoch)
        self.custom_logger.log_images(tensor2list(to_image(down_sampled)), prefix='lr', epoch=self.current_epoch)
        self.custom_logger.log_images(tensor2list(to_image(upscaled_images)), prefix='upscaled',
                                      epoch=self.current_epoch)

        metrics = self.base_metrics(upscaled_images, orig_images)
        for k, v in metrics.items():
            self.log(f'train_val_{k}', v, prog_bar=True, sync_dist=True)

        if self.debug:
            self.log('learning_rate', self.lr_schedulers().get_last_lr()[0], prog_bar=True)

        # log aggregated metrics
        metrics = self.trainer.logged_metrics
        for name, value in metrics.items():
            self.custom_logger.log_value(value.item(), name, self.current_epoch)

    def training_step(self, batch, batch_idx):

        # enable first optimizer for patch upscaling
        optimizers = self.optimizers()
        optimizer_upscale = optimizers[0] if isinstance(optimizers, list) else optimizers
        self.toggle_optimizer(optimizer_upscale, optimizer_idx=0)
        if batch_idx % self.acc_grads == 0:
            optimizer_upscale.zero_grad(set_to_none=True)

        # upscale patches and calculate loss
        loss = self.upscale_step(batch)

        # do backward for upscaler model
        self.manual_backward(loss['loss'] / self.acc_grads)

        for k, v in loss.items():
            self.log(f'train_{k}', v, prog_bar=True, sync_dist=True)

        if (batch_idx + 1) % self.acc_grads == 0:
            optimizer_upscale.step()
        self.untoggle_optimizer(optimizer_idx=0)

        # train discriminator
        if self.disc:
            # enable disc optimizer
            optimizer_disc = self.optimizers()[1]
            self.toggle_optimizer(optimizer_disc, optimizer_idx=1)
            if batch_idx % self.acc_grads == 0:
                optimizer_disc.zero_grad(set_to_none=True)

            # calculate disc loss and log it
            loss = self.disc_step(batch)
            self.log(f'disc_loss', loss, prog_bar=True, sync_dist=True)

            # do backward propagation over disc parameters
            self.manual_backward(loss / self.acc_grads)
            if (batch_idx + 1) % self.acc_grads == 0:
                optimizer_disc.step()
            self.untoggle_optimizer(optimizer_idx=1)

        # run lr schedulers
        if (batch_idx + 1) % self.acc_grads == 0:
            schedulers = self.lr_schedulers()
            schedulers = schedulers if isinstance(schedulers, list) else [schedulers]
            for sch in schedulers:
                if type(sch) is torch.optim.lr_scheduler.CosineAnnealingWarmRestarts:
                    interval = 'step'
                else:
                    interval = 'epoch'
                if interval == 'step':
                    sch.step()
                elif interval == 'epoch' and self.trainer.is_last_batch:
                    sch.step()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.use_ema:
            self.ema_model.update(self.model)

    def validation_step(self, batch, batch_idx):
        sr = batch[self.orig]
        if self.bitrate:
            down_sampled = batch[self.bitrate]
        else:
            down_sampled = torch.nn.functional.interpolate(sr, scale_factor=1 / self.scale, mode='bicubic')
        upscaled = self.upscale_images(down_sampled)
        metrics = self.base_metrics(upscaled, sr)
        for k, v in metrics.items():
            self.log(f'val_{k}', v, prog_bar=True, sync_dist=True)

    def find_lr(self, min_lr=1e-6, max_lr=1.0, steps=1000, acc_grad=1, optimizer_id=0, monitor='loss', max_factor=10.0):

        # setup optimizer
        optims, _ = self.configure_optimizers()
        optim = optims[optimizer_id]

        data_iter = iter(self.train_dataloader())
        best_loss, best_lr = 1e9, min_lr
        values, lrs = [], []
        pb_steps = tqdm(range(steps), desc='Finding learning rate')
        for step_id in pb_steps:

            lr = min_lr * np.e ** (step_id / steps * np.log(max_lr / min_lr))
            lrs.append(lr)
            for g in optim.param_groups:
                g['lr'] = lr

            optim.zero_grad(set_to_none=True)

            value = 0
            for acc_id in range(acc_grad):
                batch = {k: v.to(self.device) for k, v in next(data_iter).items()}
                loss = self.upscale_step(batch)
                (loss['loss'] / acc_grad).backward()
                value += loss[monitor].item() / acc_grad

            values.append(value)

            optim.step()

            if value < best_loss:
                best_loss = value
                best_lr = min_lr

            if step_id > 0 and value > best_loss * max_factor:
                break

            pb_steps.set_description(
                f'Step [{step_id + 1}/{steps}] [lr:{lrs[-1]:.8f}] [loss:{values[-1]:.6f}] [best_loss:{best_loss:.6f}]',
                refresh=True)
        pb_steps.close()
        self.custom_logger.log_line(x=np.asarray(lrs),
                                    y=np.clip(np.asarray(values), a_min=0.0, a_max=best_loss * max_factor),
                                    name='learning rate')
        return best_lr

    def configure_optimizers(self):
        params = list(self.model.parameters()) + (list(self.teacher_projection.parameters()) if self.teacher else [])
        optimizer = torch.optim.AdamW(lr=self.learning_rate * self.min_lr_rate, params=params, betas=(0.9, 0.99),
                                      weight_decay=0.001)
        optimizers = [optimizer]

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, self.steps, T_mult=2, eta_min=self.learning_rate),
            'interval': 'step'
        }
        schedulers = [scheduler]

        if self.disc:
            optimizers += [torch.optim.SGD(lr=self.disc_lr, params=self.disc.parameters(), momentum=0.9)]

        return optimizers, schedulers

    def on_save_checkpoint(self, checkpoint):
        del checkpoint['hyper_parameters']['dataset']
        del checkpoint['hyper_parameters']['test_dataset']

    def train_dataloader(self):
        if self.patched:
            self.dataset.reset_cache()
            batch_size = self.batch_size
        else:
            batch_size = self.full_batch_size
        return torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=not self.patched,
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
    parser.add_argument("--orig", default='hr', help="Name for original images folder")
    parser.add_argument("--test_dataset", default=[], nargs='+', help="Path to folders with test images")
    parser.add_argument("--sample", default=None, help="Path to checkpoint model")
    parser.add_argument("--videos", default=None, nargs='+', help="Path to videos to upscale")
    parser.add_argument("--video_suffix", default="upscaled", help="Suffix for upscaled videos")
    parser.add_argument("--tmp", default="tmp", help="temporary directory for logs etc")
    parser.add_argument("--model", default="resnet", choices=['unet', 'resnet', 'mobilenet'], help="model kind")
    parser.add_argument("--upscale_mode", default='nearest-exact',
                        choices=['nearest-exact', 'bilinear', 'bicubic', 'area'], help="upscaling type")
    parser.add_argument("--teacher", default=None, help="name of SwinSR checkpoint (Swin2SR_Lightweight_X2_64)")
    parser.add_argument("--cache_size", default=512, type=int, help="Cache images for each epoch")
    parser.add_argument("--train_bitrate", default=None, help="Bitrate folder for train")
    parser.add_argument("--bitrate", default='b4', help="Bitrate folder for evaluate")
    parser.add_argument("--find_lr", action='store_true', help='whether dataset was preprocessed')
    parser.add_argument("--prep", action='store_true', help='whether dataset was preprocessed')
    parser.add_argument("--patch", action='store_true', help='whether to use patches for training')
    parser.add_argument("--enhance", action='store_true', help='whether to enhance picture or predict it')
    parser.add_argument("--test_prep", action='store_true', help='whether test dataset was preprocessed')
    parser.set_defaults(prep=False, test_prep=False, patch=False, find_lr=False, enhance=False)
    # Training settings
    parser.add_argument("--lr", default=2e-3, type=float, help="Learning rate for model")
    parser.add_argument("--disc_lr", default=1e-4, type=float, help="Learning rate for discriminator")
    parser.add_argument("--disc_w", default=1e-2, type=float, help="Weight for adversarial loss")
    parser.add_argument("--disc_warmup", default=3, type=int, help="Discriminator warmup epochs")
    parser.add_argument("--ema", default=None, type=float, help="Ema weight")
    parser.add_argument("--teacher_rate", default=1.0, type=float, help="How much of teacher image to use for training")
    parser.add_argument("--teacher_w", default=1e-4, type=float, help="Weight for teacher features similarity loss")
    parser.add_argument("--teacher_dim", default=60, type=int, help="Teacher feature dimension")
    parser.add_argument("--min_lr_rate", default=0.1, type=float, help="Minimal LR ratio to decay")
    parser.add_argument("--initial_lr_rate", default=0.1, type=float, help="Initial LR ratio")
    parser.add_argument("--epochs", default=100, type=int, help="Epochs in training")
    parser.add_argument("--steps", default=1000, type=int, help="Steps in training")
    parser.add_argument("--val_steps", default=500, type=int, help="Steps in validation")
    parser.add_argument("--batch_size", default=2048, type=int, help="Batch size in training")
    parser.add_argument("--full_batch_size", default=4, type=int, help="Batch size for full images in valid")
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
    parser.add_argument("--n_layers", default=2, type=int, help="Layers per resolution")
    parser.add_argument("--disc_blocks", default=2, type=int, help="Discriminator blocks")
    parser.add_argument("--random_filters", default=72, type=int, help="Binary filters for loss")
    parser.add_argument("--random_filters_kernel", default=3, type=int, help="Binary filters kernel size")
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

        teacher = Swin2SR(upscale=2, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                        mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
        sd = torch.load(model_path, map_location='cpu')
        teacher.load_state_dict(sd, strict=True)
        teacher.eval()

    if args.patch:
        dataset = PatchSRImages(folder=args.dataset, tile=args.tile, tile_pad=args.tile_pad, scale=args.scale,
                                cache_size=args.cache_size, w=args.w, h=args.h, preprocessed=args.prep, orig=args.orig,
                                bitrate=args.train_bitrate)
    else:
        dataset = TrainSRImages(folder=args.dataset, scale=args.scale, orig=args.orig, w=args.w, h=args.h,
                                preprocessed=args.prep, bitrate=args.train_bitrate)
    test_dataset = FullSRImages(folder=args.test_dataset if args.test_dataset else args.dataset,
                                scale=args.scale, orig=args.orig, w=args.w, h=args.h,
                                preprocessed=args.test_prep, bitrate=args.bitrate)

    # update scale to * 1 in case of resnet
    if args.patch:
        if args.model == 'unet':
            input_kernel = 1 + args.tile_pad * args.scale * 2
        else:
            input_kernel = 1 + args.tile_pad * 2
        input_pad = False
    else:
        input_kernel = 3
        input_pad = True

    if args.model == 'unet':
        model = UNetEnhancer(in_channels=args.in_channels, dim=args.dim, n_blocks=args.n_blocks,
                             input_kernel=input_kernel, pad_input=input_pad, fft=args.fft,
                             enhance=args.enhance, upscale_mode=args.upscale_mode, n_layers=args.n_layers)
    elif args.model == 'resnet':
        model = ResNetEnhancer(in_channels=args.in_channels, dim=args.dim, n_blocks=args.n_blocks,
                               input_kernel=input_kernel, pad_input=input_pad, fft=args.fft,
                               enhance=args.enhance, upscale_mode=args.upscale_mode, n_layers=args.n_layers)
    else:
        model = MobileEnhancer(in_channels=args.in_channels, dim=args.dim, n_blocks=args.n_blocks,
                               input_kernel=input_kernel, pad_input=input_pad, fft=args.fft,
                               enhance=args.enhance, upscale_mode=args.upscale_mode, n_layers=args.n_layers)
    if args.disc:
        disc = SimpleDiscriminator(in_channels=args.in_channels, dim=args.disc_dim, n_blocks=args.disc_blocks, )
    else:
        disc = None

    devices = list(range(torch.cuda.device_count()))
    if args.sample:
        enhancer = ImageEnhancer.load_from_checkpoint(args.sample, model=model, dataset=dataset, test_dataset=test_dataset, clearml=logger, disc=disc,
                                 in_channels=args.in_channels, random_filters=args.random_filters, teacher=teacher,
                                 teacher_rate=args.teacher_rate, ema_weight=args.ema, learning_rate=args.lr,
                                 initial_lr_rate=args.initial_lr_rate, min_lr_rate=args.min_lr_rate,
                                 dim=args.dim, teacher_dim=args.teacher_dim, teacher_w=args.teacher_w,
                                 batch_size=args.batch_size, epochs=args.epochs, steps=args.steps * args.acc_grads,
                                 patched=args.patch, acc_grads=args.acc_grads,
                                 disc_lr=args.disc_lr, disc_w=args.disc_w, disc_warmup=args.disc_warmup,
                                 samples_epoch=args.samples_epoch, scale=args.scale, bitrate=args.bitrate,
                                 random_filters_kernel=args.random_filters_kernel, orig=args.orig,
                                 tile=args.tile, tile_pad=args.tile_pad, full_batch_size=args.full_batch_size)
        enhancer.to(f'cuda:{devices[0]}')
        if args.videos:
            for video in args.videos:
                cap = cv2.VideoCapture(video)
                video_fps = int(cap.get(cv2.CAP_PROP_FPS))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                hh, ww = height * args.scale, width * args.scale

                path = os.path.splitext(video)[0] + f'_{args.video_suffix}.avi'

                # fourcc = cv2.VideoWriter_fourcc(*'FFV1')
                # writer = cv2.VideoWriter(path, apiPreference=0, fourcc=0, fps=video_fps, frameSize=(ww, hh))
                writer = FFmpegWriter(path, outputdict={
                    # '-vcodec': 'libx264',
                    '-crf': '0',
                    '-preset': 'ultrafast',
                    '-r': f'{video_fps}',
                    '-pix_fmt': 'yuv420p'
                })
                processed = 0
                pbar = tqdm(total=nframes)
                while processed < nframes:
                    frames = []
                    for frame_id in range(args.full_batch_size):
                        # cap.set(cv2.CAP_PROP_POS_FRAMES, 1000 + frame_id)
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # frame = cv2.resize(frame, (ww, hh), interpolation=cv2.INTER_AREA)

                        frames.append(to_tensor(frame))

                    tensor = enhancer.upscale_images(torch.stack(frames, dim=0).to(enhancer.device))
                    for frame in tensor2list(to_image(tensor)):
                        # writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        writer.writeFrame(frame)

                    processed += len(frames)
                    pbar.update(len(frames))
                pbar.close()
                cap.release()
                writer.close()
        else:
            items = test_dataset.load_files(resize=True)
            orig_images = torch.stack([item[args.orig] for item in items], dim=0)
            if args.test_prep:
                lr_images = torch.stack([item[args.bitrate] for item in items], dim=0)
            else:
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
                                 batch_size=args.batch_size, epochs=args.epochs, steps=args.steps * args.acc_grads,
                                 patched=args.patch, acc_grads=args.acc_grads,
                                 disc_lr=args.disc_lr, disc_w=args.disc_w, disc_warmup=args.disc_warmup,
                                 samples_epoch=args.samples_epoch, scale=args.scale, bitrate=args.bitrate,
                                 random_filters_kernel=args.random_filters_kernel, orig=args.orig,
                                 tile=args.tile, tile_pad=args.tile_pad, full_batch_size=args.full_batch_size)

        checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=os.path.dirname(args.out_model_name),
                                                           filename=os.path.basename(args.out_model_name),
                                                           monitor='val_psnr', mode='max')
        early_stopping = pl.callbacks.EarlyStopping(monitor="val_psnr", mode="max", patience=10, min_delta=1e-3)
        trainer = Trainer(max_epochs=args.epochs, limit_train_batches=args.steps * args.acc_grads,
                          enable_model_summary=True, enable_progress_bar=True, enable_checkpointing=True,
                          strategy=DDPStrategy(find_unused_parameters=True), precision=16,
                          profiler=args.profile, check_val_every_n_epoch=1, limit_val_batches=args.val_steps,
                          accelerator='gpu', devices=devices,
                          callbacks=[checkpoint_callback, early_stopping])
        if args.find_lr:
            enhancer.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            best_lr = enhancer.find_lr(steps=args.steps, acc_grad=args.acc_grads)
            print(f'Best lr --- {best_lr}')
        else:
            trainer.fit(enhancer)
