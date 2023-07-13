import itertools
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

from random import randint, randrange, choice, choices, random
import imageio
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

import yaml
import argparse
import clearml
import os
import gc
import json
from glob import glob
from pathlib import Path
from tqdm import tqdm

import re
import cv2
import numpy as np
from datetime import datetime

import torch
from einops import rearrange

from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
import torchvision


def norm(dims, num_groups=32, min_channels_group=4):
    num_groups = min(num_groups, dims // min_channels_group)
    while dims % num_groups != 0:
        num_groups -= 1
    return torch.nn.GroupNorm(num_channels=dims, num_groups=num_groups, eps=1e-6)


def nonlinear(x):
    return torch.nn.functional.silu(x)


def grad_norm(model):
    grads = [
        param.grad.detach().flatten()
        for param in model.parameters()
        if param.grad is not None
    ]
    grads = torch.cat(grads)
    norm = grads.norm() / len(grads)
    return norm


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
    return torch.tensor(image).movedim(-1, -3)


def to_image(tensor, yuv=True):
    image = tensor.movedim(-3, -1).detach().cpu().numpy()
    if yuv:
        image = convert_YUV2RGB(image)
    image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    return image


def run_async(fun, *args, **kwargs):
    thread = threading.Thread(target=fun, args=args, kwargs=kwargs)
    thread.start()


class SimpleLogger:

    def __init__(self, clearml=None, run_async=False, tmpdir='.'):
        self.clearml = clearml
        self.run_async = run_async
        self.tmpdir = tmpdir
        if not os.path.exists(tmpdir):
            os.makedirs(tmpdir, exist_ok=True)

    def log_value(self, value, name, epoch=0, kind='val'):
        if self.clearml:
            self.clearml.report_scalar(name, kind, iteration=epoch, value=value)
        else:
            print(f'{name} --- {value}')

    def log_image(self, image, name, epoch=0):
        path = f"{self.tmpdir}/{name}.png"
        Image.fromarray(image).save(path)
        if self.clearml:
            self.clearml.report_image('valid', f"{name}", iteration=epoch, local_path=path)

    def log_images(self, images, prefix, epoch=0):
        for image_id, image in enumerate(images):
            self.log_image(image, f'{prefix}_{image_id}', epoch)

    def log_image_compare(self, images, texts, epoch, name='compare'):
        gallery = []
        for img, text in zip(images, texts):
            cv2.rectangle(img, pt1=(0, img.shape[0] // 30), pt2=(img.shape[1] // 90 * len(text), 0), color=0,
                          thickness=-1)
            cv2.putText(img, text=text, org=(0, img.shape[0] // 40), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3.0,
                        color=(255, 255, 255), thickness=3)
            gallery.append(img)
        gallery = np.concatenate(gallery, axis=1)
        if self.clearml:
            self.clearml.report_image('valid', name, iteration=epoch, image=gallery)
        else:
            Image.fromarray(gallery).save(f"{self.tmpdir}/{name}.png")

    def log_images_compare(self, images, texts, epoch, name='compare'):
        for idx, batch in enumerate(zip(*images)):
            self.log_image_compare(batch, texts, epoch=epoch, name=f'{name}_{idx}')

    def log_tensor(self, tensor, name='', depth=0):
        import lovely_tensors as lt
        if tensor.numel() > 0:
            print(f'{name} --- {lt.lovely(tensor, depth=depth)}')
        else:
            print(f'{name} --- empty tensor of shape {tensor.shape}')

    def log_plot(self, plt, name, epoch=0):
        if self.clearml:
            self.clearml.report_matplotlib_figure(title=name, series=f"valid", iteration=epoch, figure=plt)
        else:
            plt.savefig(f"{self.tmpdir}/{name}_{epoch}.png")
        plt.close()

    def log_distribution(self, values, name, epoch=0):
        sns.kdeplot(values)
        self.log_plot(plt, name, epoch)

    def log_values(self, values, name, epoch=0):
        sns.lineplot(values)
        self.log_plot(plt, name, epoch)

    def log_line(self, x, y, name, epoch=0):
        sns.lineplot(x=x, y=y)
        self.log_plot(plt, name, epoch)

    def _log_scatter2d(self, x, y, name, color=None, epoch=0):
        plt.scatter(x=x, y=y, c=None)
        self.log_plot(plt, name, epoch)

    def log_scatter2d(self, x, y, name, color=None, epoch=0):
        if self.run_async:
            run_async(self._log_scatter2d, x, y, name, color=color, epoch=epoch)
        else:
            self._log_scatter2d(x, y, name, color=color, epoch=epoch)

    def _log_scatter3d(self, x, y, z, name, color=None, epoch=0):
        ax = plt.axes(projection="3d")
        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(-1, 1)
        ax.scatter3D(x, y, z, c=color)
        self.log_plot(plt, name, epoch)

    def log_scatter3d(self, x, y, z, name, color=None, epoch=0):
        if self.run_async:
            run_async(self._log_scatter3d, x, y, z, name, color=color, epoch=epoch)
        else:
            self._log_scatter3d(x, y, z, name, color=color, epoch=epoch)

    def _log_mesh(self, vertices, faces, name, epoch=0):
        from mpl_toolkits.mplot3d import art3d
        pc = art3d.Poly3DCollection(vertices[faces],
                                    facecolors=np.ones((len(faces), 3), dtype=np.float32) * 0.75, edgecolor="gray")
        ax = plt.axes(projection="3d")
        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(-1, 1)
        ax.add_collection(pc)
        self.log_plot(plt, name, epoch)

    def log_mesh(self, vertices, faces, name, epoch=0):
        if self.run_async:
            run_async(self._log_mesh, vertices, faces, name, epoch=epoch)
        else:
            self._log_mesh(vertices, faces, name, epoch=epoch)

    def log_video(self, frames, gap, name, epoch=0):

        path = f'{self.tmpdir}/{name}_{epoch}.mp4'

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        w, h = frames[0].shape[1], frames[0].shape[0]
        writer = cv2.VideoWriter(path, apiPreference=0, fourcc=fourcc, fps=int(1 / (gap / 1000)), frameSize=(w, h))
        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame)
        writer.release()

        if self.clearml:
            self.clearml.report_media('video', name, iteration=epoch, local_path=path)

    def log_videos(self, videos_frames, gap, name, tempdir, epoch=0):
        for video_id, frames in enumerate(videos_frames):
            self.log_video(frames, gap, f'{name}_{video_id}', tempdir, epoch)

    def log_gif(self, frames, gap, name, epoch=0):
        path = f'{self.tmpdir}/{name}_{epoch}.gif'

        imageio.mimsave(path, frames, fps=int(1 / (gap / 1000)))
        if self.clearml:
            self.clearml.report_media('gifs', name, iteration=epoch, local_path=path)

    def log_gifs(self, frames_list, gap, name, tempdir, epoch=0):
        for idx, frames in enumerate(frames_list):
            self.log_gif(frames, gap, f'{name}_{idx}', tempdir, epoch)


class PatchedDataset(torch.utils.data.IterableDataset):

    def __init__(self, file_names, patch_size=16, patch_pad=2, scale=2, augment=False, orig='hr', mode='bilinear'):
        super(PatchedDataset, self).__init__()
        self.patch_size = patch_size
        self.patch_pad = patch_pad
        self.scale = scale
        self.augment = augment
        self.orig = orig
        self.file_names = file_names
        self.mode = mode

        self.objects = []

    def __load_file(self, file):
        frame = cv2.imread(file)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return to_tensor(frame).to(torch.float16)

    def load_file(self, file):
        return {self.orig: self.__load_file(file)}

    def __reset_cache(self):
        tasks = []
        with ThreadPoolExecutor(max_workers=min(8, multiprocessing.cpu_count())) as pool:
            for file in self.file_names:
                tasks.append(pool.submit(self.load_file, file=file))

            self.objects.clear()
            for task in tqdm(tasks, desc='Loading cache'):
                self.objects.append(task.result())

        gc.collect()

    def reset_cache(self):
        self.__reset_cache()

    def load_patch(self, obj):

        sr = obj[self.orig]
        c, h, w = sr.shape

        patch_size = (self.patch_size + self.patch_pad * 2) * self.scale
        i_start = randint(0, h - patch_size - 1)
        j_start = randint(0, w - patch_size - 1)
        sr_patch = sr[:, i_start:i_start + patch_size, j_start:j_start + patch_size]
        sr_patch = sr_patch.to(torch.float32)

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
                                                   mode=choice([self.mode]))[0]

        return lr_patch, sr_patch

    def __iter__(self):
        return self

    def __next__(self):
        return self.load_patch(choice(self.objects))


class GameDataset(torch.utils.data.Dataset):

    def __init__(self, file_names, mode='bilinear'):
        self.filenames = file_names
        self.mode = mode

    def transform(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image, dtype=np.float32) / 255.
        image = convert_RGB2YUV(image)
        image = torch.movedim(torch.tensor(image), -1, -3)

        return image

    def __getitem__(self, idx):

        label_path = self.filenames[idx]
        label = self.transform(cv2.imread(label_path))

        img_path = list(Path(label_path).parts)
        img_path[-2] = 'lr'
        img_path = os.path.join(*img_path)
        if os.path.exists(img_path):
            img = self.transform(cv2.imread(img_path))
        else:
            img = torch.nn.functional.interpolate(label, scale_factor=0.5, mode=self.mode)

        return img, label

    def __len__(self):
        return len(self.filenames)


def get_filenames(dataset_path, datasets_names=(), hr_subfolder='hr', pic_format='*.png'):
    filenames = []
    if len(datasets_names) == 0:
        f = glob(os.path.join(dataset_path, hr_subfolder, pic_format))
        print(dataset_path, hr_subfolder, pic_format)
        filenames.extend(f)
    elif datasets_names[0] == 'all':  # take all game folders in the root folder
        dir_all = glob(os.path.join(dataset_path, '*'))
        for dir_game in dir_all:
            print(dir_game, hr_subfolder, pic_format)
            f = glob(os.path.join(dir_game, hr_subfolder, pic_format))
            print(dir_game, hr_subfolder, pic_format)
            filenames.extend(f)
    else:  # take all files from folder games in the list dataset_names
        for game in datasets_names:
            f = glob(os.path.join(dataset_path, game, hr_subfolder, pic_format))
            filenames.extend(f)

    return filenames


class Conv(torch.nn.Module):
    def __init__(self, n_channels,
                 kernel_size):
        super().__init__()
        self.conv = torch.nn.Conv2d(n_channels, n_channels,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=(kernel_size // 2, kernel_size // 2))
        self.relu = torch.nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        return out


class ConvBlock(torch.nn.Module):

    def __init__(self, n_channels,
                 block_size,
                 kernel_size):
        super().__init__()

        layers = []
        for _ in range(block_size):
            layers.append(Conv(n_channels, kernel_size))
        self.separable_conv_block = torch.nn.Sequential(*layers)

    def forward(self, x):
        out = self.separable_conv_block(x)
        out = out + x
        return out


class DepthwiseSeparableConv(torch.nn.Module):
    def __init__(self, n_channels,
                 kernel_size,
                 quantization_friendly=False):
        super().__init__()

        self.quantization_friendly = quantization_friendly
        self.dwconv = torch.nn.Conv2d(n_channels, n_channels,
                                      kernel_size=kernel_size,
                                      stride=1,
                                      padding=(kernel_size // 2, kernel_size // 2),
                                      groups=n_channels)
        self.pwconv = torch.nn.Conv2d(n_channels, n_channels,
                                      kernel_size=1,
                                      stride=1,
                                      padding=(0, 0))
        self.relu = torch.nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.dwconv(x)
        if not self.quantization_friendly:
            out = self.relu(out)
        out = self.pwconv(out)
        out = self.relu(out)
        return out


class DepthwiseSeparableConvBlock(torch.nn.Module):

    def __init__(self, n_channels,
                 block_size,
                 kernel_size):
        super().__init__()

        layers = []
        for _ in range(block_size):
            layers.append(DepthwiseSeparableConv(n_channels, kernel_size))
        self.separable_conv_block = torch.nn.Sequential(*layers)

    def forward(self, x):
        out = self.separable_conv_block(x)
        out = out + x
        return out


class ResidualConv(torch.nn.Module):

    def __init__(self, in_channels, kernel_size):
        super().__init__()

        expand_channels = in_channels // 2

        self.pwconv_squeeze = torch.nn.Conv2d(in_channels, expand_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=(0, 0))

        self.conv = torch.nn.Conv2d(expand_channels, expand_channels,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=(kernel_size // 2, kernel_size // 2))

        self.pwconv_expand = torch.nn.Conv2d(expand_channels, in_channels,
                                             kernel_size=1,
                                             stride=1,
                                             padding=(0, 0))
        self.relu = torch.nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.pwconv_squeeze(x)
        out = self.relu(out)
        out = self.conv(out)
        out = self.relu(out)
        out = self.pwconv_expand(out)
        out = self.relu(out)
        return out


class ResidualConvBlock(torch.nn.Module):

    def __init__(self, in_channels,
                 block_size,
                 kernel_size):
        super().__init__()

        layers = []
        for _ in range(block_size):
            layers.append(ResidualConv(in_channels, kernel_size))
        self.residual_conv_block = torch.nn.Sequential(*layers)

    def forward(self, x):
        out = self.residual_conv_block(x)
        out = out + x
        return out


class InvertedResidualConv(torch.nn.Module):

    def __init__(self, in_channels, kernel_size):
        super().__init__()

        expand_channels = in_channels * 2

        self.pwconv_expand = torch.nn.Conv2d(in_channels, expand_channels,
                                             kernel_size=1,
                                             stride=1,
                                             padding=(0, 0))

        self.dwconv = torch.nn.Conv2d(expand_channels, expand_channels,
                                      kernel_size=kernel_size,
                                      stride=1,
                                      padding=(kernel_size // 2, kernel_size // 2),
                                      groups=expand_channels)

        self.pwconv_squeeze = torch.nn.Conv2d(expand_channels, in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=(0, 0))
        self.relu = torch.nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.pwconv_expand(x)
        out = self.relu(out)
        out = self.dwconv(out)
        out = self.relu(out)
        out = self.pwconv_squeeze(out)
        out = self.relu(out)
        return out


class InvertedResidualConvBlock(torch.nn.Module):

    def __init__(self, in_channels,
                 block_size,
                 kernel_size):
        super().__init__()

        layers = []
        for _ in range(block_size):
            layers.append(InvertedResidualConv(in_channels, kernel_size))
        self.inverted_residual_conv_block = torch.nn.Sequential(*layers)

    def forward(self, x):
        out = self.inverted_residual_conv_block(x)
        out = out + x
        return out


class DepthwiseInceptionBlock(torch.nn.Module):

    def __init__(self, in_channels, kernel_size):
        super().__init__()

        self.in_channels = in_channels

        if not self.in_channels % 3:
            self.mid_channels = self.in_channels // 3
            self.pass3 = ResidualConv(self.in_channels, kernel_size)

        elif not self.in_channels % 2:
            self.mid_channels = self.in_channels // 2

        self.pass1 = DepthwiseSeparableConv(self.in_channels, kernel_size)
        self.pass2 = InvertedResidualConv(self.in_channels, kernel_size)

    def forward(self, x):

        out1 = self.pass1(x)
        out2 = self.pass2(x)

        if not self.in_channels // 3:
            out3 = self.pass3(x)
            out = out1 + out2 + out3
        else:
            out = out1 + out2

        return out


class Wide3DepthwiseConv(torch.nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        squeeze_channels = in_channels // 3
        self.pwconv_squeeze = torch.nn.Conv2d(in_channels, squeeze_channels, (1, 1), (1, 1), (0, 0))

        self.conv1 = torch.nn.Conv2d(squeeze_channels, squeeze_channels, (3, 3), (1, 1), (1, 1))

        self.conv2 = torch.nn.Conv2d(squeeze_channels, squeeze_channels, (5, 5), (1, 1), (2, 2))

        self.dwconv3 = torch.nn.Conv2d(squeeze_channels, squeeze_channels, (3, 3), (1, 1), (1, 1),
                                       groups=squeeze_channels)
        self.pwconv3 = torch.nn.Conv2d(squeeze_channels, squeeze_channels, (1, 1), (1, 1), (0, 0))

        self.act = torch.nn.SiLU()

    def forward(self, x):
        x = self.pwconv_squeeze(x)
        x = self.act(x)

        x1 = self.conv1(x)
        x1 = self.act(x1)

        x2 = self.conv2(x)
        x2 = self.act(x2)

        x3 = self.dwconv3(x)
        x3 = self.act(x3)
        x3 = self.pwconv3(x3)
        x3 = self.act(x3)

        x = torch.cat((x1, x2, x3), dim=1)

        return x


class WideDepthwiseConvBlock(torch.nn.Module):

    def __init__(self, in_channels,
                 block_size,
                 kernel_size):
        super().__init__()

        layers = []
        for _ in range(block_size):
            layers.append(Wide3DepthwiseConv(in_channels))
        self.wide_depthwise_conv_block = torch.nn.Sequential(*layers)

    def forward(self, x):
        out = self.wide_depthwise_conv_block(x)
        out = out + x
        return out


class UpscalingModelBase(torch.nn.Module):

    def __init__(self, n_channels,
                 n_blocks,
                 in_channels=1,
                 block_size=2,
                 kernel_size=3,
                 upscale_factor=2,
                 upscaling_method='PixelShuffle',
                 main_act='ReLU',
                 final_act='Sigmoid',
                 main_block_type='depthwise-separable-convolution'):
        super().__init__()

        self.in_channels = in_channels
        self.scale = upscale_factor

        self.first_conv = torch.nn.Conv2d(in_channels, n_channels,
                                          kernel_size=kernel_size,
                                          stride=1,
                                          padding=(kernel_size // 2, kernel_size // 2))

        if main_act == 'ReLU':
            self.activation = torch.nn.ReLU()
        elif main_act == 'PReLU':
            self.activation = torch.nn.PReLU()
        elif main_act == 'SiLU':
            self.activation = torch.nn.SiLU()
        elif main_act == 'LeakyReLU':
            self.activation = torch.nn.LeakyReLU(negative_slope=0.1)
        elif main_act == 'HardSwish':
            self.activation = torch.nn.Hardswish(inplace=False)

        layers = []
        for _ in range(n_blocks):

            if main_block_type == 'convolution':
                layers.append(ConvBlock(n_channels, block_size, kernel_size))
            elif main_block_type == 'depthwise-separable-convolution':
                layers.append(DepthwiseSeparableConvBlock(n_channels, block_size, kernel_size))
            elif main_block_type == 'residual-convolution':
                layers.append(ResidualConvBlock(n_channels, block_size, kernel_size))
            elif main_block_type == 'inverted-residual-convolution':
                layers.append(InvertedResidualConvBlock(n_channels, block_size, kernel_size))
            elif main_block_type == 'depthwise-inception':
                layers.append(DepthwiseInceptionBlock(n_channels, kernel_size))
            elif main_block_type == 'wide-depthwise-convolution':
                layers.append(WideDepthwiseConvBlock(n_channels, block_size, kernel_size))

        self.base_blocks_seq = torch.nn.Sequential(*layers)

        if upscaling_method == 'PixelShuffle':
            self.upscaling_layer = torch.nn.PixelShuffle(self.scale)
        elif upscaling_method == 'ConvTranspose':
            self.upscaling_layer = torch.nn.ConvTranspose2d(n_channels, n_channels // (self.scale * self.scale),
                                                            kernel_size=self.scale, stride=self.scale, padding=0)

        self.final_conv = torch.nn.Conv2d(n_channels // (self.scale * self.scale), in_channels,
                                          kernel_size=kernel_size,
                                          stride=1,
                                          padding=(kernel_size // 2, kernel_size // 2))

        if final_act == 'Tanh':
            self.final_act = torch.nn.Tanh()
        elif final_act == 'Sigmoid':
            self.final_act = torch.nn.Sigmoid()

    def forward(self, x):

        x = x[:, :self.in_channels] * 2 - 1.0

        Y = self.first_conv(x)
        Y = self.activation(Y)

        Y = self.base_blocks_seq(Y)

        Y = self.upscaling_layer(Y)

        Y = self.final_conv(Y)
        Y = self.final_act(Y)

        Y = torch.clamp(Y / 2 + 0.5, min=0, max=1)

        return Y


def build_model(cfg):
    model = UpscalingModelBase(n_channels=cfg['model']['n_channels'],
                               n_blocks=cfg['model']['n_blocks'],
                               in_channels=cfg['model']['channels'],
                               block_size=cfg['model']['block_size'],
                               kernel_size=cfg['model']['kernel_size'],
                               upscale_factor=cfg['model']['upscale_factor'],
                               upscaling_method=cfg['model']['upscaling_method'],
                               main_act=cfg['model']['main_act'],
                               final_act=cfg['model']['final_act'],
                               main_block_type=cfg['model']['main_block_type'])
    return model


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


def charbonnier_loss(x, y, eps=1e-6):
    return torch.sqrt((x - y) ** 2 + eps).mean()


def multiscale_loss(x1, x2, scales=(1, 2), loss_fn=charbonnier_loss):
    loss = 0
    for scale in scales:
        __x1 = torch.nn.functional.interpolate(x1, scale_factor=1 / scale, mode='bilinear')
        __x2 = torch.nn.functional.interpolate(x2, scale_factor=1 / scale, mode='bilinear')
        loss += loss_fn(__x1, __x2)
    return loss / len(scales)


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

    def __init__(self, in_channels=1, dim=128, n_blocks=2, input_kernel=5):
        super(SimpleDiscriminator, self).__init__()
        self.in_channels = in_channels
        self.first_conv = torch.nn.Conv2d(in_channels, dim, kernel_size=input_kernel, padding=input_kernel // 2)
        self.feature_extractor = torch.nn.Sequential(*[ResBlock(dim) for _ in range(n_blocks)])
        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.utils.spectral_norm(torch.nn.Linear(dim, 1)),
        )

    def forward(self, x, return_probs=True):
        h = self.first_conv(x[:, :self.in_channels])
        features = self.feature_extractor(h)
        probs = self.classifier(features)
        return torch.sigmoid(probs.view(-1)) if return_probs else probs.view(-1)


class Losses():
    def __init__(self, loss_coeff, rbf=None):
        self.coeff = loss_coeff
        self.rbf = rbf

    def combined_loss(self, y_pred, y_true):
        loss = torch.nn.functional.mse_loss(y_true, y_pred) * (self.coeff['mse'] if 'mse' in self.coeff else 0.0) + \
               torch.nn.functional.l1_loss(y_true, y_pred) * (self.coeff['mae'] if 'mae' in self.coeff else 0.0) + \
               peak_signal_noise_ratio(y_pred, y_true, data_range=1.0) \
               * (self.coeff['psnr'] if 'psnr' in self.coeff else 0.0) + \
               structural_similarity_index_measure(y_pred, y_true, data_range=1.0) \
               * (self.coeff['ssim'] if 'ssim' in self.coeff else 0.0) + \
               ((charbonnier_loss(self.rbf(y_pred), self.rbf(y_true)) * self.coeff[
                   'rbf']) if self.rbf and 'rbf' in self.coeff else 0.0)
        return loss


class Trainer():

    def __init__(self, cfg):

        self.cfg = cfg
        self.device = torch.device('cuda', torch.cuda.current_device()) if torch.cuda.is_available() \
            else torch.device('cpu')

        pref_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.val_weights_path = cfg['saving']['ckp_folder'] + cfg['model']['name'] + '-' + pref_time \
                                + '-val-weights.pth'
        self.train_weights_path = cfg['saving']['ckp_folder'] + cfg['model']['name'] + '-' + pref_time \
                                  + '-train-weights.pth'
        self.logs_path = cfg['saving']['log_folder'] + cfg['model']['name'] + '-' + pref_time + '-logs.json'

        self.scale = self.cfg['model']['upscale_factor']
        self.channels = self.cfg['model']['channels']
        self.model = None
        self.trainloader = None
        self.validloaders = None
        self.best_train_loss = 100.0
        self.best_val_loss = 100.0
        self.best_val_psnr = 0.0
        self.losses = None
        self.optimizer = None
        self.scheduler = None

        self.configure()
        if cfg['train']['pretrained_weights']:
            self.load_pretrained()

    def configure(self):
        self.configure_model()
        self.configure_loss()
        self.configure_optimizer()
        self.prepare_val_data()

    def metrics(self, outputs, labels):
        return {
            'loss': self.losses.combined_loss(outputs[:, :self.channels], labels[:, :self.channels]),
            'psnr': peak_signal_noise_ratio(outputs[:, :1], labels[:, :1], data_range=1.0),  # gamma channel only
            'ssim': structural_similarity_index_measure(outputs[:, :1], labels[:, :1], data_range=1.0)
        }

    def forward(self, x):
        return self.model(x)

    def train(self, epoch, steps=None):

        self.model.train()
        if steps is None:
            if not self.cfg['data']['patched']:
                steps = len(self.trainloader)
            else:
                raise RuntimeError('Train dataloader has no len and num of steps was not specified')

        metrics = defaultdict(list)

        data_iter = iter(self.trainloader)
        for batch_id in range(steps):

            data = next(data_iter)
            inputs = data[0].to(self.device, non_blocking=True)
            labels = data[1].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.forward(inputs)
            __metrics = self.metrics(outputs, labels)
            __metrics['loss'].backward()
            __metrics['grad-norm'] = grad_norm(self.model)
            self.optimizer.step()

            for k, v in __metrics.items():
                metrics[k].append(v.item())

            record_step = steps // self.cfg['train']['log_freq']
            if batch_id % record_step == record_step - 1:  # record every record_step batches
                print(f'[{epoch + 1}, {batch_id + 1:5d}]',
                      *(f'{name} --- {torch.tensor(metrics[name][-record_step:]).mean().item():.6f}'
                        for name in metrics.keys()))

        # if not isinstance(self.scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
        self.scheduler.step()

        return {k: torch.tensor(v).mean().item() for k, v in metrics.items()}

    def patch_upscale(self, images):

        b, c, h_orig, w_orig = images.shape
        patch_size, patch_pad = self.cfg['data']['patch_size'], self.cfg['data']['patch_pad']
        h_add = patch_size - (h_orig - h_orig // patch_size * patch_size)
        w_add = patch_size - (w_orig - w_orig // patch_size * patch_size)
        images = torch.cat([torch.zeros_like(images[:, :, :h_add // 2 + h_add % 2]),
                            images,
                            torch.zeros_like(images[:, :, :h_add // 2])], dim=2)
        images = torch.cat([torch.zeros_like(images[:, :, :, :w_add // 2 + w_add % 2]),
                            images,
                            torch.zeros_like(images[:, :, :, :w_add // 2])], dim=3)
        h, w = images.shape[-2:]
        # divide image to patches
        __tiles = rearrange(images, 'b c (n p1) (m p2) -> b c n m p1 p2', p1=patch_size, p2=patch_size)

        # add up and down padding in height
        up_pad = __tiles[:, :, 1:, :, :patch_pad, :]
        up_pad = torch.cat([up_pad, torch.zeros_like(up_pad[:, :, :1])], dim=2)
        down_pad = __tiles[:, :, :-1, :, -patch_pad:, :]
        down_pad = torch.cat([torch.zeros_like(down_pad[:, :, :1]), down_pad], dim=2)
        __tiles = torch.cat([down_pad, __tiles, up_pad], dim=-2)

        # add left and right paddings in width
        right_pad = __tiles[:, :, :, 1:, :, :patch_pad]
        right_pad = torch.cat([right_pad, torch.zeros_like(right_pad[:, :, :, :1])], dim=3)
        left_pad = __tiles[:, :, :, :-1, :, -patch_pad:]
        left_pad = torch.cat([torch.zeros_like(left_pad[:, :, :, :1]), left_pad], dim=3)
        __tiles = torch.cat([left_pad, __tiles, right_pad], dim=-1)

        # rearrange to batch tiles
        tiles = rearrange(__tiles, 'b c n m p1 p2 -> (b n m) c p1 p2')

        upscaled = []
        for __tiles in torch.split(tiles, self.cfg['data']['batch_size'], dim=0):
            upscaled.append(self.forward(__tiles.to(self.device)).to(images.device))
            del __tiles
        gc.collect()
        torch.cuda.empty_cache()

        upscaled = torch.cat(upscaled, dim=0)
        upscaled = rearrange(upscaled, '(b n m) c p1 p2 -> b c n m p1 p2', n=h // patch_size, m=w // patch_size)
        upscaled = upscaled[:, :, :, :, patch_pad * self.scale:-patch_pad * self.scale,
                   patch_pad * self.scale:-patch_pad * self.scale]
        upscaled = rearrange(upscaled, 'b c n m p1 p2 -> b c (n p1) (m p2)', n=h // patch_size, m=w // patch_size)
        upscaled = upscaled[:, :, (h_add // 2 + h_add % 2) * self.scale: -h_add // 2 * self.scale,
                   (w_add // 2 + w_add % 2) * self.scale: -w_add // 2 * self.scale]

        return upscaled

    @torch.no_grad()
    def upscale(self, inputs):
        orig_device = inputs.device
        if self.cfg['data']['patched']:
            return self.patch_upscale(inputs)
        else:
            return self.forward(inputs.to(self.device)).to(orig_device)

    def validate(self):

        self.model.eval()
        total_metrics = []
        for valid_idx, validloader in enumerate(self.validloaders):

            metrics = defaultdict(list)
            upscaled_images = []
            with torch.no_grad():
                for batch_idx, data in enumerate(validloader):
                    inputs, labels = data[0].to(self.device), data[1].to(self.device)
                    outputs = self.upscale(inputs)

                    for k, v in self.metrics(outputs, labels).items():
                        metrics[k].append(v.item())

                    out = torch.cat([
                        outputs[:, :self.channels],
                        torch.nn.functional.interpolate(inputs[:, self.channels:],
                                                        scale_factor=self.scale, mode=self.cfg['data']['mode'])], dim=1)
                    upscaled_images.append(to_image(out))

            val_metrics = {k: torch.tensor(v).mean().item() for k, v in metrics.items()}
            total_metrics.append(val_metrics)

            if self.cfg['data']['out'] and val_metrics['psnr'] > self.best_val_psnr:

                out_path = os.path.join(self.cfg['data']['out'], f'val_{valid_idx}')
                os.makedirs(out_path, exist_ok=True)

                upscaled_images = np.concatenate(upscaled_images, axis=0)
                for image_idx in range(len(upscaled_images)):
                    cv2.imwrite(os.path.join(out_path, f'{image_idx}.png'),
                                cv2.cvtColor(upscaled_images[image_idx], cv2.COLOR_RGB2BGR))
        return total_metrics

    def save_results(self, item):

        with open(self.logs_path, mode='a', encoding='utf-8') as f:
            json.dump(item, f)
            f.write('\n')

        if item['train_loss'] < self.best_train_loss:
            print("Best train loss updated.")
            self.best_train_loss = item['train_loss']
            torch.save(self.model.state_dict(), self.train_weights_path)

        val_loss, val_psnr = [], []
        for key, value in item.items():
            if re.match(f'^val\d_loss$', key):
                val_loss.append(value)
            if re.match(f'^val\d_psnr$', key):
                val_psnr.append(value)
        val_loss = torch.tensor(val_loss).mean().item()
        if val_loss < self.best_val_loss:
            print("Best val loss updated.")
            torch.save(self.model.state_dict(), self.val_weights_path.replace('val', 'val-loss'))
            self.best_val_loss = val_loss
        val_psnr = torch.tensor(val_psnr).mean().item()
        if val_psnr > self.best_val_psnr:
            print("Best val psnr updated")
            torch.save(self.model.state_dict(), self.val_weights_path.replace('val', 'val-psnr'))
            self.best_val_psnr = val_psnr

    def get_dataset(self, folder, datasets=(), batch_size=1, patched=False):

        file_names = get_filenames(folder, datasets)
        if patched:
            dataset = PatchedDataset(file_names, patch_size=self.cfg['data']['patch_size'],
                                     patch_pad=self.cfg['data']['patch_pad'],
                                     augment=self.cfg['data']['augment'])
            dataset.reset_cache()
        else:
            dataset = GameDataset(file_names)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=not patched, num_workers=4,
                                                 prefetch_factor=2, pin_memory=True)
        return dataloader

    def prepare_train_data(self):
        self.trainloader = self.get_dataset(self.cfg['data']['train_folder'],
                                            self.cfg['data']['train_datasets'],
                                            self.cfg['data']['batch_size'], patched=self.cfg['data']['patched'])

        if not self.cfg['data']['patched']:
            train_len = len(self.trainloader) * self.cfg['data']['batch_size']
            print(f'Number of images in the train: {train_len}')

    def prepare_val_data(self):
        self.validloaders = [self.get_dataset(folder, batch_size=self.cfg['data']['val_batch_size'])
                             for folder in self.cfg['data']['val_folder']]
        for n, validloader in enumerate(self.validloaders):
            val_len = len(validloader) * self.cfg['data']['val_batch_size']
            print(f'Number of images in {n + 1} validation: {val_len}')

    def configure_model(self):

        model = build_model(self.cfg)
        print('\nModel', self.cfg['model']['name'])

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Total number of parameters: ', total_params)
        print('Number of trainable parameters: ', trainable_params)
        print(model)
        self.model = model

        print('device: ', self.device)
        self.model.to(self.device)

    def configure_loss(self):
        if self.cfg['train']['rbf_filters']:
            rbf = RandomBinaryFilter(in_channels=self.channels, filters=self.cfg['train']['rbf_filters'])
            rbf = rbf.to(self.device)
        else:
            rbf = None
        self.losses = Losses(loss_coeff=self.cfg['metrics']['loss_coeff'], rbf=rbf)

    def configure_optimizer(self):

        lr = self.cfg['train']['base_rate']
        if self.cfg['train']['optimizer'] == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        elif self.cfg['train']['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.99))

        if self.cfg['train']['sched'] == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, self.cfg['train']['sched_first'],
                T_mult=self.cfg['train']['sched_mult'], eta_min=lr * self.cfg['train']['sched_min'])
        elif self.cfg['train']['sched'] == 'multistep':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                  milestones=self.cfg['train']['sched_steps'],
                                                                  gamma=self.cfg['train']['sched_gamma'])

    def get_onnx(self):
        x = torch.randn(1, 1, 720, 1280)
        self.model.to('cpu')
        self.model.eval()

        torch.onnx.export(self.model, x, f"{self.cfg['model']['name']}.onnx",
                          export_params=True,
                          input_names=['input'],
                          output_names=['output'])

    def load_pretrained(self):
        ckp_dir = self.cfg['saving']['ckp_folder']
        file = os.path.join(ckp_dir, self.cfg['train']['pretrained_weights'])

        dict_weights = torch.load(file, map_location=self.device)

        dict_weights = {k.split('module.')[-1]: v for k, v in dict_weights.items()}
        status = self.model.load_state_dict(dict_weights)
        print('Weights loaded: ', status)

        val_metrics = self.validate()
        item = {}
        for val_idx, val_item in enumerate(val_metrics):
            item.update({f'val{val_idx}_{k}': v for k, v in val_item.items()})
        print(f'Validation metrics for loaded weights:', *(f'{k}: {v:.6f}' for k, v in item.items()))

    def find_lr(self, min_lr=1e-6, max_lr=1.0, steps=1000, acc_grad=1, monitor='loss', max_factor=10.0):

        # setup optimizer
        optim = self.optimizer

        data_iter = iter(self.trainloader)
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
                inputs, labels = (v.to(self.device) for v in next(data_iter))
                outputs = self.forward(inputs)
                __metrics = self.metrics(outputs, labels)
                (__metrics['loss'] / acc_grad).backward()
                value += __metrics[monitor].item() / acc_grad

            values.append(value)

            optim.step()

            if value < best_loss:
                best_loss = value
                best_lr = lr

            if step_id > 0 and value > best_loss * max_factor:
                break

            pb_steps.set_description(
                f'Step [{step_id + 1}/{steps}] [lr:{lrs[-1]:.8f}] [loss:{values[-1]:.6f}] [best_loss:{best_loss:.6f}]',
                refresh=True)
        pb_steps.close()
        return np.asarray(lrs), np.clip(np.asarray(values), a_min=0.0, a_max=best_loss * max_factor)


def run(cfg, logger):
    seed = 131313
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model_trainer = Trainer(cfg)
    model_trainer.prepare_train_data()  # train data didn't loaded automatically as it loads all dataset in memory

    print(f'Start training on device: ', model_trainer.device)
    for epoch in range(cfg['train']['n_epochs']):

        train_metrics = model_trainer.train(epoch, steps=cfg['train']['steps'])
        val_metrics = model_trainer.validate()

        item = {
            'Epoch': epoch,
            'lr': model_trainer.optimizer.param_groups[0]["lr"],
            **{f'train_{k}': v for k, v in train_metrics.items()},
        }
        for val_idx, val_item in enumerate(val_metrics):
            item.update({f'val{val_idx}_{k}': v for k, v in val_item.items()})

        model_trainer.save_results(item)

        print(*(f'{k}: {v:.6f}' if isinstance(v, float) else f'{k}: {v}' for k, v in item.items()))

        for key, group in itertools.groupby(
                sorted(item.keys(), key=lambda x: x.split('_')[-1]), key=lambda x: x.split('_')[-1]):
            for k in group:
                logger.report_scalar(title=key, series=k, iteration=epoch, value=item[k])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default='scripts/dd/configs/upscaling.yaml')
    parser.add_argument("--val", action='store_true', help='run only validate')
    parser.add_argument("--lr", action='store_true', help='run only learning rate finding')
    parser.set_defaults(val=False, lr=False)
    args = parser.parse_args()

    config_path = args.config_path
    cfg = yaml.load(open(config_path, 'r'), yaml.FullLoader)

    if args.val:
        # When trainer loads weights it runs validation and all val upscaling results are saved in out directory
        Trainer(cfg)
    elif args.lr:
        task = clearml.Task.init(project_name="upscaling", task_name="lr finding", reuse_last_task_id=True)
        task.connect(cfg, name='main params')
        logger = task.get_logger()
        trainer = Trainer(cfg)
        trainer.prepare_train_data()
        lr, values = trainer.find_lr()
        SimpleLogger(clearml=logger).log_line(x=lr, y=values, name='learning rate')
    else:

        task = clearml.Task.init(project_name="upscaling",
                                 task_name=f"{cfg['model']['name']}",
                                 reuse_last_task_id=True)
        logger = task.get_logger()
        task.connect(cfg, name='main params')

        run(cfg, logger)
