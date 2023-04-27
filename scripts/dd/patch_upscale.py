from modules.common.util import *
from modules.common.model import *
from modules.common.trainer import *
from scripts.dd.swin2sr import Swin2SR

import numpy as np
import os
import cv2
from random import choice, randint, shuffle
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
    image = np.clip((tensor.movedim(-3, -1) * 0.5 + 0.5).detach().cpu().numpy(), 0, 1)
    if yuv:
        image = convert_YUV2RGB(image)
    image = (image * 255).astype(np.uint8)
    return image


class FeatureBlock(torch.nn.Module):

    def __init__(self, dim, in_dim=-1, kernel_size=3):
        super(FeatureBlock, self).__init__()

        self.in_dim = in_dim if in_dim > 0 else dim

        self.conv1 = torch.nn.Conv2d(in_dim, dim, kernel_size=kernel_size, padding=1)
        self.conv2 = torch.nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=1)

    def forward(self, x):
        h = self.conv1(nonlinear(x))
        h = self.conv2(nonlinear(h))
        return (h + x) / 2 ** 0.5


class PatchEnhancer(torch.nn.Module):

    def __init__(self, in_channels=1, dim=32, tile_pad=2, n_blocks=4):
        super(PatchEnhancer, self).__init__()
        self.in_channels = in_channels
        self.tile_pad = tile_pad
        self.input_conv = torch.nn.Conv2d(in_channels, dim, kernel_size=1 + self.tile_pad * 2, padding=0, stride=1)
        self.blocks = torch.nn.ModuleList([FeatureBlock(dim, dim) for _ in range(n_blocks)])
        self.out = torch.nn.Conv2d(dim, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # extract channels
        h = x[:, :self.in_channels]

        # first conv with no padding to use tile_pad
        h = self.input_conv(h)

        # deep feature extraction
        for block in self.blocks:
            h = block(h)

        # output conv
        h = self.out(nonlinear(h))

        # get updated and remained part
        updated = x[:, :self.in_channels, self.tile_pad: -self.tile_pad, self.tile_pad: -self.tile_pad]
        remained = x[:, self.in_channels:, self.tile_pad: -self.tile_pad, self.tile_pad: -self.tile_pad]

        return torch.cat((updated + h, remained), dim=-3)


class SRTiles(torch.utils.data.IterableDataset):

    def __init__(self, folder, tile=16, tile_pad=1, scale=2, cache_size=512):
        super(SRTiles, self).__init__()
        self.tile = tile
        self.tile_pad = tile_pad
        self.scale = scale
        if type(folder) is not list:
            folder = [folder]
        self.folder = folder
        self.files = [os.path.join(f, file) for f in self.folder for file in os.listdir(f) if
                      os.path.splitext(file)[1] in ('.png', '.jpg', '.jpeg')]

        self.cache_size = cache_size
        self.cache = []

    def reset_cache(self):

        if self.cache_size <= 0:
            raise RuntimeError('Caching is not enabled, use cache_size>0 when initializing dataset')

        cache_files = [os.path.join(f, file) for f in self.folder for file in os.listdir(f) if
                       os.path.splitext(file)[1] in ('.png', '.jpg', '.jpeg')]
        shuffle(cache_files)
        cache_files = cache_files[:self.cache_size]

        with ProcessPoolExecutor() as executor:
            self.cache = executor.map(self.load_file, cache_files)
        gc.collect()

    def load_file(self, file):
        frame = cv2.imread(file)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return to_tensor(frame)

    def load_patch(self, tensor):

        c, h, w = tensor.shape

        tile = (self.tile + self.tile_pad * 2) * self.scale
        i_start = randint(0, h - tile - 1)
        j_start = randint(0, w - tile - 1)
        patch = tensor[:, i_start:i_start + tile, j_start:j_start + tile]

        return patch

    def __iter__(self):
        return self

    def __next__(self):
        if self.cache_size < 0:
            # update list in case of new files
            self.files = [os.path.join(f, file) for f in self.folder for file in os.listdir(f)
                          if os.path.splitext(file)[1] in ('.png', '.jpg', '.jpeg')]
            file = choice(self.files)
            return self.load_patch(self.load_file(file))
        else:
            return self.load_patch(choice(self.cache))


class SRImages(torch.utils.data.IterableDataset):

    def __init__(self, folder, w=2560, h=1440):
        super(SRImages, self).__init__()
        self.w = w
        self.h = h
        if type(folder) is not list:
            folder = [folder]
        self.folder = folder
        self.files = [os.path.join(f, file) for f in self.folder for file in os.listdir(f) if
                      os.path.splitext(file)[1] in ('.png', '.jpg', '.jpeg')]

    def load_file(self, file):
        frame = cv2.imread(file)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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

    def load_images(self):
        return [self.load_file(f) for f in self.files]

    def __iter__(self):
        return self

    def __next__(self):
        # update list in case of new files
        self.files = [os.path.join(f, file) for f in self.folder for file in os.listdir(f)
                      if os.path.splitext(file)[1] in ('.png', '.jpg', '.jpeg')]
        file = choice(self.files)
        return self.load_file(file)


class ImageEnhancer(pl.LightningModule):

    def __init__(self,
                 model=None,
                 dataset=None,
                 test_dataset=None,
                 test_images=None,
                 clearml=None,
                 teacher=None,
                 teacher_rate=1.0,
                 ema_weight=0.995,
                 learning_rate=1e-3,
                 initial_lr_rate=0.1,
                 min_lr_rate=0.01,
                 batch_size=1024,
                 epochs=300,
                 steps=1000,
                 samples_epoch=5,
                 scale=2,
                 tile=16,
                 tile_pad=2,
                 debug=True,
                 ):
        super(ImageEnhancer, self).__init__()

        self.save_hyperparameters(ignore=['model', 'dataset' 'test_dataset', 'test_images', 'clearml', 'teacher'])

        self.scale = scale
        self.tile = tile
        self.tile_pad = tile_pad

        self.custom_logger = SimpleLogger(clearml=clearml)
        self.debug = debug

        self.dataset = dataset
        self.test_dataset = test_dataset if exists(test_dataset) else dataset
        self.test_images = test_images

        self.teacher = teacher if exists(teacher) else None
        self.teacher_rate = teacher_rate

        self.model = model
        self.use_ema = ema_weight is not None
        if self.use_ema:
            self.ema_model = EMA(self.model, decay=ema_weight)

        self.learning_rate = learning_rate
        self.min_lr_rate = min_lr_rate
        self.initial_lr_rate = initial_lr_rate
        self.batch_size = batch_size
        self.steps = steps
        self.epochs = epochs
        self.samples_epoch = samples_epoch

    def forward(self, x, train=True, **kwargs):

        model = cases([
            (self.use_ema and (not train or not self.training), self.ema_model.module),
            self.model
        ])

        upscaled = model(x)

        return upscaled

    @torch.inference_mode()
    def get_sr(self, orig, down_sampled, window_size=8):

        # remove padding from orig batch
        orig = orig[:, :,
               self.tile_pad * self.scale: -self.tile_pad * self.scale,
               self.tile_pad * self.scale: -self.tile_pad * self.scale]

        # upscale model with teacher
        if self.teacher:
            patches = torch_convert_YUV2RGB(down_sampled * 0.5 + 0.5)

            # pad for SwinSR
            tile_size = self.tile + self.tile_pad * 2
            pad_size = (tile_size // window_size + 1) * window_size - tile_size
            patches = torch.cat([patches, torch.flip(patches, [2])], 2)[:, :, :tile_size + pad_size, :]
            patches = torch.cat([patches, torch.flip(patches, [3])], 3)[:, :, :, :tile_size + pad_size]

            patches = self.teacher(patches)[:, :,
                      self.tile_pad * self.scale: -(self.tile_pad + pad_size) * self.scale,
                      self.tile_pad * self.scale: -(self.tile_pad + pad_size) * self.scale]

            patches = torch_convert_RGB2YUV(patches) * 2 - 1.0

            orig = patches * self.teacher_rate + orig * (1 - self.teacher_rate)

            del patches
            gc.collect()
            torch.cuda.empty_cache()

        return orig

    def step(self, batch, train=True):

        # imitate low resolution
        down_sampled = torch.nn.functional.interpolate(batch, scale_factor=1 / self.scale, mode='bicubic')

        # upsample patch and enhance it
        up_sampled = torch.nn.functional.interpolate(down_sampled, scale_factor=self.scale, mode='bicubic')
        upscaled = self.forward(up_sampled, train=train)

        # get ground truth either from real SR or from teacher
        sr = self.get_sr(orig=batch, down_sampled=down_sampled)

        # return loss on real SR
        return self.get_losses(upscaled, sr)

    def get_losses(self, x, sr):

        # loss on downsampled images to reduce artifacts
        x_ds = torch.nn.functional.interpolate(x, scale_factor=1 / self.scale, mode='bicubic')
        sr_ds = torch.nn.functional.interpolate(sr, scale_factor=1 / self.scale, mode='bicubic')
        aux_loss = torch.nn.functional.l1_loss(x_ds, sr_ds)

        # loss on high frequency details compared with gaussian blurred
        x_details = x - torchvision.transforms.functional.gaussian_blur(x, kernel_size=(5, 5))
        sr_details = sr - torchvision.transforms.functional.gaussian_blur(sr, kernel_size=(5, 5))
        high_freq_loss = torch.nn.functional.l1_loss(x_details, sr_details)

        # calculate common metrics
        x_normalized = torch_convert_YUV2RGB((x * 0.5 + 0.5).clip(0, 1))
        sr_normalized = torch_convert_YUV2RGB((sr * 0.5 + 0.5).clip(0, 1))
        psnr = torchmetrics.functional.peak_signal_noise_ratio(x_normalized, sr_normalized, data_range=1.0)
        ssim = torchmetrics.functional.structural_similarity_index_measure(
            x_normalized, sr_normalized, data_range=1.0)

        return {
            'aux_loss': aux_loss,
            'high_freq': high_freq_loss,
            'psnr': psnr,
            'ssim': ssim,
            'loss': aux_loss + high_freq_loss
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
            upsampled_tiles = torch.nn.functional.interpolate(__tiles, scale_factor=self.scale, mode='bicubic')
            upscaled.append(self.forward(upsampled_tiles.to(self.device), train=False).cpu())

            del __tiles, upsampled_tiles
            gc.collect()
            torch.cuda.empty_cache()

        upscaled = torch.cat(upscaled, dim=0)
        upscaled = rearrange(upscaled, '(b n m) c p1 p2 -> b c (n p1) (m p2)', n=h // self.tile, m=w // self.tile)

        return upscaled

    def on_validation_epoch_end(self):

        # get original SR images and downsample them
        orig_images = torch.stack([next(self.test_images) for _ in range(self.samples_epoch)], dim=0)
        down_sampled = torch.nn.functional.interpolate(orig_images, scale_factor=1 / self.scale, mode='bicubic')

        # upscale images with model
        upscaled_images = self.upscale_images(down_sampled)

        # log images
        self.custom_logger.log_images_compare([tensor2list(to_image(orig_images)),
                                               tensor2list(to_image(upscaled_images))],
                                              texts=['orig', 'upscaled'],
                                              name='orig_upscaled', epoch=self.current_epoch)

        # log current lr
        if self.debug:
            self.log('learning_rate', self.lr_schedulers().get_last_lr()[0], prog_bar=True)

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        for k, v in loss.items():
            self.log(f'train_{k}', v, prog_bar=True, sync_dist=True)

        return loss['loss']

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.use_ema:
            self.ema_model.update(self.model)

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, train=False)
        for k, v in loss.items():
            self.log(f'val_{k}', v, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(lr=self.learning_rate, params=self.model.parameters(), betas=(0.9, 0.99))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.learning_rate,
                                                        pct_start=3 / self.epochs,
                                                        div_factor=1 / self.initial_lr_rate,
                                                        final_div_factor=self.initial_lr_rate / self.min_lr_rate,
                                                        epochs=self.epochs, steps_per_epoch=self.steps)
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step'
        }
        return [optimizer], [scheduler]

    def train_dataloader(self):
        self.dataset.reset_cache()
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False,
                                           num_workers=2 * torch.cuda.device_count(),
                                           pin_memory=True, prefetch_factor=2)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                                           num_workers=2 * torch.cuda.device_count(),
                                           pin_memory=True, prefetch_factor=2)


def get_parser():
    parser = ArgumentParser(description="Training patch upscaler model")
    # Input data settings
    parser.add_argument("--dataset", default=[], help="Path to folders with SR images")
    parser.add_argument("--sample", default=None, help="Path to checkpoint model")
    parser.add_argument("--tmp", default="tmp", help="temporary directory for logs etc")
    parser.add_argument("--teacher", default="Swin2SR_Lightweight_X2_64", help="name of SwinSR checkpoint")
    parser.add_argument("--cache_size", default=512, type=int, help="Cache images for each epoch")

    # Training settings
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate for diffusion")
    parser.add_argument("--ema", default=0.995, type=float, help="Ema weight")
    parser.add_argument("--teacher_rate", default=1.0, type=float, help="How much of teacher image to use for training")
    parser.add_argument("--min_lr_rate", default=0.01, type=float, help="Minimal LR ratio to decay")
    parser.add_argument("--initial_lr_rate", default=0.1, type=float, help="Initial LR ratio")
    parser.add_argument("--epochs", default=300, type=int, help="Epochs in training")
    parser.add_argument("--steps", default=1000, type=int, help="Steps in training")
    parser.add_argument("--batch_size", default=512, type=int, help="Batch size in training")
    parser.add_argument("--acc_grads", default=1, type=int,
                        help="Steps to accumulate gradients to emulate larger batch size")
    parser.add_argument("--samples_epoch", default=5, type=int, help="Samples of generator in one epoch")
    parser.add_argument("--w", default=2560, type=int, help="SR width")
    parser.add_argument("--h", default=1440, type=int, help="SR height")
    parser.add_argument("--tile", default=16, type=int, help="Tile size after downscale")
    parser.add_argument("--tile_pad", default=1, type=int, help="Tile overlap after downscale")
    parser.add_argument("--scale", default=2, type=int, help="How much down scale SR patches")

    # Model settings
    parser.add_argument("--dim", default=32, type=int, help="Base channel")
    parser.add_argument("--in_channels", default=1, type=int, choices=[1, 3], help="1 or 3 use only Luma or Cb Cr too")
    parser.add_argument("--n_blocks", default=4, type=int, help="Feature extraction blocks")
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

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=os.path.dirname(args.out_model_name),
                                                       filename=os.path.basename(args.out_model_name))

    dataset = SRTiles(folder=args.dataset, tile=args.tile, tile_pad=args.tile_pad, scale=args.scale,
                      cache_size=args.cache_size)
    images = SRImages(folder=args.dataset, w=args.w, h=args.h)

    # load pretrained upscaler
    teacher = None
    if args.teacher:
        model_path = f'{args.tmp}/{args.teacher}.pth'

        if not os.path.exists(model_path):
            url = f'https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/{args.teacher}.pth'
            r = requests.get(url, allow_redirects=True)
            open(model_path, 'wb').write(r.content)

        model = Swin2SR(upscale=2, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                        mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
        sd = torch.load(model_path, map_location='cpu')
        model.load_state_dict(sd, strict=True)
        model.eval()
        teacher = model

    model = PatchEnhancer(in_channels=args.in_channels, dim=args.dim, n_blocks=args.n_blocks,
                          tile_pad=args.tile_pad * args.scale)

    devices = list(range(torch.cuda.device_count()))
    if args.sample:
        enhancer = ImageEnhancer.load_from_checkpoint(args.sample, model=model, dataset=dataset, test_images=images,
                                                      clearml=logger, teacher=teacher,
                                                      teacher_rate=args.teacher_rate, ema_weight=args.ema,
                                                      learning_rate=args.lr,
                                                      initial_lr_rate=args.initial_lr_rate,
                                                      min_lr_rate=args.min_lr_rate,
                                                      batch_size=args.batch_size, epochs=args.epochs, steps=args.steps,
                                                      samples_epoch=args.samples_epoch, scale=args.scale,
                                                      tile=args.tile, tile_pad=args.tile_pad)
        enhancer.to(f'cuda:{devices[0]}')
        orig_images = torch.stack(images.load_images(), dim=0)
        lr_images = torch.nn.functional.interpolate(orig_images, scale_factor=1 / args.scale, mode='bicubic')
        sr_images = to_image(enhancer.upscale_images(lr_images))

        enhancer.custom_logger.log_images_compare(images=[
            tensor2list(to_image(orig_images)), tensor2list(to_image(sr_images))], texts=['orig', 'upscaled'], epoch=0)
    else:
        enhancer = ImageEnhancer(model=model, dataset=dataset, test_images=images, clearml=logger, teacher=teacher,
                                 teacher_rate=args.teacher_rate, ema_weight=args.ema, learning_rate=args.lr,
                                 initial_lr_rate=args.initial_lr_rate, min_lr_rate=args.min_lr_rate,
                                 batch_size=args.batch_size, epochs=args.epochs, steps=args.steps,
                                 samples_epoch=args.samples_epoch, scale=args.scale,
                                 tile=args.tile, tile_pad=args.tile_pad)
        trainer = Trainer(max_epochs=args.epochs, limit_train_batches=args.steps, limit_val_batches=10,
                          enable_model_summary=True, enable_progress_bar=True, enable_checkpointing=True,
                          strategy=DDPStrategy(find_unused_parameters=False), precision=16,
                          profiler=args.profile,
                          accumulate_grad_batches=args.acc_grads,
                          accelerator='gpu', devices=devices, callbacks=[checkpoint_callback])
        trainer.fit(enhancer)
