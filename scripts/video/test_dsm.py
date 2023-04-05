import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
from einops import repeat, rearrange
import cv2
import imageio
from diffusers import AutoencoderKL
import gc
import os
import random
from random import choice

from argparse import ArgumentParser

import clearml

from modules.common.util import *
from modules.common.trainer import *
from modules.common.model import *


def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


class Identity(torch.nn.Identity):

    def forward(self, input, *args, **kwargs):
        return super().forward(input)


class ConditionalNorm(torch.nn.Module):

    def __init__(self, dim, emb_dim, num_groups=32):
        super(ConditionalNorm, self).__init__()
        self.norm = norm(dim, num_groups)
        self.layer = torch.nn.Conv1d(emb_dim, dim * 2, kernel_size=1)

    def forward(self, h, emb):
        # emb (b c n)
        emb = self.layer(nonlinear(emb))
        emb = add_last_dims(emb, h)
        gamma, beta = torch.chunk(emb, 2, dim=1)  # split in channel dimension
        return self.norm(h) * (1.0 + gamma) + beta


def DownSample(in_dim, out_dim):
    return torch.nn.Conv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=(1, 4, 4),
                           padding=(0, 1, 1), stride=(1, 2, 2))


def UpSample(in_dim, out_dim):
    return torch.nn.ConvTranspose3d(in_channels=in_dim, out_channels=out_dim, kernel_size=(1, 4, 4),
                                    padding=(0, 1, 1), stride=(1, 2, 2))


class TemporalCondResBlock(torch.nn.Module):
    def __init__(self, hidden_dim, embed_dim, kernel_size=3, num_groups=32, in_dim=-1, dropout=0.0):
        super(TemporalCondResBlock, self).__init__()

        if in_dim == -1:
            in_dim = hidden_dim
        else:
            self.res_mapper = torch.nn.Conv3d(in_dim, hidden_dim, kernel_size=(1, kernel_size, kernel_size),
                                              padding=(0, kernel_size // 2, kernel_size // 2))
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.ln_1 = norm(in_dim, num_groups)
        self.layer_1 = torch.nn.Conv3d(in_dim, hidden_dim, kernel_size=(1, kernel_size, kernel_size),
                                       padding=(0, kernel_size // 2, kernel_size // 2))
        self.time_layer_1 = torch.nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(kernel_size, 1, 1),
                                            padding=(kernel_size // 2, 0, 0))
        self.conditional_norm = ConditionalNorm(hidden_dim, embed_dim, num_groups)
        self.dropout = torch.nn.Dropout3d(p=dropout)
        self.layer_2 = torch.nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(1, kernel_size, kernel_size),
                                       padding=(0, kernel_size // 2, kernel_size // 2))
        self.time_layer_2 = torch.nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(kernel_size, 1, 1),
                                            padding=(kernel_size // 2, 0, 0))

    def forward(self, x, emb, time_mask=None):

        if not exists(time_mask):
            time_mask = torch.zeros((len(x),)).type_as(x).bool()
        time_mask = add_last_dims(time_mask, x)

        h = self.layer_1(nonlinear(self.ln_1(x)))
        h = torch.where(time_mask, h, h + self.time_layer_1(h))

        h = self.conditional_norm(h, emb)
        h = self.layer_2(self.dropout(nonlinear(h)))
        h = torch.where(time_mask, h, h + self.time_layer_2(h))

        skip = x if self.in_dim == self.hidden_dim else self.res_mapper(x)
        return (h + skip) / 2 ** (1 / 2)


class TemporalAttn(torch.nn.Module):

    def __init__(self, dim, head_dim=64):
        super(TemporalAttn, self).__init__()
        self.attn = EinopsToAndFrom('b c n h w', '(b h w) n c',
                                    Attention(input_dim=dim, embed_dim=dim, head_dim=head_dim))

    def forward(self, x, time_mask=None):
        if not exists(time_mask):
            time_mask = torch.zeros((len(x),)).type_as(x).bool()
        time_mask = add_last_dims(time_mask, x)
        x = torch.where(time_mask, x, self.attn(x))
        return x


def spatial_attn(dim, head_dim=64):
    return EinopsToAndFrom('b c n h w', '(b n) (h w) c', Attention(input_dim=dim, embed_dim=dim, head_dim=head_dim))


def cross_spatial_attn(dim, head_dim=64):
    return EinopsToAndFrom('b c n h w', '(b n) (h w) c', Attention(input_dim=dim, embed_dim=dim, head_dim=head_dim),
                           dup='v')


def linear_attn(dim, head_dim=64):
    return EinopsToAndFrom('b c n h w', '(b n) c (h w)',
                           LinearAttention(input_dim=dim, embed_dim=dim, head_dim=head_dim))


def linear_attn2d(dim, head_dim=64):
    return EinopsToAndFrom('b c h w', 'b c (h w)',
                           LinearAttention(input_dim=dim, embed_dim=dim, head_dim=head_dim))


def cross_linear_attn(dim, head_dim=64):
    return EinopsToAndFrom('b c n h w', '(b n) c (h w)',
                           LinearAttention(input_dim=dim, embed_dim=dim, head_dim=head_dim), dup='v')


class SparseCausalAttention(torch.nn.Module):

    def __init__(self, attn, dim):
        super(SparseCausalAttention, self).__init__()
        self.attn = attn
        self.reduce = torch.nn.Conv3d(2 * dim, dim, kernel_size=1)

    def forward(self, x, time_mask=None):
        if not exists(time_mask):
            time_mask = torch.zeros((len(x),)).type_as(x).bool()
        time_mask = add_last_dims(time_mask, x)

        b, c, f, h, w = x.shape
        kv1 = repeat_dim(x[:, :, :1], 2, f)  # first frame of each batch
        kv2 = torch.cat([x[:, :, :1], x[:, :, :-1]], dim=2)  # previous frame for each one
        kv = torch.cat([kv1, kv2], dim=1)
        kv = torch.where(time_mask, x, self.reduce(kv))
        return self.attn(x, v=kv)


class TemporalUNet(torch.nn.Module):

    def __init__(self,
                 channels=3,
                 kernel_size=3,
                 base_ch=192,
                 res_blocks=[3],
                 ch_mults=[1, 2, 3, 4],
                 embed_features=512,
                 attn_scale=2,
                 linear_attn_scale=2,
                 num_heads=4,
                 head_dim=64,
                 dropout=0.0,
                 num_groups=32,
                 steps=4000,
                 cond='concat'):
        super(TemporalUNet, self).__init__()
        self.channels = channels
        self.cond = cond
        self.steps = steps
        # Input mapping
        self.input_mapper = torch.nn.Conv3d(2 * channels if cond == 'concat' else channels, base_ch * ch_mults[0],
                                            kernel_size=(1, kernel_size, kernel_size),
                                            padding=(0, kernel_size // 2, kernel_size // 2))
        self.input_temporal = TemporalAttn(base_ch * ch_mults[0], head_dim)

        # Time embeddings
        self.emb_features = embed_features
        self.noise_layers = torch.nn.ModuleList([
            torch.nn.Linear(embed_features // 4, embed_features),
            torch.nn.Linear(embed_features, embed_features),
        ])

        # Frame embeddings
        self.frame_layers = torch.nn.ModuleList([
            torch.nn.Linear(embed_features // 4, embed_features),
            torch.nn.Linear(embed_features, embed_features),
        ])

        # Conditional image embeddings
        if cond == 'cross':
            self.cond_input = torch.nn.Conv2d(channels, base_ch * ch_mults[0], kernel_size=kernel_size,
                                              padding=kernel_size // 2)
            self.cond_layers = torch.nn.ModuleList([])
            self.cond_downsample = torch.nn.ModuleList([])
            current_scale = 1
            for ch_id, ch_mult in enumerate(ch_mults):
                dim = base_ch * ch_mult
                blocks = res_blocks[0] if len(res_blocks) == 1 else res_blocks[ch_id]
                self.cond_layers.append(torch.nn.ModuleList(
                    [ResBlock2d(hidden_dim=dim, num_groups=num_groups, kernel_size=kernel_size) for _ in range(blocks)]
                    + [linear_attn2d(dim, head_dim)]))
                self.cond_downsample.append(
                    DownSample2d(dim, base_ch * ch_mults[ch_id + 1] if ch_id < len(ch_mults) - 1 else None,
                                 kernel_size=kernel_size) if ch_id != len(ch_mults) - 1 else torch.nn.Identity())
                current_scale = current_scale * 2 if ch_id != len(ch_mults) - 1 else current_scale

        # prepare encoder containers
        self.encoder = torch.nn.ModuleList([])
        self.downs = torch.nn.ModuleList([])
        self.encoder_spatial = torch.nn.ModuleList([])
        self.encoder_temporal = torch.nn.ModuleList([])
        self.encoder_cross = torch.nn.ModuleList([])

        current_scale = 1
        for ch_id, ch_mult in enumerate(ch_mults):
            dim = base_ch * ch_mult
            blocks = res_blocks[0] if len(res_blocks) == 1 else res_blocks[ch_id]
            self.encoder.append(torch.nn.ModuleList([
                TemporalCondResBlock(dim, embed_dim=self.emb_features, kernel_size=kernel_size,
                                     num_groups=num_groups, dropout=dropout) for _ in range(blocks)]))
            self.encoder_spatial.append(cases([
                (current_scale >= attn_scale, spatial_attn(dim, head_dim)),
                (current_scale >= linear_attn_scale, linear_attn(dim, head_dim)),
                torch.nn.Identity
            ]))
            self.encoder_cross.append(cases([
                (self.cond == 'cross' and current_scale >= attn_scale, cross_spatial_attn(dim, head_dim)),
                (self.cond == 'cross' and current_scale >= linear_attn_scale, cross_linear_attn(dim, head_dim)),
                Identity()
            ]))
            self.encoder_temporal.append(TemporalAttn(dim, head_dim))
            self.downs.append(DownSample(dim, base_ch * ch_mults[ch_id + 1] if ch_id < len(ch_mults) - 1 else None)
                              if ch_id < len(ch_mults) - 1 else torch.nn.Identity())
            current_scale = current_scale * 2 if ch_id != len(ch_mults) - 1 else current_scale

        # mid-block construction
        last_dim = base_ch * ch_mults[-1]
        self.mid_layers = torch.nn.Module()
        self.mid_layers.block_1 = TemporalCondResBlock(hidden_dim=last_dim, embed_dim=self.emb_features,
                                                       num_groups=num_groups, kernel_size=kernel_size, dropout=dropout)
        self.mid_layers.spatial = cases([
            (current_scale >= attn_scale, spatial_attn(last_dim, head_dim)),
            (current_scale >= linear_attn_scale, linear_attn(last_dim, head_dim)),
            torch.nn.Identity()
        ])
        self.mid_layers.cross = cases([
            (self.cond == 'cross' and current_scale >= attn_scale, cross_spatial_attn(last_dim, head_dim)),
            (self.cond == 'cross' and current_scale >= linear_attn_scale, cross_linear_attn(last_dim, head_dim)),
            Identity()
        ])
        self.mid_layers.temporal = TemporalAttn(last_dim, head_dim)
        self.mid_layers.block_2 = TemporalCondResBlock(hidden_dim=last_dim, embed_dim=self.emb_features,
                                                       num_groups=num_groups, kernel_size=kernel_size, dropout=dropout)

        # prepare decoder containers
        self.decoder = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        self.decoder_spatial = torch.nn.ModuleList([])
        self.decoder_temporal = torch.nn.ModuleList([])
        self.decoder_cross = torch.nn.ModuleList([])

        for ch_id, ch_mult in enumerate(reversed(ch_mults)):
            dim = base_ch * ch_mult
            blocks = res_blocks[0] if len(res_blocks) == 1 else res_blocks[- (ch_id + 1)]
            self.decoder.append(torch.nn.ModuleList([
                TemporalCondResBlock(dim, in_dim=2 * dim, embed_dim=self.emb_features,
                                     kernel_size=kernel_size, num_groups=num_groups,
                                     dropout=dropout) for _ in range(blocks)]))
            self.decoder_spatial.append(cases([
                (current_scale >= attn_scale, spatial_attn(dim, head_dim)),
                (current_scale >= linear_attn_scale, linear_attn(dim, head_dim)),
                torch.nn.Identity
            ]))
            self.decoder_cross.append(cases([
                (self.cond == 'cross' and current_scale >= attn_scale, cross_spatial_attn(dim, head_dim)),
                (self.cond == 'cross' and current_scale >= linear_attn_scale, cross_linear_attn(dim, head_dim)),
                Identity()
            ]))
            self.decoder_temporal.append(TemporalAttn(dim, head_dim))
            self.ups.append(UpSample(dim, base_ch * ch_mults[-(ch_id + 2)] if ch_id < len(ch_mults) - 1 else None)
                            if ch_id != len(ch_mults) - 1 else torch.nn.Identity())
            current_scale = current_scale // 2 if ch_id != len(ch_mults) - 1 else current_scale

        # Out latent prediction
        self.out_norm = norm(base_ch, num_groups=num_groups)
        self.out_mapper = torch.nn.Conv3d(base_ch, channels, kernel_size=(1, kernel_size, kernel_size),
                                          padding=(0, kernel_size // 2, kernel_size // 2))

    def forward(self, x, sigma, cond=None, time_mask=None):
        # funny checkpoint loading problem
        if type(x) is not torch.Tensor:
            return x
        # expand time mask to latents dim
        if not exists(time_mask):
            time_mask = torch.zeros((len(x),)).type_as(x).bool()
        time_mask = add_last_dims(time_mask, x)

        b, in_channels, temporal_dim, height, width = x.shape

        # concat to each frame condition
        x = torch.cat([repeat_dim(cond.unsqueeze(2), 2, temporal_dim), x], dim=1) if self.cond == 'concat' else x

        # map time to embeddings
        h_noise = get_timestep_encoding(sigma, self.emb_features // 4, self.steps)
        h_noise = self.noise_layers[1](nonlinear(self.noise_layers[0](h_noise)))

        # map frame embedding
        h_temporal = get_timestep_encoding(torch.arange(temporal_dim).type_as(sigma), self.emb_features // 4,
                                           temporal_dim)
        h_temporal = self.frame_layers[1](nonlinear(self.frame_layers[0](h_temporal))).transpose(0, 1)
        h_temporal = torch.where(time_mask[:, :, :, 0, 0], torch.zeros_like(h_temporal[None, :, :]),
                                 h_temporal[None, :, :])
        h_time = h_noise[:, :, None] + h_temporal

        # map input image to latent space
        conds = []
        if self.cond == 'cross':
            # Map condition image to latents
            h_cond = self.cond_input(cond)
            for blocks, ds in zip(self.cond_layers, self.cond_downsample):
                for block in blocks:
                    h_cond = block(h_cond)
                conds.append(h_cond)
                h_cond = ds(h_cond)

        # Prepare input for mapping
        h = self.input_mapper(x)
        h = self.input_temporal(h, time_mask=time_mask)

        # encode latents
        outs = []
        for layer_id, (blocks, ds, spatial, cross, temporal) in enumerate(
                zip(self.encoder, self.downs, self.encoder_spatial, self.encoder_cross, self.encoder_temporal)):
            h_cond = rearrange(repeat(conds[layer_id], 'b c h w -> b n c h w', n=temporal_dim),
                               'b n c h w -> b c n h w') if self.cond == 'cross' else None
            for block_id, block in enumerate(blocks):
                h = block(h, h_time, time_mask=time_mask)
                if block_id == len(blocks) - 1:
                    h = spatial(h, time_mask=time_mask)
                    h = cross(h, v=h_cond)
                    h = temporal(h, time_mask=time_mask)
                outs.append(h)
            h = ds(h)

        # mid layer
        h_cond = rearrange(repeat(conds[-1], 'b c h w -> b n c h w', n=temporal_dim),
                           'b n c h w -> b c n h w') if self.cond == 'cross' else None
        h = self.mid_layers.block_1(h, h_time, time_mask=time_mask)
        h = self.mid_layers.spatial(h, time_mask=time_mask)
        h = self.mid_layers.cross(h, v=h_cond)
        h = self.mid_layers.temporal(h, time_mask=time_mask)
        h = self.mid_layers.block_2(h, h_time, time_mask=time_mask)

        # decode latent
        for layer_id, (blocks, up, spatial, cross, temporal) in enumerate(
                zip(self.decoder, self.ups, self.decoder_spatial, self.decoder_cross, self.decoder_temporal)):
            h_cond = rearrange(repeat(conds[-(layer_id + 1)], 'b c h w -> b n c h w', n=temporal_dim),
                               'b n c h w -> b c n h w') if self.cond == 'cross' else None
            for block_id, block in enumerate(blocks):
                h = block(torch.cat([h, outs.pop()], dim=1), h_time, time_mask=time_mask)
                if block_id == len(blocks) - 1:
                    h = spatial(h, time_mask=time_mask)
                    h = cross(h, v=h_cond)
                    h = temporal(h, time_mask=time_mask)
            h = up(h)

        out = self.out_mapper(self.out_norm(h))

        # take only first channels in case of concat conditioning
        out = out[:, :in_channels]

        return out


class LandscapeAnimation(torch.utils.data.IterableDataset):

    def __init__(self, folder, w=512, h=256, num_frames=8, step=256):
        super(LandscapeAnimation, self).__init__()
        self.w, self.h = w, h
        self.frame_ratio = w / h
        self.frames = num_frames
        self.step = step
        self.folder = folder
        self.files = [os.path.join(self.folder, file) for file in os.listdir(self.folder) if
                      os.path.splitext(file)[1] == '.mp4']

    def __iter__(self):
        return self

    def __next__(self):
        # update list in case of new files
        self.files = [os.path.join(self.folder, file) for file in os.listdir(self.folder) if
                      os.path.splitext(file)[1] == '.mp4']

        # load videos until return valid video :)
        while True:
            try:
                while True:
                    file = choice(self.files)

                    video = cv2.VideoCapture(file)
                    if not video.isOpened():
                        video.release()
                        continue

                    video_fps = int(video.get(cv2.CAP_PROP_FPS))
                    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    frame_step = int(self.step / 1000 * video_fps)
                    frames_used = frame_step * self.frames
                    if frames_used > total_frames:
                        video.release()
                        continue

                    break

                frame_id = random.randint(0, total_frames - frames_used - 1)
                frames = []
                while len(frames) < self.frames:
                    video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                    frame_id += frame_step

                    ret, frame = video.read()
                    # try another video on fail
                    if not ret:
                        video.release()
                        return self.__next__()

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w = frame.shape[:2]
                    frame_ratio = w / h
                    if frame_ratio > self.frame_ratio:
                        # width is bigger so let's crop it
                        true_w = int(h * self.frame_ratio)
                        start_w = int((w - true_w) / 2)
                        frame = frame[:, start_w: start_w + true_w]
                    else:
                        # height is bigger
                        true_h = int(w / self.frame_ratio)
                        start_h = int((h - true_h) / 2)
                        frame = frame[start_h: start_h + true_h]
                    frame = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_AREA)
                    frame = normalize_image(frame)
                    frames.append(np.moveaxis(frame, -1, 0))
                video.release()

                # stack frames in (c f h w) order
                frames = torch.tensor(np.stack(frames, axis=1), dtype=torch.float32)
                return frames
            except Exception as e:
                print(e)


class LandscapeImages(torch.utils.data.IterableDataset):

    def __init__(self, folder, w=512, h=256):
        super(LandscapeImages, self).__init__()
        self.w, self.h = w, h
        self.frame_ratio = w / h
        self.folder = folder
        self.files = [os.path.join(self.folder, file) for file in os.listdir(self.folder) if
                      os.path.splitext(file)[1] in ('.png', '.jpg', '.jpeg')]

    def load_file(self, file):
        frame = cv2.imread(file)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w = frame.shape[:2]
        frame_ratio = w / h
        if frame_ratio > self.frame_ratio:
            # width is bigger so let's crop it
            true_w = int(h * self.frame_ratio)
            start_w = int((w - true_w) / 2)
            frame = frame[:, start_w: start_w + true_w]
        else:
            # height is bigger
            true_h = int(w / self.frame_ratio)
            start_h = int((h - true_h) / 2)
            frame = frame[start_h: start_h + true_h]
        frame = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_AREA)
        frame = normalize_image(frame)
        return torch.tensor(frame).movedim(-1, 0)

    def load_images(self):
        return [self.load_file(file) for file in self.files]

    def __iter__(self):
        return self

    def __next__(self):
        # update list in case of new files
        self.files = [os.path.join(self.folder, file) for file in os.listdir(self.folder) if
                      os.path.splitext(file)[1] in ('.png', '.jpg', '.jpeg')]

        file = choice(self.files)
        return self.load_file(file)


class LandscapeLatents(torch.utils.data.IterableDataset):

    def __init__(self, folder, num_frames=1 + 8, step=4, scale_delta=0.05):  # each step is 64 ms
        super(LandscapeLatents, self).__init__()
        self.frames = num_frames
        self.step = step
        self.folder = folder
        self.scale_delta = scale_delta
        self.files = [os.path.join(self.folder, file) for file in os.listdir(self.folder) if
                      os.path.splitext(file)[1] == '.lt']

    def __iter__(self):
        return self

    def __next__(self):
        self.files = [os.path.join(self.folder, file) for file in os.listdir(self.folder) if
                      os.path.splitext(file)[1] == '.lt']
        while True:
            try:

                file = choice(self.files)
                tensor = torch.load(file)
                if len(tensor) < self.frames * self.step:
                    del tensor
                    continue

                idx = random.randint(0, len(tensor) - self.frames * self.step)
                frames = []
                for _ in range(self.frames):
                    frames.append(tensor[idx])
                    idx += self.step

                # c n h w
                frames = torch.stack(frames, dim=1).to(torch.float32)
                frames *= (1 - torch.rand(frames.shape[0]) * self.scale_delta)[:, None, None, None]
                return frames

            except Exception as e:
                print(e)
                continue


class AnimationDiffusion(pl.LightningModule):

    def __init__(self,
                 model=None,
                 dataset=None,
                 images=None,
                 data_latent=False,
                 tmpdir='tmp',
                 test_dataset=None,
                 clearml=None,
                 vae=None,
                 vae_device=0,
                 vae_scale=0.18215,
                 channels=3,
                 w=512,
                 h=256,
                 num_frames=7 + 1,
                 gap=256,
                 ema_weight=0.995,
                 learning_rate=1e-4,
                 batch_size=16,
                 batch_vae=4,
                 min_lr_rate=1.0,
                 initial_lr_rate=0.1,
                 diffusion_steps=4000,
                 sample_steps=64,
                 samples_epoch=5,
                 steps=1000,
                 epochs=300,
                 clip_denoised=False,
                 base_noise=0.5,
                 sigma_min=0.002,
                 sigma_max=80.0,
                 logsigma_mean=-1.2,
                 logsigma_std=1.2,
                 sigma_data=0.5,
                 sample_noise=1.003,
                 sample_sigma_min=0.05,
                 sample_sigma_max=50,
                 sample_stochasticity=40 / 256,
                 clf_free=0.1,
                 clf_weight=3.0,
                 debug=True):
        super(AnimationDiffusion, self).__init__()

        self.save_hyperparameters(
            ignore=['model', 'dataset', 'images', 'test_dataset', 'clearml', 'tmpdir', 'vae', 'vae_device'])

        self.h = h
        self.w = w
        self.channels = channels
        self.num_frames = num_frames
        self.gap = gap

        self.dataset = dataset
        self.test_dataset = test_dataset if exists(test_dataset) else dataset
        self.images = iter(images) if images else None
        self.tmpdir = tmpdir
        self.data_latent = data_latent

        self.model = model
        self.use_ema = ema_weight is not None
        if self.use_ema:
            self.ema_model = EMA(self.model, decay=ema_weight)

        self.vae = [vae] if exists(vae) else None
        if exists(vae):
            self.vae_device = torch.device(f'cuda:{vae_device}')
            self.vae_scale = vae_scale

        self.learning_rate = learning_rate
        self.min_lr_rate = min_lr_rate
        self.initial_lr_rate = initial_lr_rate
        self.batch_size = batch_size
        self.steps = steps
        self.epochs = epochs
        self.samples_epoch = samples_epoch
        self.clf_free = clf_free
        self.clf_weight = clf_weight

        self.diffusion_steps = diffusion_steps
        self.sample_steps = sample_steps
        self.clip_denoised = clip_denoised
        self.base_noise = base_noise

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.logsigma_mean = logsigma_mean
        self.logsigma_std = logsigma_std
        self.sigma_data = sigma_data if not exists(vae) else 1.0

        self.batch_vae = batch_vae
        self.sample_noise = sample_noise
        self.sample_sigma_min = sample_sigma_min
        self.sample_sigma_max = sample_sigma_max
        self.sample_stochasticity = sample_stochasticity

        self.custom_logger = SimpleLogger(clearml=clearml, tmpdir=tmpdir)
        self.debug = debug

    def sample(self, num_samples, latents=None, rho=7, cond=None, clf_weight=None, **kwargs):

        if latents is None:
            latents = torch.randn(num_samples, self.channels, self.num_frames, self.h, self.w, device=self.device)

        sigma_min, sigma_max = self.sigma_min, self.sigma_max
        num_steps = self.sample_steps

        # discretize time
        step_indices = torch.arange(num_steps).type_as(latents).to(torch.float64)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

        # stochastic sampling
        x_next = latents.to(torch.float64) * t_steps[0]
        ones = torch.ones(num_samples).type_as(latents)
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1

            # increase noise
            noise_ratio = min(self.sample_stochasticity, (2 ** 0.5) - 1) \
                if self.sample_sigma_min <= t_cur <= self.sample_sigma_max else 0
            t_hat = t_cur + noise_ratio * t_cur
            x_noised = x_next + (t_hat ** 2 - t_cur ** 2).sqrt() * self.get_noise(x_next, self.sample_noise)

            # euler ode solver
            x_denoised = self.forward(x_noised, t_hat * ones, train=False, cond=cond, **kwargs).to(torch.float64)
            d_cur = (x_denoised - x_noised) / t_hat

            # correct score with classifier free guidance
            if clf_weight:
                x_uncond = self.forward(x_noised, t_hat * ones, train=False, cond=self.get_noisy_cond(cond),
                                        **kwargs).to(torch.float64)
                d_uncond = (x_uncond - x_noised) / t_hat
                d_cur = d_cur * (1 + clf_weight) - d_uncond * clf_weight
            x_tmp = x_noised + (t_hat - t_next) * d_cur

            # heun ode correction
            if i < num_steps - 1:
                x_denoised = self.forward(x_tmp, t_next * ones, train=False, cond=cond, **kwargs).to(torch.float64)
                d_next = (x_denoised - x_tmp) / t_next
                # correct score with classifier free guidance
                if clf_weight:
                    x_uncond = self.forward(x_tmp, t_next * ones, train=False, cond=self.get_noisy_cond(cond),
                                            **kwargs).to(torch.float64)
                    d_uncond = (x_uncond - x_tmp) / t_next
                    d_next = d_next * (1 + clf_weight) - d_uncond * clf_weight
                x_next = x_noised + (t_hat - t_next) * (0.5 * d_cur + 0.5 * d_next)

        return x_next

    def forward(self, x, sigma, train=True, **kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32)

        model = cases([
            (self.use_ema and (not train or not self.training), self.ema_model.module),
            self.model
        ])

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        x_correction = model(add_last_dims(c_in, x) * x, c_noise, **kwargs)
        x_denoised = add_last_dims(c_skip, x) * x + add_last_dims(c_out, x) * x_correction

        return x_denoised

    def get_noisy_cond(self, cond, sigma=None):
        if sigma is None:
            sigma = self.sigma_max
        scale = 1 / (self.sigma_data ** 2 + sigma ** 2) ** 0.5
        cond = (cond + torch.randn_like(cond) * sigma) * scale
        return cond

    def get_noise(self, x, sigma):

        base_noise = torch.randn_like(x[:, :, 0])
        base_noise = repeat(base_noise, 'b c h w -> b n c h w', n=x.shape[2])
        base_noise = rearrange(base_noise, 'b n c h w -> b c n h w')

        res_noise = torch.randn_like(x)
        noise = base_noise * self.base_noise ** 0.5 + res_noise * (1 - self.base_noise) ** 0.5

        return noise * sigma

    def step(self, batch, train=True):

        # apply vae if needed
        batch = self.encode_data(batch) if not self.data_latent else batch * self.vae_scale

        x = batch[:, :, 1:]
        cond = batch[:, :, 0]

        # sample noise schedule
        sigma = (torch.randn(len(x), ) * self.logsigma_std + self.logsigma_mean).exp().type_as(x)

        # add small noise to cond frame for augmentation
        if train:
            cond_sigma = torch.rand(len(cond)).type_as(cond) * np.exp(self.logsigma_mean - 2 * self.logsigma_std)
            cond = self.get_noisy_cond(cond, add_last_dims(cond_sigma, cond))

        # apply classifier free guidance
        if train and self.clf_free > 0:
            mask = torch.rand(len(cond), ).type_as(cond) < self.clf_free
            mask = add_last_dims(mask, cond)
            cond = torch.where(mask, self.get_noisy_cond(cond), cond)

        # noise input and predict original
        x_denoised = self.forward(x + self.get_noise(x, add_last_dims(sigma, x)), sigma, cond=cond, train=train)

        return self.get_losses(x, x_denoised, add_last_dims(sigma, x))

    def get_losses(self, x, x_denoised, sigma):

        loss_weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        loss = loss_weight * (x_denoised - x) ** 2

        return {
            'loss': loss.mean()
        }

    def load_vae(self, move_vae=True):
        if move_vae:
            self.vae[0] = self.vae[0].to(self.vae_device)
            self.model = self.model.to('cpu')
            self.ema_model = self.ema_model.to('cpu')
        return self.vae[0]

    def unload_vae(self, move_vae=True):
        if move_vae:
            self.vae[0] = self.vae[0].to('cpu')
            self.model = self.model.to(self.device)
            self.ema_model = self.ema_model.to(self.device)
        return self.vae[0]

    def encode_data(self, batch, move_vae=True):

        if self.vae:

            vae = self.load_vae(move_vae)
            mini_batches = []
            for tensor in torch.split(batch, self.batch_vae, dim=0):

                tensor = tensor.to(device=self.vae_device, dtype=torch.float16)
                if len(batch.shape) == 5:
                    tensor = rearrange(tensor, 'b c n h w -> (b n) c h w')

                with torch.no_grad():
                    tensor = vae.encode(tensor).latent_dist.mode()

                if len(batch.shape) == 5:
                    tensor = rearrange(tensor, '(b n) c h w -> b c n h w', n=batch.shape[2])
                mini_batches.append(tensor.to(device='cpu', dtype=torch.float32))
                # free mem
                del tensor
                gc.collect()
                torch.cuda.empty_cache()

            batch = torch.cat(mini_batches, dim=0)

            self.unload_vae(move_vae)

            batch = batch * self.vae_scale

        return batch

    def decode_data(self, batch, move_vae=True):
        if self.vae:
            batch = batch / self.vae_scale
            vae = self.load_vae(move_vae)

            mini_batches = []
            for tensor in torch.split(batch, self.batch_vae, dim=0):

                tensor = tensor.to(device=self.vae_device, dtype=torch.float16)
                if len(batch.shape) == 5:
                    tensor = rearrange(tensor, 'b c n h w -> (b n) c h w')

                with torch.no_grad():
                    tensor = vae.decode(tensor).sample

                if len(batch.shape) == 5:
                    tensor = rearrange(tensor, '(b n) c h w -> b c n h w', n=batch.shape[2])
                mini_batches.append(tensor.to(device='cpu', dtype=torch.float32))
                # free mem
                del tensor
                gc.collect()
                torch.cuda.empty_cache()

            batch = torch.cat(mini_batches, dim=0)

            self.unload_vae(move_vae)

        return batch

    def sample_from_images(self, conds, latent=False):

        conds = self.encode_data(conds) if not latent else conds
        # sample latents based on conditional frames
        latents = []
        for cond in torch.split(conds, self.batch_size, dim=0):
            with torch.no_grad():
                x = self.sample(num_samples=len(cond), cond=cond.to(self.device),
                                clf_weight=self.clf_weight).to(device='cpu', dtype=torch.float32)
            latents.append(torch.cat([cond.unsqueeze(2).to('cpu'), x], dim=2))
            del x, cond
        gc.collect()
        torch.cuda.empty_cache()
        # decode latents back to frames
        latents = torch.cat(latents, dim=0)
        videos = self.decode_data(latents)
        return rearrange(videos, 'b c n h w -> b n c h w')

    def on_validation_epoch_end(self):

        # animate predefined images
        if self.images:
            images = torch.stack([next(self.images) for _ in range(self.samples_epoch)], dim=0)
            animations = prepare_torch_images(self.sample_from_images(images, latent=False))
            for video_id in range(len(animations)):
                self.custom_logger.log_gif(tensor2list(animations[video_id]), self.gap,
                                           f'animation_{video_id}', epoch=self.current_epoch)

        data_iter = iter(self.trainer.val_dataloaders[0])

        # load conditional frames from data
        data = [next(data_iter) for _ in range(self.samples_epoch // self.batch_size
                                               + int(self.samples_epoch % self.batch_size != 0))]
        data = torch.cat(data, dim=0)

        cond = data[:, :, 0] if not self.data_latent else data[:, :, 0] * self.vae_scale
        videos = self.sample_from_images(cond, latent=self.data_latent)
        videos = prepare_torch_images(videos)

        # prepare train examples for log
        train_videos = self.decode_data(data * self.vae_scale) if self.data_latent else data
        train_videos = prepare_torch_images(rearrange(train_videos, 'b c n h w -> b n c h w'))

        # log all videos separately
        for video_id in range(len(train_videos)):
            self.custom_logger.log_gif(tensor2list(videos[video_id]), self.gap,
                                       f'sample_{video_id}', epoch=self.current_epoch)
            self.custom_logger.log_gif(tensor2list(train_videos[video_id]), self.gap,
                                       f'orig_{video_id}', epoch=self.current_epoch)

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
        # optimizer = Lion(lr=self.learning_rate, params=self.model.parameters(), betas=(0.95, 0.98))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.learning_rate,
                                                        pct_start=3 / self.epochs, div_factor=1 / self.initial_lr_rate,
                                                        final_div_factor=self.initial_lr_rate / self.min_lr_rate,
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
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                                           num_workers=2 * torch.cuda.device_count(),
                                           pin_memory=True, prefetch_factor=2)


def get_parser():
    parser = ArgumentParser(description="Training diffusion model")
    # Input data settings
    parser.add_argument("--dataset", default="", help="Path to folder with videos")
    parser.add_argument("--sample", default=None, help="Path to checkpoint model")
    parser.add_argument("--images", default=None, help="Path to debug images")
    parser.add_argument("--tmp", default="tmp", help="temporary directory for logs etc")
    parser.add_argument("--cond", default="concat", choices=['cross', 'concat'], help="Image condition type")
    parser.add_argument("--vae", default="stabilityai/stable-diffusion-2-1",
                        help="name of stability pipeline for autoencoder")
    parser.add_argument("--vae_device", default="same", choices=["same", "other"],
                        help="whether use pretrained model on the same device")

    # Training settings
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate for diffusion")
    parser.add_argument("--ema", default=0.995, type=float, help="Ema weight")
    parser.add_argument("--min_lr_rate", default=1.0, type=float, help="Minimal LR ratio to decay")
    parser.add_argument("--initial_lr_rate", default=0.1, type=float, help="Initial LR ratio")
    parser.add_argument("--epochs", default=300, type=int, help="Epochs in training")
    parser.add_argument("--steps", default=5000, type=int, help="Steps in training")
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size in training")
    parser.add_argument("--acc_grads", default=16, type=int,
                        help="Steps to accumulate gradients to emulate larger batch size")
    parser.add_argument("--samples_epoch", default=5, type=int, help="Samples of generator in one epoch")
    parser.add_argument("--frames", default=7, type=int, help="number of frames per batch to generate")
    parser.add_argument("--gap", default=256, type=int, help="gap between frames in ms")
    parser.add_argument("--w", default=512, type=int, help="frame width")
    parser.add_argument("--h", default=256, type=int, help="frame height ")

    # Model settings
    parser.add_argument("--ch_mults", default=[1, 2, 3, 4], nargs='+', type=int, help="Multipliers for base channel")
    parser.add_argument("--res_blocks", default=[3], nargs='+', type=int, help="blocks per channel")
    parser.add_argument("--base_ch", default=192, type=int, help="Base attention channel")
    parser.add_argument("--attention_dim", default=16, type=int, help="Width till the one attention would be done")
    parser.add_argument("--linear_attention_dim", default=64, type=int,
                        help="Width till the one attention would be done")
    parser.add_argument("--head_dim", default=64, type=int, help="Attention head dim")
    parser.add_argument("--embed_dim", default=512, type=int, help="Dim for temporal and time embedding")
    parser.add_argument("--diffusion_steps", default=4000, type=int, help="Steps to do diffusion")
    parser.add_argument("--sample_steps", default=64, type=int, help="Steps for sampling")
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout regularization for model")
    parser.add_argument("--clf_free", default=0.0, type=float, help="Classifier free guidance rate")
    parser.add_argument("--clf_weight", default=None, type=float, help="Classifier free guidance weight sampling")
    parser.add_argument("--base_noise", default=0.5, type=float, help="How much of residual noise to add")

    # Meta settings
    parser.add_argument("--out_model_name", default="landscape_diffusion", help="Name of output model path")
    parser.add_argument("--profile", default=None, choices=['simple', 'advanced'], help="Whether to use profiling")
    parser.add_argument("--task_name", default="Landscape diffusion", help="ClearML task name")
    parser.add_argument("--clearml", action='store_true')
    parser.set_defaults(clearml=False)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    if args.clearml:
        print("Initializing ClearML")
        task = clearml.Task.init(project_name='animation', task_name=args.task_name, reuse_last_task_id=True,
                                 auto_connect_frameworks=False)
        task.connect(args, name='config')
        logger = task.get_logger()
    else:
        logger = None

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=os.path.dirname(args.out_model_name),
                                                       filename=os.path.basename(args.out_model_name))

    # dataset = LandscapeAnimation(folder=args.dataset, w=args.w, h=args.h, num_frames=args.frames + 1, step=args.gap)
    dataset = LandscapeLatents(folder=args.dataset, num_frames=args.frames + 1, step=4) if not args.sample else None
    images = args.images
    if images or args.sample:
        images = LandscapeImages(folder=images, w=args.w, h=args.h)

    vae = AutoencoderKL.from_pretrained(args.vae, subfolder='vae', torch_dtype=torch.float16) if args.vae else None
    assert exists(vae) or args.vae_device == 'same' or torch.cuda.device_count() > 1, \
        'At least 2 device needed for in case of other vae device'
    devices = list(range(torch.cuda.device_count()))
    vae_device = devices[-1]
    if exists(vae) and args.vae_device == 'other':
        devices.pop()

    actual_h = args.h // 8 if exists(vae) else args.h
    actual_w = args.w // 8 if exists(vae) else args.w
    channels = 4 if exists(vae) else 3
    model = TemporalUNet(channels=channels, base_ch=args.base_ch, ch_mults=args.ch_mults, res_blocks=args.res_blocks,
                         head_dim=args.head_dim, dropout=args.dropout, steps=args.diffusion_steps,
                         embed_features=args.embed_dim, cond=args.cond,
                         attn_scale=actual_h // args.attention_dim,
                         linear_attn_scale=actual_h // args.linear_attention_dim)

    if args.sample:
        diffusion = AnimationDiffusion.load_from_checkpoint(args.sample, model=model, dataset=dataset, images=images,
                                                            tmpdir=args.tmp, clearml=logger, vae=vae,
                                                            vae_device=vae_device, h=actual_h, w=actual_w,
                                                            channels=channels,
                                                            num_frames=args.frames + 1, gap=args.gap, data_latent=True,
                                                            learning_rate=args.lr, min_lr_rate=args.min_lr_rate,
                                                            batch_size=args.batch_size,
                                                            epochs=args.epochs, steps=args.steps, ema_weight=args.ema,
                                                            clf_free=args.clf_free, clf_weight=args.clf_weight,
                                                            sample_steps=args.sample_steps,
                                                            samples_epoch=args.samples_epoch,
                                                            diffusion_steps=args.diffusion_steps,
                                                            base_noise=args.base_noise)
        diffusion.to(f'cuda:{devices[0]}')
        frames = torch.stack(images.load_images(), dim=0).to(diffusion.device)
        videos = prepare_torch_images(diffusion.sample_from_images(frames))

        # log all videos separately
        for video_id in range(len(videos)):
            diffusion.custom_logger.log_gif(tensor2list(videos[video_id]), args.gap,
                                            f'sample_{video_id}', epoch=0)
    else:
        diffusion = AnimationDiffusion(model=model, dataset=dataset, images=images, tmpdir=args.tmp, clearml=logger,
                                       vae=vae,
                                       vae_device=vae_device, h=actual_h, w=actual_w, channels=channels,
                                       num_frames=args.frames + 1, gap=args.gap, data_latent=True,
                                       learning_rate=args.lr, min_lr_rate=args.min_lr_rate, batch_size=args.batch_size,
                                       epochs=args.epochs, steps=args.steps, ema_weight=args.ema,
                                       clf_free=args.clf_free, clf_weight=args.clf_weight,
                                       sample_steps=args.sample_steps, samples_epoch=args.samples_epoch,
                                       diffusion_steps=args.diffusion_steps, base_noise=args.base_noise)
        trainer = Trainer(max_epochs=args.epochs, limit_train_batches=args.steps, limit_val_batches=10,
                          enable_model_summary=True, enable_progress_bar=True, enable_checkpointing=True,
                          strategy=DDPStrategy(find_unused_parameters=False), precision=16,
                          profiler=args.profile,
                          accumulate_grad_batches=args.acc_grads,
                          accelerator='gpu', devices=devices, callbacks=[checkpoint_callback])
        trainer.fit(diffusion)
