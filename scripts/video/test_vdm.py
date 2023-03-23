import copy
import gc
import math
import os
import random
from functools import partial
from pathlib import Path
from random import choice

import PIL
import clearml
import cv2
import imageio
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from PIL import ImageFile, Image
from diffusers import AutoencoderKL
from einops import rearrange, repeat
from einops_exts import check_shape, rearrange_many
from rotary_embedding_torch import RotaryEmbedding
from torch import nn, einsum
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam
from torch.utils import data
from torchvision import transforms as T, utils
from tqdm import tqdm
import lovely_tensors as lt

ImageFile.LOAD_TRUNCATED_IMAGES = True
PIL.Image.MAX_IMAGE_PIXELS = 93312000000

# VDM

def exists(x):
    return x is not None


def noop(*args, **kwargs):
    pass


def is_odd(n):
    return (n % 2) == 1


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cycle(dl):
    idl = iter(dl)
    while True:
        yield next(idl)


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    return groups, remainder


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])


# relative positional bias

class RelativePositionBias(nn.Module):
    def __init__(
            self,
            heads=8,
            num_buckets=32,
            max_distance=128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
                torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype=torch.long, device=device)
        k_pos = torch.arange(n, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets=self.num_buckets,
                                                   max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')


# small helper modules

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return (self.fn(x, *args, **kwargs) + x) / (2 ** 0.5)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def Upsample(dim):
    return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


def Downsample(dim):
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0.0):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding=(0, 1, 1))
        self.tproj = nn.Conv3d(dim_out, dim_out, (3, 1, 1), padding=(1, 0, 0))
        self.norm = nn.GroupNorm(groups, dim)
        self.act = nn.SiLU()
        self.dropout = torch.nn.Dropout3d(p=dropout)

    def forward(self, x, scale_shift=None, focus_present_mask=None):

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        else:
            x = self.norm(x)

        x = self.act(x)
        x = self.dropout(x)
        x = self.proj(x)
        x = torch.where(focus_present_mask[:, None, None, None, None], x, x + self.tproj(x))
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(time_emb_dim, dim * 2, kernel_size=1)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups, dropout=0.1)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, focus_present_mask=None):
        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c f -> b c f 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift, focus_present_mask=focus_present_mask)

        h = self.block2(h, focus_present_mask=focus_present_mask)
        return (h + self.res_conv(x)) / (2 ** 0.5)


class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = dim // dim_head
        hidden_dim = dim
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = rearrange_many(qkv, 'b (h c) x y -> b h c (x y)', h=self.heads)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        out = self.to_out(out)
        return rearrange(out, '(b f) c h w -> b c f h w', b=b)


# attention along space and time

class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        x = rearrange(x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            heads=4,
            dim_head=32,
            rotary_emb=None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = dim // dim_head
        hidden_dim = dim

        self.rotary_emb = rotary_emb
        self.to_q = nn.Linear(dim, hidden_dim, bias=False)
        self.to_k = nn.Linear(dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(dim, hidden_dim, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(
            self,
            x, xx=None,
            pos_bias=None,
            focus_present_mask=None
    ):
        n, device = x.shape[-2], x.device

        # qkv = self.to_qkv(x).chunk(3, dim=-1)
        if xx is None:
            xx = x
        q, k, v = self.to_q(x), self.to_k(xx), self.to_v(xx)

        if exists(focus_present_mask) and focus_present_mask.all():
            # if all batch samples are focusing on present
            # it would be equivalent to passing that token's values through to the output
            # values = qkv[-1]
            # return self.to_out(values)
            return self.to_out(q)

        # split out heads

        q, k, v = rearrange_many([q, k, v], '... n (h d) -> ... h n d', h=self.heads)

        # scale

        q = q * self.scale

        # rotate positions into queries and keys for time attention

        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # similarity

        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)

        # relative positional bias

        if exists(pos_bias):
            sim = sim + pos_bias[:self.heads]

        if exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = torch.ones((n, n), device=device, dtype=torch.bool)
            attend_self_mask = torch.eye(n, device=device, dtype=torch.bool)

            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # numerical stability

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)


class SparseCausalAttention(Attention):

    def __init__(self, dim, *args, **kwargs):
        super(SparseCausalAttention, self).__init__(dim, *args, **kwargs)
        self.map = torch.nn.Linear(2 * dim, dim)

    def forward(
            self,
            x, xx=None,  # b f hw c
            pos_bias=None,
            focus_present_mask=None
    ):
        b, f, hw, c = x.shape
        kv1 = repeat_dim(x[:, :1], 1, f)  # first frame of each batch
        kv2 = torch.cat([x[:, :1], x[:, :-1]], dim=1)  # previous frame for each one
        kv = torch.cat([kv1, kv2], dim=-1)
        kv = torch.where(focus_present_mask[:, None, None, None], x, self.map(kv))
        return super().forward(x, xx=kv, pos_bias=pos_bias, focus_present_mask=focus_present_mask)


# model

class Unet3D(nn.Module):
    def __init__(
            self,
            dim,
            cond_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            attn_heads=16,
            attn_dim_head=64,
            use_bert_text_cond=False,
            init_dim=None,
            init_kernel_size=7,
            use_sparse_linear_attn=True,
            block_type='resnet',
            resnet_groups=16
    ):
        super().__init__()
        self.channels = channels

        # temporal attention and its relative positional encoding

        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))

        temporal_attn = lambda dim: EinopsToAndFrom('b c f h w', 'b (h w) f c',
                                                    Attention(dim, heads=attn_heads, dim_head=attn_dim_head,
                                                              rotary_emb=rotary_emb))

        self.time_rel_pos_bias = RelativePositionBias(heads=attn_heads,
                                                      max_distance=32)  # realistically will not be able to generate that many frames of video... yet

        # initial conv

        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)

        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(channels, init_dim, (1, init_kernel_size, init_kernel_size),
                                   padding=(0, init_padding, init_padding))

        self.init_temporal_attn = Residual(PreNorm(init_dim, temporal_attn(init_dim)))

        # dimensions

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time conditioning

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # frame conditioning
        self.frame_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # text conditioning

        self.has_cond = exists(cond_dim) or use_bert_text_cond
        cond_dim = 64 if use_bert_text_cond else cond_dim

        self.null_cond_emb = nn.Parameter(torch.randn(1, cond_dim)) if self.has_cond else None

        cond_dim = time_dim + int(cond_dim or 0)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)

        # block type

        block_klass = partial(ResnetBlock, groups=resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim=cond_dim)

        # modules for all layers

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),
                block_klass_cond(dim_out, dim_out),
                block_klass_cond(dim_out, dim_out),
                block_klass_cond(dim_out, dim_out),
                Residual(PreNorm(dim_out, SpatialLinearAttention(dim_out,
                                                                 heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_out, temporal_attn(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)

        spatial_attn = EinopsToAndFrom('b c f h w', 'b f (h w) c', Attention(mid_dim, heads=attn_heads))

        self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_temporal_attn = Residual(PreNorm(mid_dim, temporal_attn(mid_dim)))

        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                block_klass_cond(dim_out * 2, dim_in),
                block_klass_cond(dim_in, dim_in),
                block_klass_cond(dim_in, dim_in),
                block_klass_cond(dim_in, dim_in),
                Residual(PreNorm(dim_in, SpatialLinearAttention(dim_in,
                                                                heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_in, temporal_attn(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim),
            nn.Conv3d(dim, out_dim, 1)
        )

    def forward_with_cond_scale(
            self,
            *args,
            cond_scale=2.,
            **kwargs
    ):
        logits = self.forward(*args, null_cond_prob=0., **kwargs)
        if cond_scale == 1 or not self.has_cond:
            return logits

        null_logits = self.forward(*args, null_cond_prob=1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
            self,
            x,
            time,
            cond=None,
            null_cond_prob=0.,
            focus_present_mask=None,
            prob_focus_present=0.
            # probability at which a given batch sample will focus on the present (0. is all off, 1. is completely arrested attention across time)
    ):
        assert not (self.has_cond and not exists(cond)), 'cond must be passed in if cond_dim specified'
        batch, device = x.shape[0], x.device

        focus_present_mask = default(focus_present_mask,
                                     lambda: prob_mask_like((batch,), prob_focus_present, device=device))

        time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device=x.device)

        x = self.init_conv(x)

        x = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)

        r = x.clone()

        t = self.time_mlp(time) if exists(self.time_mlp) else None
        frame_cond = self.frame_mlp(torch.arange(x.shape[2]).type_as(t))
        frame_cond = frame_cond.transpose(0, 1)[None, :, :]
        t = t[:, :, None] + torch.where(focus_present_mask[:, None, None], torch.zeros_like(frame_cond), frame_cond)

        # classifier free guidance

        if self.has_cond:
            batch, device = x.shape[0], x.device
            mask = prob_mask_like((batch,), null_cond_prob, device=device)
            cond = torch.where(rearrange(mask, 'b -> b 1'), self.null_cond_emb, cond)
            t = torch.cat((t, cond), dim=-1)

        h = []

        for block1, block2, block3, block4, spatial_attn, temporal_attn, downsample in self.downs:
            x = block1(x, t, focus_present_mask=focus_present_mask)
            x = block2(x, t, focus_present_mask=focus_present_mask)
            x = block3(x, t, focus_present_mask=focus_present_mask)
            x = block4(x, t, focus_present_mask=focus_present_mask)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t, focus_present_mask=focus_present_mask)
        x = self.mid_spatial_attn(x, focus_present_mask=focus_present_mask)
        x = self.mid_temporal_attn(x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
        x = self.mid_block2(x, t, focus_present_mask=focus_present_mask)

        for block1, block2, block3, block4, spatial_attn, temporal_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t, focus_present_mask=focus_present_mask)
            x = block2(x, t, focus_present_mask=focus_present_mask)
            x = block3(x, t, focus_present_mask=focus_present_mask)
            x = block4(x, t, focus_present_mask=focus_present_mask)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        return self.final_conv[1](self.final_conv[0](x, t, focus_present_mask=focus_present_mask))


# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)


def beta_schedule(timesteps, min=0.00085, max=0.012):
    betas = torch.linspace(min ** 0.5, max ** 0.5, timesteps, dtype=torch.float64) ** 2
    return torch.clip(betas, 0, 0.9999)


class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            denoise_fn,
            *,
            w=256, h=128,
            num_frames,
            text_use_bert_cls=False,
            channels=3,
            timesteps=1000,
            loss_type='l1',
            use_dynamic_thres=False,  # from the Imagen paper
            dynamic_thres_percentile=0.9
    ):
        super().__init__()
        self.channels = channels
        self.w = w
        self.h = h
        self.num_frames = num_frames
        self.denoise_fn = denoise_fn

        # betas = cosine_beta_schedule(timesteps)
        betas = beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # register buffer helper function that casts float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # text conditioning parameters

        self.text_use_bert_cls = text_use_bert_cls

        # dynamic thresholding when sampling

        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, cond=None, cond_frame=None, cond_scale=1.):
        eps_cond = self.denoise_fn.forward_with_cond_scale(torch.cat([x, cond_frame], dim=1), t,
                                                           cond=cond, cond_scale=cond_scale)
        # eps_uncond = self.denoise_fn.forward_with_cond_scale(
        #     torch.cat([x, self.q_sample(cond_frame, torch.ones_like(t) * (self.num_timesteps - 1))], dim=1), t,
        #     cond=cond, cond_scale=cond_scale)
        eps = eps_cond
        x_recon = self.predict_start_from_noise(x, t=t, noise=eps[:, :self.channels])
        # x_recon = x_recon.clamp(-5.0, 5.0)
        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim=-1
                )

                s.clamp_(min=1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            # clip by threshold, depending on whether static or dynamic
            x_recon = x_recon.clamp(-s, s) / s

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.inference_mode()
    def p_sample(self, x, t, cond_frame=None, cond=None, cond_scale=1., clip_denoised=True):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised, cond=cond,
                                                                 cond_frame=cond_frame,
                                                                 cond_scale=cond_scale)

        b, c, f, h, w = x.shape
        base_noise = repeat(torch.randn(b, c, h, w).type_as(x), 'b c h w -> b f c h w', f=f).transpose(1, 2)
        residual_noise = torch.randn_like(x)
        noise = (base_noise + residual_noise) / (2 ** 0.5)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.inference_mode()
    def p_sample_loop(self, shape, cond_frame=None, cond=None, cond_scale=1., clip_denoised=True):
        device = self.betas.device

        # b = shape[0]
        # img = torch.randn(shape, device=device)

        b, c, f, h, w = shape
        base_noise = repeat(torch.randn(b, c, h, w, device=device), 'b c h w -> b f c h w', f=f).transpose(1, 2)
        residual_noise = torch.randn(shape, device=device)
        img = (base_noise + residual_noise) / (2 ** 0.5)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), cond=cond,
                                cond_frame=cond_frame, clip_denoised=clip_denoised,
                                cond_scale=cond_scale)

        # return unnormalize_img(img)
        return img

    @torch.inference_mode()
    def sample(self, cond_frame, cond=None, cond_scale=1., batch_size=16, clip_denoised=True):
        device = next(self.denoise_fn.parameters()).device

        batch_size = cond.shape[0] if exists(cond) else batch_size
        channels = self.channels
        num_frames = self.num_frames
        return self.p_sample_loop((batch_size, channels, num_frames, self.h, self.w), cond_frame=cond_frame,
                                  cond=cond, cond_scale=cond_scale, clip_denoised=clip_denoised)

    @torch.inference_mode()
    def interpolate(self, x1, x2, cond_frame=None, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), cond_frame=cond_frame)

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, cond_frame=None, cond=None, noise=None, **kwargs):
        b, c, f, h, w, device = *x_start.shape, x_start.device
        base_noise = repeat(torch.randn(b, c, h, w).type_as(x_start), 'b c h w -> b f c h w', f=f).transpose(1, 2)
        residual_noise = default(noise, lambda: torch.randn_like(x_start))
        noise = (base_noise + residual_noise) / (2 ** 0.5)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # if random.random() < 0.0:
        # cond_frame = self.q_sample(cond_frame, torch.ones_like(t) * (random.randint(2, 8)))
        x_recon = self.denoise_fn(torch.cat([x_noisy, cond_frame], dim=1), t, cond=cond, **kwargs)[:, :self.channels]

        if self.loss_type == 'l1':
            loss = F.l1_loss(noise, x_recon)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, *args, **kwargs):
        b, device, w, h = x.shape[0], x.device, self.w, self.h
        check_shape(x, 'b c f h w', c=self.channels, f=self.num_frames, h=h, w=w)
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        # x = normalize_img(x)
        return self.p_losses(x, t, *args, **kwargs)


# trainer class

CHANNELS_TO_MODE = {
    1: 'L',
    3: 'RGB',
    4: 'RGBA'
}


def seek_all_images(img, channels=3):
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]

    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1


# tensor of shape (channels, frames, height, width) -> gif

def video_tensor_to_gif(tensor, path, duration=256, loop=0, optimize=True):
    # images = map(T.ToPILImage(), tensor.unbind(dim=1))
    # first_img, *rest_imgs = images
    # first_img.save(path, save_all=True, append_images=rest_imgs, duration=duration, loop=loop, optimize=optimize)
    images = list(map(lambda t: t.numpy().astype(np.uint8),
                      ((tensor.moveaxis(0, -1).clip(-1.0, 1.0) + 1.0) / 2 * 255.0).cpu().unbind(dim=0)))
    imageio.mimsave(path, images, fps=int(1 / (duration / 1000)))
    return images


# gif -> (channels, frame, height, width) tensor

def repeat_dim(tensor, dim, n):
    shape = []
    for idx in range(len(tensor.shape)):
        if idx == dim:
            shape.append(n)
        else:
            shape.append(1)

    return tensor.repeat(*shape)


def gif_to_tensor(path, channels=3, transform=T.ToTensor()):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels=channels)))
    # cond_tensor = tensors[0].unsqueeze(1)
    tensors = torch.stack(tensors, dim=1)
    # tensors = torch.cat([repeat_dim(cond_tensor, 1, tensors.shape[1]), tensors], dim=0)
    return tensors


def identity(t, *args, **kwargs):
    return t


def normalize_img(t):
    return t * 2 - 1


def unnormalize_img(t):
    return (t + 1) * 0.5


def cast_num_frames(t, *, frames):
    f = t.shape[1]

    if f == frames:
        return t

    if f > frames:
        return t[:, :frames]

    return F.pad(t, (0, 0, 0, 0, 0, frames - f))


class Dataset(data.IterableDataset):
    def __init__(
            self,
            folder,
            w=256, h=128,
            channels=3,
            num_frames=16,
            horizontal_flip=False,
            force_num_frames=True,
            exts=['gif']
    ):
        super().__init__()
        self.folder = folder
        self.w = w
        self.h = h
        self.channels = channels
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.cast_num_frames_fn = partial(cast_num_frames, frames=num_frames) if force_num_frames else identity

        self.transform = T.Compose([
            T.Resize((h, w)),
            T.RandomHorizontalFlip() if horizontal_flip else T.Lambda(identity),
            T.CenterCrop((h, w)),
            T.ToTensor()
        ])

    def __iter__(self):
        return self

    # def __len__(self):
    #     return len(self.paths)

    def __next__(self):
        self.paths = [p for ext in ['gif'] for p in Path(f'{self.folder}').glob(f'**/*.{ext}')]
        while True:
            try:
                path = random.choice(self.paths)
                tensor = gif_to_tensor(path, self.channels, transform=self.transform)
                return self.cast_num_frames_fn(tensor)
            except Exception as e:
                print(e)


class LandscapeAnimation(torch.utils.data.IterableDataset):

    def __init__(self, folder, w=256, h=128, num_frames=1 + 8, step=500):
        super(LandscapeAnimation, self).__init__()
        self.w, self.h = w, h
        self.frames = num_frames
        self.step = step
        self.frame_ratio = w / h
        self.folder = folder
        self.files = [os.path.join(self.folder, file) for file in os.listdir(self.folder) if
                      os.path.splitext(file)[1] == '.mp4']

    def __iter__(self):
        return self

    def __next__(self):
        self.files = [os.path.join(self.folder, file) for file in os.listdir(self.folder) if
                      os.path.splitext(file)[1] == '.mp4']

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
                    frame = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
                    frame = normalize_img(frame)
                    frames.append(np.moveaxis(frame, -1, 0))
                video.release()

                frames = torch.tensor(np.stack(frames, axis=1), dtype=torch.float32)
                return frames
            except Exception as e:
                print(e)


class LandscapeLatents(torch.utils.data.IterableDataset):

    def __init__(self, folder, num_frames=1 + 8, step=4):  # each step is 64 ms
        super(LandscapeLatents, self).__init__()
        self.frames = num_frames
        self.step = step
        self.folder = folder
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

                idx = random.randint(0, len(tensor) - self.frames * self.step - 1)
                frames = []
                for _ in range(self.frames):
                    frames.append(tensor[idx])
                    idx += self.step

                frames = torch.stack(frames, dim=1).to(torch.float32)
                return frames

            except Exception as e:
                print(e)
                continue


# trainer class

class Trainer(object):
    def __init__(
            self,
            diffusion_model,
            folder,
            *,
            w=512,
            h=256,
            ema_decay=0.995,
            num_frames=16,
            step=256,
            train_batch_size=32,
            train_lr=1e-4,
            train_num_steps=100000,
            gradient_accumulate_every=2,
            amp=False,
            step_start_ema=2000,
            update_ema_every=10,
            save_and_sample_every=1000,
            results_folder='./tmp',
            num_sample_rows=2,
            max_grad_norm=None,
            logger=None
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.logger = logger
        self.frames = diffusion_model.num_frames
        self.duration = step

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.w, self.h = w, h
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        channels = diffusion_model.channels
        num_frames = diffusion_model.num_frames

        # self.ds = Dataset(folder, w=w, h=h, channels=channels, num_frames=num_frames + 1)
        # self.ds = LandscapeAnimation(folder, w=w, h=h, num_frames=num_frames + 1, step=step)
        self.ds = LandscapeLatents(folder, num_frames=num_frames + 1, step=4)

        self.dl = cycle(data.DataLoader(self.ds, batch_size=train_batch_size, shuffle=False, pin_memory=True,
                                        num_workers=8, prefetch_factor=2, ))
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=(0.9, 0.99))

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled=amp)
        self.max_grad_norm = max_grad_norm

        self.num_sample_rows = num_sample_rows
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True, parents=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone, **kwargs):
        if milestone == -1:
            all_milestones = [int(p.stem.split('-')[-1]) for p in Path(self.results_folder).glob('**/*.pt')]
            assert len(
                all_milestones) > 0, 'need to have at least one milestone to load from latest checkpoint (milestone == -1)'
            milestone = max(all_milestones)

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))

        self.step = data['step']
        self.model.load_state_dict(data['model'], **kwargs)
        self.ema_model.load_state_dict(data['ema'], **kwargs)
        self.scaler.load_state_dict(data['scaler'])

    def train(
            self,
            vae,
            vae_scale=0.18215,
            vae_device=torch.device('cpu'),
            model_device=torch.device('cpu'),
            prob_focus_present=0.5,
            focus_present_mask=None,
            log_fn=noop
    ):
        assert callable(log_fn)

        while self.step < self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                # encode data to latents

                # bs = 4
                # datas = torch.split(next(self.dl).to(torch.float16), bs, dim=0)
                # bdata = []
                # for data in datas:
                #     data = data.to(vae_device)
                #     data = rearrange(data, 'b c n h w -> (b n) c h w')
                #     with torch.no_grad():
                #         latents = vae.encode(data).latent_dist.mode() * vae_scale
                #     data = rearrange(latents, '(b n) c h w -> b c n h w', b=bs).to(model_device).to(torch.float32)
                #     bdata.append(data)
                # data = torch.cat(bdata, dim=0)
                data = next(self.dl).to(model_device) * vae_scale

                with autocast(enabled=self.amp):
                    loss = self.model(
                        data[:, :, 1:],
                        cond_frame=repeat_dim(data[:, :, :1], 2, self.frames),
                        prob_focus_present=prob_focus_present,
                        focus_present_mask=focus_present_mask,
                    )

                    self.scaler.scale(loss / self.gradient_accumulate_every).backward()
                del data
                print(f'{self.step}: {loss.item()}')

            log = {'loss': loss.item()}

            if exists(self.max_grad_norm):
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_and_sample_every == 0 and self.step != 1:
                milestone = self.step // self.save_and_sample_every
                num_samples = self.num_sample_rows ** 2

                bs = 4
                groups, remainder = num_to_groups(num_samples, bs)

                # log examples from train dataset
                # train_ex = [next(self.dl)[:bs] for _ in range(groups)]
                # if remainder > 0:
                #     train_ex.append(next(self.dl)[:remainder])
                # train_videos_list = torch.cat(train_ex, dim=0)
                # train_videos_list = F.pad(train_videos_list, (2, 2, 2, 2))
                # train_gif = rearrange(train_videos_list, '(i j) c f h w -> c f (i h) (j w)', i=self.num_sample_rows)
                # video_path = str(self.results_folder / str(f'{milestone}_train.gif'))
                # video_tensor_to_gif(train_gif, video_path, duration=self.duration)
                # self.logger.report_media('gifs', f'{milestone}_train.gif', iteration=milestone, local_path=video_path)

                orig_batches = [next(self.dl)[:bs, :, :1] for _ in range(groups)]
                if remainder > 0:
                    orig_batches.append(next(self.dl)[:remainder, :, :1])
                # batches = []
                # for cond in orig_batches:
                #     cond = cond * vae_scale
                #     cond = cond.to(torch.float16).to(vae_device)
                #     cond = rearrange(cond, 'b c n h w -> (b n) c h w')
                #     with torch.no_grad():
                #         latents = vae.encode(cond).latent_dist.mode()
                #     latents = rearrange(latents, '(b n) c h w -> b c n h w', b=bs)
                #     batches.append(latents.to(model_device).to(torch.float32))

                batches = [cond * vae_scale for cond in orig_batches]
                latents_list = list(map(lambda cond: self.ema_model.sample(
                    cond_frame=repeat_dim(cond.to(model_device), 2, self.frames), batch_size=len(cond),
                    clip_denoised=False).cpu(), batches))
                gc.collect()
                torch.cuda.empty_cache()

                vae = vae.to(vae_device)

                all_videos_list = []
                for latent, cond in zip(latents_list, batches):
                    cond = cond.to(torch.float16).to(vae_device)
                    latent = latent.to(torch.float16).to(vae_device)
                    latent = torch.cat([cond, latent], dim=2)
                    latent = latent / vae_scale
                    latent = rearrange(latent, 'b c n h w -> (b n) c h w')
                    print('predicted latent', lt.lovely(latent))
                    with torch.no_grad():
                        video = vae.decode(latent).sample
                    print('decoded video', lt.lovely(video))
                    video = rearrange(video, '(b n) c h w -> b c n h w', b=bs)
                    all_videos_list.append(video.cpu().to(torch.float32))

                    # free memory
                    del cond, latent, video
                    gc.collect()
                    torch.cuda.empty_cache()

                vae = vae.to('cpu')
                gc.collect()
                torch.cuda.empty_cache()

                # all_videos_list = torch.cat([torch.cat(orig_batches, dim=0), torch.cat(all_videos_list, dim=0)], dim=2)
                all_videos_list = torch.cat(all_videos_list, dim=0)

                all_videos_list = F.pad(all_videos_list, (2, 2, 2, 2))

                one_gif = rearrange(all_videos_list, '(i j) c f h w -> c f (i h) (j w)', i=self.num_sample_rows)
                video_path = str(self.results_folder / str(f'{milestone}.gif'))
                video_tensor_to_gif(one_gif, video_path, duration=self.duration)
                self.logger.report_media('gifs', f'{milestone}.gif', iteration=milestone, local_path=video_path)

                log = {**log, 'sample': video_path}
                self.save(milestone)

            log_fn(log)
            self.step += 1

        print('training completed')


nframes = 7 + 1
channels = 4
test_dir = '/dsk1/danil/3d/nerf/data/landscape-animation'
w, h = 512, 256
model_device = torch.device('cuda:0')
vae_device = torch.device('cuda:0')

# run training
model = Unet3D(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    channels=channels * 2,
)

diffusion = GaussianDiffusion(
    model,
    w=w // 8,
    h=h // 8,
    channels=channels,
    num_frames=nframes - 1,
    use_dynamic_thres=True,
    timesteps=256,  # number of steps
    loss_type='l2'  # L1 or L2
).to(model_device)

vae = AutoencoderKL.from_pretrained('stabilityai/stable-diffusion-2', subfolder='vae', torch_dtype=torch.float16)

print("Initializing ClearML")
task = clearml.Task.init(project_name='animation', task_name='Test video diffusion latents', reuse_last_task_id=True,
                         auto_connect_frameworks=False)
# task.connect(args, name='config')
logger = task.get_logger()

trainer = Trainer(
    diffusion,
    test_dir,
    logger=logger,
    step=256,
    # this folder path needs to contain all your training data, as .gif files, of correct image size and number of frames
    train_batch_size=20,
    train_lr=1e-4,
    results_folder='./tmp2',
    save_and_sample_every=256,
    update_ema_every=1,
    num_sample_rows=2,
    train_num_steps=1000000,  # total training steps
    gradient_accumulate_every=4,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    amp=True  # turn on mixed precision
)
# trainer.load(25)

gc.collect()
torch.cuda.empty_cache()

trainer.train(vae, vae_device=vae_device, model_device=model_device)
