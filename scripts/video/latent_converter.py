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

vae = AutoencoderKL.from_pretrained('stabilityai/stable-diffusion-2', subfolder='vae', torch_dtype=torch.float16).to(
    'cuda:0')

nframes = 7 + 1
video_frames = 48
root = '/dsk1/danil/3d/nerf/data/tumblr'
ww, hh = 512, 256
orig_frame_ratio = ww / hh
step = 64
batch_size = 24

files = [os.path.join(root, file) for file in os.listdir(root) if os.path.splitext(file)[1] in ('.mp4', '.gif')]
for file in tqdm(files):

    file_parts = os.path.splitext(os.path.basename(file))
    name = file_parts[0] + '.pt'
    ext = file_parts[1]
    path = os.path.join(root, name)

    if os.path.exists(path):
        continue

    try:

        frames = []
        if ext == '.mp4':
            # load video
            video = cv2.VideoCapture(file)
            if not video.isOpened():
                video.release()
                continue

            video_fps = int(video.get(cv2.CAP_PROP_FPS))
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_step = int(step / 1000 * video_fps)
            frames_used = frame_step * nframes
            if frames_used > total_frames:
                video.release()
                continue

            # take frames
            frame_id = 0
            while len(frames) < min(video_frames, total_frames // frame_step):
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                frame_id += frame_step

                ret, frame = video.read()
                # try another video on fail
                if not ret:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            video.release()
        elif ext == '.gif':
            gif = Image.open(file)
            fps = int(1000 / gif.info['duration'])
            orig_frames = [frame.convert('RGB') for frame in PIL.ImageSequence.Iterator(gif)]
            frame_step = max(int(step / 1000 * fps), 1)
            frames_used = frame_step * nframes
            if frames_used > len(orig_frames):
                continue
            frame_id = 0
            while len(frames) < min(video_frames, len(orig_frames) // frame_step):
                frames.append(np.array(orig_frames[frame_id]))
                frame_id += frame_step

        if len(frames) < nframes:
            continue

        processed_frames = []
        for frame in frames:

            h, w = frame.shape[:2]
            frame_ratio = w / h
            if frame_ratio > orig_frame_ratio:
                # width is bigger so let's crop it
                true_w = int(h * orig_frame_ratio)
                start_w = int((w - true_w) / 2)
                frame = frame[:, start_w: start_w + true_w]
            else:
                # height is bigger
                true_h = int(w / orig_frame_ratio)
                start_h = int((h - true_h) / 2)
                frame = frame[start_h: start_h + true_h]
            frame = cv2.resize(frame, (ww, hh), interpolation=cv2.INTER_AREA)
            frame = frame.astype(np.float32) / 127.5 - 1.0
            processed_frames.append(np.moveaxis(frame, -1, 0))

        frames = torch.tensor(np.stack(processed_frames, axis=0), dtype=torch.float16)

        latents = []
        for batch in torch.split(frames, batch_size, dim=0):
            batch = batch.to('cuda:0')
            with torch.no_grad():
                latent = vae.encode(batch).latent_dist.mode()
            latents.append(latent.cpu())
        latents = torch.cat(latents, dim=0)
        torch.save(latents, path)

    except Exception as e:
        print(e)
        continue
