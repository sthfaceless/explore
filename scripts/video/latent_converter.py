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


vae = AutoencoderKL.from_pretrained('stabilityai/stable-diffusion-2', subfolder='vae', torch_dtype=torch.float16).to(
    'cuda:0')

nframes = 7 + 1
video_frames = 48
root = '/dsk1/danil/3d/nerf/data/landscape-animation'
ww, hh = 512, 256
orig_frame_ratio = ww / hh
step = 64
batch_size = 24

# ds = LandscapeAnimation(root, w=w, h=h, step=step)
# dl = iter(data.DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True,
#                                         num_workers=8, prefetch_factor=2))

files = [os.path.join(root, file) for file in os.listdir(root) if os.path.splitext(file)[1] == '.mp4']
for file in tqdm(files):

    name = os.path.splitext(os.path.basename(file))[0] + '.pt'
    path = os.path.join(root, name)

    if os.path.exists(path):
        continue

    try:
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
        frames = []
        while len(frames) < video_frames:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            frame_id += frame_step

            ret, frame = video.read()
            # try another video on fail
            if not ret:
                video.release()
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
            frames.append(np.moveaxis(frame, -1, 0))
        video.release()

        frames = torch.tensor(np.stack(frames, axis=0), dtype=torch.float16)


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

