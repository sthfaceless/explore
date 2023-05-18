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
from tqdm import tqdm
from argparse import ArgumentParser
import clearml
import lpips


def get_parser():
    parser = ArgumentParser(description="Calculate lpips for videos")
    # Input data settings
    parser.add_argument("--videos", default=[], nargs='+', help="List here pairs of videos")
    parser.add_argument("--batch", default=8, type=int, help="Num of frames to calculate LPIPS")

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    metric_fn = lpips.LPIPS(net='alex').to(device)

    agg_metrics = []
    for orig, distorted in tqdm(zip(args.videos[::2], args.videos[1::2])):

        orig_video = cv2.VideoCapture(orig)
        distorted_video = cv2.VideoCapture(distorted)

        nframes = int(distorted_video.get(cv2.CAP_PROP_FRAME_COUNT))
        metrics = []

        processed = 0
        pbar = tqdm(total=nframes)
        while processed < nframes:

            orig_frames = []
            distorted_frames = []
            for frame_id in range(args.batch):
                ret1, frame1 = orig_video.read()
                ret2, frame2 = distorted_video.read()

                if not (ret1 and ret2):
                    break

                orig_frames.append(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
                distorted_frames.append(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))

            orig_frames = torch.tensor(np.stack(orig_frames, axis=0)).to(dtype=torch.float32, device=device)
            distorted_frames = torch.tensor(np.stack(distorted_frames, axis=0)).to(dtype=torch.float32, device=device)

            orig_frames = (orig_frames / 127.5 - 1.0).movedim(-1, -3)
            distorted_frames = (distorted_frames / 127.5 - 1.0).movedim(-1, -3)

            with torch.no_grad():
                metrics.append(metric_fn(orig_frames, distorted_frames).mean())

            processed += len(orig_frames)
            pbar.update(len(orig_frames))
        pbar.close()

        orig_video.release()
        distorted_video.release()


        metrics = torch.stack(metrics)
        print(f'LPIPS for video {distorted}, mean --- {metrics.mean()} std --- {metrics.std()}')
        agg_metrics.append(metrics.mean())

    agg_metrics = torch.stack(agg_metrics)
    print(f'LPIPS average for videos {agg_metrics.mean()}')
    print(f'LPIPS std for videos {agg_metrics.std()}')