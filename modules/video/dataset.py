import os
import random
from random import choice

import cv2
import numpy as np
import torch


class LandscapeAnimation(torch.utils.data.IterableDataset):

    def __init__(self, root, w=256, h=128, frames=1 + 8, step=500):
        super(LandscapeAnimation, self).__init__()
        self.w, self.h = w, h
        self.frames = frames
        self.step = step
        self.frame_ratio = w / h
        self.files = [os.path.join(root, file) for file in os.listdir(root) if os.path.splitext(file)[1] == '.mp4']

    def __iter__(self):
        return self

    def __next__(self):

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
            frame = (cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_AREA).astype(
                np.float32) / 255.0 - 1.0) * 2
            frames.append(np.moveaxis(frame, -1, 0))
        video.release()

        return {
            'frames': np.stack(frames, axis=0)
        }
