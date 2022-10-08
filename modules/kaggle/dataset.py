import os
from random import random, randint

import cv2
import numpy as np
import torch


class KagglePokemons(torch.utils.data.IterableDataset):

    def __init__(self, root='/kaggle/input/pokemon-images-dataset/pokemon/pokemon', img_size=64):
        self.root = root
        self.w, self.h = img_size, img_size
        self.image_paths = os.listdir(self.root)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.root, self.image_paths[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64))
        if random() < 0.5:
            img = img[:, ::-1]
        img = img.astype(np.float32) / (255.0 / 2) - 1.0
        img = np.moveaxis(img, -1, 0)
        return img

    def __iter__(self):
        return self

    def __next__(self):
        return self[randint(0, len(self.image_paths) - 1)]
