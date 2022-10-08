from random import randint

import numpy as np
import torch


class ColorSquares(torch.utils.data.IterableDataset):

    def __init__(self, img_size=64, n_squares=256):
        self.img_size = img_size
        self.squares = []
        for _ in range(n_squares):
            r = randint(0, 255)
            g = randint(0, 255)
            b = randint(0, 255)
            colors = np.array([r, g, b]).reshape((3, 1, 1))
            self.squares.append(np.tile(colors, (1, img_size, img_size)).astype(np.float32) / (255.0 / 2) - 1.0)

    def __getitem__(self, idx):
        return self.squares[idx]

    def __iter__(self):
        return self

    def __next__(self):
        return self[randint(0, len(self.squares) - 1)]
