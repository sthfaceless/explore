import math
import os
import random

import msgpack
import numpy as np

from common import *
from scenes import *
import pyngp as ngp


class TestbedSnapshot:

    def __init__(self):
        self.testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
        self.testbed.nerf.sharpen = 0.0
        self.testbed.exposure = 0.0
        self.testbed.nerf.render_with_camera_distortion = True
        self.testbed.shall_train = False

        self.testbed.fov_axis = 0
        self.focal = 1.5
        self.img_size = 800
        focal_screen = self.focal * self.img_size / 2
        self.testbed.fov = math.atan(self.img_size / (focal_screen * 2)) * 2 * 180 / np.pi
        self.default_cams = [[[-1.0, -0.0, -0.0, -0.0],
                              [0.0, 0.866, 0.5, 0.5],
                              [-0.0, 0.5, -0.866, -0.866]],
                             [[-1.0, -0.0, -0.0, -0.0],
                              [0.0, 0.866, -0.5, -0.5],
                              [-0.0, -0.5, -0.866, -0.866]],
                             [[1.0, -0.0, 0.0, 0.0],
                              [-0.0, 0.866, 0.5, 0.5],
                              [-0.0, -0.5, 0.866, 0.866]],
                             [[1.0, 0.0, 0.0, 0.0],
                              [0.0, 0.866, -0.5, -0.5],
                              [-0.0, 0.5, 0.866, 0.866]],
                             [[0.0, 0.5, -0.866, -0.866],
                              [-0.0, 0.866, 0.5, 0.5],
                              [1.0, -0.0, 0.0, 0.0]],
                             [[0.0, -0.5, -0.866, -0.866],
                              [0.0, 0.866, -0.5, -0.5],
                              [1.0, 0.0, 0.0, 0.0]],
                             [[0.0, -0.5, 0.866, 0.866],
                              [0.0, 0.866, 0.5, 0.5],
                              [-1.0, -0.0, 0.0, 0.0]],
                             [[0.0, 0.5, 0.866, 0.866],
                              [-0.0, 0.866, -0.5, -0.5],
                              [-1.0, 0.0, 0.0, 0.0]]]

    def linear_to_srgb(self, img):
        limit = 0.0031308
        return np.where(img > limit, 1.055 * (img ** (1.0 / 2.4)) - 0.055, 12.92 * img)

    def get_screenshots(self, snappath, img_size=800):

        focal_screen = self.focal * img_size / 2
        self.testbed.fov = math.atan(img_size / (focal_screen * 2)) * 2 * 180 / np.pi

        self.testbed.load_snapshot(snappath)

        images = []
        for cam in self.default_cams:
            self.testbed.set_nerf_camera_matrix(np.array(cam))
            image = self.testbed.render(img_size, img_size, 32, True)
            image[..., 0:3] = np.divide(image[..., 0:3], image[..., 3:4], out=np.zeros_like(image[..., 0:3]),
                                        where=image[..., 3:4] != 0)
            image = np.clip(self.linear_to_srgb(image[..., 0:3]), 0.0, 1.0)
            images.append(image)

        return images

    def get_snapshot_body(self, dataset_root, snapshot_name='snapshot.msgpack'):

        items = [os.path.join(dataset_root, item) for item in os.listdir(dataset_root)]
        items = [item for item in items if os.path.isdir(item) and os.listdir(item)]

        with open(os.path.join(items[0], snapshot_name), 'rb') as f:
            obj = msgpack.loads(f.read())

        del obj['snapshot']['nerf']['dataset']['paths']
        del obj['snapshot']['nerf']['dataset']['metadata']
        del obj['snapshot']['nerf']['dataset']['xforms']

        obj['snapshot']['nerf']['dataset']['xforms'] = []
        for image_id in range(obj['snapshot']['nerf']['dataset']['n_images']):
            obj['snapshot']['nerf']['dataset']['xforms'] \
                .append({'end': np.eye(3, 4).tolist(), 'start': np.eye(3, 4).tolist()})

        return obj

    def get_random_snapshots(self, dataset_root, n=5):
        items = [os.path.join(dataset_root, item) for item in os.listdir(dataset_root)]
        items = [item for item in items if os.path.isdir(item) and os.listdir(item)]
        return random.choices(items, k=n)
