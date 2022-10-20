import glob
import os
from random import sample, choices, randint

import cv2
from tqdm import tqdm

from modules.common.util import *
from modules.nerf.util import *


class NerfScene(torch.utils.data.IterableDataset):

    def __init__(self, scene_root, batch_rays, n_images=-1, transforms='transforms.json', images_dir='images',
                 color_ratio=0.75, w=128, h=128):
        super(NerfScene, self).__init__()
        self.scene_root = scene_root
        self.batch_rays = batch_rays
        self.color_sample = int(self.batch_rays * color_ratio)
        self.all_sample = self.batch_rays - self.color_sample

        # images_dir = 'train'
        # transforms = 'transforms_train.json'

        with open(os.path.join(self.scene_root, transforms), 'r') as f:
            meta = json.load(f)

        self.imgs = []
        self.poses = []
        self.non_black_pixels = []
        self.non_black_ids = []
        if n_images < 0:
            frames = meta['frames']
        else:
            frames = sample(meta['frames'], k=n_images)
        for image_id, frame in enumerate(frames):
            # read image and resize to desired resolution
            img = cv2.imread(os.path.join(self.scene_root, images_dir, os.path.basename(frame['file_path'])))
            img = cv2.resize(img, (h, w))

            # find black pixels at the image to ignore them in sampling
            # self.non_black_pixels.append(np.transpose((img.min(axis=-1) > 0).nonzero()))
            self.non_black_pixels.append(
                np.mgrid[int(h / 4):int(3 * h / 4), int(w / 4):int(3 * w / 4)].reshape(2, -1).transpose())

            self.non_black_ids.append(np.ones(len(self.non_black_pixels[-1]), dtype=np.int32) * image_id)
            # save camera poses of images
            self.imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
            self.poses.append(np.array(frame['transform_matrix']).astype(np.float32)[:-1])
        self.imgs = np.array(self.imgs)  # n_images h w 3
        self.poses = np.array(self.poses)  # n_images 3 4
        self.non_black_pixels = np.concatenate(self.non_black_pixels, axis=0)
        self.non_black_ids = np.concatenate(self.non_black_ids, axis=0)

        self.h, self.w = self.imgs[0].shape[:2]

        # center all cameras at (0,0,0)
        self.poses[:, :, -1] -= self.get_mean_point().reshape(1, 3)
        # rescale coordinates to out system
        scale = 4.0 / self.get_mean_distance()
        self.poses[:, :, -1] *= scale

        self.camera_angle_x = float(meta['camera_angle_x'])
        self.focal_x = 0.5 / np.tan(0.5 * self.camera_angle_x)
        if 'camera_angle_y' in meta:
            self.camera_angle_y = float(meta['camera_angle_y'])
            self.focal_y = 0.5 / np.tan(0.5 * self.camera_angle_y)
        else:
            self.focal_y = self.focal_x

    def __iter__(self):
        return self

    def __next__(self):

        # indices for all rays
        img_pixels = self.w * self.h
        indices = np.random.randint(low=0, high=img_pixels * len(self.imgs), size=self.all_sample)
        img_ids = indices // img_pixels

        ray_ids = indices % img_pixels
        ray_h = ray_ids // self.w
        ray_w = ray_ids % self.w

        # indices for color rays
        indices = np.random.randint(low=0, high=len(self.non_black_pixels), size=self.color_sample)
        ray_h = np.concatenate([ray_h, self.non_black_pixels[indices, 0]], axis=0)
        ray_w = np.concatenate([ray_w, self.non_black_pixels[indices, 1]], axis=0)
        img_ids = np.concatenate([img_ids, self.non_black_ids[indices]], axis=0)

        pixels = self.imgs[img_ids, ray_h, ray_w]
        poses = self.poses[img_ids]

        dirs = get_image_coords(h=self.h, w=self.w, focal_x=self.focal_x, focal_y=self.focal_y)

        pixel_coords = dirs[torch.from_numpy(ray_h), torch.from_numpy(ray_w)]

        return {
            'pixels': torch.from_numpy(pixels),
            'poses': torch.from_numpy(poses),
            'pixel_coords': pixel_coords,
            'near': torch.ones(self.batch_rays) * (4.0 - 3 ** (1 / 2)),
            'far': torch.ones(self.batch_rays) * (4.0 + 3 ** (1 / 2)),
            'base_radius': torch.ones(self.batch_rays) / (self.focal_x * self.w * 3 ** (1 / 2))
        }

    def get_focal(self):
        return self.focal_x

    def get_mean_distance(self):
        return float(np.mean(np.linalg.norm(self.poses[:, :, -1], axis=-1)))

    def get_mean_point(self):
        mid_dir = np.array([0, 0, -1], dtype=np.float32)
        mid_point = np.mean(self.poses[:, :, -1] + (self.poses[:, :, :3] @ mid_dir) * self.get_mean_distance(), axis=0)
        return mid_point


class NerfClass(torch.utils.data.IterableDataset):

    def __init__(self, class_path, batch_rays, images_per_scene=-1, batch_objects=1, cache_size=-1,
                 transforms='transforms.json', images_dir='images', w=128, h=128):
        self.batch_rays = batch_rays
        self.batch_objects = batch_objects
        self.w = w
        self.h = h
        self.class_path = class_path
        self.transforms = transforms
        self.images_dir = images_dir
        self.images_per_scene = images_per_scene
        self.items = [item for item in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, item))]

        if cache_size == -1:
            self.cache_size = len(self.items)
        else:
            self.cache_size = cache_size
        self.cache = []
        self.cache_keys = []
        self.reset_cache()

    def load_nerf_scene(self, object_id):
        return NerfScene(os.path.join(self.class_path, self.items[object_id]), self.batch_rays,
                         transforms=self.transforms, w=self.w, h=self.h,
                         images_dir=self.images_dir, n_images=self.images_per_scene)

    def reset_cache(self):
        self.cache_keys = sample(range(len(self.items)), k=self.cache_size)
        self.cache = {obj_id: self.load_nerf_scene(obj_id) for obj_id in tqdm(self.cache_keys)}

    def __iter__(self):
        return self

    def get_items(self):
        return self.items

    def get_focal(self):
        return self.load_nerf_scene(0).get_focal()

    def get_camera_distance(self):
        return self.load_nerf_scene(0).get_mean_distance()

    def __next__(self):
        cache_index = randint(a=0, b=self.cache_size - 1)
        id = self.cache_keys[cache_index]
        scene = next(self.cache[id])
        scene['id'] = torch.LongTensor([id])
        return scene


class NerfViews(torch.utils.data.IterableDataset):

    def __init__(self, path, batch_size, sample=10, n_views=64, image_size=128, cache_size=-1):
        super(NerfViews, self).__init__()
        self.path = path
        self.image_size = image_size

        self.batch_size = batch_size
        self.n_views = n_views
        self.sample = sample

        items = [item for item in os.listdir(path) if os.path.isdir(os.path.join(path, item))]
        self.items = [item for item in items if len(glob.glob(f'{os.path.join(path, item)}/*.png')) == n_views]
        if cache_size == -1:
            self.cache_size = len(self.items)
        else:
            self.cache_size = cache_size
        self.cache = np.array([])

    def reset_cache(self):
        cache_items = []
        ids = choices(range(len(self.items)), k=self.cache_size)
        for idx in ids:
            class_path = os.path.join(self.path, self.items[idx])
            class_images = []
            for img_path in os.listdir(class_path):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.image_size, self.image_size))
                class_images.append(normalize_image(img))
            cache_items.append(np.array(class_images).moveaxis(-1, 1))  # n h w 3 -> n 3 h w
        self.cache = np.array(cache_items)

    def get_cache_indexes(self, size):
        return np.random.randint(low=0, high=len(self.cache), size=size)

    def get_cache_items(self, cache_indexes, views):
        return self.cache[cache_indexes, views]

    def __iter__(self):
        return self

    def __next__(self):
        view_ids = choices(range(self.n_views), k=self.sample)
        cache_indexes = self.get_cache_indexes(self.batch_size)
        batch = {
            'ids': view_ids,
            'data': [
                        torch.from_numpy(self.cache[cache_indexes, np.random.randint(low=0, high=self.n_views,
                                                                                     size=self.batch_size)])]
                    + [torch.from_numpy(self.cache[cache_indexes, np.full(self.batch_size, fill_value=view_id,
                                                                          dtype=np.int32)])
                       for view_id in view_ids]
        }
        return batch


class NerfWeights(torch.utils.data.Dataset):

    def __init__(self, paths, batch_size, device=torch.device('cpu'),
                 layers=('input_layer', 'rgb_layer', 'density_layer', 'blocks.1.0', 'blocks.0.0', 'blocks.2.0',
                         'blocks.3.0', 'blocks.3.1', 'blocks.2.1', 'blocks.1.1', 'blocks.0.1')):
        super(NerfWeights, self).__init__()
        self.layers = layers
        self.batch_size = batch_size

        self.shapes = []
        self.weights = []
        for i, path in enumerate(paths):
            model_state = torch.load(path, map_location=device)
            model_weights = []
            for layer in layers:
                w = torch.cat([model_state[f'{layer}.0.weight'], model_state[f'{layer}.0.bias'].unsqueeze(1)], dim=1)

                if i == 0:
                    self.shapes.append(tuple(w.shape))

                # we need something more smart because there is no invert operation for that normalization
                # w = (w - w.min()) / (w.max() - w.min()) * 2 - 1.0
                model_weights.append(w.view(-1))
            self.weights.append(torch.cat(model_weights, dim=0))
        self.shapes = tuple(self.shapes)
        # self.weights = torch.stack(self.weights, dim=0)

    def get_shapes(self):
        return self.shapes

    def get_layers(self):
        return self.layers

    def __iter__(self):
        return self

    def __next__(self):
        rand_indexes = torch.randint(low=0, high=len(self.weights) - 1, size=self.batch_size)
        return {
            'weights': self.weights[rand_indexes]  # (b, features)
        }

    def __len__(self):
        return len(self.weights)

    def __getitem__(self, idx):
        return {
            'weights': self.weights[idx]
        }


class NerfLatents(torch.utils.data.IterableDataset):

    def __init__(self, latents_checkpoint, latent_shape=(32, 8, 8)):
        super(NerfLatents, self).__init__()
        latents = torch.load(latents_checkpoint, map_location='cpu')['state_dict']['latents.weight']
        latents = latents.view(-1, *latent_shape).numpy()
        self.latents = [latents[idx] for idx in range(len(latents))]

    def __getitem__(self, idx):
        return self.latents[idx]

    def __iter__(self):
        return self

    def __next__(self):
        return self[randint(0, len(self.latents) - 1)]


class PairViews(torch.utils.data.IterableDataset):

    def __init__(self, class_path, images_per_scene=-1, transforms='transforms.json', images_dir='images',
                 cache_size=-1, w=128, h=128):
        super(PairViews, self).__init__()
        # images_dir = 'train'
        # transforms = 'transforms_train.json'
        self.class_path = class_path
        self.images_dir = images_dir
        self.transforms = transforms
        self.h = h
        self.w = w
        self.images_per_scene = images_per_scene
        self.items = [item for item in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, item))]
        if cache_size == -1:
            self.cache_size = len(self.items)
        else:
            self.cache_size = cache_size
        self.images, self.poses, self.focals = [], [], []
        self.reset_cache()

    def reset_cache(self):
        self.images, self.poses, self.focals = [], [], []
        items = sample(self.items, k=self.cache_size)
        for scene in tqdm(items):
            scene_root = os.path.join(self.class_path, scene)
            with open(os.path.join(scene_root, self.transforms), 'r') as f:
                meta = json.load(f)
            self.focals.append(0.5 / np.tan(0.5 * meta['camera_angle_x']))
            if self.images_per_scene < 0:
                frames = meta['frames']
            else:
                frames = sample(meta['frames'], k=self.images_per_scene)
            images, poses = [], []
            for image_id, frame in enumerate(frames):
                # read image and resize to desired resolution
                img = cv2.imread(os.path.join(scene_root, self.images_dir, os.path.basename(frame['file_path'])))
                img = cv2.resize(img, (self.h, self.w))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # save camera poses of images
                images.append(normalize_image(img).swapaxes(0, -1))
                poses.append(np.array(frame['transform_matrix']).astype(np.float32)[:-1])
            self.images.append(images)
            # some standardizations
            poses = np.array(poses)
            self.mean_distance = np.mean(np.linalg.norm(poses[:, :, -1], axis=-1))
            mid_dir = np.array([0, 0, -1], dtype=np.float32)
            mid_point = np.mean(poses[:, :, -1] + (poses[:, :, :3] @ mid_dir) * self.mean_distance, axis=0)
            poses[:, :, -1] -= mid_point.reshape(1, 3)
            poses[:, :, -1] *= 4.0 / self.mean_distance
            self.poses.append(poses)

    def get_mean_distance(self):
        return self.mean_distance

    def __iter__(self):
        return self

    def __next__(self):
        idx = randint(0, self.cache_size - 1)
        views = self.images[idx]
        pair = sample(range(len(views)), k=2)
        return {
            'view': views[pair[0]],
            'view_poses': self.poses[idx][pair[0]],
            'cond': views[pair[1]],
            'cond_poses': self.poses[idx][pair[1]],
            'focal': self.focals[idx].astype(np.float32),
        }
