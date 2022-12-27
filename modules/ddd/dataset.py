import os

from modules.ddd.util import *


class ShapenetPointClouds(torch.utils.data.Dataset):

    def __init__(self, shapenet_root, n_points=100000, cache_dir=None, categories=['plane'], classes=['02691156'],
                 noise=1e-2, cache_scenes=512, smoothed_suffix=None):
        super(ShapenetPointClouds, self).__init__()
        self.shapenet_root = shapenet_root
        self.classes = [cls for cls in os.listdir(shapenet_root) if cls in classes]
        self.items = []
        for cls in self.classes:
            if smoothed_suffix is None:
                smoothed_suffix = ''
            paths = [os.path.join(shapenet_root, cls, item, 'models', f'model_normalized.obj{smoothed_suffix}')
                     for item in os.listdir(os.path.join(shapenet_root, cls))]
            self.items.extend([path for path in paths if os.path.exists(path)])
        assert len(self.items) > 0, 'Loaded empty list of objects from shapenet'
        self.noise = noise

    def __len__(self):
        # return len(self.pc_ds)
        return len(self.items)

    def __getitem__(self, idx):

        mesh_path = self.items[idx]
        vertices, faces = read_obj(mesh_path)
        vertices = normalize_points(vertices) * 0.95

        # sdf_grid = torch.load(os.path.join(os.path.dirname(mesh_path), 'sdf'))

        return {
            'vertices': vertices,
            'faces': faces,
            'sdf': None
        }
