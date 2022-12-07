import os
from argparse import ArgumentParser

import torch
from tqdm import tqdm

from modules.ddd.util import *


def parse_args():
    parser = ArgumentParser(description="Calculates SDF grids for shapenet meshes")

    parser.add_argument("--shapenet", default="", help="Path to shapenet root")
    parser.add_argument("--classes", default=['02691156'], nargs='+', help="Particular classes to generate dataset for")
    parser.add_argument("--res", default=128, type=int, help="SDF grid resolution")
    parser.add_argument("--scans", default=100, type=int, help="Count of cameras scans")
    parser.add_argument("--scan_res", default=1024, type=int, help="Resolution of camera scans")

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()

    shapenet_root = args.shapenet
    classes = [cls for cls in os.listdir(shapenet_root) if cls in args.classes]
    items = []
    for cls in classes:
        paths = [os.path.join(shapenet_root, cls, item, 'models', 'model_normalized.obj')
                 for item in os.listdir(os.path.join(shapenet_root, cls))]
        items.extend([path for path in paths if os.path.exists(path)])

    assert len(items) > 0, 'Loaded empty list of objects from shapenet'

    grid_resolution = args.res
    x, y, z = torch.meshgrid(torch.arange(grid_resolution), torch.arange(grid_resolution),
                             torch.arange(grid_resolution))
    x, y, z = x.reshape(-1), y.reshape(-1), z.reshape(-1)
    points = normalize_points(torch.stack([x, y, z], dim=-1))

    for mesh_path in tqdm(items, desc='Processed mesh files'):
        try:
            vertices, faces = read_obj(mesh_path)
            vertices = normalize_points(vertices) * 0.95
            sdf = calculate_sdf(points.unsqueeze(0).type_as(vertices), vertices.unsqueeze(0), faces, scan_count=args.scans,
                                scan_resolution=args.scan_res)[0]
            sdf_grid = sdf.view(grid_resolution, grid_resolution, grid_resolution)
            out_path = os.path.join(os.path.dirname(mesh_path), 'sdf')
            torch.save(sdf, out_path)
        except Exception as e:
            print(f'Exception occured during processed mesh file {str(e)}')

