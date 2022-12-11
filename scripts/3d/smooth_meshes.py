import os
from argparse import ArgumentParser

from tqdm import tqdm

import kaolin
from modules.ddd.util import *


def parse_args():
    parser = ArgumentParser(
        description="Smooths shapenet meshes with voxelizing and devoxelizing + laplacian smoothing")

    parser.add_argument("--shapenet", default="", help="Path to shapenet root")
    parser.add_argument("--suffix", default=".sm", help="Suffix for smoothed files")
    parser.add_argument("--classes", default=['02691156'], nargs='+', help="Particular classes to generate dataset for")
    parser.add_argument("--res", default=100, type=int, help="voxelize resolution")
    parser.add_argument("--smooths", default=1, type=int, help="number of laplacian smoothing")

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

    # determine device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    grid_res = args.res
    for mesh_path in tqdm(items, desc='Processed mesh files'):
        try:
            vertices, faces = read_obj(mesh_path)
            vertices, faces = vertices.to(device), faces.to(device)
            _vertices = (normalize_points(vertices) + 1) / 2
            # convert triangle mesh to voxels
            voxels = kaolin.ops.conversions.trianglemeshes_to_voxelgrids(_vertices.unsqueeze(0), faces, grid_res)
            # remove extra non surface voxels
            voxels = kaolin.ops.voxelgrid.extract_surface(voxels, mode='wide')
            # return to mesh
            _vertices, _faces = kaolin.ops.conversions.voxelgrids_to_trianglemeshes(voxels)
            _vertices, _faces = (_vertices[0] / (grid_res) - 0.5) * 2, _faces[0]
            for _ in range(args.smooths):
                _vertices = kaolin.metrics.trianglemesh.uniform_laplacian_smoothing(_vertices.unsqueeze(0), _faces)[0]
            out_path = mesh_path + args.suffix
            save_obj(out_path, _vertices, _faces)
        except Exception as e:
            print(f'Exception occured during processed mesh file {str(e)}')
