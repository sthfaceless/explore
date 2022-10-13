# requires Pytorch3D installation
import argparse
import gc
import json
import math
import os
import shutil

import cv2
import numpy as np
import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    AmbientLights
)
from pytorch3d.renderer.blending import BlendParams
from tqdm import tqdm


def save_image(i, image, transform_matrices, batch_id, batch_size, out_path, out):
    P = np.array([[-1.0, 0, 0],
                  [0, 1.0, 0],
                  [0, 0, -1.0]], dtype=np.float32)

    w2c = transform_matrices[i].transpose()
    w2c[:3, :3] = P.transpose() @ w2c[:3, :3] @ P
    w2c[:3, 3] = P @ w2c[:3, 3]
    c2w = np.linalg.inv(w2c)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = (image * 255).astype(np.uint8)

    item_id = batch_id * batch_size + i
    cv2.imwrite(f'{out_path}/images/{item_id}.png', image)
    out['frames'].append({
        'file_path': f'./images/{item_id}.png',
        'transform_matrix': c2w
    })


def generate_images(n_views, shapenet_root, obj_class, obj_id, out_path, batch_size=16,
                    img_size=800, aabb=1.0, focal=1.5, random=False):
    # determine device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    obj_filename = shapenet_root + '/' + obj_class + '/' + obj_id
    obj_filename = f'{obj_filename}/models/model_normalized.obj'

    # load object mesh
    mesh = load_objs_as_meshes([obj_filename], device=device,
                               create_texture_atlas=True, load_textures=True, texture_atlas_size=4,
                               texture_wrap='repeat')

    raster_settings = RasterizationSettings(
        image_size=img_size,
        blur_radius=0.0,
        faces_per_pixel=2,
        bin_size=-1,
    )

    # create renderer
    R, T = look_at_view_transform(dist=2.0 * aabb, elev=0, azim=0)
    cameras = PerspectiveCameras(device=device, R=R, T=T, focal_length=focal, principal_point=((0.0, 0.0),))
    lights = AmbientLights(device=device, ambient_color=(1.0, 1.0, 1.0))
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            blend_params=BlendParams(background_color=(0.0, 0.0, 0.0))
        )
    )

    if random:
        elev = np.random.uniform(-85, 85, n_views).reshape(1, -1)
        azim = np.random.uniform(-180, 180, n_views).reshape(1, -1)
    else:
        elev = np.linspace(-85, 85, n_views).reshape(1, -1)
        azim = np.linspace(0, 6 * 360, n_views).reshape(1, -1)

    grid = np.concatenate([elev, azim], axis=0)

    os.makedirs(f'{out_path}/images', exist_ok=True)

    focal_screen = focal * img_size / 2
    out = {
        "camera_angle_x": math.atan(img_size / (focal_screen * 2)) * 2,
        "fl_x": focal_screen,
        "cx": img_size / 2,
        "cy": img_size / 2,
        "w": img_size,
        "h": img_size,
        "scale": 1,
        "black_transparent": True,
        "aabb_scale": aabb,
        "frames": [],
    }

    for batch_id in range(len(grid[0]) // batch_size + min(1, len(grid[0]) % batch_size)):

        R, T = look_at_view_transform(dist=2.0 * aabb, elev=grid[0][batch_id * batch_size: (batch_id + 1) * batch_size],
                                      azim=grid[1][batch_id * batch_size: (batch_id + 1) * batch_size])
        cameras = PerspectiveCameras(device=device, R=R, T=T, focal_length=focal, principal_point=((0.0, 0.0),))

        meshes = mesh.extend(min(batch_size, len(grid[0]) - batch_id * batch_size))
        with torch.no_grad():
            images = renderer(meshes, cameras=cameras, lights=lights).cpu().numpy()[..., :3]

        transform_matrices = cameras.get_world_to_view_transform().get_matrix().cpu().numpy()
        for i, image in enumerate(images):
            save_image(i, image, transform_matrices, batch_id, batch_size, out_path, out)

        del images, transform_matrices, meshes, cameras, R, T
        gc.collect()

    for f in out["frames"]:
        f["transform_matrix"] = f["transform_matrix"].tolist()

    with open(f'{out_path}/transforms.json', "w") as outfile:
        json.dump(out, outfile, indent=2)

    # if at least half of images wasn't rendered then remove this attempt
    if len(os.listdir(f'{out_path}/images')) < n_views // 2:
        shutil.rmtree(out_path)

    del out
    gc.collect()


def parse_args():
    parser = argparse.ArgumentParser(description="Generates images from shapenet in NeRF format")

    parser.add_argument("--shapenet", default="", help="Path to shapenet root")
    parser.add_argument("--nclasses", default=-1, help="Number of classes to generate dataset")
    parser.add_argument("--classes", default=[], nargs='+', help="Particular classes to generate dataset for")
    parser.add_argument("--out", default="", help="Output directory for images")
    parser.add_argument("--views", default=64, type=int, help="number of views generated with renderer")
    parser.add_argument("--picture_res", default=128, type=int, help="generated pictures resolution")
    parser.add_argument("--batch_size", default=32, type=int, help="picture generation batch")
    parser.add_argument("--aabb", default=1.0, type=float, help="Scene aabb scale")
    parser.add_argument("--focal", default=2.0, type=float, help="Camera focal")
    parser.add_argument("--random", action='store_true')
    parser.set_defaults(clearml=False)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    shapenet_root = args.shapenet
    out_dir = args.out

    obj_classes = [file for file in os.listdir(shapenet_root) if os.path.isdir(os.path.join(shapenet_root, file))]
    if args.classes:
        obj_classes = [cls for cls in args.classes if cls in obj_classes]
    elif args.nclasses > 0:
        obj_classes = obj_classes[:args.nclasses]

    for obj_class in obj_classes:
        class_root = os.path.join(shapenet_root, obj_class)
        obj_ids = [file for file in os.listdir(class_root) if os.path.isdir(os.path.join(class_root, file))]
        for obj_id in tqdm(obj_ids):
            obj_root = os.path.join(class_root, obj_id)
            obj_out = os.path.join(out_dir, obj_class, obj_id)
            os.makedirs(obj_out, exist_ok=True)
            try:
                generate_images(n_views=args.views, shapenet_root=shapenet_root, obj_class=obj_class, obj_id=obj_id,
                                out_path=obj_out, batch_size=args.batch_size, img_size=args.picture_res,
                                aabb=args.aabb, focal=args.focal, random=args.random)
                # check that texture was applied
                img = cv2.imread(os.path.join(obj_out, 'images', f'{args.views // 2}.png'))
                colors = np.unique(np.mean(img.astype(np.float32), axis=-1).astype(np.uint8)).tolist()
                if len(colors) <= 3:
                    shutil.rmtree(obj_out, ignore_errors=True)
            except Exception as e:
                print(e)
                shutil.rmtree(obj_out, ignore_errors=True)

        print(f"Passed class {obj_class}")
