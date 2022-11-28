import torch
# # rendering components
from pytorch3d.renderer import (
    Textures, look_at_view_transform,
    RasterizationSettings, MeshRenderer, MeshRasterizer,
    HardPhongShader, BlendParams,
    SoftPhongShader, AmbientLights, PerspectiveCameras
)
# # datastructures
from pytorch3d.structures import Meshes
# # 3D transformations functions
import numpy as np


# io utils

def render_mesh(verts, faces, img_size=256, n_views=8, device=torch.device('cpu'), random=False, focal=1.5):
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = Textures(verts_rgb=verts_rgb.to(device))
    mesh = Meshes(
        verts=[verts.to(device)],
        faces=[faces.to(device)],
        textures=textures
    )
    raster_settings = RasterizationSettings(
        image_size=img_size,
        blur_radius=0.0,
        faces_per_pixel=2,
        bin_size=-1,
    )

    if random:
        elev = np.random.uniform(-85, 85, n_views).reshape(1, -1)
        azim = np.random.uniform(-180, 180, n_views).reshape(1, -1)
    else:
        elev = np.linspace(-85, 85, n_views).reshape(1, -1)
        azim = np.linspace(0, 6 * 360, n_views).reshape(1, -1)
    # create renderer
    R, T = look_at_view_transform(dist=2.0, elev=elev, azim=azim)
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
            blend_params=BlendParams(background_color=(0.99, 0.99, 0.99))
        )
    )

    meshes = mesh.extend(n_views)
    with torch.no_grad():
        images = renderer(meshes, cameras=cameras, lights=lights).cpu().numpy()[..., :3]
    images = (images * 255).astype(np.uint8)
    return images
