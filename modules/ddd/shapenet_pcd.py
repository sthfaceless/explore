import torch

import kaolin
from modules.ddd.util import normalize_points

def preprocessing_transform(inputs):
    mesh = inputs['mesh']
    vertices = mesh.vertices.unsqueeze(0)
    faces = mesh.faces

    # Some materials don't contain an RGB texture map, so we are considering the single value
    # to be a single pixel texture map (1, 3, 1, 1)
    # we apply a modulo 1 on the UVs because ShapeNet follows GL_REPEAT behavior (see: https://open.gl/textures)
    uvs = torch.nn.functional.pad(mesh.uvs.unsqueeze(0) % 1, (0, 0, 0, 1)) * 2. - 1.
    uvs[:, :, 1] = -uvs[:, :, 1]
    face_uvs_idx = mesh.face_uvs_idx
    materials_order = mesh.materials_order
    materials = [m['map_Kd'].permute(2, 0, 1).unsqueeze(0).float() / 255. if 'map_Kd' in m else
                 m['Kd'].reshape(1, 3, 1, 1)
                 for m in mesh.materials]

    nb_faces = faces.shape[0]
    num_consecutive_materials = \
        torch.cat([
            materials_order[1:, 1],
            torch.LongTensor([nb_faces])
        ], dim=0) - materials_order[:, 1]

    face_material_idx = kaolin.ops.batch.tile_to_packed(
        materials_order[:, 0],
        num_consecutive_materials
    ).squeeze(-1)
    mask = face_uvs_idx == -1
    face_uvs_idx[mask] = 0
    face_uvs = kaolin.ops.mesh.index_vertices_by_faces(
        uvs, face_uvs_idx
    )
    face_uvs[:, mask] = 0.

    outputs = {
        'vertices': vertices,
        'faces': faces,
        'face_areas': kaolin.ops.mesh.face_areas(vertices, faces),
        'face_uvs': face_uvs,
        'materials': materials,
        'face_material_idx': face_material_idx,
        'name': inputs['name']
    }

    return outputs


class SamplePointsTransform(object):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __call__(self, inputs):
        coords, face_idx, feature_uvs = kaolin.ops.mesh.sample_points(
            inputs['vertices'],
            inputs['faces'],
            num_samples=self.num_samples,
            areas=inputs['face_areas'],
            face_features=inputs['face_uvs']
        )
        coords = coords.squeeze(0)
        face_idx = face_idx.squeeze(0)
        feature_uvs = feature_uvs.squeeze(0)

        # Interpolate the RGB values from the texture map
        point_materials_idx = inputs['face_material_idx'][face_idx]
        all_point_colors = torch.zeros((self.num_samples, 3))
        for i, material in enumerate(inputs['materials']):
            mask = point_materials_idx == i
            point_color = torch.nn.functional.grid_sample(
                material,
                feature_uvs[mask].reshape(1, 1, -1, 2),
                mode='bilinear',
                align_corners=False,
                padding_mode='border')
            all_point_colors[mask] = point_color[0, :, 0, :].permute(1, 0)

        outputs = {
            'coords': coords,
            'face_idx': face_idx,
            'colors': all_point_colors,
            'name': inputs['name']
        }
        return outputs


class ShapenetPointClouds(torch.utils.data.Dataset):

    def __init__(self, shapenet_root, n_points=100000, cache_dir=None, categories=['plane'], noise=1e-2,
                 cache_scenes=512):
        super(ShapenetPointClouds, self).__init__()
        # Make ShapeNet dataset with preprocessing transform
        self.ds = kaolin.io.shapenet.ShapeNetV2(root=shapenet_root,
                                                categories=categories,
                                                train=False,
                                                split=0.0,
                                                with_materials=True,
                                                output_dict=True,
                                                transform=preprocessing_transform)

        # Cache the result of the preprocessing transform
        # and apply the sampling at runtime
        self.pc_ds = kaolin.io.dataset.CachedDataset(self.ds,
                                                     cache_dir=cache_dir,
                                                     save_on_disk=cache_dir is not None,
                                                     num_workers=torch.cuda.device_count() * 2,
                                                     transform=SamplePointsTransform(n_points),
                                                     cache_at_runtime=cache_scenes,
                                                     force_overwrite=True)
        self.noise = noise

    def __len__(self):
        return len(self.pc_ds)

    def __getitem__(self, idx):
        pcd = self.pc_ds[idx]['coords']
        pcd = normalize_points(pcd)
        return pcd
