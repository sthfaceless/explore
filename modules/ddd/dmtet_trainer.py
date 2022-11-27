from random import shuffle

import numpy as np
import pytorch_lightning as pl

import kaolin
from modules.common.trainer import SimpleLogger
from modules.ddd.model import *
from render_util import render_mesh


class PCD2Mesh(pl.LightningModule):

    def __init__(self, dataset=None, clearml=None, timelapse=None, train_rate=0.8, grid_resolution=64,
                 learning_rate=1e-4,
                 steps_schedule=(1000, 20000, 50000, 100000), min_lr_rate=1.0, encoder_dims=(64, 128, 256),
                 sdf_dims=(256, 256, 128, 64), disc_dims=(32, 64, 128, 256), sdf_clamp=0.03,
                 n_volume_division=1, n_surface_division=1, chamfer_samples=5000,
                 sdf_weight=0.4, gcn_dims=(256, 128), gcn_hidden=(128, 64), delta_weight=1.0, disc_weight=10,
                 curvature_threshold=torch.pi / 16, curvature_samples=10, disc_sdf_grid=16, disc_sdf_scale=0.1,
                 disc_v_noise=1e-3, chamfer_weight=500, normal_weight=1e-6,
                 encoder_grids=(32, 16, 8), batch_size=16, pe_powers=16, noise=0.02):
        super(PCD2Mesh, self).__init__()
        self.save_hyperparameters(ignore=['dataset', 'clearml', 'timelapse'])

        self.timelapse = timelapse
        self.simple_logger = SimpleLogger(clearml)
        self.dataset = dataset
        if dataset is not None:
            idxs = list(range(len(dataset)))
            shuffle(idxs)
            n_train_items = int(train_rate * len(dataset))
            self.train_idxs = idxs[:n_train_items]
            self.val_idxs = idxs[n_train_items:]

        self.steps_schedule = steps_schedule
        self.learning_rate = learning_rate
        self.min_lr_rate = min_lr_rate
        self.batch_size = batch_size

        tet_vertexes, tetrahedras = get_tetrahedras_grid(grid_resolution)
        self.register_buffer('tet_vertexes', tet_vertexes)
        self.register_buffer('tetrahedras', tetrahedras)
        self.n_tetrahedra_vertexes = len(tet_vertexes)

        self.pe_powers = pe_powers
        self.input_dim = pe_powers * 3 + 3
        self.noise = noise
        self.sdf_clamp = sdf_clamp
        self.n_volume_division = n_volume_division
        self.n_surface_division = n_surface_division

        self.sdf_weight = sdf_weight
        self.delta_weight = delta_weight
        self.disc_weight = disc_weight
        self.chamfer_weight = chamfer_weight
        self.chamfer_samples = chamfer_samples
        self.normal_weight = normal_weight

        self.curvature_threshold = curvature_threshold
        self.curvature_samples = curvature_samples
        self.disc_sdf_grid = disc_sdf_grid
        self.disc_sdf_scale = disc_sdf_scale
        self.disc_v_noise = disc_v_noise

        self.sdf_points_encoder = MultiPointVoxelCNN(input_dim=self.input_dim, dims=encoder_dims, grids=encoder_grids,
                                                     do_points_map=False)
        self.sdf_model = SimpleMLP(input_dim=sum(encoder_dims), out_dim=1 + sdf_dims[-1], hidden_dims=sdf_dims)

        self.ref_points_encoder = MultiPointVoxelCNN(input_dim=self.input_dim, dims=encoder_dims,
                                                     grids=encoder_grids, do_points_map=False)
        self.ref1 = TetConv(input_dim=3 + 1 + sdf_dims[-1] + sum(encoder_dims), out_dim=3 + 1 + sdf_dims[-1],
                            gcn_dims=gcn_dims, mlp_dims=gcn_hidden)
        self.ref2 = TetConv(input_dim=3 + 1 + sdf_dims[-1] + sum(encoder_dims), out_dim=3 + 1 + sdf_dims[-1],
                            gcn_dims=gcn_dims, mlp_dims=gcn_hidden)
        self.surface_ref = MeshConv(input_dim=3 + sum(encoder_dims), out_dim=4, gcn_dims=gcn_dims,
                                    mlp_dims=gcn_hidden)

        self.sdf_disc = SDFDiscriminator(input_dim=1 + sum(encoder_dims), hidden_dims=disc_dims)

    def get_mesh_sdf(self, bpoints, bvertices, bfaces):
        bsdf = []
        for points, vertices, faces in zip(bpoints, bvertices, bfaces):
            points, vertices = points.unsqueeze(0), vertices.unsqueeze(0)
            face_vertices = kaolin.ops.mesh.index_vertices_by_faces(vertices, faces)
            dists, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(points, face_vertices)
            signs = kaolin.ops.mesh.check_sign(vertices, faces, points).squeeze(0)
            sdf = dists.squeeze(0) * (signs.type_as(dists) - 0.5) * 2 * (-1)  # inside is -1
            bsdf.append(sdf)
        return bsdf

    def calculate_sdf_loss(self, btet_vertexes, btet_sdf, bvertices, bfaces):
        btrue_sdf = self.get_mesh_sdf(btet_vertexes, bvertices, bfaces)
        sdf_loss = None
        for tet_vertexes, tet_sdf, true_sdf in zip(btet_vertexes, btet_sdf, btrue_sdf):
            true_sdf = torch.clamp(true_sdf, min=-self.sdf_clamp, max=self.sdf_clamp)
            loss = torch.sum((tet_sdf - true_sdf) ** 2)
            if sdf_loss is None:
                sdf_loss = loss
            else:
                sdf_loss += loss
        sdf_loss = sdf_loss / len(btet_vertexes)
        return sdf_loss

    @torch.no_grad()
    def get_mesh(self, pcd_noised, n_volume_division=None, n_surface_division=None):
        pe_features = [torch.cat([_pcd_noised, get_positional_encoding(_pcd_noised, self.pe_powers * 3)], dim=-1)
                       for _pcd_noised in pcd_noised]

        sdf_grids = self.sdf_points_encoder.bvoxelize(pcd_noised, pe_features)
        tet_vertexes = [self.tet_vertexes for _ in range(len(pcd_noised))]
        pos_vertex_features = self.sdf_points_encoder.bdevoxelize(tet_vertexes, sdf_grids)
        tet_out = self.sdf_model.bforward(pos_vertex_features)
        tet_sdf, tet_features = [out[:, 0] for out in tet_out], [out[:, 1:] for out in tet_out]

        tetrahedras = [self.tetrahedras for _ in range(len(tet_vertexes))]
        tet_vertexes, tetrahedras, tet_sdf, tet_features, extra_vertexes, extra_sdf \
            = get_surface_tetrahedras(tet_vertexes, tetrahedras, tet_sdf, tet_features)
        # if we're training only on SDF
        if self.global_step < self.steps_schedule[0]:
            mesh_vertices, mesh_faces = [], []
            for vertexes, tets, sdf in zip(tet_vertexes, tetrahedras, tet_sdf):
                vertices, faces = kaolin.ops.conversions.marching_tetrahedra(vertexes.unsqueeze(0), tets,
                                                                             sdf.unsqueeze(0))
                mesh_vertices.append(vertices[0])
                mesh_faces.append(faces[0])
        # add volume subdivision and refinement
        if self.global_step >= self.steps_schedule[0]:
            # first refinement step
            grids = self.ref_points_encoder.bvoxelize(pcd_noised, pe_features)
            pos_features = self.ref_points_encoder.bdevoxelize(tet_vertexes, grids)
            delta_v, delta_s, tet_features = self.ref1([
                torch.cat([v, s.unsqueeze(-1), f, pf], dim=-1)
                for v, s, f, pf in zip(tet_vertexes, tet_sdf, tet_features, pos_features)], tetrahedras)
            # update vertexes
            tet_vertexes = [vertexes + delta_vertexes for vertexes, delta_vertexes in zip(tet_vertexes, delta_v)]
            tet_sdf = [sdf + delta_sdf for sdf, delta_sdf in zip(tet_sdf, delta_s)]

            if n_volume_division is None:
                n_volume_division = self.n_volume_division
            for div_id in range(n_volume_division):
                new_tetrahedras, new_tet_vertexes, new_tet_sdf, new_tet_features = [], [], [], []
                for vertexes, tets, sdf, features in zip(tet_vertexes, tetrahedras, tet_sdf, tet_features):
                    # volume subdivision
                    _vertexes, _tets, _out = kaolin.ops.mesh.subdivide_tetmesh(
                        vertexes.unsqueeze(0), tets, torch.cat([sdf.unsqueeze(-1), features], dim=-1).unsqueeze(0))
                    new_tet_vertexes.append(_vertexes.squeeze(0))
                    new_tetrahedras.append(_tets)
                    new_tet_sdf.append(_out[0, :, 0])
                    new_tet_features.append(_out[0, :, 1:])

                # take only surface tetrahedras and add sdf loss to others
                tet_vertexes, tetrahedras, tet_sdf, tet_features, extra_vertexes, extra_sdf \
                    = get_surface_tetrahedras(new_tet_vertexes, new_tetrahedras, new_tet_sdf, new_tet_features)

            # additional volume refinement step
            pos_features = self.ref_points_encoder.bdevoxelize(tet_vertexes, grids)
            delta_v, delta_s, tet_features = self.ref2([
                torch.cat([v, s.unsqueeze(-1), f, pf], dim=-1)
                for v, s, f, pf in zip(tet_vertexes, tet_sdf, tet_features, pos_features)], tetrahedras)
            # update vertexes
            tet_vertexes = [vertexes + delta_vertexes for vertexes, delta_vertexes in zip(tet_vertexes, delta_v)]
            tet_sdf = [sdf + delta_sdf for sdf, delta_sdf in zip(tet_sdf, delta_s)]

            mesh_vertices, mesh_faces = [], []
            for vertexes, tets, sdf in zip(tet_vertexes, tetrahedras, tet_sdf):
                vertices, faces = kaolin.ops.conversions.marching_tetrahedra(vertexes.unsqueeze(0), tets,
                                                                             sdf.unsqueeze(0))
                mesh_vertices.append(vertices[0])
                mesh_faces.append(faces[0])

        # surface subdivision
        if n_surface_division is None:
            n_surface_division = self.n_surface_division
        if self.global_step >= self.steps_schedule[2]:
            pos_features = self.ref_points_encoder.bdevoxelize(mesh_vertices, grids)
            delta_v, alphas = self.surface_ref([torch.cat([v, pf], dim=-1)
                                                for v, pf in zip(mesh_vertices, pos_features)], mesh_faces)
            mesh_vertices = [vertices + delta_vertices for vertices, delta_vertices in zip(mesh_vertices, delta_v)]
            new_mesh_vertices, new_mesh_faces = [], []
            for vertices, faces, alpha in zip(mesh_vertices, mesh_faces, alphas):
                vertices, faces = kaolin.ops.mesh.subdivide_trianglemesh(
                    vertices.unsqueeze(0), faces, iterations=n_surface_division, alpha=alpha.unsqueeze(0))
                new_mesh_vertices.append(vertices[0])
                new_mesh_faces.append(faces)
            mesh_vertices, mesh_faces = new_mesh_vertices, new_mesh_faces

        return mesh_vertices, mesh_faces

    def mesh_step(self, bvertices, bfaces, n_volume_division=None, n_surface_division=None):
        # make features based on conditional point cloud and predict initial sdf
        bpcd, bpcd_faces, pcd_noised, pe_features = [], [], [], []
        for vertices, faces in zip(bvertices, bfaces):
            pcd, faces_ids = kaolin.ops.mesh.sample_points(vertices.unsqueeze(0), faces, self.chamfer_samples)
            pcd = pcd.squeeze(0)
            bpcd.append(pcd)
            bpcd_faces.append(faces_ids.squeeze(0))
            _pcd_noised = pcd + torch.randn_like(pcd) * self.noise
            pcd_noised.append(_pcd_noised)
            pe_features.append(torch.cat([_pcd_noised,
                                          get_positional_encoding(_pcd_noised, self.pe_powers * 3)], dim=-1))

        sdf_grids = self.sdf_points_encoder.bvoxelize(pcd_noised, pe_features)
        tet_vertexes = [self.tet_vertexes for _ in range(len(bpcd))]
        pos_vertex_features = self.sdf_points_encoder.bdevoxelize(tet_vertexes, sdf_grids)
        tet_out = self.sdf_model.bforward(pos_vertex_features)
        tet_sdf, tet_features = [out[:, 0] for out in tet_out], [out[:, 1:] for out in tet_out]

        out = {
            'sdf_grids': sdf_grids
        }
        # if we're training only on SDF
        if self.global_step < self.steps_schedule[0]:
            true_sdf = self.get_mesh_sdf(tet_vertexes, bvertices, bfaces)
            out['sdf_loss'] = torch.mean(torch.stack([torch.sum((t_sdf - p_sdf) ** 2)
                                                      for t_sdf, p_sdf in zip(true_sdf, tet_sdf)], dim=0), dim=0)
            out['loss'] = out['sdf_loss']
        # add volume subdivision and refinement
        if self.global_step >= self.steps_schedule[0]:
            # calc sdf loss for extra tetrahedras
            tetrahedras = [self.tetrahedras for _ in range(len(tet_vertexes))]
            tet_vertexes, tetrahedras, tet_sdf, tet_features, extra_vertexes, extra_sdf \
                = get_surface_tetrahedras(tet_vertexes, tetrahedras, tet_sdf, tet_features)
            out['sdf_loss'] = self.calculate_sdf_loss(extra_vertexes, extra_sdf, bvertices, bfaces)
            # first refinement step
            grids = self.ref_points_encoder.bvoxelize(pcd_noised, pe_features)
            pos_features = self.ref_points_encoder.bdevoxelize(tet_vertexes, grids)
            delta_v, delta_s, tet_features = self.ref1([
                torch.cat([v, s.unsqueeze(-1), f, pf], dim=-1)
                for v, s, f, pf in zip(tet_vertexes, tet_sdf, tet_features, pos_features)], tetrahedras)
            # update vertexes
            tet_vertexes = [vertexes + delta_vertexes for vertexes, delta_vertexes in zip(tet_vertexes, delta_v)]
            tet_sdf = [sdf + delta_sdf for sdf, delta_sdf in zip(tet_sdf, delta_s)]
            # add regularization on vertex delta
            out['delta_vertex'] = torch.mean(torch.stack(
                [torch.sum(delta_vertexes ** 2, dim=[1, 2]) for delta_vertexes in delta_v], dim=0), dim=0)

            if n_volume_division is None:
                n_volume_division = self.n_volume_division
            for div_id in range(n_volume_division):
                new_tetrahedras, new_tet_vertexes, new_tet_sdf, new_tet_features = [], [], [], []
                for vertexes, tets, sdf, features in zip(tet_vertexes, tetrahedras, tet_sdf, tet_features):
                    # volume subdivision
                    _vertexes, _tets, _out = kaolin.ops.mesh.subdivide_tetmesh(
                        vertexes.unsqueeze(0), tets, torch.cat([sdf.unsqueeze(-1), features], dim=-1).unsqueeze(0))
                    new_tet_vertexes.append(_vertexes.squeeze(0))
                    new_tetrahedras.append(_tets)
                    new_tet_sdf.append(_out[0, :, 0])
                    new_tet_features.append(_out[0, :, 1:])

                # take only surface tetrahedras and add sdf loss to others
                tet_vertexes, tetrahedras, tet_sdf, tet_features, extra_vertexes, extra_sdf \
                    = get_surface_tetrahedras(new_tet_vertexes, new_tetrahedras, new_tet_sdf, new_tet_features)
                out['sdf_loss'] += self.calculate_sdf_loss(extra_vertexes, extra_sdf, bvertices, bfaces)

            # additional volume refinement step
            pos_features = self.ref_points_encoder.bdevoxelize(tet_vertexes, grids)
            delta_v, delta_s, tet_features = self.ref2([
                torch.cat([v, s.unsqueeze(-1), f, pf], dim=-1)
                for v, s, f, pf in zip(tet_vertexes, tet_sdf, tet_features, pos_features)], tetrahedras)
            # update vertexes
            tet_vertexes = [vertexes + delta_vertexes for vertexes, delta_vertexes in zip(tet_vertexes, delta_v)]
            tet_sdf = [sdf + delta_sdf for sdf, delta_sdf in zip(tet_sdf, delta_s)]
            # add regularization on vertex delta
            out['delta_vertex'] += torch.mean(torch.stack(
                [torch.sum(delta_vertexes ** 2, dim=[1, 2]) for delta_vertexes in delta_v], dim=0), dim=0)

            mesh_vertices, mesh_faces = [], []
            for vertexes, tets, sdf in zip(tet_vertexes, tetrahedras, tet_sdf):
                vertices, faces = kaolin.ops.conversions.marching_tetrahedra(vertexes.unsqueeze(0), tets,
                                                                             sdf.unsqueeze(0))
                mesh_vertices.append(vertices[0])
                mesh_faces.append(faces[0])

            out['loss'] += out['delta_vertex'] * self.delta_weight
            out['loss'] += out['sdf_loss'] * self.sdf_weight

        # surface subdivision
        if n_surface_division is None:
            n_surface_division = self.n_surface_division
        if self.global_step >= self.steps_schedule[2]:
            pos_features = self.ref_points_encoder.bdevoxelize(mesh_vertices, grids)
            delta_v, alphas = self.surface_ref([torch.cat([v, pf], dim=-1)
                                                for v, pf in zip(mesh_vertices, pos_features)], mesh_faces)
            mesh_vertices = [vertices + delta_vertices for vertices, delta_vertices in zip(mesh_vertices, delta_v)]
            new_mesh_vertices, new_mesh_faces = [], []
            for vertices, faces, alpha in zip(mesh_vertices, mesh_faces, alphas):
                vertices, faces = kaolin.ops.mesh.subdivide_trianglemesh(
                    vertices.unsqueeze(0), faces, iterations=n_surface_division, alpha=alpha.unsqueeze(0))
                new_mesh_vertices.append(vertices[0])
                new_mesh_faces.append(faces)
            mesh_vertices, mesh_faces = new_mesh_vertices, new_mesh_faces

        out['mesh_vertices'] = mesh_vertices
        out['mesh_faces'] = mesh_faces

        if self.global_step >= self.steps_schedule[0]:
            chamfer_loss, normal_loss = None, None
            for pred_vertices, pred_faces, true_vertices, true_faces, true_pcd, true_faces_ids in zip(mesh_vertices,
                                                                                                      mesh_faces,
                                                                                                      bvertices, bfaces,
                                                                                                      bpcd, bpcd_faces):
                # calculate pcd chamfer loss
                pred_pcd, faces_ids = kaolin.ops.mesh.sample_points(
                    pred_vertices.unsqueeze(0), pred_faces, self.chamfer_samples)
                loss = kaolin.metrics.pointcloud.chamfer_distance(pred_pcd, true_pcd.unsqueeze(0),
                                                                  w1=len(pred_pcd), w2=len(true_pcd))
                if chamfer_loss is None:
                    chamfer_loss = loss
                else:
                    chamfer_loss += loss
                # calculate mesh normal loss with finding normal for each point in pcd
                pred_normals = kaolin.ops.mesh.face_normals(
                    kaolin.ops.mesh.index_vertices_by_faces(pred_vertices.unsqueeze(0), pred_faces),
                    unit=True).squeeze(0)
                pred_normals = pred_normals[faces_ids.squeeze(0)]
                # finding normal of the closest points
                dist, point_ids = kaolin.metrics.pointcloud.sided_distance(pred_pcd, true_pcd.unsqueeze(0))
                true_normals = kaolin.ops.mesh.face_normals(
                    kaolin.ops.mesh.index_vertices_by_faces(true_vertices.unsqueeze(0), true_faces),
                    unit=True).squeeze(0)
                true_normals = true_normals[true_faces_ids[point_ids.squeeze(0)]]
                loss = torch.sum(1 - torch.abs(
                    torch.matmul(pred_normals.unsqueeze(1), true_normals.unsqueeze(2)).view(-1)))
                if normal_loss is None:
                    normal_loss = loss
                else:
                    normal_loss += loss

            out['chamfer_loss'] = chamfer_loss / len(bpcd)
            out['normal_loss'] = normal_loss / len(bpcd)
            out['loss'] += out['chamfer_loss'] * self.chamfer_weight + out['normal_loss'] * self.normal_weight

        return out

    def shared_step(self, bvertices, bfaces, optimizer_idx, n_volume_division=None, n_surface_division=None):

        if optimizer_idx == 1:  # for optimizing discriminator
            with torch.no_grad():
                out = self.mesh_step(bvertices, bfaces, n_volume_division=n_volume_division,
                                     n_surface_division=n_surface_division)
        else:
            out = self.mesh_step(bvertices, bfaces, n_volume_division=n_volume_division,
                                 n_surface_division=n_surface_division)

        # discriminator
        if self.global_step >= self.steps_schedule[1]:
            mesh_vertices, mesh_faces = out['mesh_vertices'], out['mesh_faces']
            curvatures = [calculate_gaussian_curvature(vertices, faces) for vertices, faces in zip(bvertices, bfaces)]
            indexes = [torch.arange(len(curvature)).type_as(curvature).long()[curvature >= self.curvature_threshold]
                       for curvature in curvatures]
            indexes = [index[torch.randint(0, len(index), (self.curvature_samples,)).type_as(index).long()]
                       for index in indexes]
            grid = torch.stack(torch.meshgrid(
                torch.linspace(-1, 1, self.disc_sdf_grid).type_as(mesh_vertices),
                torch.linspace(-1, 1, self.disc_sdf_grid).type_as(mesh_vertices),
                torch.linspace(-1, 1, self.disc_sdf_grid).type_as(mesh_vertices), ), dim=-1) * self.disc_sdf_scale
            bgrids = [(vertices[index] + torch.randn_like(vertices[index]) * self.disc_v_noise)
                      .view(self.curvature_samples, 1, 1, 1, 3) + grid.unsqueeze(0)
                      for vertices, index in zip(bvertices, indexes)]
            pred_sdf_features, true_sdf_features = [], []
            sdf_grids = out['sdf_grids']
            for item_idx, pred_vertices, pred_faces, true_vertices, true_faces, grids in zip(range(len(mesh_vertices)),
                                                                                             mesh_vertices, mesh_faces,
                                                                                             bvertices, bfaces, bgrids):

                for grid in grids:
                    grid_points = grid.view(self.disc_sdf_grid ** 3, 3)
                    pos_features = self.sdf_points_encoder.bdevoxelize([grid_points], [sdf_grids[item_idx]])[0]. \
                        view(1, self.disc_sdf_grid, self.disc_sdf_grid, self.disc_sdf_grid)
                    pred_sdf_features.append(torch.cat([
                        self.get_mesh_sdf([grid_points], [pred_vertices], [pred_faces])[0]
                        .view(1, self.disc_sdf_grid, self.disc_sdf_grid, self.disc_sdf_grid), pos_features], dim=1))
                    true_sdf_features.append(torch.cat([
                        self.get_mesh_sdf([grid_points], [true_vertices], [true_faces])[0]
                        .view(1, self.disc_sdf_grid, self.disc_sdf_grid, self.disc_sdf_grid), pos_features], dim=1))
            pred_sdf_features, true_sdf_features = torch.stack(pred_sdf_features, dim=0), torch.stack(true_sdf_features,
                                                                                                      dim=0)
            disc_preds = self.sdf_disc(torch.cat([pred_sdf_features, true_sdf_features], dim=0))
            out['adv_loss'] = torch.mean((disc_preds[:len(pred_sdf_features)] - 1) ** 2)
            out['loss'] += out['adv_loss'] * self.disc_weight
            out['disc_loss'] = torch.mean(disc_preds[:len(pred_sdf_features)] ** 2) \
                               + torch.mean((1 - disc_preds[len(pred_sdf_features):]) ** 2)

        return out

    def on_validation_epoch_end(self):

        data_iter = iter(self.trainer.val_dataloaders[0])
        batch = next(data_iter)
        pcd = batch['pcd']
        pcd_noised = [_pcd + torch.randn_like(_pcd) * self.noise for _pcd in pcd]
        mesh_vertices, mesh_faces = self.get_mesh(pcd_noised)

        self.timelapse.add_pointcloud_batch(category='input_noised',
                                            pointcloud_list=[_pcd.cpu() for _pcd in pcd],
                                            points_type="usd_geom_points",
                                            iteration=self.global_step)
        self.timelapse.add_pointcloud_batch(category='input_noised',
                                            pointcloud_list=[_pcd.cpu() for _pcd in pcd_noised],
                                            points_type="usd_geom_points",
                                            iteration=self.global_step)
        self.timelapse.add_mesh_batch(
            iteration=self.global_step,
            category='predicted_mesh',
            vertices_list=[v.cpu() for v in mesh_vertices],
            faces_list=[f.cpu() for f in mesh_faces]
        )

        images = np.stack([render_mesh(v.detach().cpu().numpy(), f.detach().cpu().numpy())
                           for v, f in zip(mesh_vertices, mesh_faces)], axis=0)
        b, views, h, w = images.shape
        images = images.reshape((b, 2, views // 2, h, w)).moveaxis(1, 3).reshape((b, 2 * h, views // 2 * w))
        images = [images[idx] for idx in range(len(images))]
        self.simple_logger.log_images(images, 'rendered_mesh', self.global_step)

    def training_step(self, batch, batch_idx, optimizer_idx):
        # generator optimization
        if optimizer_idx == 0:
            out = self.shared_step(batch['vertices'], batch['faces'], optimizer_idx)
            return out['loss']
        # discriminator optimization
        if optimizer_idx == 1:
            out = self.shared_step(batch['vertices'], batch['faces'], optimizer_idx)
            return out['disc_loss']

    def configure_optimizers(self):
        gen_optimizer = torch.optim.Adam(lr=self.learning_rate,
                                         params=list(self.sdf_points_encoder.parameters())
                                                + list(self.sdf_model.parameters())
                                                + list(self.ref_points_encoder.parameters())
                                                + list(self.ref1.parameters()) + list(self.ref2.parameters())
                                                + list(self.surface_ref.parameters()),
                                         betas=(0.9, 0.99))
        gen_scheduler = torch.optim.lr_scheduler.OneCycleLR(gen_optimizer, max_lr=self.learning_rate,
                                                            pct_start=self.steps_schedule[0] / self.steps_schedule[-1],
                                                            div_factor=2.0,
                                                            final_div_factor=1 / (2.0 * self.min_lr_rate),
                                                            total_steps=self.steps_schedule[-1])
        gen_scheduler = {
            'scheduler': gen_scheduler,
            'interval': 'step'
        }
        dis_optimizer = torch.optim.Adam(lr=self.learning_rate, params=self.sdf_disc.parameters(), betas=(0.9, 0.99))
        dis_steps = self.steps_schedule[-1] - self.steps_schedule[1]
        dis_scheduler = torch.optim.lr_scheduler.OneCycleLR(dis_optimizer, max_lr=self.learning_rate,
                                                            pct_start=self.steps_schedule[0] / dis_steps,
                                                            div_factor=2.0,
                                                            final_div_factor=1 / (2.0 * self.min_lr_rate),
                                                            total_steps=dis_steps)
        dis_scheduler = {
            'scheduler': dis_scheduler,
            'interval': 'step'
        }
        return [gen_optimizer, dis_optimizer], [gen_scheduler, dis_scheduler]

    def train_dataloader(self):
        train_items = RandomIndexedListWrapper(self.dataset, self.train_idxs)
        return torch.utils.data.DataLoader(train_items, batch_size=self.batch_size, shuffle=False,
                                           num_workers=2 * torch.cuda.device_count(), collate_fn=lambda lst: lst,
                                           pin_memory=True, drop_last=False, prefetch_factor=2)

    def val_dataloader(self):
        val_items = RandomIndexedListWrapper(self.dataset, self.val_idxs)
        return torch.utils.data.DataLoader(val_items, batch_size=self.batch_size, shuffle=False,
                                           num_workers=2 * torch.cuda.device_count(), collate_fn=lambda lst: lst,
                                           pin_memory=True, drop_last=False, prefetch_factor=2)
