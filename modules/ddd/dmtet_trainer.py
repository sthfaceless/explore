import functools
from random import shuffle

import pytorch_lightning as pl

import kaolin
from modules.common.trainer import SimpleLogger
from modules.ddd.model import *
from modules.ddd.render_util import render_mesh


class PCD2Mesh(pl.LightningModule):

    def __init__(self, dataset=None, clearml=None, timelapse=None, train_rate=0.8, grid_resolution=64,
                 learning_rate=1e-4, debug_interval=-1,
                 steps_schedule=(1000, 20000, 50000, 100000), min_lr_rate=1.0, encoder_dims=(64, 128, 256),
                 sdf_dims=(256, 256, 128, 64), disc_dims=(32, 64, 128, 256), sdf_clamp=0.03,
                 n_volume_division=1, n_surface_division=1, chamfer_samples=5000,
                 sdf_weight=0.4, gcn_dims=(256, 128), gcn_hidden=(128, 64), delta_weight=1.0, disc_weight=10,
                 curvature_threshold=torch.pi / 16, curvature_samples=10, disc_sdf_grid=16, disc_sdf_scale=0.1,
                 disc_v_noise=1e-3, chamfer_weight=500, normal_weight=1e-6,
                 encoder_grids=(32, 16, 8), batch_size=16, pe_powers=16, noise=0.02):
        super(PCD2Mesh, self).__init__()
        self.save_hyperparameters(ignore=['dataset', 'clearml', 'timelapse'])

        self.debug = debug_interval > 0
        self.debug_interval = debug_interval
        self.debug_state = False

        self.timelapse = timelapse
        self.lg = SimpleLogger(clearml)
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
        self.pos_features = sum(encoder_dims)
        self.noise = noise
        self.sdf_clamp = sdf_clamp
        self.n_volume_division = n_volume_division
        self.n_surface_division = n_surface_division
        self.encoder_grids = encoder_grids

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
        self.sdf_model = SimpleMLP(input_dim=self.pos_features, out_dim=1 + sdf_dims[-1], hidden_dims=sdf_dims)

        self.ref_points_encoder = MultiPointVoxelCNN(input_dim=self.input_dim, dims=encoder_dims,
                                                     grids=encoder_grids, do_points_map=False)
        self.ref1 = TetConv(input_dim=3 + 1 + sdf_dims[-1] + self.pos_features, out_dim=3 + 1 + sdf_dims[-1],
                            gcn_dims=gcn_dims, mlp_dims=gcn_hidden)
        self.ref2 = TetConv(input_dim=3 + 1 + sdf_dims[-1] + self.pos_features, out_dim=3 + 1 + sdf_dims[-1],
                            gcn_dims=gcn_dims, mlp_dims=gcn_hidden)
        self.surface_ref = MeshConv(input_dim=3 + self.pos_features, out_dim=4, gcn_dims=gcn_dims,
                                    mlp_dims=gcn_hidden)

        self.sdf_disc = SDFDiscriminator(input_dim=1 + self.pos_features, hidden_dims=disc_dims)

        # state variables
        self.volume_refinement = False
        self.adversarial_training = False
        self.surface_subdivision = False

        if self.debug:
            print(self.lg.log_tensor(tet_vertexes, 'Tetrahedras grid vertices'))
            print(self.lg.log_tensor(tetrahedras, 'Tetrahedras grid faces'))
            self.lg.log_scatter3d(tn(tet_vertexes[:, 0]), tn(tet_vertexes[:, 1]), tn(tet_vertexes[:, 2]),
                                  'tetrahedras_vertex')
            _v, _f = tetrahedras2mesh(tet_vertexes, tetrahedras)
            indexes = torch.randint(low=0, high=len(_f), size=(50000,)).type_as(_f).long()
            self.lg.log_mesh(tn(_v.cpu()), tn(_f[indexes]), 'tetrahedras_grid')

    def get_mesh_sdf(self, points, vertices, faces):
        if len(faces) == 0 or len(points[0]) == 0:
            return torch.zeros(1, len(points)).type_as(points)
        face_vertices = kaolin.ops.mesh.index_vertices_by_faces(vertices, faces)
        dists, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(points, face_vertices)
        signs = kaolin.ops.mesh.check_sign(vertices, faces, points)
        sdf = dists * (signs.type_as(dists) - 0.5) * 2 * (-1)  # inside is -1
        return sdf

    def calculate_sdf_loss(self, tet_vertexes, tet_sdf, vertices, faces):
        if len(tet_vertexes[0]) == 0:
            return torch.tensor(0).type_as(tet_sdf)
        true_sdf = self.get_mesh_sdf(tet_vertexes, vertices, faces)
        true_sdf = torch.clamp(true_sdf, min=-self.sdf_clamp, max=self.sdf_clamp)
        loss = torch.mean((tet_sdf - true_sdf) ** 2)
        return loss

    def on_train_batch_start(self, batch, batch_idx):
        if self.global_step >= self.steps_schedule[0]:
            self.volume_refinement = True
        if self.global_step >= self.steps_schedule[1]:
            self.adversarial_training = True
        if self.global_step >= self.steps_schedule[2]:
            self.surface_subdivision = True
        if self.debug and self.global_step % self.debug_interval == 0:
            self.debug_state = True
        else:
            self.debug_state = False

    def surf_batchify(self, out):
        tet_vertexes, tetrahedras, tet_sdf, tet_features, extra_vertexes, extra_sdf = out
        return tet_vertexes.unsqueeze(0), tetrahedras, tet_sdf.unsqueeze(0), tet_features.unsqueeze(0), \
               extra_vertexes.unsqueeze(0), extra_sdf.unsqueeze(0)

    def single_mesh_step(self, pcd_noised=None, vertices=None, faces=None, n_volume_division=None,
                         n_surface_division=None):
        # create output dict
        out = {}
        # sample pcd for train
        if exists(vertices, faces):
            # make vertices batched as all models and losses are expecting batched input
            vertices = vertices.unsqueeze(0)
            pcd, true_faces_ids = kaolin.ops.mesh.sample_points(vertices, faces, self.chamfer_samples)
            pcd_noised = torch.clamp(pcd + torch.randn_like(pcd) * self.noise, min=-1.0, max=1.0)

        # NeRF like encoding
        pe_features = torch.cat([pcd_noised, get_positional_encoding(pcd_noised, self.pe_powers * 3)], dim=-1)

        if self.debug_state and exists(vertices, faces):
            self.lg.log_tensor(pcd[0], 'Input point cloud')
            self.lg.log_tensor(pcd_noised[0], 'Noised onput point cloud')
            self.lg.log_tensor(pe_features[0], 'NeRF like input features')

        # create first positional encoding grids for SDF prediction and save it for SDF discriminator
        sdf_grids = self.sdf_points_encoder.voxelize(pcd_noised, pe_features)
        out['sdf_grids'] = sdf_grids

        # create feature for each tetrahedras vertex and predict initial sdf + features
        tet_vertexes = self.tet_vertexes.unsqueeze(0)  # make batched
        pos_vertex_features = self.sdf_points_encoder.devoxelize(tet_vertexes, sdf_grids)
        tet_out = self.sdf_model.forward(pos_vertex_features)
        tet_sdf, tet_features = tet_out[:, :, 0], tet_out[:, :, 1:]

        if self.debug_state:
            for grid_idx, grid in enumerate(sdf_grids):
                self.lg.log_tensor(grid, f'SDF volume feature grid {self.encoder_grids[grid_idx]}')
            self.lg.log_tensor(tet_sdf, 'First predicted sdf')
            self.lg.log_tensor(tet_features, 'First predicted features')
            indexes = torch.abs(tet_sdf[0]) < 0.05
            self.lg.log_scatter3d(tn(tet_vertexes[0, indexes, 0]), tn(tet_vertexes[0, indexes, 1]),
                                  tn(tet_vertexes[0, indexes, 2]), 'predicted_sdf',
                                  color=tn((torch.ones(len(tet_sdf[0]), 3).type_as(tet_sdf)
                                            * tet_sdf[0].unsqueeze(1) + 0.5).clamp(max=1.0, min=0.0)[indexes]),
                                  epoch=self.global_step)

        # if we're training only on SDF
        if not self.volume_refinement and exists(vertices, faces):
            true_sdf = self.get_mesh_sdf(tet_vertexes, vertices, faces)
            out['sdf_loss'] = torch.mean((true_sdf - tet_sdf) ** 2)
            out['loss'] = out['sdf_loss']

        # add volume subdivision and refinement
        if self.volume_refinement:
            tet_vertexes, tetrahedras, tet_sdf, tet_features, extra_vertexes, extra_sdf \
                = self.surf_batchify(get_surface_tetrahedras(tet_vertexes[0], self.tetrahedras,
                                                             tet_sdf[0], tet_features[0]))
            if exists(vertices, faces):
                out['sdf_loss'] = self.calculate_sdf_loss(extra_vertexes, extra_sdf, vertices, faces)

            # encode same tetrahedras with another volume encoder
            grids = self.ref_points_encoder.voxelize(pcd_noised, pe_features)
            pos_features = self.ref_points_encoder.devoxelize(tet_vertexes, grids)
            delta_v, delta_s, tet_features = self.ref1(torch.cat([tet_vertexes[0], tet_sdf.unsqueeze(-1)[0],
                                                                  tet_features[0], pos_features[0]], dim=-1),
                                                       tetrahedras)
            tet_features = tet_features.unsqueeze(0)

            # update vertexes
            tet_vertexes = tet_vertexes + delta_v.unsqueeze(0)
            tet_sdf = tet_sdf + delta_s.unsqueeze(0)

            # add regularization on vertex delta
            if len(delta_v) > 0:
                out['delta_vertex'] = torch.mean(torch.sum(delta_v ** 2, dim=1))
            else:
                out['delta_vertex'] = torch.tensor(0).type_as(delta_v)

            if self.debug_state:
                for grid_idx, grid in enumerate(grids):
                    self.lg.log_tensor(grid, f'Refinement volume feature grid {self.encoder_grids[grid_idx]}')
                self.lg.log_tensor(delta_v, 'First refimenent delta vertices')
                self.lg.log_tensor(delta_s, 'First refimenent delta sdf')
                self.lg.log_tensor(tet_features, 'First refimenent features')
                self.lg.log_scatter3d(tn(tet_vertexes[0, :, 0]), tn(tet_vertexes[0, :, 1]), tn(tet_vertexes[0, :, 2]),
                                      'first_refined_tetrahedras', epoch=self.global_step)
                _, debug_faces = tetrahedras2mesh(tet_vertexes[0], tetrahedras)
                if len(debug_faces) > 50000:
                    indexes = torch.randint(low=0, high=len(debug_faces), size=(50000,)).type_as(debug_faces)
                    debug_faces = debug_faces[indexes]
                self.lg.log_mesh(tn(tet_vertexes[0]), tn(debug_faces), 'first_refined_tetrahedras_mesh', epoch=self.global_step)

            if n_volume_division is None:
                n_volume_division = self.n_volume_division
            for div_id in range(n_volume_division):

                # volume subdivision
                tet_vertexes, tetrahedras, out_features = kaolin.ops.mesh.subdivide_tetmesh(
                    tet_vertexes, tetrahedras, torch.cat([tet_sdf.unsqueeze(-1), tet_features], dim=-1))
                tet_sdf, tet_features = out_features[:, :, 0], out_features[:, :, 1:]

                # take only surface tetrahedras and add sdf loss to others
                tet_vertexes, tetrahedras, tet_sdf, tet_features, extra_vertexes, extra_sdf \
                    = self.surf_batchify(get_surface_tetrahedras(tet_vertexes[0], tetrahedras,
                                                                 tet_sdf[0], tet_features[0]))
                if exists(vertices, faces):
                    out['sdf_loss'] += self.calculate_sdf_loss(extra_vertexes, extra_sdf, vertices, faces)

            if self.debug_state:
                self.lg.log_tensor(tet_vertexes, 'subdivided vertexes')
                self.lg.log_tensor(tet_features, 'subdivided features')
                self.lg.log_scatter3d(tn(tet_vertexes[0, :, 0]), tn(tet_vertexes[0, :, 1]), tn(tet_vertexes[0, :, 2]),
                                      'subdivided_tetrahedras', epoch=self.global_step)
                _, debug_faces = tetrahedras2mesh(tet_vertexes[0], tetrahedras)
                if len(debug_faces) > 50000:
                    indexes = torch.randint(low=0, high=len(debug_faces), size=(50000,)).type_as(debug_faces)
                    debug_faces = debug_faces[indexes]
                self.lg.log_mesh(tn(tet_vertexes[0]), tn(debug_faces), 'subdivided_tetrahedras_mesh', epoch=self.global_step)

            # additional volume refinement step
            pos_features = self.ref_points_encoder.devoxelize(tet_vertexes, grids)
            delta_v, delta_s, tet_features = self.ref2(torch.cat([tet_vertexes[0], tet_sdf.unsqueeze(-1)[0],
                                                                  tet_features[0], pos_features[0]], dim=-1),
                                                       tetrahedras)
            tet_features = tet_features.unsqueeze(0)

            # update vertexes
            tet_vertexes = tet_vertexes + delta_v.unsqueeze(0)
            tet_sdf = tet_sdf + delta_s.unsqueeze(0)

            if self.debug_state:
                self.lg.log_tensor(delta_v, 'Second refimenent delta vertices')
                self.lg.log_tensor(delta_s, 'Second refimenent delta sdf')
                self.lg.log_tensor(tet_features, 'Second refimenent features')
                self.lg.log_scatter3d(tn(tet_vertexes[0, :, 0]), tn(tet_vertexes[0, :, 1]), tn(tet_vertexes[0, :, 2]),
                                      'subdivided_refined_tetrahedras', epoch=self.global_step)
                _, debug_faces = tetrahedras2mesh(tet_vertexes[0], tetrahedras)
                if len(debug_faces) > 50000:
                    indexes = torch.randint(low=0, high=len(debug_faces), size=(50000,)).type_as(debug_faces)
                    debug_faces = debug_faces[indexes]
                self.lg.log_mesh(tn(tet_vertexes[0]), tn(debug_faces), 'subdivided_refined_tetrahedras_mesh',
                                 epoch=self.global_step)

            # add all sdf loss on all remaining vertices
            if exists(vertices, faces):
                out['sdf_loss'] += self.calculate_sdf_loss(tet_vertexes, tet_sdf, vertices, faces)
                out['loss'] = out['sdf_loss'] * self.sdf_weight

            # add regularization on vertex delta
            if exists(vertices, faces) and len(delta_v) > 0:
                out['delta_vertex'] += torch.mean(torch.sum(delta_v ** 2, dim=1))
                out['loss'] += out['delta_vertex'] * self.delta_weight

            # apply marching tetrahedra on surface tetrahedras to extract mesh
            mesh_vertices, mesh_faces = kaolin.ops.conversions.marching_tetrahedra(tet_vertexes, tetrahedras, tet_sdf)
            mesh_vertices, mesh_faces = mesh_vertices[0].unsqueeze(0), mesh_faces[0]
            out['mesh_vertices'] = mesh_vertices
            out['mesh_faces'] = mesh_faces

            if self.debug_state:
                debug_faces = mesh_faces
                if len(debug_faces) > 50000:
                    indexes = torch.randint(low=0, high=len(debug_faces), size=(50000,)).type_as(debug_faces)
                    debug_faces = debug_faces[indexes]
                self.lg.log_mesh(tn(mesh_vertices[0]), tn(debug_faces), 'first_predicted_mesh', epoch=self.global_step)

        # surface subdivision
        if n_surface_division is None:
            n_surface_division = self.n_surface_division
        if self.surface_subdivision:
            # positional features for mesh vertices
            pos_features = self.ref_points_encoder.devoxelize(mesh_vertices, grids)

            # learnable surface subdivision predicts changed vertices and alpha smoothing factor
            delta_v, alphas = self.surface_ref(torch.cat([mesh_vertices[0], pos_features[0]], dim=-1), mesh_faces)
            mesh_vertices = mesh_vertices + delta_v.unsqueeze(0)
            if len(mesh_faces) > 0:
                mesh_vertices, mesh_faces = kaolin.ops.mesh.subdivide_trianglemesh(
                    mesh_vertices, mesh_faces, iterations=n_surface_division, alpha=alphas.unsqueeze(0))

            out['mesh_vertices'] = mesh_vertices
            out['mesh_faces'] = mesh_faces

            if self.debug_state:
                self.lg.log_tensor(delta_v, 'Surface subdivision delta vertices')
                self.lg.log_tensor(alphas, 'Surface subdivision alphas')
                debug_faces = mesh_faces
                if len(debug_faces) > 50000:
                    indexes = torch.randint(low=0, high=len(debug_faces), size=(50000,)).type_as(debug_faces)
                    debug_faces = debug_faces[indexes]
                self.lg.log_mesh(tn(mesh_vertices[0]), tn(debug_faces), 'subdivided_predicted_mesh', epoch=self.global_step)

        # calculate losses on predicted mesh
        if self.volume_refinement and exists(vertices, faces):
            if len(mesh_faces) > 0:
                # calculate pcd chamfer loss
                face_areas = torch.nan_to_num(kaolin.ops.mesh.face_areas(mesh_vertices, mesh_faces), nan=1.0)
                pred_pcd, faces_ids = kaolin.ops.mesh.sample_points(mesh_vertices, mesh_faces, self.chamfer_samples,
                                                                    areas=face_areas)
                chamfer_loss = kaolin.metrics.pointcloud.chamfer_distance(pred_pcd, pcd)[0]

                # calculate mesh normal loss with finding normal for each point in pcd
                pred_normals = kaolin.ops.mesh.face_normals(
                    kaolin.ops.mesh.index_vertices_by_faces(mesh_vertices, mesh_faces), unit=True)[0]
                pred_normals = pred_normals[faces_ids[0]]

                # finding normal of the closest points
                dist, point_ids = kaolin.metrics.pointcloud.sided_distance(pred_pcd, pcd)
                true_normals = kaolin.ops.mesh.face_normals(
                    kaolin.ops.mesh.index_vertices_by_faces(vertices, faces), unit=True)[0]
                true_normals = true_normals[true_faces_ids[0][point_ids[0]]]
                normal_loss = torch.mean(1 - torch.abs(
                    torch.matmul(pred_normals.unsqueeze(1), true_normals.unsqueeze(2)).view(-1)))

                if self.debug_state:
                    self.lg.log_tensor(pred_pcd, 'Predicted point cloud on mesh')
                    self.lg.log_tensor(dist, 'Predicted - True pcd dists')
                    self.lg.log_tensor(pred_normals, 'Predicted mesh normals')
                out['chamfer_loss'] = chamfer_loss
                out['loss'] += out['chamfer_loss'] * self.chamfer_weight
                out['normal_loss'] = normal_loss
                out['loss'] += out['normal_loss'] * self.normal_weight

        return out

    def shared_step(self, vertices, faces, optimizer_idx, n_volume_division=None, n_surface_division=None,
                    kind='train'):

        if optimizer_idx == 1:  # for optimizing discriminator
            with torch.no_grad():
                out = self.single_mesh_step(vertices=vertices, faces=faces, n_volume_division=n_volume_division,
                                            n_surface_division=n_surface_division)
        else:
            out = self.single_mesh_step(vertices=vertices, faces=faces, n_volume_division=n_volume_division,
                                        n_surface_division=n_surface_division)

        # discriminator
        if self.adversarial_training:
            # find high curvature points on mesh
            curvatures = calculate_gaussian_curvature(vertices, faces)
            indexes = torch.arange(len(curvatures)).type_as(curvatures).long()[curvatures >= self.curvature_threshold]
            if len(indexes) > 0:
                indexes = indexes[torch.randint(0, len(indexes), (self.curvature_samples,)).type_as(indexes).long()]

                # create uniform grid nearby selected point
                grid = torch.stack(torch.meshgrid(
                    torch.linspace(-1, 1, self.disc_sdf_grid).type_as(vertices),
                    torch.linspace(-1, 1, self.disc_sdf_grid).type_as(vertices),
                    torch.linspace(-1, 1, self.disc_sdf_grid).type_as(vertices), ), dim=-1) * self.disc_sdf_scale
                grids = (vertices[indexes] + torch.randn_like(vertices[indexes]) * self.disc_v_noise) \
                            .view(self.curvature_samples, 1, 1, 1, 3) + grid.unsqueeze(0)

                # extract mesh output of generator
                mesh_vertices, mesh_faces, sdf_grids = out['mesh_vertices'], out['mesh_faces'], out['sdf_grids']
                pred_sdf_features, true_sdf_features = [], []

                # generate true sdf and positional features for each grid
                for grid_idx in range(len(grids)):
                    # make each grid as n points as expected by models
                    grid_points = grids[grid_idx].view(1, self.disc_sdf_grid ** 3, 3)
                    pos_features = self.sdf_points_encoder.devoxelize(grid_points, sdf_grids). \
                        view(self.disc_sdf_grid, self.disc_sdf_grid, self.disc_sdf_grid, self.pos_features)
                    true_grid_sdf = self.get_mesh_sdf(grid_points, vertices.unsqueeze(0), faces)[0] \
                        .view(self.disc_sdf_grid, self.disc_sdf_grid, self.disc_sdf_grid, 1)
                    true_sdf_features.append(torch.cat([true_grid_sdf, pos_features], dim=-1))
                    if len(mesh_faces) > 0:
                        pred_grid_sdf = self.get_mesh_sdf(grid_points, mesh_vertices, mesh_faces)[0] \
                            .view(self.disc_sdf_grid, self.disc_sdf_grid, self.disc_sdf_grid, 1)
                        pred_sdf_features.append(torch.cat([pred_grid_sdf, pos_features], dim=-1))

                    if self.debug_state:
                        self.lg.log_tensor(true_grid_sdf, 'True sdf grid at high curvature')
                        tsp = grid_points.view(-1, 3)
                        tsg = true_grid_sdf.view(-1, 1)
                        self.lg.log_scatter3d(tn(tsp[:, 0]), tn(tsp[:, 1]), tn(tsp[:, 2]), f'true_sdf_grid_{grid_idx}',
                                              color=tn(torch.ones(len(tsp), 3).type_as(tsg)
                                                       * tsg.clamp(min=-0.5, max=0.5) + 0.5), epoch=self.global_step)
                        if len(mesh_faces) > 0:
                            self.lg.log_tensor(pred_grid_sdf, 'Predicted sdf grid at high curvature')
                            psg = pred_grid_sdf.view(-1, 1)
                            self.lg.log_scatter3d(tn(tsp[:, 0]), tn(tsp[:, 1]), tn(tsp[:, 2]), f'pred_sdf_grid_{grid_idx}',
                                                  color=tn(torch.ones(len(tsp), 3).type_as(tsg)
                                                           * psg.clamp(min=-0.5, max=0.5) + 0.5), epoch=self.global_step)

                sdf_features = torch.stack(true_sdf_features, dim=0)
                if len(pred_sdf_features) > 0:
                    sdf_features = torch.cat([sdf_features, torch.stack(pred_sdf_features, dim=0)], dim=0)
                disc_preds = self.sdf_disc(sdf_features.movedim(-1, 1))
                if len(pred_sdf_features) > 0:
                    out['adv_loss'] = torch.mean((1 - disc_preds[:len(pred_sdf_features)]) ** 2)
                    out['loss'] += out['adv_loss'] * self.disc_weight
                    out['disc_loss'] = torch.mean(disc_preds[:len(pred_sdf_features)] ** 2) \
                                       + torch.mean((1 - disc_preds[len(pred_sdf_features):]) ** 2)
                else:
                    out['disc_loss'] = torch.mean((1 - disc_preds[len(pred_sdf_features):]) ** 2)

                if self.debug_state:
                    self.lg.log_tensor(curvatures, 'Calculated gaussian curvatures')

        for k, v in out.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                self.log(f'{kind}_{k}', v, prog_bar=True, sync_dist=True, batch_size=1)
        return out

    def on_validation_epoch_end(self):

        data_iter = iter(self.trainer.val_dataloaders[0])
        batch = next(data_iter)
        pcd = [_pcd.to(self.device) for _pcd in batch['pcd']]
        pcd_noised = [torch.clamp(_pcd + torch.randn_like(_pcd) * self.noise, min=-1.0, max=1.0) for _pcd in pcd]

        if self.debug:
            for batch_idx in range(len(pcd_noised)):
                points = pcd_noised[batch_idx]
                self.lg.log_scatter3d(tn(points[:, 0]), tn(points[:, 1]), tn(points[:, 2]), f'pcd_noised_{batch_idx}',
                                      epoch=self.global_step)

        self.timelapse.add_pointcloud_batch(category='input',
                                            pointcloud_list=[_pcd.cpu() for _pcd in pcd],
                                            points_type="usd_geom_points",
                                            iteration=self.global_step)
        self.timelapse.add_pointcloud_batch(category='input_noised',
                                            pointcloud_list=[_pcd.cpu() for _pcd in pcd_noised],
                                            points_type="usd_geom_points",
                                            iteration=self.global_step)
        mesh_vertices, mesh_faces = [], []
        with torch.no_grad():
            for _pcd_noised in pcd_noised:
                out = self.single_mesh_step(pcd_noised=_pcd_noised.unsqueeze(0))
                if 'mesh_vertices' in out and 'mesh_faces' in out:
                    mesh_vertices.append(out['mesh_vertices'][0])
                    mesh_faces.append(out['mesh_faces'])

        if len(mesh_faces) == 0:
            return

        self.timelapse.add_mesh_batch(
            iteration=self.global_step,
            category='predicted_mesh',
            vertices_list=[v.cpu() for v in mesh_vertices],
            faces_list=[f.cpu() for f in mesh_faces]
        )

        if self.debug:
            for batch_idx, (v, f) in enumerate(zip(mesh_vertices, mesh_faces)):
                self.lg.log_mesh(tn(v), tn(f), f'rendered_mesh_{batch_idx}', epoch=self.global_step)

        rendered_images = [render_mesh(v, f, device=self.device) for v, f in zip(mesh_vertices, mesh_faces)
                           if len(f) > 0]
        if len(rendered_images) > 0:
            images = np.stack(rendered_images, axis=0)
            b, views, h, w = images.shape
            images = images.reshape((b, 2, views // 2, h, w)).moveaxis(1, 3).reshape((b, 2 * h, views // 2 * w))
            images = [images[idx] for idx in range(len(images))]
            self.simple_logger.log_images(images, 'rendered_mesh', self.global_step)

    def validation_step(self, batch, batch_idx):
        loss = functools.reduce(lambda l1, l2: l1 + l2,
                                [self.shared_step(v, f, 0, kind='val')['loss']
                                 for v, f in zip(batch['vertices'], batch['faces'])])
        return loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        # generator optimization
        if optimizer_idx == 0:
            loss = functools.reduce(lambda l1, l2: l1 + l2,
                                    [self.shared_step(v, f, optimizer_idx)['loss']
                                     for v, f in zip(batch['vertices'], batch['faces'])])
            return loss
        # discriminator optimization
        if optimizer_idx == 1:
            if not self.adversarial_training:
                return None
            outs = [self.shared_step(v, f, optimizer_idx)['disc_loss']
                                     for v, f in zip(batch['vertices'], batch['faces'])]
            losses = [out['disc_loss'] for out in outs if 'disc_loss' in out]
            if len(losses) == 0:
                return None
            loss = functools.reduce(lambda l1, l2: l1 + l2, losses)
            return loss

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
                                           num_workers=2 * torch.cuda.device_count(), collate_fn=collate_dicts,
                                           pin_memory=True, drop_last=False, prefetch_factor=2)

    def val_dataloader(self):
        val_items = RandomIndexedListWrapper(self.dataset, self.val_idxs)
        return torch.utils.data.DataLoader(val_items, batch_size=self.batch_size, shuffle=False,
                                           num_workers=2 * torch.cuda.device_count(), collate_fn=collate_dicts,
                                           pin_memory=True, drop_last=False, prefetch_factor=2)
