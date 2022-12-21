import functools
from random import shuffle

import pytorch_lightning as pl

import kaolin
from modules.common.trainer import SimpleLogger
from modules.ddd.model import *
from modules.ddd.rast_util import Rasterizer


class PCD2Mesh(pl.LightningModule):

    def __init__(self, dataset=None, clearml=None, timelapse=None, train_rate=0.8, grid_resolution=64,
                 learning_rate=1e-4, debug_interval=100, ref='gcn', disc=True, use_rasterizer=True,
                 n_views=8, view_resolution=256,
                 steps_schedule=(1000, 20000, 50000, 100000), min_lr_rate=1.0, encoder_dims=(64, 128, 256),
                 encoder_out=256, with_norm=False, delta_scale=1 / 2.0, res_features=64,
                 sdf_dims=(256, 256, 128, 64), disc_dims=(32, 64, 128, 256), sdf_clamp=0.03,
                 n_volume_division=1, n_surface_division=1, chamfer_samples=5000, sdf_sign_reg=1e-4, sdf_value_reg=1e-2,
                 continuous_reg=1e-2,
                 sdf_weight=0.4, gcn_dims=(256, 128), gcn_hidden=(128, 64), delta_weight=1.0, disc_weight=10,
                 curvature_threshold=torch.pi / 16, curvature_samples=10, disc_sdf_grid=16, disc_sdf_scale=0.1,
                 disc_v_noise=1e-3, chamfer_weight=500, normal_weight=1e-4, lap_reg=0.5,
                 encoder_grids=(32, 16, 8), batch_size=16, pe_powers=16, noise=0.02):
        super(PCD2Mesh, self).__init__()
        self.save_hyperparameters(ignore=['dataset', 'clearml', 'timelapse'])

        self.debug = debug_interval > 0
        self.debug_interval = debug_interval
        self.debug_state = False

        self.use_rasterizer = use_rasterizer
        self.n_views = n_views
        self.view_resolution = view_resolution
        if use_rasterizer:
            self.rasterizer = Rasterizer(torch.device('cuda:0'))

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
        self.grid_res = grid_resolution
        self.true_grid_res = grid_resolution / delta_scale

        self.pe_powers = pe_powers
        self.input_dim = pe_powers * 3 + 3
        self.pos_features = encoder_out + self.input_dim
        self.noise = noise
        self.sdf_clamp = sdf_clamp
        self.n_volume_division = 0
        self.true_volume_division = n_volume_division
        self.n_surface_division = n_surface_division
        self.encoder_grids = encoder_grids
        self.res_features = res_features

        self.sdf_weight = sdf_weight
        self.delta_weight = delta_weight
        self.disc_weight = disc_weight
        self.chamfer_weight = chamfer_weight
        self.chamfer_samples = chamfer_samples
        self.normal_weight = normal_weight
        self.lap_reg = lap_reg
        self.sdf_value_reg = sdf_value_reg
        self.sdf_sign_reg = sdf_sign_reg
        self.continuous_reg = continuous_reg

        self.curvature_threshold = curvature_threshold
        self.curvature_samples = curvature_samples
        self.disc_sdf_grid = disc_sdf_grid
        self.disc_sdf_scale = disc_sdf_scale
        self.disc_v_noise = disc_v_noise

        self.sdf_points_encoder = MultiPointVoxelCNN(input_dim=self.input_dim, dim=encoder_out, dims=encoder_dims,
                                                     grids=encoder_grids, do_points_map=False, with_norm=True,
                                                     skip=False, n_layers=3)
        self.sdf_model = SimpleMLP(input_dim=self.pos_features, out_dim=1 + res_features, hidden_dims=sdf_dims,
                                   with_norm=True)

        self.ref_points_encoder = MultiPointVoxelCNN(input_dim=self.input_dim, dim=encoder_out, dims=encoder_dims,
                                                     grids=encoder_grids, do_points_map=False, with_norm=True,
                                                     skip=False, n_layers=3)
        self.ref = ref
        if ref == 'gcn':
            self.ref1 = GCNConv(input_dim=1 + self.pos_features + res_features, out_dim=3 + 1 + res_features,
                                gcn_dims=gcn_dims, mlp_dims=gcn_hidden, with_norm=with_norm)
            self.ref2 = GCNConv(input_dim=1 + self.pos_features + res_features, out_dim=3 + 1 + res_features,
                                gcn_dims=gcn_dims, mlp_dims=gcn_hidden, with_norm=with_norm)
            self.surface_ref = GCNConv(input_dim=self.pos_features, out_dim=3 + 1, gcn_dims=gcn_dims,
                                       mlp_dims=gcn_hidden, with_norm=with_norm)
        elif ref == 'conv':
            self.ref1 = PointVoxelCNN(input_dim=1 + self.pos_features + res_features, out_dim=3 + 1 + res_features,
                                      grid_res=encoder_grids[0], dim=encoder_dims[0], dim_points=gcn_dims[-1],
                                      n_layers=4, with_norm=with_norm, skip=True)
            self.ref2 = PointVoxelCNN(input_dim=1 + self.pos_features + res_features, out_dim=3 + 1 + res_features,
                                      grid_res=encoder_grids[0], dim=encoder_dims[0], dim_points=gcn_dims[-1],
                                      n_layers=4, with_norm=with_norm, skip=True)
            self.surface_ref = PointVoxelCNN(input_dim=self.pos_features, out_dim=3 + 1,
                                             grid_res=encoder_grids[0], dim=encoder_dims[0], dim_points=gcn_dims[-1],
                                             n_layers=4, with_norm=with_norm, skip=True)
        elif ref == 'linear':
            self.ref1 = SimpleMLP(input_dim=1 + self.pos_features + res_features, out_dim=3 + 1 + res_features,
                                  hidden_dims=gcn_dims + gcn_hidden, with_norm=with_norm)
            self.ref2 = SimpleMLP(input_dim=1 + self.pos_features + res_features, out_dim=3 + 1 + res_features,
                                  hidden_dims=gcn_dims + gcn_hidden, with_norm=with_norm)
            self.surface_ref = SimpleMLP(input_dim=self.pos_features, out_dim=3 + 1, hidden_dims=gcn_dims + gcn_hidden,
                                         with_norm=with_norm)
        else:
            raise NotImplementedError

        self.disc = disc
        self.sdf_disc = SDFDiscriminator(input_dim=1 + self.pos_features, hidden_dims=disc_dims, with_norm=True)

        # state variables
        self.volume_refinement = False
        self.adversarial_training = False
        self.surface_subdivision = False

        ####################### DEBUG CODE #######################
        if self.debug:
            print(self.lg.log_tensor(tet_vertexes.transpose(0, 1), 'Tetrahedras grid vertices', depth=1))
            print(self.lg.log_tensor(tetrahedras, 'Tetrahedras grid faces'))
            self.lg.log_scatter3d(tn(tet_vertexes[:, 0]), tn(tet_vertexes[:, 1]), tn(tet_vertexes[:, 2]),
                                  'tetrahedras_vertex')
            _v, _f = tetrahedras2mesh(tet_vertexes, tetrahedras)
            indexes = torch.randint(low=0, high=len(_f), size=(50000,)).type_as(_f).long()
            self.lg.log_mesh(tn(_v.cpu()), tn(_f[indexes]), 'tetrahedras_grid')
        ############################################################

    @torch.no_grad()
    def get_mesh_sdf(self, points, vertices, faces, pcd=None, num_samples=5000):
        if faces.numel() == 0 or points.numel() == 0:
            return torch.zeros(1, len(points[0])).type_as(points)
        if pcd is None:
            pcd, _ = kaolin.ops.mesh.sample_points(vertices, faces, num_samples=num_samples)
        dists, _ = kaolin.metrics.pointcloud.sided_distance(points, pcd)
        signs = kaolin.ops.mesh.check_sign(vertices, faces, points)
        sdf = dists * (signs.type_as(dists) - 0.5) * 2 * (-1)  # inside is -1
        return sdf

    @torch.no_grad()
    def get_mesh_udf(self, points, vertices, faces, pcd=None, num_samples=5000):
        if faces.numel() == 0 or points.numel() == 0:
            return torch.zeros(1, len(points[0])).type_as(points)
        if pcd is None:
            pcd, _ = kaolin.ops.mesh.sample_points(vertices, faces, num_samples=num_samples)
        dists, _ = kaolin.metrics.pointcloud.sided_distance(points, pcd)
        return dists

    def calculate_sdf_loss(self, tet_vertexes, tet_sdf, vertices, faces, true_sdf=None):
        if tet_vertexes.numel() == 0 or faces.numel() == 0:
            return torch.tensor(0).type_as(tet_sdf)
        # true_sdf = self.get_mesh_sdf(tet_vertexes, vertices, faces)
        true_sdf = calculate_sdf(tet_vertexes, vertices, faces, true_sdf=true_sdf)
        true_sdf = torch.clamp(true_sdf, min=-self.sdf_clamp, max=self.sdf_clamp)
        loss = torch.mean((tet_sdf - true_sdf) ** 2)
        return loss

    def on_train_batch_start(self, batch, batch_idx):
        if self.global_step >= self.steps_schedule[0]:
            self.volume_refinement = True
            self.n_volume_division = self.true_volume_division
        if self.global_step >= self.steps_schedule[1] and self.disc:
            self.adversarial_training = True
        if self.global_step >= self.steps_schedule[2]:
            self.surface_subdivision = True
        if self.debug and self.global_step % self.debug_interval == 0:
            self.debug_state = True
        else:
            self.debug_state = False

    def surf_batchify(self, out):
        tet_vertexes, tetrahedras, tet_sdf, tet_features, extra_vertexes, extra_tets, extra_sdf = out
        return tet_vertexes.unsqueeze(0), tetrahedras, tet_sdf.unsqueeze(0), tet_features.unsqueeze(0), \
               extra_vertexes.unsqueeze(0), extra_tets, extra_sdf.unsqueeze(0)

    def single_mesh_step(self, pcd_noised=None, vertices=None, faces=None, n_volume_division=None, true_sdf=None,
                         n_surface_division=None):

        is_train = exists(vertices, faces)

        ### STEP 0 --- Sampling noisy point cloud from mesh
        if is_train:
            # make vertices batched as all models and losses are expecting batched input
            vertices = vertices.unsqueeze(0)
            # sample pcd for train
            pcd, true_faces_ids = kaolin.ops.mesh.sample_points(vertices, faces, self.chamfer_samples)
            pcd_noised = pcd + torch.randn_like(pcd) * self.noise
            pcd_noised = torch.clamp(pcd_noised, min=-1.0, max=1.0)

        ### STEP 1 --- Initial sdf prediction
        # create output dict
        out = {
            'loss': torch.tensor(0).type_as(pcd_noised)
        }
        # NeRF like encoding
        pe_features = torch.cat([pcd_noised, get_positional_encoding(pcd_noised, self.pe_powers * 3)], dim=-1)
        # create first positional encoding grids for SDF prediction and save it for SDF discriminator
        sdf_grids = self.sdf_points_encoder.voxelize(pcd_noised, pe_features)
        out['sdf_grids'] = sdf_grids

        # create feature for each tetrahedras vertex and predict initial sdf + features
        tetrahedras = self.tetrahedras
        tet_vertexes = self.tet_vertexes.unsqueeze(0)  # make batched
        pos_vertex_features = self.sdf_points_encoder.devoxelize(tet_vertexes, sdf_grids)
        pos_vertex_features = torch.cat([tet_vertexes, get_positional_encoding(tet_vertexes, self.pe_powers * 3),
                                         pos_vertex_features], dim=-1)
        tet_out = self.sdf_model.forward(pos_vertex_features)
        tet_sdf, tet_features = tet_out[:, :, 0], tet_out[:, :, 1:]

        ####################### DEBUG CODE #######################
        if self.debug_state:
            if is_train:
                self.lg.log_tensor(pcd[0], 'Input point cloud')
                self.lg.log_tensor(pcd_noised[0], 'Noised input point cloud')
                self.lg.log_tensor(pe_features[0], 'NeRF like input features')
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
        ############################################################

        ### STEP 2.1 --- First surface refinement
        # encode same tetrahedras with another volume encoder
        grids = self.ref_points_encoder.voxelize(pcd_noised, pe_features)
        pos_features = self.ref_points_encoder.devoxelize(tet_vertexes, grids)
        pos_features = torch.cat([tet_vertexes, get_positional_encoding(tet_vertexes, self.pe_powers * 3),
                                  pos_features], dim=-1)
        ref1_features = torch.cat([tet_sdf.unsqueeze(-1), tet_features, pos_features], dim=-1)
        if self.ref == 'gcn':
            ref1_out = self.ref1(ref1_features[0], get_tetrahedras_edges(tetrahedras))
        elif self.ref == 'conv':
            ref1_out = self.ref1(tet_vertexes, ref1_features)[0]
        elif self.ref == 'linear':
            ref1_out = self.ref1(ref1_features)[0]
        else:
            raise NotImplementedError
        delta_v, delta_s, tet_features = torch.tanh(ref1_out[:, :3]) / self.true_grid_res, \
                                         torch.tanh(ref1_out[:, 3]), ref1_out[:, 4:]
        tet_features = tet_features.unsqueeze(0)

        # update vertexes
        tet_vertexes = tet_vertexes + delta_v.unsqueeze(0)
        tet_sdf = tet_sdf + delta_s.unsqueeze(0)

        # add regularization on vertex delta
        if len(delta_v) > 0:
            out['delta_vertex'] = torch.mean(torch.sum(delta_v ** 2, dim=1))
            out['delta_laplace'] = delta_laplace_loss_tetrahedras(delta_v, tetrahedras)
            out['delta_sdf'] = sdf_value_reg(delta_s, get_tetrahedras_edges(tetrahedras, unique=True), self.grid_res)
        else:
            out['delta_vertex'] = torch.tensor(0).type_as(delta_v)
            out['delta_laplace'] = torch.tensor(0).type_as(delta_v)
            out['delta_sdf'] = torch.tensor(0).type_as(delta_s)

        ####################### DEBUG CODE #######################
        if self.debug_state:
            for grid_idx, grid in enumerate(grids):
                self.lg.log_tensor(grid, f'Refinement volume feature grid {self.encoder_grids[grid_idx]}')
            self.lg.log_tensor(tet_vertexes.transpose(1, 2), 'First refined vertexes', depth=1)
            self.lg.log_tensor(delta_v.transpose(0, 1), 'First refimenent delta vertices', depth=1)
            self.lg.log_tensor(delta_s, 'First refimenent delta sdf')
            self.lg.log_tensor(tet_features, 'First refimenent features')
            self.lg.log_scatter3d(tn(tet_vertexes[0, :, 0]), tn(tet_vertexes[0, :, 1]), tn(tet_vertexes[0, :, 2]),
                                  'first_refined_tetrahedras', epoch=self.global_step)
            debug_tet_vertexes, debug_tetrahedras, _, _, _, _, _ = self.surf_batchify(
                get_surface_tetrahedras(tet_vertexes[0], tetrahedras, tet_sdf[0], tet_features[0]))
            if debug_tetrahedras.numel() > 0:
                _, debug_faces = tetrahedras2mesh(debug_tet_vertexes[0], debug_tetrahedras)
                if len(debug_faces) > 50000:
                    indexes = torch.randint(low=0, high=len(debug_faces), size=(50000,)).type_as(debug_faces)
                    debug_faces = debug_faces[indexes]
                self.lg.log_mesh(tn(tet_vertexes[0]), tn(debug_faces), 'first_refined_tetrahedras_mesh',
                                 epoch=self.global_step)
        ############################################################

        ### STEP 2.2 --- Volume subdivision
        if n_volume_division is None:
            n_volume_division = self.n_volume_division
        not_subdivided_vertexes, not_subdivided_sdf, not_subdivided_tets = [], [], []
        if len(get_only_surface_tetrahedras(tetrahedras, tet_sdf[0])) <= 0.1 * len(tetrahedras):

            for div_id in range(n_volume_division):
                # we have to remove non-surface tetrahedras as volume subdivision is expensive
                tet_vertexes, tetrahedras, tet_sdf, tet_features, __not_subdivided_vertexes, __not_subdivided_tets, \
                __not_subdivided_sdf = self.surf_batchify(get_surface_tetrahedras(tet_vertexes[0], tetrahedras,
                                                                                  tet_sdf[0], tet_features[0]))
                not_subdivided_vertexes.append(__not_subdivided_vertexes)
                not_subdivided_sdf.append(__not_subdivided_sdf)
                not_subdivided_tets.append(__not_subdivided_tets)

                # volume subdivision
                tet_vertexes, tetrahedras, out_features = kaolin.ops.mesh.subdivide_tetmesh(
                    tet_vertexes, tetrahedras, torch.cat([tet_sdf.unsqueeze(-1), tet_features], dim=-1))
                tet_sdf, tet_features = out_features[:, :, 0], out_features[:, :, 1:]

        ####################### DEBUG CODE #######################
        if self.debug_state:
            self.lg.log_tensor(tet_vertexes, 'subdivided vertexes')
            self.lg.log_tensor(tet_features, 'subdivided features')
            self.lg.log_scatter3d(tn(tet_vertexes[0, :, 0]), tn(tet_vertexes[0, :, 1]), tn(tet_vertexes[0, :, 2]),
                                  'subdivided_tetrahedras', epoch=self.global_step)
            debug_tet_vertexes, debug_tetrahedras, _, _, _, _, _ = self.surf_batchify(
                get_surface_tetrahedras(tet_vertexes[0], tetrahedras, tet_sdf[0], tet_features[0]))
            if debug_tetrahedras.numel() > 0:
                _, debug_faces = tetrahedras2mesh(debug_tet_vertexes[0], debug_tetrahedras)
                if len(debug_faces) > 50000:
                    indexes = torch.randint(low=0, high=len(debug_faces), size=(50000,)).type_as(debug_faces)
                    debug_faces = debug_faces[indexes]
                self.lg.log_mesh(tn(debug_tet_vertexes[0]), tn(debug_faces), 'subdivided_tetrahedras_mesh',
                                 epoch=self.global_step)
        ############################################################

        ### STEP 2.3 --- Additional surface refinement
        if tet_vertexes.numel() > 0 and tetrahedras.numel() > 0:
            pos_features = self.ref_points_encoder.devoxelize(tet_vertexes, grids)
            pos_features = torch.cat([tet_vertexes, get_positional_encoding(tet_vertexes, self.pe_powers * 3),
                                      pos_features], dim=-1)
            ref2_features = torch.cat([tet_sdf.unsqueeze(-1), tet_features, pos_features], dim=-1)
            if self.ref == 'gcn':
                ref2_out = self.ref2(ref2_features[0], get_tetrahedras_edges(tetrahedras))
            elif self.ref == 'conv':
                ref2_out = self.ref2(tet_vertexes, ref2_features)[0]
            elif self.ref == 'linear':
                ref2_out = self.ref2(ref2_features)[0]
            else:
                raise NotImplementedError
            delta_v, delta_s, tet_features = torch.tanh(ref2_out[:, :3]) / \
                                             (self.true_grid_res * (2 ** self.n_volume_division)), \
                                             torch.tanh(ref2_out[:, 3]), ref2_out[:, 4:]
            tet_features = tet_features.unsqueeze(0)

            # update vertexes
            tet_vertexes = tet_vertexes + delta_v.unsqueeze(0)
            tet_sdf = tet_sdf + delta_s.unsqueeze(0)

            # add regularization on vertex delta
            if len(delta_v) > 0:
                out['delta_vertex'] += torch.mean(torch.sum(delta_v ** 2, dim=1))
                out['delta_laplace'] += delta_laplace_loss_tetrahedras(delta_v, tetrahedras)
                out['delta_sdf'] += sdf_value_reg(delta_s, get_tetrahedras_edges(tetrahedras, unique=True),
                                                  self.grid_res * 2 ** self.n_volume_division)

            ####################### DEBUG CODE #######################
            if self.debug_state:
                self.lg.log_tensor(tet_vertexes.transpose(1, 2), 'Second refined vertexes', depth=1)
                self.lg.log_tensor(delta_v.transpose(0, 1), 'Second refimenent delta vertices', depth=1)
                self.lg.log_tensor(delta_s, 'Second refimenent delta sdf')
                self.lg.log_tensor(tet_features, 'Second refimenent features')
                self.lg.log_scatter3d(tn(tet_vertexes[0, :, 0]), tn(tet_vertexes[0, :, 1]),
                                      tn(tet_vertexes[0, :, 2]),
                                      'subdivided_refined_tetrahedras', epoch=self.global_step)
                debug_tet_vertexes, debug_tetrahedras, _, _, _, _, _ = self.surf_batchify(
                    get_surface_tetrahedras(tet_vertexes[0], tetrahedras, tet_sdf[0], tet_features[0]))
                if debug_tetrahedras.numel() > 0:
                    _, debug_faces = tetrahedras2mesh(debug_tet_vertexes[0], debug_tetrahedras)
                    if len(debug_faces) > 50000:
                        indexes = torch.randint(low=0, high=len(debug_faces), size=(50000,)).type_as(debug_faces)
                        debug_faces = debug_faces[indexes]
                    self.lg.log_mesh(tn(debug_tet_vertexes[0]), tn(debug_faces),
                                     'subdivided_refined_tetrahedras_mesh',
                                     epoch=self.global_step)
            ############################################################

        ### STEP 2.4 ---  Apply marching tetrahedra on surface tetrahedras to extract mesh
        mesh_vertices, mesh_faces, tet_indices = kaolin.ops.conversions.marching_tetrahedra(
            tet_vertexes, tetrahedras, tet_sdf, return_tet_idx=True)
        mesh_vertices, mesh_faces, tet_indices = mesh_vertices[0].unsqueeze(0), mesh_faces[0], tet_indices[0]
        out['mesh_vertices'] = mesh_vertices
        out['mesh_faces'] = mesh_faces

        ### LOSS --- sdf
        if is_train:
            total_vertexes = torch.cat([tet_vertexes] + not_subdivided_vertexes, dim=1)
            total_sdf = torch.cat([tet_sdf] + not_subdivided_sdf, dim=1)
            sdf_weight = self.sdf_weight if self.volume_refinement else 1.0
            if abs(sdf_weight) > 1e-6:
                out['sdf_loss'] = self.calculate_sdf_loss(total_vertexes, total_sdf, vertices, faces, true_sdf=true_sdf)
                out['loss'] += out['sdf_loss'] * sdf_weight

        ### REGULARIZATION --- close tetrahedras must have same sdf sign
        if self.volume_refinement and is_train:
            out['sdf_sign_reg'] = functools.reduce(lambda l1, l2: l1 + l2, [
                sdf_sign_reg(sdf[0], get_tetrahedras_edges(tets, unique=True))
                for sdf, tets in zip([tet_sdf] + not_subdivided_sdf, [tetrahedras] + not_subdivided_tets)])
            out['loss'] += out['sdf_sign_reg'] * self.sdf_sign_reg

        ####################### DEBUG CODE #######################
        if self.debug_state:
            if mesh_faces.numel() > 0:
                debug_faces = mesh_faces
                if len(debug_faces) > 50000:
                    indexes = torch.randint(low=0, high=len(debug_faces), size=(50000,)).type_as(debug_faces)
                    debug_faces = debug_faces[indexes]
                self.lg.log_mesh(tn(mesh_vertices[0]), tn(debug_faces), 'first_predicted_mesh',
                                 epoch=self.global_step)
        ############################################################

        ### STEP 3 --- Learnable surface subdivision
        if n_surface_division is None:
            n_surface_division = self.n_surface_division
        if self.surface_subdivision:
            if mesh_faces.numel() > 0 and mesh_vertices.numel() > 0:

                mesh_vertices = laplace_smoothing(mesh_vertices[0], get_mesh_edges(mesh_faces)).unsqueeze(0)

                # positional features for mesh vertices
                pos_features = self.ref_points_encoder.devoxelize(mesh_vertices, grids)
                pos_features = torch.cat([mesh_vertices, get_positional_encoding(mesh_vertices, self.pe_powers * 3),
                                          pos_features], dim=-1)

                # learnable surface subdivision predicts changed vertices and alpha smoothing factor
                if self.ref == 'gcn':
                    surface_ref_out = self.surface_ref(pos_features[0], get_mesh_edges(mesh_faces))
                elif self.ref == 'conv':
                    surface_ref_out = self.surface_ref(mesh_vertices, pos_features)[0]
                elif self.ref == 'linear':
                    surface_ref_out = self.surface_ref(pos_features)[0]
                else:
                    raise NotImplementedError
                delta_v, alphas = torch.tanh(surface_ref_out[:, :3]) / (self.true_grid_res), \
                                  torch.sigmoid(surface_ref_out[:, 3])
                mesh_vertices = mesh_vertices + delta_v.unsqueeze(0)

                # add regularization on vertex delta
                if len(delta_v) > 0:
                    out['delta_vertex'] += torch.mean(torch.sum(delta_v ** 2, dim=1))
                    out['delta_laplace'] += delta_laplace_loss_mesh(delta_v, mesh_faces)

                mesh_vertices, mesh_faces = kaolin.ops.mesh.subdivide_trianglemesh(
                    mesh_vertices, mesh_faces, iterations=n_surface_division, alpha=alphas.unsqueeze(0))

                out['mesh_vertices'] = mesh_vertices
                out['mesh_faces'] = mesh_faces

                ####################### DEBUG CODE #######################
                if self.debug_state:
                    self.lg.log_tensor(delta_v, 'Surface subdivision delta vertices')
                    self.lg.log_tensor(alphas, 'Surface subdivision alphas')
                    if mesh_faces.numel() > 0:
                        debug_faces = mesh_faces
                        if len(debug_faces) > 50000:
                            indexes = torch.randint(low=0, high=len(debug_faces), size=(50000,)).type_as(
                                debug_faces)
                            debug_faces = debug_faces[indexes]
                        self.lg.log_mesh(tn(mesh_vertices[0]), tn(debug_faces), 'subdivided_predicted_mesh',
                                         epoch=self.global_step)
                ############################################################

        ### REGULARIZATION --- delta vertexes, sdf and laplace
        if self.volume_refinement and is_train:
            out['loss'] += out['delta_vertex'] * self.delta_weight
            out['loss'] += out['delta_laplace'] * self.lap_reg
            out['loss'] += out['delta_sdf'] * self.sdf_value_reg

        ### LOSS --- Chamfer loss and smoothness normal loss
        if self.volume_refinement and is_train:
            if mesh_faces.numel() > 0 and mesh_vertices.numel() > 0:
                # calculate pcd chamfer loss
                face_areas = torch.nan_to_num(kaolin.ops.mesh.face_areas(mesh_vertices, mesh_faces), nan=1.0)
                pred_pcd, faces_ids = kaolin.ops.mesh.sample_points(mesh_vertices, mesh_faces,
                                                                    self.chamfer_samples * 10, areas=face_areas)
                chamfer_loss = kaolin.metrics.pointcloud.chamfer_distance(pred_pcd, pcd)[0]
                normal_loss = smoothness_loss(mesh_vertices[0], mesh_faces)

                ####################### DEBUG CODE #######################
                if self.debug_state:
                    self.lg.log_tensor(face_areas, 'Face areas of predicted mesh')
                    self.lg.log_tensor(pred_pcd, 'Predicted point cloud on mesh')
                ############################################################

                out['chamfer_loss'] = chamfer_loss
                out['loss'] += out['chamfer_loss'] * self.chamfer_weight
                out['normal_loss'] = normal_loss
                out['loss'] += out['normal_loss'] * self.normal_weight

        return out

    def shared_step(self, vertices, faces, optimizer_idx, n_volume_division=None, n_surface_division=None,
                    true_sdf=None, kind='train'):

        if optimizer_idx == 1:  # for optimizing discriminator
            with torch.no_grad():
                out = self.single_mesh_step(vertices=vertices, faces=faces, n_volume_division=n_volume_division,
                                            n_surface_division=n_surface_division, true_sdf=true_sdf)
        else:
            out = self.single_mesh_step(vertices=vertices, faces=faces, n_volume_division=n_volume_division,
                                        n_surface_division=n_surface_division, true_sdf=true_sdf)

        ### STEP 4 --- Discriminate output
        if self.adversarial_training:
            # find high curvature points on mesh
            curvatures = calculate_gaussian_curvature(vertices, faces)
            indexes = torch.arange(len(curvatures)).type_as(curvatures).long()[curvatures >= self.curvature_threshold]
            if indexes.numel() > 0:
                indexes = indexes[torch.randint(0, len(indexes), (self.curvature_samples,)).type_as(indexes).long()]
                selected_vertices = vertices[indexes]
                # create uniform grid nearby selected point
                grid = torch.stack(torch.meshgrid(
                    torch.linspace(-1, 1, self.disc_sdf_grid).type_as(vertices),
                    torch.linspace(-1, 1, self.disc_sdf_grid).type_as(vertices),
                    torch.linspace(-1, 1, self.disc_sdf_grid).type_as(vertices), ), dim=-1) * self.disc_sdf_scale
                grids = (selected_vertices + torch.randn_like(selected_vertices) * self.disc_v_noise) \
                            .view(self.curvature_samples, 1, 1, 1, 3) + grid.unsqueeze(0)

                # extract mesh output of generator
                mesh_vertices, mesh_faces, sdf_grids = out['mesh_vertices'], out['mesh_faces'], out['sdf_grids']

                # create grid features
                grid_points = grids.view(self.curvature_samples, self.disc_sdf_grid ** 3, 3)
                pos_features = torch.cat([grid_points, get_positional_encoding(grid_points, self.pe_powers * 3),
                                          self.sdf_points_encoder.devoxelize(grid_points, sdf_grids)], dim=-1) \
                    .view(self.curvature_samples, self.disc_sdf_grid, self.disc_sdf_grid, self.disc_sdf_grid,
                          self.pos_features)
                true_grid_udf = torch.stack([self.get_mesh_udf(grid_points[grid_idx].unsqueeze(0),
                                                               vertices.unsqueeze(0),
                                                               get_close_faces(selected_vertices[grid_idx],
                                                                               vertices, faces, self.disc_sdf_scale),
                                                               num_samples=self.chamfer_samples) \
                                            .view(self.disc_sdf_grid, self.disc_sdf_grid, self.disc_sdf_grid, 1)
                                             for grid_idx in range(self.curvature_samples)], dim=0)
                true_udf_features = torch.cat([pos_features, true_grid_udf], dim=-1)
                if mesh_faces.numel() > 0:
                    pred_grid_udf = torch.stack([self.get_mesh_udf(grid_points[grid_idx].unsqueeze(0), mesh_vertices,
                                                                   get_close_faces(selected_vertices[grid_idx],
                                                                                   mesh_vertices[0], mesh_faces,
                                                                                   self.disc_sdf_scale),
                                                                   num_samples=self.chamfer_samples) \
                                                .view(self.disc_sdf_grid, self.disc_sdf_grid, self.disc_sdf_grid, 1)
                                                 for grid_idx in range(self.curvature_samples)], dim=0)
                    pred_udf_features = torch.cat([pos_features, pred_grid_udf], dim=-1)
                else:
                    pred_udf_features = []

                # concat true and pred udf features to one batch for discriminator
                sdf_features = true_udf_features
                if len(pred_udf_features) > 0:
                    sdf_features = torch.cat([sdf_features, pred_udf_features], dim=0)
                disc_preds = self.sdf_disc(sdf_features.movedim(-1, 1))
                if len(pred_udf_features) > 0:
                    if self.adversarial_training:
                        out['adv_loss'] = torch.mean((1 - disc_preds[:len(pred_udf_features)]) ** 2)
                        out['loss'] += out['adv_loss'] * self.disc_weight
                    out['disc_loss'] = torch.mean(disc_preds[:len(pred_udf_features)] ** 2) \
                                       + torch.mean((1 - disc_preds[len(pred_udf_features):]) ** 2)
                else:
                    out['disc_loss'] = torch.mean((1 - disc_preds[len(pred_udf_features):]) ** 2)

                ####################### DEBUG CODE #######################
                if self.debug_state:
                    self.lg.log_tensor(curvatures, 'Calculated gaussian curvatures')
                    self.lg.log_tensor(disc_preds, 'Discriminator predictions')
                    self.lg.log_tensor(true_grid_udf, 'True udf grid at high curvature')
                    tup = grid_points[0].view(-1, 3)
                    tug = true_grid_udf[0].view(-1, 1)
                    self.lg.log_scatter3d(tn(tup[:, 0]), tn(tup[:, 1]), tn(tup[:, 2]), f'true_udf_grid_{0}',
                                          color=tn(torch.ones(len(tup), 3).type_as(tug)
                                                   * tug.clamp(min=-0.5, max=0.5) + 0.5), epoch=self.global_step)
                    if mesh_faces.numel() > 0:
                        self.lg.log_tensor(pred_grid_udf, 'Predicted sdf grid at high curvature')
                        pug = pred_grid_udf[0].view(-1, 1)
                        self.lg.log_scatter3d(tn(tup[:, 0]), tn(tup[:, 1]), tn(tup[:, 2]), f'pred_udf_grid_{0}',
                                              color=tn(torch.ones(len(tup), 3).type_as(tug)
                                                       * pug.clamp(min=-0.5, max=0.5) + 0.5),
                                              epoch=self.global_step)
                ############################################################

        for k, v in out.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                self.log(f'{kind}_{k}', v, prog_bar=True, sync_dist=True, batch_size=1)
        return out

    def on_validation_epoch_end(self):

        data_iter = iter(self.trainer.val_dataloaders[0])
        batch = next(data_iter)
        vertices = [_vertices.to(self.device) for _vertices in batch['vertices']]
        faces = [_faces.to(self.device) for _faces in batch['faces']]
        pcd = [kaolin.ops.mesh.sample_points(_vertices.unsqueeze(0), _faces, self.chamfer_samples)[0][0]
               for _vertices, _faces in zip(vertices, faces)]
        pcd_noised = [torch.clamp(_pcd + torch.randn_like(_pcd) * self.noise, min=-1.0, max=1.0) for _pcd in pcd]

        ####################### DEBUG CODE #######################
        if self.debug:
            for batch_idx in range(len(pcd_noised)):
                points = pcd_noised[batch_idx]
                self.lg.log_scatter3d(tn(points[:, 0]), tn(points[:, 1]), tn(points[:, 2]), f'pcd_noised_{batch_idx}',
                                      epoch=self.global_step)
        ############################################################

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
                if 'mesh_vertices' in out and 'mesh_faces' in out and out['mesh_faces'].numel() > 0:
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

        ####################### DEBUG CODE #######################
        if self.debug:
            for batch_idx, (v, f) in enumerate(zip(mesh_vertices, mesh_faces)):
                self.lg.log_mesh(tn(v), tn(f), f'rendered_mesh_{batch_idx}', epoch=self.global_step)
        ############################################################

        if self.use_rasterizer:
            rendered_images = []
            for v, f in zip(mesh_vertices, mesh_faces):
                poses = get_random_view(torch.ones(self.n_views).type_as(v) * 2.0)
                projections = projection_matrix(near=torch.ones(self.n_views).type_as(v) * 1.0,
                                                far=torch.ones(self.n_views).type_as(v) * 3.0)
                out = self.rasterizer.rasterize(v, f, poses, projections, res=self.view_resolution)
                rendered_images.append(tn(out[..., 3].unsqueeze(-1).repeat(1, 1, 1, 3).bool().float() * 255)
                                       .astype(np.uint8))
            if len(rendered_images) > 0:
                images = np.stack(rendered_images, axis=0)
                b, views, h, w, _ = images.shape
                images = np.moveaxis(images.reshape((b, 2, views // 2, h, w, 3)), 2, 3).reshape(
                    (b, 2 * h, views // 2 * w, 3))
                images = [images[idx] for idx in range(len(images))]
                self.lg.log_images(images, 'rendered_mesh', self.global_step)

    def validation_step(self, batch, batch_idx):
        loss = functools.reduce(lambda l1, l2: l1 + l2,
                                [self.shared_step(v, f, 0, kind='val')['loss']
                                 for v, f in zip(batch['vertices'], batch['faces'])])
        return loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        # generator optimization
        if optimizer_idx == 0:
            loss = functools.reduce(lambda l1, l2: l1 + l2,
                                    [self.shared_step(v, f, optimizer_idx, true_sdf=sdf)['loss']
                                     for v, f, sdf in zip(batch['vertices'], batch['faces'], batch['sdf'])])
            return loss
        # discriminator optimization
        if optimizer_idx == 1:
            if not self.disc or not self.volume_refinement:
                return None
            outs = [self.shared_step(v, f, optimizer_idx, true_sdf=sdf)
                    for v, f, sdf in zip(batch['vertices'], batch['faces'], batch['sdf'])]
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
