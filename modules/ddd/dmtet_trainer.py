from random import shuffle

import pytorch_lightning as pl

import kaolin
from modules.common.trainer import SimpleLogger
from modules.ddd.model import *


class PCD2Mesh(pl.LightningModule):

    def __init__(self, dataset=None, clearml=None, train_rate=0.8, grid_resolution=128, learning_rate=1e-4,
                 steps_schedule=(1000, 20000, 50000, 100000), min_lr_rate=1.0, encoder_dims=(64, 128, 256),
                 sdf_dims=(256, 256, 128, 64), disc_dims=(32, 64, 128, 256), sdf_clamp=0.03,
                 sdf_weight=0.4, gcn_dims=(256, 128), gcn_hidden=(128, 64), delta_weight=1.0, disc_weight=10,
                 curvature_threshold=torch.pi / 16, curvature_samples=10, disc_sdf_grid=16, disc_sdf_scale=0.02,
                 chamfer_weight=500,
                 encoder_grids=(32, 16, 8), batch_size=16, pe_powers=16, noise=0.02):
        super(PCD2Mesh, self).__init__()
        self.save_hyperparameters(ignore=['dataset', 'clearml'])

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

        self.sdf_weight = sdf_weight
        self.delta_weight = delta_weight
        self.disc_weight = disc_weight
        self.chamfer_weight = chamfer_weight

        self.curvature_threshold = curvature_threshold
        self.curvature_samples = curvature_samples
        self.disc_sdf_grid = disc_sdf_grid
        self.disc_sdf_scale = disc_sdf_scale

        self.sdf_points_encoder = MultiPointVoxelCNN(input_dim=self.input_dim, dims=encoder_dims, grids=encoder_grids,
                                                     do_points_map=False)
        self.sdf_model = SimpleMLP(input_dim=sum(encoder_dims), out_dim=1 + sdf_dims[-1], hidden_dims=sdf_dims)

        self.ref_points_encoder = MultiPointVoxelCNN(input_dim=self.input_dim, dims=encoder_dims,
                                                     grids=encoder_grids, do_points_map=False)
        self.ref1 = TetConv(input_dim=3 + 1 + sdf_dims[-1] + sum(encoder_dims), out_dim=3 + 1 + sdf_dims[-1],
                            gcn_dims=gcn_dims, mlp_dims=gcn_hidden)
        self.ref2 = TetConv(input_dim=3 + 1 + sdf_dims[-1] + sum(encoder_dims), out_dim=3 + 1 + sdf_dims[-1],
                            gcn_dims=gcn_dims, mlp_dims=gcn_hidden)
        self.surface_ref = MeshConv(input_dim=3 + 1 + sdf_dims[-1] + sum(encoder_dims),
                                    out_dim=4, gcn_dims=gcn_dims, mlp_dims=gcn_hidden)

        self.sdf_disc = SFDDiscriminator(hidden_dims=disc_dims)

    def loss(self, out):
        return out

    def get_mesh_sdf(self, points, vertices, faces):
        face_vertices = kaolin.ops.mesh.index_vertices_by_faces(vertices, faces)
        dists, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(points, face_vertices)
        signs = kaolin.ops.mesh.check_sign(vertices, faces, points)
        sdf = dists * (signs.type_as(dists) - 0.5) * 2 * (-1)  # inside is -1
        return sdf

    def shared_step(self, bpcd, bvertices, bfaces):
        out = {
            'loss': torch.zeros(1).type_as(bpcd)
        }
        # kaolin ops limits us to one unique set of faces per batch
        b, n_points, _ = bvertices.shape
        faces = bfaces[0]
        vertices = bvertices[0].view(1, n_points, 3).repeat(b, 1, 1)
        b, n_points, _ = bpcd.shape
        pcd = bpcd[0].view(1, n_points, 3).repeat(b, 1, 1)

        # make features based on conditional point cloud
        pcd_noised = pcd + torch.randn_like(pcd) * self.noise
        pe_features = torch.cat([pcd_noised, get_positional_encoding(pcd_noised, self.pe_powers * 3)], dim=-1)
        grids = self.sdf_points_encoder.voxelize(pcd_noised, pe_features)

        # predict initial sdf for each tetrahedral vertex
        tet_vertexes = self.tet_vertexes.view(1, self.n_tetrahedra_vertexes, 3).repeat(len(grids), 1, 1)
        pos_vertex_features = self.sdf_points_encoder.devoxelize(tet_vertexes, grids)
        tet_out = self.sdf_model(pos_vertex_features)
        tet_sdf, tet_features = tet_out[:, :, 0], tet_out[:, :, 1:]
        # calculate true sdf for these vertices
        true_sdf = torch.clamp(self.get_mesh_sdf(tet_vertexes, vertices, faces), min=-self.sdf_clamp,
                               max=self.sdf_clamp)

        # if we're training only on SDF
        if self.global_step < self.steps_schedule[0]:
            out['sdf_loss'] = torch.mean(torch.sum((tet_sdf - true_sdf) ** 2, dim=1), dim=0)
            out['loss'] += out['sdf_loss']
        # add volume subdivision and refinement
        if self.global_step >= self.steps_schedule[0]:
            surface_tetrahedras, non_surface_tetrahedras = get_surface_tetrahedras(self.tetrahedras, tet_sdf)
            # first refinement step
            grids = self.ref_points_encoder.voxelize(pcd_noised, pe_features)
            ref_vertex_features = self.ref_points_encoder.devoxelize(tet_vertexes, grids)
            delta_v, delta_sdf, ref_features = self.ref1(
                torch.cat([tet_vertexes, tet_out, ref_vertex_features], dim=-1), surface_tetrahedras)
            out['delta_vertex'] = torch.mean(torch.sum(delta_v ** 2, dim=[1, 2]), dim=0)

            ref_vertexes = tet_vertexes + delta_v
            ref_sdf = tet_sdf + delta_sdf

            _surface_tetrahedras = []
            _surface_vertexes = []
            _surface_sdf = []
            _surface_features = []
            for item_idx, tet in enumerate(surface_tetrahedras):
                ntet_vertexes, ntetrahedras, nfeatures = kaolin.ops.mesh.subdivide_tetmesh(
                    ref_vertexes[item_idx].unsqueeze(0), tet,
                    torch.cat([ref_sdf[item_idx].unsqueeze(-1), ref_sdf[item_idx]], dim=-1))
                _surface_tetrahedras.append(get_surface_tetrahedras(ntetrahedras, nfeatures[:, 0]))
                _surface_vertexes.append(ntet_vertexes)
                _surface_sdf.append(nfeatures[:, 0])
                _surface_features.append(nfeatures[:, 1:])
            delta_v, delta_sdf, ref_features = self.ref2(
                torch.cat([tet_vertexes, tet_out, ref_vertex_features], dim=-1), _surface_tetrahedras)
            out['delta_vertex'] += torch.mean(torch.sum(delta_v ** 2, dim=[1, 2]), dim=0)

            out['loss'] += out['delta_vertex'] * self.delta_weight

            meshes = [kaolin.ops.conversions.marching_tetrahedra(v.unsqueeze(0), t, s)
                      for v, t, s in zip(_surface_vertexes, _surface_tetrahedras, _surface_sdf)]
            mesh_vertices = [mesh[0][0] for mesh in meshes]
            mesh_faces = [mesh[1][0] for mesh in meshes]

        # surface subdivision
        if self.global_step >= self.steps_schedule[2]:
            pass

        if self.global_step >= self.steps_schedule[0]:
            pred_points = [kaolin.ops.mesh.sample_points(vertices.unsqueeze(0), faces, len(pcd))
                           for vertices, faces in zip(mesh_vertices, mesh_faces)]
            out['chamfer_loss'] = torch.mean(torch.stack([kaolin.metrics.pointcloud.chamfer_distance(p1, p2)
                                                          for p1, p2 in zip(pcd, pred_points)], dim=0))
            out['loss'] += out['chamfer_loss'] * self.chamfer_weight

        # discriminator
        if self.global_step >= self.steps_schedule[1]:
            curvatures = [calculate_gaussian_curvature(vertices, faces) for vertices, faces in zip(bvertices, bfaces)]
            indexes = [torch.arange(len(curvature)).type_as(curvature).long()[curvature >= self.curvature_threshold]
                       for curvature in curvatures]
            indexes = [index[torch.randint(0, len(index), (self.curvature_samples,)).type_as(index).long()]
                       for index in indexes]
            grid = torch.stack(torch.meshgrid(
                torch.linspace(-1, 1, self.disc_sdf_grid).type_as(mesh_vertices),
                torch.linspace(-1, 1, self.disc_sdf_grid).type_as(mesh_vertices),
                torch.linspace(-1, 1, self.disc_sdf_grid).type_as(mesh_vertices), ), dim=-1) * self.disc_sdf_scale
            grids = [vertices[index].view(self.curvature_samples, 1, 1, 1, 3) + grid.unsqueeze(0)
                     for vertices, index in zip(bvertices, indexes)]
            sdf_grids = torch.cat([self.get_mesh_sdf(grid.view(self.curvature_samples, self.disc_sdf_grid ** 3, 3),
                                                     vertices, faces)
                                  .view(self.curvature_samples, 1, self.disc_sdf_grid, self.disc_sdf_grid,
                                        self.disc_sdf_grid)
                                   for vertices, faces, grid in zip(mesh_vertices, mesh_faces, grids)], dim=0)
            disc_preds = self.sdf_disc(sdf_grids)
            out['disc_loss'] = torch.mean((disc_preds - 1) ** 2) / 2
            out['loss'] += out['disc_loss'] * self.disc_weight

        return out

    def training_step(self, batch, batch_idx, optimizer_idx):
        # generator optimization
        if optimizer_idx == 0:
            out = self.shared_step(batch['pcd'], batch['vertices'], batch['faces'])
            return out['loss']
        # discriminator optimization
        if optimizer_idx == 1:
            out = self.shared_step(batch['pcd'], batch['vertices'], batch['faces'])
            return out['disc_loss']

    def configure_optimizers(self):
        gen_optimizer = torch.optim.Adam(lr=self.learning_rate,
                                         params=list(self.sdf_points_encoder.parameters())
                                                + list(self.sdf_model.parameters())
                                                + list(self.ref_points_encoder.parameters())
                                                + list(self.ref1.parameters()) + list(self.ref2.parameters()),
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
        train_items = IndexedListWrapper(self.dataset, self.train_idxs)
        return torch.utils.data.DataLoader(train_items, batch_size=self.batch_size, shuffle=True,
                                           num_workers=2 * torch.cuda.device_count(),
                                           pin_memory=True, drop_last=False, prefetch_factor=2)

    def val_dataloader(self):
        val_items = IndexedListWrapper(self.dataset, self.val_idxs)
        return torch.utils.data.DataLoader(val_items, batch_size=self.batch_size, shuffle=True,
                                           num_workers=2 * torch.cuda.device_count(),
                                           pin_memory=True, drop_last=False, prefetch_factor=2)
