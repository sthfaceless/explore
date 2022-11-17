import torch


# from stl import mesh


# Positional Encoding from https://github.com/yenchenlin/nerf-pytorch/blob/1f064835d2cca26e4df2d7d130daa39a8cee1795/run_nerf_helpers.py
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires):
    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class Decoder(torch.nn.Module):

    def __init__(self, input_dims=3, internal_dims=128, output_dims=4, hidden=5, multires=2):
        super().__init__()
        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            input_dims = input_ch

        net = (torch.nn.Linear(input_dims, internal_dims, bias=False), torch.nn.ReLU())
        for i in range(hidden - 1):
            net = net + (torch.nn.Linear(internal_dims, internal_dims, bias=False), torch.nn.ReLU())
        net = net + (torch.nn.Linear(internal_dims, output_dims, bias=False),)
        self.net = torch.nn.Sequential(*net)

    def forward(self, p):
        if self.embed_fn is not None:
            p = self.embed_fn(p)
        out = self.net(p)
        return out

    def pre_train_sphere(self, iter):
        print("Initialize SDF to sphere")
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(list(self.parameters()), lr=1e-4)

        for i in range(iter):
            p = torch.rand((1024, 3), device='cuda') - 0.5
            ref_value = torch.sqrt((p ** 2).sum(-1)) - 0.3
            output = self(p)
            loss = loss_fn(output[..., 0], ref_value)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Pre-trained MLP", loss.item())


def kao():
    import numpy as np

    import kaolin

    # path to the point cloud to be reconstructed
    pcd_path = "/dsk1/kaolin/examples/samples/bear_pointcloud.usd"
    # pcd_path  = '/dsk1/kaolin/examples/samples/untitled.ply'

    # pcd_path = "/dsk1/get3d/untitled_low.txt"
    # path to the output logs (readable with the training visualizer in the omniverse app)
    logs_path = '/dsk1/kaolin/examples/samples/logs'

    # We initialize the timelapse that will store USD for the visualization apps
    timelapse = kaolin.visualize.Timelapse(logs_path)

    # arguments and hyperparameters
    device = 'cuda'
    lr = 1e-3
    laplacian_weight = 0.01
    iterations = 5000
    save_every = 100
    multires = 2
    grid_res = 128

    # file = open(pcd_path).readlines()
    # tri = []
    # answ = []
    # n = 1
    # for num in file:
    #     real_num =  round(float(num[:-1]), 5)
    #     tri.append(real_num)

    #     if n%3 == 0:
    #         if tri in answ:
    #             pass
    #         else:
    #             answ.append(tri)
    #         tri = []
    #     n+=1

    # points = torch.FloatTensor(np.array(answ)).to(device)

    # surface = mesh.Mesh.from_file('/dsk1/get3d/96k.stl')
    # point_list = surface.points.reshape([-1, 3])
    # point_list = np.around(point_list,2)
    # vertices, vertex_indices = np.unique(point_list, return_inverse=True, axis=0)
    # points = torch.FloatTensor(vertices).to(device)

    points = kaolin.io.usd.import_pointclouds(pcd_path)[0].points.to(device)
    # print(points.shape)

    if points.shape[0] > 125000:
        idx = list(range(points.shape[0]))
        np.random.shuffle(idx)
        idx = torch.tensor(idx[:125000], device=points.device, dtype=torch.long)
        points = points[idx]

    # The reconstructed object needs to be slightly smaller than the grid to get watertight surface after MT.
    center = (points.max(0)[0] + points.min(0)[0]) / 2
    max_l = (points.max(0)[0] - points.min(0)[0]).max()
    points = ((points - center) / max_l) * 0.9
    timelapse.add_pointcloud_batch(category='input',
                                   pointcloud_list=[points.cpu()], points_type="usd_geom_points")

    tet_verts = torch.tensor(np.load('/dsk1/kaolin/examples/samples/{}_verts.npz'.format(grid_res))['data'],
                             dtype=torch.float, device=device)
    tets = torch.tensor(
        ([np.load('/dsk1/kaolin/examples/samples/{}_tets_{}.npz'.format(grid_res, i))['data'] for i in range(4)]),
        dtype=torch.long, device=device).permute(1, 0)

    # Initialize model and create optimizer
    model = Decoder(multires=multires).to(device)
    model.pre_train_sphere(1000)

    # Laplacian regularization using umbrella operator (Fujiwara / Desbrun).
    # https://mgarland.org/class/geom04/material/smoothing.pdf
    def laplace_regularizer_const(mesh_verts, mesh_faces):
        term = torch.zeros_like(mesh_verts)
        norm = torch.zeros_like(mesh_verts[..., 0:1])

        v0 = mesh_verts[mesh_faces[:, 0], :]
        v1 = mesh_verts[mesh_faces[:, 1], :]
        v2 = mesh_verts[mesh_faces[:, 2], :]

        term.scatter_add_(0, mesh_faces[:, 0:1].repeat(1, 3), (v1 - v0) + (v2 - v0))
        term.scatter_add_(0, mesh_faces[:, 1:2].repeat(1, 3), (v0 - v1) + (v2 - v1))
        term.scatter_add_(0, mesh_faces[:, 2:3].repeat(1, 3), (v0 - v2) + (v1 - v2))

        # just assign equal weights in laplace regularizer
        two = torch.ones_like(v0) * 2.0
        norm.scatter_add_(0, mesh_faces[:, 0:1], two)
        norm.scatter_add_(0, mesh_faces[:, 1:2], two)
        norm.scatter_add_(0, mesh_faces[:, 2:3], two)

        term = term / torch.clamp(norm, min=1.0)

        return torch.mean(term ** 2)

    def loss_f(mesh_verts, mesh_faces, points, it):
        pred_points = kaolin.ops.mesh.sample_points(mesh_verts.unsqueeze(0), mesh_faces, 50000)[0][0]
        # print(mesh_faces)
        chamfer = kaolin.metrics.pointcloud.chamfer_distance(pred_points.unsqueeze(0), points.unsqueeze(0)).mean()
        if it > iterations // 2:
            lap = laplace_regularizer_const(mesh_verts, mesh_faces)
            return chamfer + lap * laplacian_weight
        return chamfer

    vars = [p for _, p in model.named_parameters()]
    optimizer = torch.optim.Adam(vars, lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: max(0.0, 10 ** (
                -x * 0.0002)))  # LR decay over time

    for it in range(iterations):
        pred = model(tet_verts)  # predict SDF and per-vertex deformation
        sdf, deform = pred[:, 0], pred[:, 1:]
        verts_deformed = tet_verts + torch.tanh(deform) / grid_res  # constraint deformation to avoid flipping tets

        mesh_verts, mesh_faces = kaolin.ops.conversions.marching_tetrahedra(verts_deformed.unsqueeze(0), tets,
                                                                            sdf.unsqueeze(
                                                                                0))  # running MT (batched) to extract surface mesh
        mesh_verts, mesh_faces = mesh_verts[0], mesh_faces[0]
        loss = loss_f(mesh_verts, mesh_faces, points, it)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (it) % save_every == 0 or it == (iterations - 1):
            print('Iteration {} - loss: {}, # of mesh vertices: {}, # of mesh faces: {}'.format(it, loss,
                                                                                                mesh_verts.shape[0],
                                                                                                mesh_faces.shape[0]))
            # save reconstructed mesh

            timelapse.add_mesh_batch(
                iteration=it + 1,
                category=f'extracted_mesh_{it}',
                vertices_list=[mesh_verts.cpu()],
                faces_list=[mesh_faces.cpu()]
            )


if __name__ == "__main__":
    kao()