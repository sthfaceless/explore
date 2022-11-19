import pytorch_lightning as pl
import torch
from modules.dmtet.util import *


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

        net = torch.nn.ModuleList([torch.nn.Linear(input_dims, internal_dims, bias=False)])
        for i in range(hidden - 1):
            net = net + [torch.nn.Linear(internal_dims, internal_dims, bias=False)]
        self.net = net
        self.out = torch.nn.Linear(internal_dims, output_dims, bias=False)

    def forward(self, p):
        if self.embed_fn is not None:
            p = self.embed_fn(p)
        for block in self.net:
            p = torch.nn.functional.leaky_relu(block(p))
        out = self.out(p)
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


class PcdDenoiser(pl.LightningModule):

    def __init__(self, dataset=None, grid_resolution=128, learning_rate=1e-3, epochs=30, steps=10000, min_lr_rate=0.1,
                 batch_size=32, pe_encodings=2, hidden_dim=128, hidden_layers=5, sphere_pretraining=1000):
        super(PcdDenoiser, self).__init__()

        self.dataset = dataset
        self.save_hyperparameters(ignore=['dataset'])

        self.epochs = epochs
        self.steps = steps
        self.learning_rate = learning_rate
        self.min_lr_rate = min_lr_rate
        self.batch_size = batch_size

        self.model = Decoder(multires=pe_encodings, internal_dims=hidden_dim, hidden=hidden_layers)
        self.model.pre_train_sphere(sphere_pretraining)

        self.grid_resolution = grid_resolution
        self.tet_verts = get_tetrahedra_verts(grid_resolution)
        self.tets = get_tetrahedras(grid_resolution)


    def step(self, batch):

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(lr=self.learning_rate, params=self.parameters(), betas=(0.9, 0.99))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.learning_rate,
                                                        pct_start=3 / self.epochs, div_factor=2.0,
                                                        final_div_factor=1 / (2.0 * self.min_lr_rate),
                                                        epochs=self.epochs, steps_per_epoch=self.steps)
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step'
        }
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,
                                           num_workers=2 * torch.cuda.device_count(),
                                           pin_memory=True, drop_last=False, prefetch_factor=2)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,
                                           num_workers=2 * torch.cuda.device_count(),
                                           pin_memory=True, drop_last=False, prefetch_factor=2)
