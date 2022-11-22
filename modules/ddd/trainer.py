import pytorch_lightning as pl

from modules.common.util import *
from modules.ddd.util import *
from modules.common.trainer import SimpleLogger
from modules.ddd.model import *
from random import shuffle

class PCD2Mesh(pl.LightningModule):

    def __init__(self, dataset=None, clearml=None, train_rate=0.8, grid_resolution=128, learning_rate=1e-4,
                 steps_schedule=(1000, 20000, 50000, 100000), min_lr_rate=1.0,
                 batch_size=16, pe_encodings=2, hidden_dim=128, hidden_layers=5, sphere_pretraining=1000):
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
        self.tet_vertexes = tet_vertexes
        self.tetrahedras = tetrahedras

    def step(self, batch):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(lr=self.learning_rate, params=self.parameters(), betas=(0.9, 0.99))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.learning_rate,
                                                        pct_start=self.steps_schedule[0] / self.steps_schedule[-1],
                                                        div_factor=2.0, final_div_factor=1 / (2.0 * self.min_lr_rate),
                                                        total_steps=self.steps_schedule[-1])
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step'
        }
        return [optimizer], [scheduler]

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
