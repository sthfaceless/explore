import pytorch_lightning as pl
import torch.nn

from modules.bio.model import *


class ProteinMutationTrainer(pl.LightningModule):

    def __init__(self, train_dataset=None, val_dataset=None, encoder_dims=(32, 64, 128), encoder_grids=(64, 32, 16),
                 learning_rate=1e-4, min_lr_rate=0.5, epochs=30, steps=1000, batch_size=32,
                 unique_atoms=36, atoms_embedding_dim=128, generated_features=48):
        super(ProteinMutationTrainer, self).__init__()

        self.save_hyperparameters(ignore=['train_dataset', 'val_dataset'])

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.learning_rate = learning_rate
        self.min_lr_rate = min_lr_rate
        self.epochs = epochs
        self.steps = steps
        self.batch_size = batch_size

        self.atom_embeds = torch.nn.Embedding(num_embeddings=unique_atoms + 1, embedding_dim=atoms_embedding_dim,
                                              padding_idx=unique_atoms)
        self.point_encoder = MultiPointVoxelCNN(input_dim=generated_features * 2 + atoms_embedding_dim * 2,
                                                dims=encoder_dims, grids=encoder_grids, do_points_map=True)
        self.point_regression = PointRegression(input_dim=3 * sum(encoder_dims))

    def forward(self, batch):
        wt_atoms, mut_atoms = self.atom_embeds(batch['wt_atom_ids'].long()), \
                              self.atom_embeds(batch['mut_atom_ids'].long())
        wt_features = torch.cat([batch['wt_features'], batch['wt_features'] - batch['mut_features'], wt_atoms,
                                 wt_atoms - mut_atoms], dim=-1)
        wt_features = self.point_encoder(batch['wt_points'], wt_features)
        mut_features = torch.cat([batch['mut_features'], batch['mut_features'] - batch['wt_features'], mut_atoms,
                                  mut_atoms - wt_atoms], dim=-1)
        mut_features = self.point_encoder(batch['mut_points'], mut_features)
        features = torch.cat([wt_features, mut_features, mut_features - wt_features], dim=-1)
        preds = self.point_regression(batch['wt_points'], features)
        return {
            'pred': preds
        }

    def shared_step(self, batch, kind='train'):
        out = self.forward(batch)
        loss = torch.nn.functional.mse_loss(batch['dT'], out['pred'])
        self.log(f'{kind}_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch)

    def configure_optimizers(self):
        opt = torch.optim.Adam(params=self.parameters(), lr=self.learning_rate, betas=(0.5, 0.9))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.learning_rate,
                                                        pct_start=3 / self.epochs, div_factor=2.0,
                                                        final_div_factor=1 / (2.0 * self.min_lr_rate),
                                                        epochs=self.epochs, steps_per_epoch=self.steps)
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step'
        }
        return [opt], [scheduler]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False,
                                           num_workers=torch.cuda.device_count() * 2,
                                           pin_memory=True, drop_last=False, prefetch_factor=2)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                                           num_workers=torch.cuda.device_count() * 2,
                                           pin_memory=True, drop_last=False, prefetch_factor=2)
