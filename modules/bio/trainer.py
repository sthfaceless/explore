import pytorch_lightning as pl
import torch.nn

from modules.bio.model import *
from modules.ddd.model import *


class ProteinMutationTrainer(pl.LightningModule):

    def __init__(self, train_dataset=None, val_dataset=None, encoder_dims=(32, 64, 128), encoder_grids=(32, 16, 8),
                 encoder_dim=128, learning_rate=1e-4, min_lr_rate=0.5, epochs=30, steps=1000, batch_size=32,
                 unique_atoms=36, atoms_embedding_dim=128, generated_features=48, seq_len=400, regression_blocks=8,
                 pe_powers=16):
        super(ProteinMutationTrainer, self).__init__()

        self.save_hyperparameters(ignore=['train_dataset', 'val_dataset'])

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.learning_rate = learning_rate
        self.min_lr_rate = min_lr_rate
        self.epochs = epochs
        self.steps = steps
        self.batch_size = batch_size
        self.pe_powers = pe_powers

        self.atom_embeds = torch.nn.Embedding(num_embeddings=unique_atoms + 1, embedding_dim=atoms_embedding_dim,
                                              padding_idx=unique_atoms)
        self.point_encoder = MultiPointVoxelCNN(input_dim=generated_features * 2 + atoms_embedding_dim * 2,
                                                dim=encoder_dim,
                                                dims=encoder_dims, grids=encoder_grids, do_points_map=True)
        self.point_regression = PointRegression(input_dim=3 * (encoder_dim + pe_powers * 3), seq_len=seq_len,
                                                n_blocks=regression_blocks)

    def forward(self, batch):
        # get embeddings for atoms in space
        wt_atoms, mut_atoms = self.atom_embeds(batch['wt_atom_ids'].long()), \
                              self.atom_embeds(batch['mut_atom_ids'].long())
        # construct features with atom embeddings and nerf like positional encodings
        # WILD-TYPE
        wt_features = torch.cat([batch['wt_features'], batch['wt_features'] - batch['mut_features'], wt_atoms,
                                 wt_atoms - mut_atoms], dim=-1)
        wt_grids = self.point_encoder.voxelize(batch['wt_points'], wt_features, mask=batch['wt_mask'])
        wt_features = self.point_encoder.devoxelize(batch['wt_alpha_points'], wt_grids, mask=batch['wt_alpha_mask'])
        wt_pos_features = get_positional_encoding(batch['wt_alpha_points'], self.pe_powers * 3)
        wt_pos_features = torch.where(batch['wt_alpha_mask'], wt_pos_features, torch.zeros_like(wt_pos_features))
        wt_features = torch.cat([wt_features, wt_pos_features], dim=-1)
        # MUTANT encoding
        mut_features = torch.cat([batch['mut_features'], batch['mut_features'] - batch['wt_features'], mut_atoms,
                                  mut_atoms - wt_atoms], dim=-1)
        mut_grids = self.point_encoder.voxelize(batch['mut_points'], mut_features, mask=batch['mut_mask'])
        mut_features = self.point_encoder.devoxelize(batch['mut_alpha_points'], mut_grids, mask=batch['mut_alpha_mask'])
        mut_pos_features = get_positional_encoding(batch['mut_alpha_points'], self.pe_powers * 3)
        mut_pos_features = torch.where(batch['mut_alpha_mask'], mut_pos_features, torch.zeros_like(mut_pos_features))
        mut_features = torch.cat([mut_features, mut_pos_features], dim=-1)
        # Features concat
        features = torch.cat([wt_features, mut_features, mut_features - wt_features], dim=-1)
        # do regression on volume features with masking
        feature_mask = torch.logical_or(batch['wt_alpha_mask'], batch['mut_alpha_mask'])
        preds = self.point_regression(features, mask=feature_mask)
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
