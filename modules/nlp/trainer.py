import pytorch_lightning as pl
import torch


class SessionTrainer(pl.LightningModule):

    def __init__(self, dataset, n_users, n_events, test_dataset=None, embed_dim=128, batch_size=128, learning_rate=1e-2,
                 padding_idx=0, session_size=10, hidden_dim=128, time_norm=False):
        super(SessionTrainer, self).__init__()
        self.dataset = dataset
        self.test_dataset = test_dataset if test_dataset is not None else dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.session_size = session_size

        self.user_embed = torch.nn.Embedding(num_embeddings=n_users, embedding_dim=embed_dim, padding_idx=padding_idx,
                                             max_norm=1)
        self.event_embed = torch.nn.Embedding(num_embeddings=n_events, embedding_dim=embed_dim, padding_idx=padding_idx,
                                              max_norm=1)

        self.hidden_dim = hidden_dim
        self.time_dim = hidden_dim // 4
        self.time_norm = time_norm
        if time_norm:
            self.time_layers = torch.nn.ModuleList([
                torch.nn.Linear(self.time_dim, hidden_dim),
                torch.nn.Linear(hidden_dim, 1)
            ])

    def get_time_encoding(self, t, max_time=60 * 60):
        powers = max_time ** (2 / self.time_dim * torch.arange(self.time_dim // 2).type_as(t))
        invert_powers = 1 / powers
        x = torch.matmul(t.unsqueeze(-1), invert_powers.unsqueeze(0))
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        if self.time_dim % 2 == 1:
            x = torch.nn.functional.pad(x, pad=(0, 1), value=0)
        return x

    def build_context(self, events_embed, times=None):
        # get context vector from events b (1+neg) d
        if self.time_norm and times is not None:
            times_embed = self.get_time_encoding(times)
            weights = self.time_layers[1](torch.nn.functional.silu(self.time_layers[0](times_embed)))
            weights = torch.softmax(weights, dim=-2)
            return (weights * events_embed).sum(dim=-2)
        else:
            return torch.mean(events_embed, dim=-2)

    def pred(self, events, times=None):

        users_embed = self.user_embed(torch.arange(self.user_embed.num_embeddings).type_as(events))  # u d
        events_embed = self.build_context(self.event_embed(events), times)  # b d

        preds = torch.softmax(torch.sum(users_embed.unsqueeze(0) * events_embed.unsqueeze(1), dim=-1), dim=-1)

        return preds  # b u

    def forward(self, users, events, times=None):
        users_embed = self.user_embed(users)  # b d
        events_embed = self.event_embed(events)  # b (1+neg) t d
        events_embed = self.build_context(events_embed, times)
        # do scalar product of user (b 1 d) with session embed (b (1+neg) d)
        preds = torch.softmax(torch.sum(users_embed.unsqueeze(-2) * events_embed, dim=-1), dim=-1)

        return preds

    def step(self, batch):

        preds = self.forward(batch['users'], batch['events'], batch['times'])

        return {
            # in every batch zero user is a true class
            'loss': torch.nn.functional.cross_entropy(preds, torch.zeros(len(preds)).type_as(batch['events'])),
            'preds': preds
        }

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log(f'train_loss', loss['loss'], prog_bar=True, sync_dist=True)

        return loss['loss']

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log(f'val_loss', loss['loss'], prog_bar=True, sync_dist=True)

        events = batch['events'][:, 0]
        times = batch['times'][:, 0]
        true_users = batch['users']

        preds = self.pred(events, times)
        pred_ids = torch.argmax(preds, dim=-1)

        accuracy = (true_users == pred_ids).float().mean()
        self.log(f'val_accuracy', accuracy, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        params = list(self.user_embed.parameters()) + list(self.event_embed.parameters())
        if self.time_norm:
            params += list(self.time_layers.parameters())
        optimizer = torch.optim.SGD(lr=self.learning_rate, params=params)
        return optimizer

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False,
                                           num_workers=max((2 * torch.cuda.device_count(), 2)),
                                           pin_memory=False, prefetch_factor=1)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                                           num_workers=max((2 * torch.cuda.device_count(), 2)),
                                           pin_memory=False, prefetch_factor=1)
