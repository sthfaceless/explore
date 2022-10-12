import argparse
import gc
import json
import os
from collections import defaultdict
import time

import clearml
import msgpack
import numpy as np
import torch
from tqdm import tqdm

from instant_utils import TestbedSnapshot


class ReshapeLayer(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(*self.shape)


class NgpWeightsDiscriminator(torch.nn.Module):

    def __init__(self, model_offsets, density_res, num_layers, layer_features, hidden_dims=[1024, 512, 256]):
        super().__init__()
        self.density_network_offset = model_offsets[0]
        self.rgb_network_offset = model_offsets[1]
        self.pos_enc_offset = model_offsets[2]
        self.density_res = density_res
        self.num_layers = num_layers
        self.layer_features = layer_features

        self.layers_hidden_dims = [4, 8, 16, 32]
        self.layer_encoders = torch.nn.ModuleList([])
        layer_encoders_out = 0
        for layer_id in range(self.num_layers):

            layers = []
            for prev_dim, dim in zip([self.layer_features] + self.layers_hidden_dims, self.layers_hidden_dims):
                layers.append(torch.nn.Sequential(
                    torch.nn.Conv3d(prev_dim, dim, kernel_size=3, stride=2, padding=1),
                    torch.nn.BatchNorm3d(dim),
                    torch.nn.LeakyReLU(),
                    torch.nn.Conv3d(dim, dim, kernel_size=3, padding=1),
                    torch.nn.BatchNorm3d(dim),
                    torch.nn.LeakyReLU()
                ))
            layers.append(torch.nn.Flatten())
            layer_encoder = torch.nn.Sequential(*layers)
            self.layer_encoders.append(layer_encoder)

            layer_encoders_out += 1 ** 3 * self.layers_hidden_dims[-1]
            self.layers_hidden_dims.append(self.layers_hidden_dims[-1] * 2)

        layers = []
        for prev_dim, dim in zip([layer_encoders_out] + hidden_dims, hidden_dims):
            layers.append(torch.nn.Sequential(
                torch.nn.Linear(prev_dim, dim),
                torch.nn.BatchNorm1d(dim),
                torch.nn.LeakyReLU()
            ))
        self.head = torch.nn.Sequential(
            *layers,
            torch.nn.Linear(hidden_dims[-1], 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        embeddings = []
        for layer_id in self.num_layers:
            embeddings.append(self.layer_encoders[layer_id](x[f'layer_{layer_id}']))
        embedding = torch.cat(embeddings, dim=1)
        pred = self.head(embedding)
        return pred


class BaseVae(torch.nn.Module):

    def __init__(self, latent_dim, model_offsets, density_res, num_layers, layer_features):
        super().__init__()
        # setting model params
        self.density_network_offset = model_offsets[0]
        self.rgb_network_offset = model_offsets[1]
        self.pos_enc_offset = model_offsets[2]
        self.density_res = density_res
        self.num_layers = num_layers
        self.layer_features = layer_features

        self.latent = latent_dim

        ### AutoEncoder for head
        # self.network_hidden_dims = [512, 256, 128]
        # self.network_input = model_offsets[1]
        # layers = []
        # for prev_dim, dim in zip([self.network_input] + self.network_hidden_dims[:-1], self.network_hidden_dims):
        #     layers.append(torch.nn.Sequential(
        #         torch.nn.Linear(prev_dim, dim),
        #         torch.nn.BatchNorm1d(dim),
        #         torch.nn.LeakyReLU()
        #     ))
        # self.network_encoder = torch.nn.Sequential(*layers)
        #
        # layers = []
        # for prev_dim, dim in zip([self.latent] + self.network_hidden_dims[::-1], self.network_hidden_dims[::-1]):
        #     layers.append(torch.nn.Sequential(
        #         torch.nn.Linear(prev_dim, dim),
        #         torch.nn.BatchNorm1d(dim),
        #         torch.nn.LeakyReLU()
        #     ))
        # self.network_decoder = torch.nn.Sequential(*layers,
        #                                            torch.nn.Linear(self.network_hidden_dims[0], self.network_input),
        #                                            torch.nn.BatchNorm1d(self.network_input),
        #                                            torch.nn.Tanh())

        ### AutoEncoder for density grid
        # self.density_hidden_dims = [1 << (i + 1) for i in range(7)]  # 64 32 16 8 4
        # layers = []
        # for prev_dim, dim in zip([1] + self.density_hidden_dims, self.density_hidden_dims):
        #     layers.append(torch.nn.Sequential(
        #         torch.nn.Conv3d(prev_dim, dim, kernel_size=5, stride=2, padding=2),
        #         torch.nn.BatchNorm3d(dim),
        #         torch.nn.LeakyReLU(),
        #         torch.nn.Conv3d(dim, dim, kernel_size=5, padding=2),
        #         torch.nn.BatchNorm3d(dim),
        #         torch.nn.LeakyReLU()
        #     ))
        # layers.append(torch.nn.Flatten())
        # self.density_encoder = torch.nn.Sequential(*layers)
        #
        # layers = [torch.nn.Sequential(
        #     torch.nn.Linear(self.latent, 1 ** 3 * self.density_hidden_dims[-1]),
        #     torch.nn.LeakyReLU(),
        #     ReshapeLayer(shape=(-1, self.density_hidden_dims[-1], 1, 1, 1))
        # )]
        # for prev_dim, dim in zip(self.density_hidden_dims[::-1], self.density_hidden_dims[-2::-1] + [1]):
        #     layers.append(torch.nn.Sequential(
        #         torch.nn.ConvTranspose3d(prev_dim, dim, kernel_size=5, stride=2, padding=2, output_padding=1),
        #         torch.nn.BatchNorm3d(dim),
        #         torch.nn.LeakyReLU(),
        #         torch.nn.Conv3d(dim, dim, kernel_size=5, padding=2),
        #         torch.nn.BatchNorm3d(dim),
        #         torch.nn.LeakyReLU()
        #     ))
        # self.density_decoder = torch.nn.Sequential(*layers,
        #                                            torch.nn.Conv3d(1, 1, kernel_size=5, padding=2),
        #                                            torch.nn.BatchNorm3d(1),
        #                                            torch.nn.Tanh())

        ### AutoEncoder for positional encoding grid
        self.layers_hidden_dims = [4, 8, 16, 32]
        self.layer_encoders = torch.nn.ModuleList([])
        self.layer_decoders = torch.nn.ModuleList([])
        layer_encoders_out = 0
        for layer_id in range(self.num_layers):

            layers = []
            for prev_dim, dim in zip([self.layer_features] + self.layers_hidden_dims, self.layers_hidden_dims):
                layers.append(torch.nn.Sequential(
                    torch.nn.Conv3d(prev_dim, dim, kernel_size=3, stride=2, padding=1),
                    torch.nn.BatchNorm3d(dim),
                    torch.nn.LeakyReLU(),
                    torch.nn.Conv3d(dim, dim, kernel_size=3, padding=1),
                    torch.nn.BatchNorm3d(dim),
                    torch.nn.LeakyReLU()
                ))
            layers.append(torch.nn.Flatten())
            layer_encoder = torch.nn.Sequential(*layers)
            self.layer_encoders.append(layer_encoder)
            layer_encoders_out += 1 ** 3 * self.layers_hidden_dims[-1]

            layers = [torch.nn.Sequential(
                torch.nn.Linear(self.latent, 1 ** 3 * self.layers_hidden_dims[-1]),
                torch.nn.LeakyReLU(),
                ReshapeLayer(shape=(-1, self.layers_hidden_dims[-1], 1, 1, 1))
            )]
            for prev_dim, dim in zip(self.layers_hidden_dims[::-1],
                                     self.layers_hidden_dims[-2::-1] + [self.layer_features]):
                layers.append(torch.nn.Sequential(
                    torch.nn.ConvTranspose3d(prev_dim, dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                    torch.nn.BatchNorm3d(dim),
                    torch.nn.LeakyReLU(),
                    torch.nn.Conv3d(dim, dim, kernel_size=3, padding=1),
                    torch.nn.BatchNorm3d(dim),
                    torch.nn.LeakyReLU()
                ))

            layer_decoder = torch.nn.Sequential(*layers,
                                                torch.nn.Conv3d(
                                                    self.layer_features, self.layer_features,
                                                    kernel_size=3, padding=1),
                                                torch.nn.BatchNorm3d(self.layer_features),
                                                torch.nn.Tanh())
            self.layer_decoders.append(layer_decoder)

            self.layers_hidden_dims.append(self.layers_hidden_dims[-1] * 2)

        encoder_out_dim = layer_encoders_out
        # encoder_out_dim += 1 ** 3 * self.density_hidden_dims[-1]
        # encoder_out_dim += self.network_hidden_dims[-1]

        mu_hidden_dim = 512
        self.mu_network = torch.nn.Sequential(
            torch.nn.Linear(encoder_out_dim, self.latent),
            # torch.nn.BatchNorm1d(mu_hidden_dim),
            # torch.nn.LeakyReLU(),
            # torch.nn.Linear(mu_hidden_dim, mu_hidden_dim),
            # torch.nn.BatchNorm1d(mu_hidden_dim),
            # torch.nn.LeakyReLU(),
            # torch.nn.Linear(mu_hidden_dim, self.latent),
        )

        sigma_hidden_dim = 512
        self.sigma_network = torch.nn.Sequential(
            torch.nn.Linear(encoder_out_dim, self.latent),
            # torch.nn.BatchNorm1d(sigma_hidden_dim),
            # torch.nn.LeakyReLU(),
            # torch.nn.Linear(sigma_hidden_dim, sigma_hidden_dim),
            # torch.nn.BatchNorm1d(sigma_hidden_dim),
            # torch.nn.LeakyReLU(),
            # torch.nn.Linear(sigma_hidden_dim, self.latent),
        )

    def encode(self, x):
        # network_embedding = self.network_encoder(x['network_weights'])
        # density_embedding = self.density_encoder(x['density_grid'])
        layers_embedding = [self.layer_encoders[layer_id](x[f'layer_{layer_id}']) for layer_id in
                            range(self.num_layers)]

        embedding = torch.cat([] + layers_embedding, dim=1)
        mu = self.mu_network(embedding)
        logvar = self.sigma_network(embedding)
        return mu, logvar

    def decode(self, z, x):
        # res = {
        #     'network_weights': self.network_decoder(z),
        #     'density_grid': self.density_decoder(z)
        # }
        res = {
            'network_weights': x['network_weights'],
            'density_grid': torch.zeros_like(x['density_grid'], device=z.device)
        }

        for layer_id in range(self.num_layers):
            res[f'layer_{layer_id}'] = self.layer_decoders[layer_id](z)

        return res

    def reparametrize(self, mu, logvar):
        sigma = torch.exp(logvar)
        return sigma * torch.rand_like(sigma) + mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        pred = self.decode(z, x)
        return x, pred, mu, logvar

    def loss(self, *args):

        x = args[0]
        pred = args[1]
        mu = args[2]
        logvar = args[3]

        losses = {
            'network_rec_loss': torch.nn.functional.mse_loss(x['network_weights'], pred['network_weights']),
            'density_grid_rec_loss': torch.nn.functional.mse_loss(x['density_grid'], pred['density_grid'])
        }
        rec_loss = losses['network_rec_loss']
        # rec_loss += losses['density_grid_rec_loss']

        for layer_id in range(self.num_layers):
            interpolated_density = (torch.nn.functional.interpolate(x['density_grid'], x[f'layer_{layer_id}'].shape[2:],
                                                                   mode='trilinear') + 1.0) / 2.0
            losses[f'layer_{layer_id}_rec_loss'] = torch.mean(((x[f'layer_{layer_id}'] - pred[f'layer_{layer_id}']) ** 2
                                                               ) * interpolated_density)
            # losses[f'layer_{layer_id}_rec_loss'] = torch.nn.functional.mse_loss(x[f'layer_{layer_id}'],
            #                                                                     pred[f'layer_{layer_id}'])

            rec_loss += losses[f'layer_{layer_id}_rec_loss']

        rec_loss /= (0 + self.num_layers)
        losses['rec_loss'] = rec_loss

        kl_loss = torch.mean(torch.sum(
            -logvar + (mu ** 2 + torch.exp(2 * logvar)) / 2 - 0.5, dim=1), dim=0)
        losses['kl_loss'] = kl_loss

        latent_l1_reg = (torch.norm(mu, 1) + torch.norm(logvar, 1)) / (2.0 * self.latent)
        losses['latent_l1_reg'] = latent_l1_reg

        losses['loss'] = rec_loss + 1e-2 * kl_loss + 1e-6 * latent_l1_reg

        return losses

    def sample(self, n_items, device, x):
        points = torch.randn(n_items, self.latent, device=device)
        with torch.no_grad():
            samples = self.decode(points, x)
        return samples

    def log_epoch(self, logger):
        pass


class NgpWeightsDataset(torch.utils.data.Dataset):

    def __init__(self, items, dataset_root,
                 weights_suffix='weights.npz', params_key='params', density_key='density', **kwargs):
        super().__init__()
        self.dataset_root = dataset_root
        self.items = items
        self.weights_suffix = weights_suffix
        self.params_key = params_key
        self.density_key = density_key
        self.config = kwargs

    def __len__(self):
        return len(self.items)

    def postprocess(self, samples, batched=True):

        for name, w in samples.items():
            samples[name] = w.cpu().numpy()
            if not batched:
                samples[name] = np.expand_dims(samples[name], axis=0)

        n_items = samples['density_grid'].shape[0]
        samples['density_grid'] = np.exp(samples['density_grid'] * 13.0).astype(np.float16)
        # samples['density_grid'] = np.ones_like(samples['density_grid']).astype(np.float16)
        samples['network_weights'] = (samples['network_weights'] * 3.0).astype(np.float16)
        for layer_id in range(model.num_layers):
            samples[f'layer_{layer_id}'] = (samples[f'layer_{layer_id}'] * 10.0).astype(np.float16) \
                .transpose((0, 2, 3, 4, 1))  # batch dimension same and channel dimension last

        params = []
        params.append(samples['network_weights'].reshape(n_items, -1))
        for layer_id in range(model.num_layers):
            params.append(samples[f'layer_{layer_id}'].reshape(n_items, -1))

        density_grid = samples['density_grid'].reshape(n_items, -1)
        params = np.concatenate(params, axis=1)

        if not batched:
            params = np.squeeze(params, axis=0)
            density_grid = np.squeeze(density_grid, axis=0)

        return params, density_grid

    def preprocess(self, params, density_grid, batched=True):

        if not batched:
            params = np.expand_dims(params, axis=0)
            density_grid = np.expand_dims(density_grid, axis=0)

        n_items = density_grid.shape[0]
        density_res = self.config['density_res']
        item = {
            'density_grid': np.log(1e-6 + density_grid.reshape((n_items, 1, density_res, density_res, density_res))
                                   .clip(0.0, 1e5).astype(np.float32)) / 13.0
        }

        model_offsets = self.config['model_offsets']
        item['network_weights'] = params[:, :model_offsets[1]].clip(-3.0, 3.0).astype(np.float32) / 3.0

        layer_res = self.config['layer_res']
        layer_features = self.config['layer_features']
        prev_offset = model_offsets[1]
        for layer_id in range(self.config['num_layers']):
            res = layer_res[layer_id]
            n_weights = res ** 3 * layer_features
            item[f'layer_{layer_id}'] \
                = params[:, prev_offset:prev_offset + n_weights].reshape(n_items, res, res, res, layer_features) \
                      .transpose((0, 4, 1, 2, 3)).clip(-10.0, 10.0).astype(np.float32) / 10.0
            # make the features dimension first to be compatible with torch convolutions
            prev_offset += n_weights

        if not batched:
            for name, arr in item.items():
                item[name] = np.squeeze(arr, axis=0)

        return item

    def __getitem__(self, idx):
        path = self.items[idx]
        file = np.load(os.path.join(self.dataset_root, path, self.weights_suffix))
        item = self.preprocess(file[self.params_key], file[self.density_key], batched=False)
        return item


class NpgWeightsTrainer:

    def __init__(self, model, dataset_root, save_model, **kwargs):
        self.model = model
        self.dataset_root = dataset_root
        self.kwargs = kwargs
        self.save_model = save_model

    def gallery(self, array, ncols=4):
        nindex, height, width, color = array.shape
        nrows = nindex // ncols
        result = (array.reshape(nrows, ncols, height, width, color)
                  .swapaxes(1, 2)
                  .reshape(height * nrows, width * ncols, color))
        return result

    def get_galleries(self, params, density_grid):
        testbed = TestbedSnapshot()
        obj = testbed.get_snapshot_body(self.dataset_root)
        samples = params.shape[0]
        galleries = []
        for sample_id in range(samples):
            obj['snapshot']['density_grid_binary'] = density_grid[sample_id].tobytes(order='C')
            obj['snapshot']['params_binary'] = params[sample_id].tobytes(order='C')

            timestamp = int(time.time())
            snap_name = f'tmp_{timestamp}.msgpack'
            with open(snap_name, 'wb') as f:
                packed = msgpack.packb(obj)
                f.write(packed)

            images = np.array(testbed.get_screenshots(snap_name, img_size=128))
            img = self.gallery(images, ncols=4)
            galleries.append(img)
            os.remove(snap_name)

        del testbed, obj
        gc.collect()
        torch.cuda.empty_cache()

        return galleries

    def fit(self, epochs, logger, device=torch.device('cpu'), batch_size=1, workers=8,
            learning_rate=1e-2, decay_rate=0.95, train_factor=0.7, loss_eps=1e-6, patience=3, val_samples=5):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)

        items = [item for item in os.listdir(self.dataset_root)
                 if os.path.isdir(os.path.join(self.dataset_root, item)) and os.listdir(
                os.path.join(self.dataset_root, item))]
        train_size = int(len(items) * train_factor)
        test_size = len(items) - train_size

        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        history = defaultdict(lambda: defaultdict(lambda: []))
        best_loss = 1e6
        for epoch in range(epochs):

            print(f"Begin {epoch + 1} epoch")

            train_items, val_items = torch.utils.data.random_split(items, [train_size, test_size])

            # train model on train data
            train_dataset = NgpWeightsDataset(train_items, self.dataset_root, **self.kwargs)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                       shuffle=True, num_workers=workers)
            train_history = defaultdict(lambda: [])
            for batch_idx, batch in tqdm(enumerate(train_loader)):

                for key, tensor in batch.items():
                    batch[key] = tensor.to(device)

                optimizer.zero_grad()
                pred = self.model.forward(batch)
                out = self.model.loss(*pred)
                out['loss'].backward()

                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0 * num_params)
                optimizer.step()

                for key, value in out.items():
                    train_history[key].append(value.item())

            for key, values in train_history.items():
                value = float(np.mean(values))
                print(f"--- train {key} - {value}")
                history["train"][key].append(value)
                logger.report_scalar(title=key, series="train", value=value, iteration=epoch)

            # calculate loss and metrics on valid data
            val_dataset = NgpWeightsDataset(val_items, self.dataset_root, **self.kwargs)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                     shuffle=True, num_workers=workers)
            val_history = defaultdict(lambda: [])
            image_logged = False
            for batch_idx, batch in tqdm(enumerate(val_loader)):

                for key, tensor in batch.items():
                    batch[key] = tensor.to(device)

                with torch.no_grad():
                    pred = self.model.forward(batch)
                    out = self.model.loss(*pred)

                for key, value in out.items():
                    val_history[key].append(value.item())

                if not image_logged:
                    orig_params, orig_density_grid = val_dataset.postprocess(pred[0])
                    orig_galleries = self.get_galleries(orig_params[:val_samples], orig_density_grid[:val_samples])
                    pred_params, pred_density_grid = val_dataset.postprocess(pred[1])
                    pred_galleries = self.get_galleries(pred_params[:val_samples], pred_density_grid[:val_samples])
                    for gallery_id, (orig_gallery, pred_gallery) in enumerate(zip(orig_galleries, pred_galleries)):
                        img = self.gallery(np.array([orig_gallery, pred_gallery]), ncols=1)
                        logger.report_image('valid', f'rec_{gallery_id}', iteration=epoch, image=img,
                                            max_image_history=-1)

                    image_logged = True

            # decay learning rate
            lr_scheduler.step()
            # log examples of model
            self.model.log_epoch(logger)

            for key, values in val_history.items():
                value = float(np.mean(values))
                print(f"--- val {key} - {value}")
                history["val"][key].append(value)
                logger.report_scalar(title=key, series="valid", value=value, iteration=epoch)

            # model checkpoint
            if history['val']['loss'][-1] < best_loss:
                best_loss = history['val']['loss'][-1]
                if os.path.exists(self.save_model):
                    os.remove(self.save_model)
                torch.save(model.state_dict(), self.save_model)

            val_dataset = NgpWeightsDataset(val_items, self.dataset_root, **self.kwargs)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_samples, shuffle=False,
                                                     num_workers=1, prefetch_factor=1)
            # sample and render items from latent space
            params, density_grid = val_dataset.postprocess(model.sample(n_items=val_samples, device=device,
                                                                        x=next(iter(val_loader))))
            galleries = self.get_galleries(params, density_grid)
            for gallery_id, gallery in enumerate(galleries):
                logger.report_image('valid', f'sample_{gallery_id}',
                                    iteration=epoch, image=gallery, max_image_history=5)

            # early stop if needed
            if epoch + 1 > patience:
                losses = history["val"]["loss"][-patience:]
                mean_loss = sum(losses) / len(losses)
                if max(losses) - mean_loss < loss_eps and mean_loss - min(losses) < loss_eps:
                    break
            print(f"Finished {epoch + 1} epochs")

            gc.collect()
            torch.cuda.empty_cache()

        return history


class SnapshotSampler:

    def __init__(self, dataset_root, model_path, snapshot_name='snapshot.msgpack'):
        self.dataset_root = dataset_root
        self.snapshot_name = snapshot_name
        self.model_path = model_path

    def sample(self, out, n=1, device=torch.device('cpu')):
        with open(f'{self.model_path}.cfg') as f:
            cfg = json.load(f)
        model = BaseVae(**cfg)
        model.load_state_dict(torch.load(self.model_path))
        model.to(device)

        dataset = NgpWeightsDataset(items=[], dataset_root=self.dataset_root)
        params, density_grid = dataset.postprocess(model.sample(n, device))

        testbed = TestbedSnapshot()
        obj = testbed.get_snapshot_body(self.dataset_root, self.snapshot_name)

        for sample_id in tqdm(range(n)):
            obj['snapshot']['density_grid_binary'] = density_grid[sample_id].tobytes(order='C')
            obj['snapshot']['params_binary'] = params[sample_id].tobytes(order='C')

            with open(f"{out}_{sample_id}.msgpack", 'wb') as f:
                packed = msgpack.packb(obj)
                f.write(packed)


def parse_args():
    parser = argparse.ArgumentParser(description="Creates simple VAE on weights from instant ngp")

    parser.add_argument("--sample", default="", help="Sample snapshots from pretrained model")
    parser.add_argument("--n_samples", default=1, type=int, help="Amount of sample snapshots")
    parser.add_argument("--load_model", default="", help="Path to pretrained model")

    parser.add_argument("--dataset", default="", help="Path to instant weights dataset")
    parser.add_argument("--save_model", default="model", help="Path to save model")
    parser.add_argument("--task_name", default="VAE Training", help="ClearML task name")

    parser.add_argument("--latent_dim", default=128, type=int, help="Latent dim in VAE")
    parser.add_argument("--hidden_dims", default=[1024, 512, 256], nargs='+', type=int, help="Hidden dims in VAE")

    parser.add_argument("--epochs", default=100, type=int, help="Epochs to train model")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate in model")
    parser.add_argument("--decay_rate", default=0.95, type=float, help="LR decay in exponential decay")
    parser.add_argument("--train_factor", default=0.7, type=float, help="Factor of training data")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size for data")
    parser.add_argument("--workers", default=4, type=int, help="Workers to load data")
    parser.add_argument("--loss_eps", default=1e-6, type=float, help="Loss epsilon for early stopping")
    parser.add_argument("--patience", default=3, type=int, help="Epochs to wait until stop training when"
                                                                " learning rate doesn't changes")

    parser.add_argument("--weights_suffix", default="weights.npz", help="Name of weights file in object dir")
    parser.add_argument("--params_key", default="params", help="Key of model params in weights file")
    parser.add_argument("--density_key", default="density", help="Key of density grid in weights file")

    parser.add_argument("--model_offsets", default=[1536, 8192, 4800512], type=int, nargs='+',
                        help="Offsets for weights in model[density network weights, rgb network weights, positional encoding weights]")
    parser.add_argument("--layer_res", default=[16, 32, 64, 128], type=int, nargs='+', help="Layer resolutions")
    parser.add_argument("--pos_encoding_layers", default=4, type=int, help="Number of layers in positional encoding")
    parser.add_argument("--pos_encoding_features", default=2, type=int,
                        help="Number of features in positional encoding")
    parser.add_argument("--pos_encoding_scale", default=2.0, type=float, help="Per level scale in positional encoding")
    parser.add_argument("--density_res", default=128, type=int, help="Resolution of density grid")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.sample:
        print("Creating snapshot sampler")
        sampler = SnapshotSampler(dataset_root=args.dataset, model_path=args.load_model)
        print("Begin snapshot sampling")
        sampler.sample(out=args.sample, n=args.n_samples, device=device)
        print("End snapshot sampling")
    else:

        print("Initializing ClearML")
        task = clearml.Task.init(project_name='3dGen', task_name=args.task_name, reuse_last_task_id=True)
        task.connect(args, name='config')

        print("Creating model and it's trainer")
        model = BaseVae(latent_dim=args.latent_dim, model_offsets=args.model_offsets, density_res=args.density_res,
                        num_layers=args.pos_encoding_layers, layer_features=args.pos_encoding_features)
        model.to(device)
        # Saving model config to instantiate it in future
        model_config = {
            'latent_dim': args.latent_dim,
            'model_offsets': args.model_offsets,
            'density_res': args.density_res,
            'num_layers': args.pos_encoding_layers,
            'layer_features': args.pos_encoding_features
        }
        with open(f'{args.save_model}.cfg', 'w') as fp:
            json.dump(model_config, fp)

        trainer = NpgWeightsTrainer(model=model, dataset_root=args.dataset, save_model=args.save_model,
                                    model_offsets=args.model_offsets, layer_res=args.layer_res,
                                    num_layers=args.pos_encoding_layers, layer_features=args.pos_encoding_features,
                                    level_scale=args.pos_encoding_scale, density_res=args.density_res,
                                    weights_suffix=args.weights_suffix, params_key=args.params_key,
                                    density_key=args.density_key)

        print("Fit model")
        history = trainer.fit(epochs=args.epochs, logger=task.get_logger(), device=device,
                              learning_rate=args.learning_rate,
                              batch_size=args.batch_size, workers=args.workers,
                              decay_rate=args.decay_rate, train_factor=args.train_factor, loss_eps=args.loss_eps)

        # Saving training history
        with open(f'{args.save_model}.history', 'w') as fp:
            json.dump(history, fp)

# CUDA_VISIBLE_DEVICES=1 http_proxy= nohup python scripts/instantvae.py --dataset /dsk1/danil/3d/instant_16_4L_x2 --save_model /dsk1/danil/3d/instant_16_4L_x2/model > vae.log & disown
