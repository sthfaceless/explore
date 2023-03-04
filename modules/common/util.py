import json
import random
import threading

import numpy as np
import torch


class IndexedListWrapper:

    def __init__(self, lst, indexes):
        self.lst = lst
        self.indexes = indexes

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        return self.lst[self.indexes[idx]]


class RandomIndexedListWrapper(torch.utils.data.IterableDataset):

    def __init__(self, lst, indexes):
        self.lst = lst
        self.indexes = indexes

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                idx = random.randint(0, len(self.indexes) - 1)
                return self.lst[self.indexes[idx]]
            except Exception as e:
                print(f'Exception occured during loading dataset object {str(e)}')


def merge_history(history, outs, key='train'):
    if key not in history:
        history[key] = {}
    values = {}
    for out in outs:
        for k, val in out.items():
            if k in values:
                values[k].append(val)
            else:
                values[k] = [val]
    for k, vals in values.items():
        if k not in history[key]:
            history[key][k] = [float(np.mean(np.array(vals)))]
        else:
            history[key][k].append(float(np.mean(np.array(vals))))


def early_stop(history, patience=5, eps=1e-4, monitor='test', metric='loss'):
    if len(history[monitor][metric]) < patience:
        return False
    loss_history = np.array(history[monitor][metric][-patience:])
    if np.mean(loss_history - np.mean(loss_history)) < eps:
        return True
    else:
        return False


def save_best(history, model, model_name, monitor='test', metric='loss'):
    if len(history[monitor][metric]) == 1 or history[monitor][metric][-1] < np.min(
            np.array(history[monitor][metric][:-1])):
        torch.save(model.state_dict(), model_name)
        with open(f'{model_name}.history', 'w') as f:
            json.dump(history, f)


def normalize_image(image):
    return image.astype(np.float32) / (255 / 2) - 1.0


def denormalize_image(image):
    return ((image + 1.0) / 2 * 255).astype(np.uint8)


def get_timestep_encoding(t, dim, steps):
    t = t.float()
    powers = steps ** (2 / dim * torch.arange(dim // 2).type_as(t))
    invert_powers = 1 / powers
    x = torch.matmul(t.unsqueeze(-1), invert_powers.unsqueeze(0))
    x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
    if dim % 2 == 1:
        x = torch.nn.functional.pad(x, pad=(0, 1), value=0)
    return x  # (b dim)


def get_positional_encoding(x, features):
    dim = x.shape[-1]
    x = x.float()
    powers = 2 ** torch.arange(features // (2 * dim)).type_as(x)
    h = torch.matmul(x.unsqueeze(-1), powers.unsqueeze(0))
    h = torch.cat([torch.sin(h), torch.cos(h)], dim=-1)
    h = h.view(*h.shape[:-2], -1)  # dim, p -> dim * p
    if h.shape[-1] < features:
        h = torch.nn.functional.pad(h, pad=(0, features - h.shape[-1]), value=0)
    return h


def get_numpy_positional_encoding(points, pe_features=16):
    powers = 2 ** np.arange(pe_features // 2, dtype=np.float32)
    h = np.matmul(np.expand_dims(points, axis=-1), np.expand_dims(powers, axis=0))
    h = np.concatenate([np.sin(h), np.cos(h)], axis=-1)
    h = h.reshape((*h.shape[:-2], -1))  # dim, p -> dim * p
    return h


def approx_standard_normal_cdf(x):
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def log_tensor_size(x, name):
    size_bytes = x.element_size() * x.nelement()
    size_mb = size_bytes / 1e6
    print(f'{name} size --- {size_mb:.2f}mb')


def log_tensors_size(tensors, name):
    for idx, tensor in enumerate(tensors):
        log_tensor_size(tensor, f'{name}_{idx}')


def collate_dicts(dicts):
    if len(dicts) == 0:
        return {}
    result = {}
    for d in dicts:
        for k, v in d.items():
            if k in result:
                result[k].append(v)
            else:
                result[k] = [v]

    return result


def exists(*args):
    exist = True
    for arg in args:
        exist = exist and arg is not None
    return exist


def normalize_vector(v):
    return v / torch.norm(v, p=2, dim=-1, keepdim=True)


def run_async(fun, *args, **kwargs):
    thread = threading.Thread(target=fun, args=args, kwargs=kwargs)
    thread.start()


def tn(tensor):
    return tensor.detach().cpu().numpy()


def weight_loss(loss, weight):
    return torch.nan_to_num(loss, nan=0, posinf=0, neginf=0) * weight


def repeat_dim(tensor, dim, n):
    shape = []
    for idx in range(len(tensor.shape)):
        if idx == dim:
            shape.append(n)
        else:
            shape.append(1)

    return tensor.repeat(*shape)


def prepare_torch_images(images):
    images = images.detach().cpu().numpy()
    images = np.clip(np.moveaxis(images, -3, -1), -1.0, 1.0)
    images = denormalize_image(images)
    return images


def tensor2list(tensor):
    return [tensor[idx] for idx in range(len(tensor))]
