import json

import numpy as np
import torch


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


def approx_standard_normal_cdf(x):
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))
