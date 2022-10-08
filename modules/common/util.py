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
