from copy import deepcopy

import torch
from PIL import Image

class EMA(torch.nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super(EMA, self).__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)
        for param in self.module.parameters():
            param.requires_grad_(False)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class SimpleLogger:

    def __init__(self, clearml=None):
        self.clearml = clearml

    def log_image(self, image, name, epoch=0):
        if self.clearml:
            self.clearml.report_image('valid', f"{name}", iteration=epoch, image=image)
        else:
            Image.fromarray(image).save(f"{name}.png")

    def log_images(self, images, prefix, epoch=0):
        for image_id, image in enumerate(images):
            self.log_image(image, f'{prefix}_{image_id}', epoch)
