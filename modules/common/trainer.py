from copy import deepcopy

import lovely_tensors as lt
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from mpl_toolkits.mplot3d import art3d

from modules.common.util import *


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

    def __init__(self, clearml=None, run_async=False):
        self.clearml = clearml
        self.run_async = run_async

    def log_image(self, image, name, epoch=0):
        if self.clearml:
            self.clearml.report_image('valid', f"{name}", iteration=epoch, image=image)
        else:
            Image.fromarray(image).save(f"{name}.png")

    def log_images(self, images, prefix, epoch=0):
        for image_id, image in enumerate(images):
            self.log_image(image, f'{prefix}_{image_id}', epoch)

    def log_tensor(self, tensor, name=''):
        if tensor.numel() > 0:
            print(f'{name} --- {lt.lovely(tensor)}')
        else:
            print(f'{name} --- empty tensor of shape {tensor.shape}')

    def log_plot(self, plt, name, epoch=0):
        if self.clearml:
            self.clearml.report_matplotlib_figure(title=name, series="valid", iteration=epoch, figure=plt)
        else:
            plt.savefig(f"{name}_{epoch}.png")
        plt.close()

    def log_distribution(self, values, name, epoch=0):
        sns.kdeplot(values)
        self.log_plot(plt, name, epoch)

    def log_values(self, values, name, epoch=0):
        sns.lineplot(values)
        self.log_plot(plt, name, epoch)

    def _log_scatter2d(self, x, y, name, color=None, epoch=0):
        plt.scatter(x=x, y=y, c=None)
        self.log_plot(plt, name, epoch)

    def log_scatter2d(self, x, y, name, color=None, epoch=0):
        if self.run_async:
            run_async(self._log_scatter2d, x, y, name, color=color, epoch=epoch)
        else:
            self._log_scatter2d(x, y, name, color=color, epoch=epoch)
    def _log_scatter3d(self, x, y, z, name, color=None, epoch=0):
        ax = plt.axes(projection="3d")
        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(-1, 1)
        ax.scatter3D(x, y, z, c=color)
        self.log_plot(plt, name, epoch)

    def log_scatter3d(self, x, y, z, name, color=None, epoch=0):
        if self.run_async:
            run_async(self._log_scatter3d, x, y, z, name, color=color, epoch=epoch)
        else:
            self._log_scatter3d(x, y, z, name, color=color, epoch=epoch)

    def _log_mesh(self, vertices, faces, name, epoch=0):
        pc = art3d.Poly3DCollection(vertices[faces],
                                    facecolors=np.ones((len(faces), 3), dtype=np.float32) * 0.75, edgecolor="gray")
        ax = plt.axes(projection="3d")
        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(-1, 1)
        ax.add_collection(pc)
        self.log_plot(plt, name, epoch)

    def log_mesh(self, vertices, faces, name, epoch=0):
        if self.run_async:
            run_async(self._log_mesh, vertices, faces, name, epoch=epoch)
        else:
            self._log_mesh(vertices, faces, name, epoch=epoch)
