import os
from copy import deepcopy

import cv2
import imageio
import lovely_tensors as lt
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from mpl_toolkits.mplot3d import art3d

from modules.common.util import *

import torch
import numpy as np


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

    def __init__(self, clearml=None, run_async=False, tmpdir='.'):
        self.clearml = clearml
        self.run_async = run_async
        self.tmpdir = tmpdir
        if not os.path.exists(tmpdir):
            os.makedirs(tmpdir, exist_ok=True)

    def log_value(self, value, name, epoch=0, kind='val'):
        if self.clearml:
            self.clearml.report_scalar(name, kind, iteration=epoch, value=value)
        else:
            print(f'{name} --- {value}')

    def log_image(self, image, name, epoch=0):
        path = f"{self.tmpdir}/{name}.png"
        Image.fromarray(image).save(path)
        if self.clearml:
            self.clearml.report_image('valid', f"{name}", iteration=epoch, local_path=path)

    def log_images(self, images, prefix, epoch=0):
        for image_id, image in enumerate(images):
            self.log_image(image, f'{prefix}_{image_id}', epoch)

    def log_image_compare(self, images, texts, epoch, name='compare'):
        gallery = []
        for img, text in zip(images, texts):
            cv2.rectangle(img, pt1=(0, img.shape[0] // 30), pt2=(img.shape[1] // 90 * len(text), 0), color=0,
                          thickness=-1)
            cv2.putText(img, text=text, org=(0, img.shape[0] // 40), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3.0,
                        color=(255, 255, 255), thickness=3)
            gallery.append(img)
        gallery = np.concatenate(gallery, axis=1)
        if self.clearml:
            self.clearml.report_image('valid', name, iteration=epoch, image=gallery)
        else:
            Image.fromarray(gallery).save(f"{self.tmpdir}/{name}.png")

    def log_images_compare(self, images, texts, epoch, name='compare'):
        for idx, batch in enumerate(zip(*images)):
            self.log_image_compare(batch, texts, epoch=epoch, name=f'{name}_{idx}')

    def log_tensor(self, tensor, name='', depth=0):
        if tensor.numel() > 0:
            print(f'{name} --- {lt.lovely(tensor, depth=depth)}')
        else:
            print(f'{name} --- empty tensor of shape {tensor.shape}')

    def log_plot(self, plt, name, epoch=0):
        if self.clearml:
            self.clearml.report_matplotlib_figure(title=name, series=f"valid", iteration=epoch, figure=plt)
        else:
            plt.savefig(f"{self.tmpdir}/{name}_{epoch}.png")
        plt.close()

    def log_distribution(self, values, name, epoch=0):
        sns.kdeplot(values)
        self.log_plot(plt, name, epoch)

    def log_values(self, values, name, epoch=0):
        sns.lineplot(values)
        self.log_plot(plt, name, epoch)

    def log_line(self, x, y, name, epoch=0):
        sns.lineplot(x=x, y=y)
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

    def log_video(self, frames, gap, name, epoch=0):

        path = f'{self.tmpdir}/{name}_{epoch}.mp4'

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        w, h = frames[0].shape[1], frames[0].shape[0]
        writer = cv2.VideoWriter(path, apiPreference=0, fourcc=fourcc, fps=int(1 / (gap / 1000)), frameSize=(w, h))
        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame)
        writer.release()

        if self.clearml:
            self.clearml.report_media('video', name, iteration=epoch, local_path=path)

    def log_videos(self, videos_frames, gap, name, tempdir, epoch=0):
        for video_id, frames in enumerate(videos_frames):
            self.log_video(frames, gap, f'{name}_{video_id}', tempdir, epoch)

    def log_gif(self, frames, gap, name, epoch=0):
        path = f'{self.tmpdir}/{name}_{epoch}.gif'

        imageio.mimsave(path, frames, fps=int(1 / (gap / 1000)))
        if self.clearml:
            self.clearml.report_media('gifs', name, iteration=epoch, local_path=path)

    def log_gifs(self, frames_list, gap, name, tempdir, epoch=0):
        for idx, frames in enumerate(frames_list):
            self.log_gif(frames, gap, f'{name}_{idx}', tempdir, epoch)


class Lion(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        """Initialize the hyperparameters.
        Args:
          params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
          lr (float, optional): learning rate (default: 1e-4)
          betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.99))
          weight_decay (float, optional): weight decay coefficient (default: 0)
        """

        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
          closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
        Returns:
          the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                torch.nan_to_num(p.grad, nan=0, posinf=1e5, neginf=-1e5, out=p.grad)
                # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                grad = p.grad
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                # Weight update
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group['lr'])
                # Decay the momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss
