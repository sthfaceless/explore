from argparse import ArgumentParser

import clearml
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer

from modules.nerf.dataset import NerfLatents
from modules.nerf.trainer import LatentDiffusion, NerfClassTrainer, SimpleLogger
from modules.nerf.util import render_latent_nerf


def get_parser():
    parser = ArgumentParser(description="Training NeRF latent sampler")
    # Input data settings
    parser.add_argument("--latent_path", default="", help="Path to latents checkpoint")
    parser.add_argument("--sampler", default="", help="Path to latent sampler")

    # Training settings
    parser.add_argument("--learning_rate", default=5 * 1e-5, type=float, help="Learning rate for decoder and nerf")
    parser.add_argument("--min_lr_rate", default=0.1, type=float, help="Learning rate for decoder and nerf")
    parser.add_argument("--max_beta", default=0.0195, type=float, help="Max noise schedule")
    parser.add_argument("--min_beta", default=0.0015, type=float, help="Min noise schedule")
    parser.add_argument("--epochs", default=100, type=int, help="Epochs in training")
    parser.add_argument("--steps", default=10000, type=int, help="Steps per epoch")
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size")
    parser.add_argument("--diffusion_steps", default=1000, type=int, help="Steps in diffusion")
    parser.add_argument("--samples_epoch", default=10, type=int, help="Samples of generator in one epoch")
    parser.add_argument("--latent_shape", default=(32, 16, 16), type=int, nargs='+', help="Shape of generated latents")

    # Model settings
    parser.add_argument("--hidden_dims", default=[128, 128, 256, 256, 512, 512, 1024, 1024], nargs='+', type=int,
                        help="Hidden dims for UNet")
    parser.add_argument("--attention_dim", default=8, type=int, help="Width till the one attention would be done")
    parser.add_argument("--img_size", default=128, type=int, help="Image size for decoder rendering")
    parser.add_argument("--focal", default=1.0, type=float, help="Focal for decoder rendering")

    # Meta settings
    parser.add_argument("--out_model_name", default="nerf_latent_sampler", help="Name of output model path")
    parser.add_argument("--task_name", default="Nerf Latents Diffusion", help="ClearML task name")
    parser.add_argument("--clearml", action='store_true')
    parser.set_defaults(clearml=False)
    return parser


if __name__ == "__main__":

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    if args.clearml:
        print("Initializing ClearML")
        task = clearml.Task.init(project_name='3dGen', task_name=args.task_name, reuse_last_task_id=True)
        task.connect(args, name='config')
        logger = task.get_logger()
    else:
        logger = None

    if args.sampler:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        renderer = NerfClassTrainer.load_from_checkpoint(args.decoder_path, map_location=device, dataset=None) \
            .to(device)
        sampler = LatentDiffusion.load_from_checkpoint(args.sampler, map_location=device, dataset=None)
        galleries = []
        for batch_idx in range(args.samples_epoch // args.batch_size + min(1, args.samples_epoch % args.batch_size)):
            size = min(args.batch_size, args.samples_epoch - batch_idx * args.batch_size)
            h = sampler.sample(size)
            galleries.extend(render_latent_nerf(h, renderer, w=args.img_size, h=args.img_size, focal=args.focal))
        SimpleLogger(logger).log_images(galleries, 'samples', epoch=0)
    else:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=args.out_model_name,
                                                           filename='best', save_on_train_epoch_end=True)
        dataset = NerfLatents(latents_checkpoint=args.latent_path, latent_shape=args.latent_shape)
        model = LatentDiffusion(shape=args.latent_shape, unet_hiddens=args.hidden_dims, dataset=dataset,
                                decoder_path=args.latent_path, img_size=args.img_size, focal=args.focal,
                                attention_dim=args.attention_dim, min_lr_rate=args.min_lr_rate, min_beta=args.min_beta,
                                max_beta=args.max_beta, epochs=args.epochs, steps=args.steps,
                                diffusion_steps=args.diffusion_steps, learning_rate=args.learning_rate,
                                log_samples=args.samples_epoch, clearml=logger, batch_size=args.batch_size)
        trainer = Trainer(max_epochs=args.epochs, limit_train_batches=args.steps, limit_val_batches=args.steps // 10,
                          reload_dataloaders_every_n_epochs=1, enable_model_summary=True,
                          enable_progress_bar=True, enable_checkpointing=True,
                          accelerator='gpu', gpus=[0], callbacks=[checkpoint_callback], check_val_every_n_epoch=5)
        trainer.fit(model)
