import os
from argparse import ArgumentParser

import clearml
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy

from modules.nerf.dataset import PairViews
from modules.nerf.trainer import NVSDiffusion


def get_parser():
    parser = ArgumentParser(description="Training NeRF latents")
    # Input data settings
    parser.add_argument("--dataset", default="", help="Path to shapenet class")

    # Training settings
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning rate for decoder and nerf")
    parser.add_argument("--min_lr_rate", default=0.5, type=float, help="Minimal learning rate ratio")
    parser.add_argument("--epochs", default=100, type=int, help="Epochs in training")
    parser.add_argument("--steps", default=10000, type=int, help="Epochs in training")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size in training")
    parser.add_argument("--acc_grads", default=1, type=int,
                        help="Steps to accumulate gradients to emulate larger batch size")
    parser.add_argument("--images_batch", default=64, type=int, help="NeRF images in one batch")
    parser.add_argument("--cache_size", default=-1, type=int, help="NeRF scenes loaded in memory for epoch")
    parser.add_argument("--samples_epoch", default=5, type=int, help="Samples of generator in one epoch")
    parser.add_argument("--samples_length", default=8, type=int, help="One sample length")
    parser.add_argument("--img_size", default=128, type=int, help="Image size to train and render")

    # Model settings
    parser.add_argument("--hidden_dims", default=[128, 128, 256, 256, 512, 512, 1024, 1024], nargs='+', type=int,
                        help="Hidden dims for decoder")
    parser.add_argument("--attention_dim", default=32, type=int, help="Width till the one attention would be done")
    parser.add_argument("--diffusion_steps", default=1000, type=int, help="Steps to do diffusion")
    parser.add_argument("--sample_steps", default=100, type=int, help="Steps for sampling")
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout regularization for model")
    parser.add_argument("--clf_free", default=0.1, type=float, help="Classifier free guidance rate")
    parser.add_argument("--focal", default=1.5, type=float, help="Focal for rendering and dataset")

    # Meta settings
    parser.add_argument("--out_model_name", default="nvs_diffusion", help="Name of output model path")
    parser.add_argument("--task_name", default="Nerf Latents Training", help="ClearML task name")
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

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=os.path.dirname(args.out_model_name),
                                                       filename=os.path.basename(args.out_model_name))

    dataset = PairViews(class_path=args.dataset, images_per_scene=args.images_batch, w=args.img_size, h=args.img_size,
                        cache_size=args.cache_size)
    model = NVSDiffusion(clearml=logger, shape=(3, args.img_size, args.img_size), dataset=dataset,
                         attention_dim=args.attention_dim, xunet_hiddens=args.hidden_dims, dropout=args.dropout,
                         classifier_free=args.clf_free, batch_size=args.batch_size, min_lr_rate=args.min_lr_rate,
                         diffusion_steps=args.diffusion_steps, log_samples=args.samples_epoch, focal=args.focal,
                         log_length=args.samples_length, learning_rate=args.learning_rate,
                         sample_steps=args.sample_steps)
    trainer = Trainer(max_epochs=args.epochs, limit_train_batches=args.steps, limit_val_batches=args.steps // 10,
                      enable_model_summary=True, enable_progress_bar=True, enable_checkpointing=True,
                      strategy=DDPStrategy(find_unused_parameters=False), precision=16,
                      accumulate_grad_batches=args.acc_grads,
                      accelerator='gpu', devices=1, callbacks=[checkpoint_callback],
                      reload_dataloaders_every_n_epochs=1)
    trainer.fit(model)
