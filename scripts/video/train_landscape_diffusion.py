import os
from argparse import ArgumentParser

import clearml
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy

from modules.video.dataset import LandscapeAnimation
from modules.video.trainer import LandscapeDiffusion


def get_parser():
    parser = ArgumentParser(description="Training NeRF latents")
    # Input data settings
    parser.add_argument("--dataset", default="", help="Path to folder with videos")
    parser.add_argument("--tmp", default="tmp", help="temporary directory for logs etc")

    # Training settings
    parser.add_argument("--base_lr", default=1e-6, type=float, help="Learning rate for decoder and nerf")
    parser.add_argument("--min_lr_rate", default=0.5, type=float, help="Minimal learning rate ratio")
    parser.add_argument("--epochs", default=30, type=int, help="Epochs in training")
    parser.add_argument("--steps", default=10000, type=int, help="Epochs in training")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size in training")
    parser.add_argument("--acc_grads", default=4, type=int,
                        help="Steps to accumulate gradients to emulate larger batch size")
    parser.add_argument("--samples_epoch", default=5, type=int, help="Samples of generator in one epoch")
    parser.add_argument("--frames", default=8, type=int, help="number of frames per batch to generate")
    parser.add_argument("--gap", default=128, type=int, help="gap between frames in ms")
    parser.add_argument("--w", default=256, type=int, help="frame width")
    parser.add_argument("--h", default=128, type=int, help="frame height ")

    # Model settings
    parser.add_argument("--hidden_dims", default=[128, 128, 256, 256, 384, 384, 512, 512], nargs='+', type=int,
                        help="Hidden dims for decoder")
    parser.add_argument("--attention_dim", default=32, type=int, help="Width till the one attention would be done")
    parser.add_argument("--local_attention_dim", default=64, type=int,
                        help="Width till the one local attention would be done")
    parser.add_argument("--local_attention_patch", default=8, type=int, help="Local attention patch size")
    parser.add_argument("--diffusion_steps", default=4000, type=int, help="Steps to do diffusion")
    parser.add_argument("--sample_steps", default=128, type=int, help="Steps for sampling")
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout regularization for model")
    parser.add_argument("--clf_free", default=0.1, type=float, help="Classifier free guidance rate")
    parser.add_argument("--clf_weight", default=3.0, type=float, help="Classifier free guidance weight sampling")

    # Meta settings
    parser.add_argument("--out_model_name", default="landscape_diffusion", help="Name of output model path")
    parser.add_argument("--task_name", default="Landscape diffusion", help="ClearML task name")
    parser.add_argument("--clearml", action='store_true')
    parser.set_defaults(clearml=False)
    return parser


if __name__ == "__main__":

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    if args.clearml:
        print("Initializing ClearML")
        task = clearml.Task.init(project_name='animation', task_name=args.task_name, reuse_last_task_id=True)
        task.connect(args, name='config')
        logger = task.get_logger()
    else:
        logger = None

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=os.path.dirname(args.out_model_name),
                                                       filename=os.path.basename(args.out_model_name))

    dataset = LandscapeAnimation(root=args.dataset, w=args.w, h=args.h, frames=args.frames + 1, step=args.gap)
    learning_rate = min(args.base_lr * args.batch_size * args.acc_grads * args.frames, 1e-4)
    model = LandscapeDiffusion(clearml=logger, shape=(3, args.h, args.w), dataset=dataset,
                               tempdir=args.tmp, local_attn_dim=args.local_attention_dim,
                               local_attn_patch=args.local_attention_patch,
                               attention_dim=args.attention_dim, frames=args.frames, gap=args.gap,
                               unet_hiddens=args.hidden_dims, dropout=args.dropout,
                               classifier_free=args.clf_free, batch_size=args.batch_size, min_lr_rate=args.min_lr_rate,
                               diffusion_steps=args.diffusion_steps, log_samples=args.samples_epoch,
                               learning_rate=learning_rate, clf_weight=args.clf_weight,
                               sample_steps=args.sample_steps, steps=args.steps, epochs=args.epochs)
    trainer = Trainer(max_epochs=args.epochs, limit_train_batches=args.steps, limit_val_batches=args.steps // 1000,
                      enable_model_summary=True, enable_progress_bar=True, enable_checkpointing=True,
                      strategy=DDPStrategy(find_unused_parameters=False),
                      accumulate_grad_batches=args.acc_grads,
                      accelerator='gpu', devices=1, callbacks=[checkpoint_callback])
    trainer.fit(model)
