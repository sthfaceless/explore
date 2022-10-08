from argparse import ArgumentParser

import clearml
import pytorch_lightning
from pytorch_lightning import Trainer

from modules.nerf.trainer import MultiVAETrainer


def get_parser():
    parser = ArgumentParser(description="Training NeRF latents")
    # Input data settings
    parser.add_argument("--dataset", default="", help="Path to dir with scenes")

    # Training settings
    parser.add_argument("--learning_rate", default=5 * 1e-4, type=float, help="Learning rate for decoder and nerf")
    parser.add_argument("--initial_lr_ratio", default=0.1, type=float, help="Initial learning rate ratio")
    parser.add_argument("--min_lr_ratio", default=0.001, type=float, help="Minimal learning rate ratio")
    parser.add_argument("--train_rate", default=0.8, type=float, help="Train data rate")
    parser.add_argument("--epochs", default=100, type=int, help="Epochs in training")
    parser.add_argument("--warmup_epochs", default=3, type=int, help="Epochs to warmup learning rate")
    parser.add_argument("--steps", default=2000, type=int, help="Epochs in training")
    parser.add_argument("--acc_grads", default=1, type=int,
                        help="Steps to accumulate gradients to emulate larger batch size")
    parser.add_argument("--batch_size", default=64, type=int, help="Images in one batch")
    parser.add_argument("--sample_views", default=10, type=int, help="Sample views for batch and decoders")
    parser.add_argument("--n_views", default=64, type=int, help="Total views per scene")
    parser.add_argument("--val_samples", default=5, type=int, help="Validation samples per epoch")
    parser.add_argument("--cache_size", default=-1, type=int, help="NeRF scenes loaded in memory for epoch")
    parser.add_argument("--img_size", default=128, type=int, help="Image size to train and render")

    # Model settings
    parser.add_argument("--hidden_dims", default=[16, 32, 32, 64, 64, 64, 64, 128, 128], nargs='+', type=int,
                        help="Hidden dims for encoder and decoder")
    parser.add_argument("--latent", default=256, type=int, help="Latent dim size")
    parser.add_argument("--kl_weight", default=5 * 1e-4, type=int, help="KL loss weight of latent")
    parser.add_argument("--attention_dim", default=16, type=int, help="Width till the one attention would be done")

    # Meta settings
    parser.add_argument("--out_model_name", default="vaes", help="Name of output model path")
    parser.add_argument("--task_name", default="Multi View VAE", help="ClearML task name")
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

    checkpoint_callback = pytorch_lightning.callbacks.ModelCheckpoint(dirpath=args.out_model_name)
    trainer = Trainer(max_epochs=args.epochs, limit_train_batches=args.steps, limit_val_batches=args.steps,
                      enable_model_summary=True, enable_progress_bar=True, enable_checkpointing=True,
                      accelerator='gpu', devices=1, callbacks=[checkpoint_callback],
                      reload_dataloaders_every_n_epochs=1)
    model = MultiVAETrainer(clearml=logger, path=args.dataset, steps=args.steps, epochs=args.epochs,
                            learning_rate=args.learning_rate, hidden_dims=args.hidden_dims,
                            min_lr_ratio=args.min_lr_ratio, initial_lr_ratio=args.initial_lr_ratio,
                            batch_size=args.batch_size, kl_weight=args.kl_weight,
                            latent_dim=args.latent, n_views=args.n_views, sample_views=args.sample_views,
                            warmup_epochs=args.warmup_epochs, val_samples=args.val_samples,
                            img_size=args.img_size, cache_size=args.cache_size)
    trainer.fit(model)
