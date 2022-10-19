import os
from argparse import ArgumentParser

import clearml
import pytorch_lightning
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy

from modules.nerf.dataset import NerfClass
from modules.nerf.trainer import NerfClassTrainer


def get_parser():
    parser = ArgumentParser(description="Training NeRF latents")
    # Input data settings
    parser.add_argument("--dataset", default="", help="Path to shapenet class")

    # Training settings
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning rate for decoder and nerf")
    parser.add_argument("--lr_embed", default=1e-3, type=float, help="Learning rate for embeddings")
    parser.add_argument("--initial_lr_ratio", default=0.5, type=float, help="Initial learning rate ratio")
    parser.add_argument("--min_lr_ratio", default=0.1, type=float, help="Minimal learning rate ratio")
    parser.add_argument("--train_rate", default=0.8, type=float, help="Train data rate")
    parser.add_argument("--epochs", default=100, type=int, help="Epochs in training")
    parser.add_argument("--steps", default=10000, type=int, help="Epochs in training")
    parser.add_argument("--acc_grads", default=4, type=int,
                        help="Steps to accumulate gradients to emulate larger batch size")
    parser.add_argument("--batch_objects", default=2, type=int, help="NeRF objects in one batch")
    parser.add_argument("--images_batch", default=64, type=int, help="NeRF images in one batch")
    parser.add_argument("--cache_size", default=-1, type=int, help="NeRF scenes loaded in memory for epoch")
    parser.add_argument("--samples_epoch", default=10, type=int, help="Samples of generator in one epoch")
    parser.add_argument("--log_every", default=100, type=int, help="Log every steps")
    parser.add_argument("--img_size", default=128, type=int, help="Image size to train and render")

    # Model settings
    parser.add_argument("--hidden_dims", default=[32, 32, 64, 64, 128, 128, 128, 128, 256, 256], nargs='+', type=int,
                        help="Hidden dims for decoder")
    parser.add_argument("--latent", default=[32, 16, 16], type=int, help="Latent dim shape")
    parser.add_argument("--feature_dim", default=32, type=int, help="Positional feature dim")
    parser.add_argument("--attention_dim", default=8, type=int, help="Width till the one attention would be done")
    parser.add_argument("--embed_noise", default=0.1, type=float, help="Noise added to embedding")

    # Nerf settings
    parser.add_argument("--near", default=4-3**(1/2), type=float, help="Hidden dim shape")
    parser.add_argument("--far", default=4+3**(1/2), type=float, help="Hidden dim shape")
    parser.add_argument("--nerf_hidden", default=128, type=int, help="Hidden dim shape")
    parser.add_argument("--nerf_blocks", default=4, type=int, help="NeRF residual blocks")
    parser.add_argument("--nerf_pe", default=12, type=int, help="L for positional encoding")
    parser.add_argument("--nerf_spp", default=128, type=int, help="Samples per pixel in nerf")
    parser.add_argument("--nerf_batch", default=2048, type=int, help="Rays in one batch for NeRF")

    # Meta settings
    parser.add_argument("--out_model_name", default="nerf_gen", help="Name of output model path")
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

    checkpoint_callback = pytorch_lightning.callbacks.ModelCheckpoint(dirpath=os.path.dirname(args.out_model_name),
                                                                      filename=os.path.basename(args.out_model_name))

    dataset = NerfClass(class_path=args.dataset, batch_rays=args.nerf_batch, batch_objects=args.batch_objects,
                        images_per_scene=args.images_batch, w=args.img_size, h=args.img_size,
                        cache_size=args.cache_size)
    model = NerfClassTrainer(clearml=logger, dataset=dataset, n_objects=len(dataset.get_items()),
                             steps=args.steps, epochs=args.epochs,
                             attention_dim=args.attention_dim, embed_noise=args.embed_noise,
                             embed_shape=args.latent, lr_embed=args.lr_embed, nerf_hidden=args.nerf_hidden,
                             nerf_blocks=args.nerf_blocks, nerf_spp=args.nerf_spp, nerf_pe=args.nerf_pe,
                             batch_rays=args.nerf_batch, batch_objects=args.batch_objects,
                             learning_rate=args.learning_rate, decoder_hiddens=args.hidden_dims,
                             positional_dim=args.feature_dim, near=args.near, far=args.far,
                             model_out=os.path.join(os.path.dirname(args.out_model_name),
                                                    os.path.basename(args.out_model_name) + '.parts'),
                             min_lr_ratio=args.min_lr_ratio, initial_lr_ratio=args.initial_lr_ratio,
                             val_samples=args.samples_epoch, image_size=args.img_size,
                             accumulate_gradients=args.acc_grads)
    trainer = Trainer(max_epochs=args.epochs, limit_train_batches=args.steps, limit_val_batches=args.steps,
                      enable_model_summary=True, enable_progress_bar=True, enable_checkpointing=True,
                      strategy=DDPStrategy(find_unused_parameters=False), precision=16,
                      accelerator='gpu', devices=1, callbacks=[checkpoint_callback],
                      reload_dataloaders_every_n_epochs=1)
    trainer.fit(model)
