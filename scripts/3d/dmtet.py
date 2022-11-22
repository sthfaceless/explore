import os
from argparse import ArgumentParser

import clearml
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy

from modules.ddd.shapenet_pcd import ShapenetPointClouds
from modules.ddd.trainer import PCD2Mesh


def get_parser():
    parser = ArgumentParser(description="Training NeRF latents")
    # Input data settings
    parser.add_argument("--dataset", default="", help="Path to shapenet")
    parser.add_argument("--cats", default=['plane'], nargs='+', help="Shapenet categories to train")
    parser.add_argument("--noise", default=1e-2, type=float, help="How much noise to add to point cloud")
    parser.add_argument("--train_rate", default=0.8, type=float, help="Train data split")
    parser.add_argument("--n_points", default=50000, type=int, help="How much sample to point cloud")
    parser.add_argument("--cache_size", default=512, type=int, help="Cache for ShapeNet objects")
    parser.add_argument("--cache_dir", default=None, help="Cache dir for ShapeNet objects")

    # Training settings
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning rate for decoder and nerf")
    parser.add_argument("--min_lr_rate", default=0.5, type=float, help="Minimal learning rate ratio")
    parser.add_argument("--steps_schedule", default=[1000, 20000, 50000, 100000], nargs='+', type=int,
                        help="sdf train --- surface train --- adversarial train --- subdivision train")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size in training")
    parser.add_argument("--validations", default=100, type=int, help="Total numbers of validations during training")
    parser.add_argument("--acc_grads", default=1, type=int,
                        help="Steps to accumulate gradients to emulate larger batch size")
    parser.add_argument("--samples_epoch", default=5, type=int, help="Samples of generator in one epoch")
    parser.add_argument("--img_size", default=128, type=int, help="Image size to train and render")

    # Model settings
    parser.add_argument("--hidden_dims", default=[64, 64, 128, 128, 256, 256, 512, 512], nargs='+', type=int,
                        help="Hidden dims for decoder")
    parser.add_argument("--grid", default=128, type=int, help="tetrahedra grid resolution")

    # Meta settings
    parser.add_argument("--out_model_name", default="dmtet", help="Name of output model path")
    parser.add_argument("--task_name", default="DMTet training", help="ClearML task name")
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

    dataset = ShapenetPointClouds(shapenet_root=args.dataset, n_points=args.n_points, categories=args.cats,
                                  noise=args.noise, cache_dir=args.cache_dir, cache_scenes=args.cache_size)
    model = PCD2Mesh(dataset=dataset, clearml=logger, train_rate=args.train_rate, grid_resolution=args.grid)
    trainer = Trainer(max_steps=args.steps_schedule[-1], val_check_interval=1 / args.validations,
                      enable_model_summary=True, enable_progress_bar=True, enable_checkpointing=True,
                      strategy=DDPStrategy(find_unused_parameters=False),
                      accumulate_grad_batches=args.acc_grads,
                      accelerator='gpu', devices=1, callbacks=[checkpoint_callback])
    trainer.fit(model)
