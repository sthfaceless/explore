import os
from argparse import ArgumentParser

import clearml
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy

import kaolin
from modules.ddd.dmtet_trainer import PCD2Mesh
from modules.ddd.shapenet_pcd import ShapenetPointClouds


def get_parser():
    parser = ArgumentParser(description="Training NeRF latents")
    # Input data settings
    parser.add_argument("--dataset", default="", help="Path to shapenet")
    parser.add_argument("--cats", default=['plane'], nargs='+', help="Shapenet categories to train")
    parser.add_argument("--noise", default=1e-2, type=float, help="How much noise to add to point cloud")
    parser.add_argument("--train_rate", default=0.8, type=float, help="Train data split")
    parser.add_argument("--n_points", default=5000, type=int, help="How much sample to point cloud")
    parser.add_argument("--cache_size", default=512, type=int, help="Cache for ShapeNet objects")
    parser.add_argument("--cache_dir", default=None, help="Cache dir for ShapeNet objects")

    # Training settings
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning rate for decoder and nerf")
    parser.add_argument("--min_lr_rate", default=0.5, type=float, help="Minimal learning rate ratio")
    parser.add_argument("--steps", default=[1000, 20000, 50000, 100000], nargs='+', type=int,
                        help="sdf train --- surface train --- adversarial train --- subdivision train")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size in training")
    parser.add_argument("--validations", default=100, type=int, help="Total numbers of validations during training")
    parser.add_argument("--acc_grads", default=8, type=int,
                        help="Steps to accumulate gradients to emulate larger batch size")
    parser.add_argument("--samples_epoch", default=5, type=int, help="Samples of generator in one epoch")
    parser.add_argument("--img_size", default=128, type=int, help="Image size to train and render")

    # Model settings
    parser.add_argument("--encoder_dims", default=[64, 128, 256], nargs='+', type=int, help="Hidden dims for volume encoder")
    parser.add_argument("--encoder_grids", default=[32, 16, 8], nargs='+', type=int, help="Grid size for volume encoder")
    parser.add_argument("--sdf_dims", default=[256, 256, 128, 64], nargs='+', type=int, help="Hidden dims for SDF mlp")
    parser.add_argument("--disc_dims", default=[32, 64, 128, 256], nargs='+', type=int, help="SDF discriminator hidden dims")
    parser.add_argument("--gcn_dims", default=[256, 128], nargs='+', type=int, help="Graph Convolutional refinement conv dims")
    parser.add_argument("--gcn_hidden", default=[128, 64], nargs='+', type=int, help="Graph Convolutional refinement linear dims")
    parser.add_argument("--grid", default=64, type=int, help="tetrahedra grid resolution")

    parser.add_argument("--disc_weight", default=10, type=float, help="Weight for discriminator loss")
    parser.add_argument("--chamfer_weight", default=500, type=float, help="Weight for chamfer distance")
    parser.add_argument("--normal_weight", default=1e-6, type=float, help="Weight for faces normal reg")
    parser.add_argument("--delta_weight", default=1, type=float, help="Weight for veretexes delta change reg")
    parser.add_argument("--sdf_weight", default=0.4, type=float, help="Weight for sdf prediction reg")

    parser.add_argument("--sdf_clamp", default=0.03, type=float, help="Max absolute true sdf value")
    parser.add_argument("--curvature_threshold", default=3.1415926/16, type=float, help="Vertices gaussian curvature threshold")
    parser.add_argument("--disc_sdf_scale", default=0.1, type=float, help="SDF discriminator grid size")
    parser.add_argument("--curvature_samples", default=10, type=int, help="SDF discriminator vertex samples")
    parser.add_argument("--disc_sdf_grid", default=16, type=int, help="SDF discriminator grid resolution")
    parser.add_argument("--disc_v_noise", default=1e-3, type=float, help="SDF discriminator origin vertex noise")

    parser.add_argument("--n_surface_division", default=1, type=int, help="Surface subdivision steps")
    parser.add_argument("--n_volume_division", default=1, type=int, help="Volume subdivision steps")

    # Meta settings
    parser.add_argument("--logs_path", default="./", help="Kaolin logs dir")
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
    timelapse = kaolin.visualize.Timelapse(args.logs_path)
    model = PCD2Mesh(dataset=dataset, clearml=logger, timelapse=timelapse, train_rate=args.train_rate, grid_resolution=args.grid,
                     steps_schedule=args.steps, min_lr_rate=args.min_lr_rate, encoder_dims=args.encoder_dims, encoder_grids=args.encoder_grids,
                     sdf_dims=args.sdf_dims, disc_dims=args.disc_dims, gcn_dims=args.gcn_dims, gcn_hidden=args.gcn_hidden,
                     sdf_weight=args.sdf_weight, disc_weight=args.disc_weight, chamfer_weight=args.chamfer_weight,
                     normal_weight=args.normal_weight, delta_weight=args.delta_weight, learning_rate=args.learning_rate,
                     n_volume_division=args.n_volume_subdivision, n_surface_division=args.n_surface_subdivision,
                     chamfer_samples=args.n_points, sdf_clamp=args.sdf_clamp, disc_sdf_scale=args.disc_sdf_scale,
                     disc_sdf_grid=args.disc_sdf_grid, curvature_samples=args.curvature_samples,
                     curvature_threshold=args.curvature_threshold, disc_v_noise=args.disc_v_noise,
                     noise=args.noise, batch_size=args.batch_size)
    trainer = Trainer(max_steps=args.steps_schedule[-1], val_check_interval=1 / args.validations,
                      enable_model_summary=True, enable_progress_bar=True, enable_checkpointing=True,
                      strategy=DDPStrategy(find_unused_parameters=True),
                      accumulate_grad_batches=args.acc_grads,
                      accelerator='gpu', devices=1, callbacks=[checkpoint_callback])
    trainer.fit(model)