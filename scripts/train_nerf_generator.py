import argparse
import os

import clearml
import torch

from modules.nerf.dataset import NerfWeights
from modules.nerf.model import NerfVAEBlocks, NerfWeightsBlockDiscriminator, Nerf
from modules.nerf.trainer import NerfGenTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Run NeRF generator")

    # Input data settings
    parser.add_argument("--dataset", default="", help="Dataset with weights")
    parser.add_argument("--weights_name", default="model", help="Name of NeRF saved weights")
    parser.add_argument("--classes", default=[], nargs='+', help="Shapenet classes to run on")

    # VAE settings
    parser.add_argument("--hidden_dims", default=[1024, 512, 256], nargs='+', type=int,
                        help="Hidden dims for VAE")
    parser.add_argument("--latent", default=256, type=int, help="Latent dim of VAE")
    parser.add_argument("--kl", default=5 * 1e-4, type=float, help="KL loss weight")
    parser.add_argument("--l1", default=1e-6, type=float, help="L1 latent reg weight")

    # Training settings
    parser.add_argument("--learning_rate", default=5 * 1e-4, type=float, help="Learning rate")
    parser.add_argument("--train_rate", default=0.7, type=float, help="Train data rate")
    parser.add_argument("--epochs", default=100, type=int, help="Epochs in training")
    parser.add_argument("--patience", default=5, type=int, help="Epochs when learning rate doesn't changes")
    parser.add_argument("--disc_warmup", default=5, type=int, help="Epochs to wait before applying discriminator")
    parser.add_argument("--disc_weight", default=1e-2, type=float, help="Weights of discriminator loss")
    parser.add_argument("--batch_size", default=32, type=int, help="Nerf batch size")
    parser.add_argument("--samples_epoch", default=5, type=int, help="Samples of generator in one epoch")
    parser.add_argument("--max_lr", default=1e-3, type=float, help="Max learning rate in OneCycleLR")
    parser.add_argument("--stop_eps", default=1e-4, type=float, help="Learning rate stuck epsilon")
    parser.add_argument("--pct_start", default=0.3, type=float, help="Where would be peak of OneCycleLR")

    # Nerf settings
    parser.add_argument("--nerf_hidden", default=32, type=int, help="Hidden dim shape")
    parser.add_argument("--nerf_blocks", default=4, type=int, help="NeRF residual blocks")
    parser.add_argument("--nerf_pe", default=12, type=int, help="L for positional encoding")
    parser.add_argument("--nerf_spp", default=128, type=int, help="Samples per pixel in nerf")
    parser.add_argument("--nerf_batch", default=4096, type=int, help="Rays in one batch for NeRF")

    # Meta settings
    parser.add_argument("--out_model_name", default="nerf_gen", help="Name of output model file")
    parser.add_argument("--type", choices=['vae'], default="vae", help="Type of generator")
    parser.add_argument("--task_name", default="NerfGeneration", help="ClearML task name")
    parser.add_argument("--clearml", action='store_true')
    parser.set_defaults(clearml=False)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    if args.clearml:
        print("Initializing ClearML")
        task = clearml.Task.init(project_name='3dGen', task_name=args.task_name, reuse_last_task_id=True)
        task.connect(args, name='config')
        logger = task.get_logger()
    else:
        logger = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nerf = Nerf(hidden_dim=args.nerf_hidden, num_blocks=args.nerf_blocks, pe_powers=args.nerf_pe, density_noise=0.0)
    nerf = nerf.to(device)
    for cls in args.classes:
        path = os.path.join(args.dataset, cls)
        print(f"Begin training generator on {cls} \n path - {path}")

        # determine nerf weights layout
        models = [os.path.join(path, obj, args.weights_name) for obj in os.listdir(path)]
        sample_scene = NerfWeights(paths=models[:1], batch_size=1)
        layers = sample_scene.get_layers()
        shapes = sample_scene.get_shapes()
        print(f"Nerf layers names: {layers}")
        print(f"Nerf layers layout: {shapes}")

        # run train
        vae = NerfVAEBlocks(layers=layers, shapes=shapes, hidden_dims=args.hidden_dims,
                      latent_dim=args.latent, kl_weight=args.kl, l1=args.l1, disc_weight=args.disc_weight)
        vae = vae.to(device)
        disc = NerfWeightsBlockDiscriminator(layers=layers, shapes=shapes)
        disc = disc.to(device)
        trainer = NerfGenTrainer(gen_model=vae, gen_disc=disc, nerf_model=nerf)
        trainer.train(class_path=path, epochs=args.epochs, batch_size=args.batch_size, train_rate=args.train_rate,
                      stop_patience=args.patience, stop_eps=args.stop_eps, model_name=args.out_model_name,
                      nerf_spp=args.nerf_spp, nerf_batch=args.nerf_batch, weights_name=args.weights_name,
                      learning_rate=args.learning_rate, max_lr=args.max_lr, pct_start=args.pct_start,
                      log_samples=args.samples_epoch, logger=logger, disc_warmup=args.disc_warmup,
                      device=device)
