import os
from argparse import ArgumentParser

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
from sklearn.model_selection import GroupShuffleSplit

from modules.bio.dataset import ProteinMutations
from modules.bio.trainer import ProteinMutationTrainer


def get_parser():
    parser = ArgumentParser(description="Training mutations thermostability")
    # Input data settings
    parser.add_argument("--dataset", default="", help="Path to training csv")
    parser.add_argument("--pdbs", default="", help="Path to directory with pdbs")

    # Training settings
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning rate for decoder and nerf")
    parser.add_argument("--train_rate", default=0.7, type=float, help="Training data rate")
    parser.add_argument("--min_lr_rate", default=0.5, type=float, help="Minimal learning rate ratio")
    parser.add_argument("--steps", default=1000, type=int, help="Training steps per epoch")
    parser.add_argument("--epochs", default=30, type=int, help="Training epochs")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size in training")
    parser.add_argument("--acc_grads", default=8, type=int,
                        help="Steps to accumulate gradients to emulate larger batch size")

    # Model settings
    parser.add_argument("--encoder_dims", default=[64, 128, 256], nargs='+', type=int,
                        help="Hidden dims for volume encoder")
    parser.add_argument("--encoder_grids", default=[32, 16, 8], nargs='+', type=int,
                        help="Grid size for volume encoder")

    # Meta settings
    parser.add_argument("--out_model_name", default="thermo", help="Name of output model path")
    parser.set_defaults(clearml=False)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # train test dataset splitting
    df = pd.read_csv(args.dataset)
    splitter = GroupShuffleSplit(train_size=args.train_rate, n_splits=1)
    split = splitter.split(df, groups=df['group'])
    train_inds, val_inds = next(split)
    train_dataset = ProteinMutations(df=df.iloc[train_inds], pdb_root=args.pdbs)
    val_dataset = ProteinMutations(df=df.iloc[val_inds], pdb_root=args.pdbs)
    # creating model
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=os.path.dirname(args.out_model_name),
                                                       filename=os.path.basename(args.out_model_name))
    model = ProteinMutationTrainer(train_dataset=train_dataset, val_dataset=val_dataset,
                                   min_lr_rate=args.min_lr_rate, encoder_dims=args.encoder_dims,
                                   encoder_grids=args.encoder_grids, learning_rate=args.learning_rate,
                                   batch_size=args.batch_size)
    trainer = Trainer(max_epochs=args.epochs, limit_train_batches=args.steps, limit_val_batches=args.steps,
                      enable_model_summary=True, enable_progress_bar=True, enable_checkpointing=True,
                      strategy=DDPStrategy(find_unused_parameters=False),
                      accumulate_grad_batches=args.acc_grads, precision=16,
                      accelerator='gpu', devices=1, callbacks=[checkpoint_callback])
    trainer.fit(model)
