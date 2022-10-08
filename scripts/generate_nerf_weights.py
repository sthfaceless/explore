import argparse
import os

import clearml
import torch
from tqdm import tqdm

from modules.nerf.model import Nerf
from modules.nerf.trainer import NerfTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Run NeRF on shapenet dataset")

    parser.add_argument("--dataset", default="", help="Shapenet scenes")
    parser.add_argument("--nclasses", default=-1, type=int, help="Num of shapenet classes to run on")
    parser.add_argument("--classes", default=[], nargs='+', help="Shapenet classes to run on")
    parser.add_argument("--batch_rays", default=1 << 14, type=int, help="Rays in batch")
    parser.add_argument("--n_steps", type=int, default=3000, help="Number of steps to train each NeRF")
    parser.add_argument("--log_every", type=int, default=300, help="Number steps to log every")

    parser.add_argument("--learning_rate", default=5 * 1e-4, type=float, help="Learning rate")
    parser.add_argument("--max_lr", default=1e-3, type=float, help="Max learning rate in OneCycleLR")
    parser.add_argument("--pct_start", default=0.1, type=float, help="Where would be peak of OneCycleLR")
    parser.add_argument("--steps_epoch", default=500, type=int, help="Steps per epoch in OneCycleLR")

    parser.add_argument("--hidden_dim", default=32, type=int, help="Hidden dim shape")
    parser.add_argument("--nerf_blocks", default=4, type=int, help="NeRF residual blocks")
    parser.add_argument("--pe_powers", default=12, type=int, help="L for positional encoding")

    parser.add_argument("--spp", default=128, type=int, help="Samples per pixel in NeRF rendering")
    parser.add_argument("--coarse_weight", default=0.1, type=float, help="Weight of coarse loss")

    parser.add_argument("--out_model_name", default="model", help="Name of output model file")
    parser.add_argument("--task_name", default="Nerf weights generation", help="ClearML task name")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    print("Initializing ClearML")
    task = clearml.Task.init(project_name='3dGen', task_name=args.task_name, reuse_last_task_id=True)
    task.connect(args, name='config')

    dataset_root = args.dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obj_classes = [file for file in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, file))]
    if args.classes:
        obj_classes = [cls for cls in obj_classes if cls in args.classes]
    elif args.nclasses > 0:
        obj_classes = obj_classes[:args.nclasses]

    print(f"Founded {len(obj_classes)} classes to train")
    iteration_id = 0
    for num_class, obj_class in enumerate(obj_classes):
        class_root = os.path.join(dataset_root, obj_class)
        obj_ids = [file for file in os.listdir(class_root) if os.path.isdir(os.path.join(class_root, file))]
        print(F"Begin training on class {obj_class}, items: {len(obj_ids)}")
        for num_obj, obj_id in tqdm(enumerate(obj_ids)):
            obj_root = os.path.join(class_root, obj_id)

            print(f"--- Begin training on obj {obj_id}")
            model = Nerf(hidden_dim=args.hidden_dim, num_blocks=args.nerf_blocks, pe_powers=args.pe_powers).to(device)
            trainer = NerfTrainer(model=model, spp=args.spp, coarse_weight=args.coarse_weight, pe_powers=args.pe_powers)
            trainer.train(scene_root=obj_root, steps=args.n_steps, learning_rate=args.learning_rate,
                          batch_rays=args.batch_rays, max_lr=args.max_lr, steps_epoch=args.steps_epoch,
                          pct_start=args.pct_start,
                          log_every=args.log_every, logger=task.get_logger(), iteration_id=iteration_id, device=device)

            torch.save(model.state_dict(), os.path.join(obj_root, args.out_model_name))
            iteration_id += 1
        print(f"Finished training on class {obj_class}")
