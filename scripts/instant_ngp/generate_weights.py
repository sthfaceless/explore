#!/usr/bin/env python3

# Creates instant-ngp weights from shapenet dataset

import argparse
import os
import time

import numpy as np
import torch
import shutil
import gc
import random

import msgpack
from common import *
from scenes import *
import pyngp as ngp  # noqa
from generate_images import generate_images
from tqdm import tqdm


def parse_args():
	parser = argparse.ArgumentParser(
		description="Run neural graphics primitives testbed with additional configuration & output options")

	parser.add_argument("--shapenet", "--training_data", default="",
						help="The scene to load. Can be the scene's name or a full path to the training data.")

	parser.add_argument("--out", default="", help="Output directory for weights and images")
	parser.add_argument("--views", default=128, type=int, help="number of views generated with renderer")
	parser.add_argument("--picture_res", default=1024, type=int, help="generated pictures resolution")
	parser.add_argument("--batch_size", default=32, type=int, help="picture generation batch")

	parser.add_argument("--save_mesh", default="",
						help="Output a marching-cubes based mesh from the NeRF or SDF model. Supports OBJ and PLY format.")
	parser.add_argument("--marching_cubes_res", default=256, type=int,
						help="Sets the resolution for the marching cubes grid.")

	parser.add_argument("--n_steps", type=int, default=-1, help="Number of steps to train for before quitting.")

	args = parser.parse_args()
	return args


if __name__ == "__main__":
	args = parse_args()

	mode = ngp.TestbedMode.Nerf
	network = os.path.join(ROOT_DIR, "configs", "../nerf", "base.json")
	network_stem = os.path.splitext(os.path.basename(network))[0]

	SHAPENET_ROOT = args.shapenet
	RESULT_DIR = args.out

	old_training_step = 0
	n_steps = args.n_steps
	if n_steps < 0:
		n_steps = 3500

	classes = [name for name in os.listdir(SHAPENET_ROOT) if os.path.isdir(os.path.join(SHAPENET_ROOT, name))]
	for num, obj_class in enumerate(classes):
		class_path = os.path.join(SHAPENET_ROOT, obj_class)
		ids = [name for name in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, name))]
		for obj_id in random.choices(ids, k=10):
			out_path = os.path.join(RESULT_DIR, f"{obj_class}_{obj_id}")
			try:
				os.makedirs(out_path, exist_ok=True)

				generate_images(args.views, SHAPENET_ROOT, obj_class, obj_id, out_path,
								batch_size=args.batch_size, img_size=args.picture_res)

				testbed = ngp.Testbed(mode)
				testbed.nerf.sharpen = float(0)
				testbed.exposure = 0

				testbed.load_training_data(out_path)

				testbed.reload_network_from_file(network)
				testbed.shall_train = True
				testbed.nerf.render_with_camera_distortion = True

				# Optionally match nerf paper behaviour and train on a
				# fixed white bg. We prefer training on random BG colors.
				# testbed.background_color = [1.0, 1.0, 1.0, 1.0]
				# testbed.nerf.training.random_bg_color = False

				### TRAINING NERF
				tqdm_last_update = 0
				if n_steps > 0:
					with tqdm(desc="Training", total=n_steps, unit="step") as t:
						while testbed.frame():
							if testbed.want_repl():
								repl(testbed)
							# What will happen when training is done?
							if testbed.training_step >= n_steps:
								break

							# Update progress bar
							if testbed.training_step < old_training_step or old_training_step == 0:
								old_training_step = 0
								t.reset()

							now = time.monotonic()
							if now - tqdm_last_update > 0.1:
								t.update(testbed.training_step - old_training_step)
								t.set_postfix(loss=testbed.loss)
								old_training_step = testbed.training_step
								tqdm_last_update = now

				### SAVING WEIGHTS
				snap_path = os.path.join(out_path, "snapshot.msgpack")
				testbed.save_snapshot(snap_path, False)
				with open(snap_path, 'rb') as f:
					data = msgpack.loads(f.read())
				density_grid = np.frombuffer(data['snapshot']['density_grid_binary'], dtype=np.float16)
				params = np.frombuffer(data['snapshot']['params_binary'], dtype=np.float16)
				np.savez(os.path.join(out_path, "weights.npz"), density=density_grid, params=params)

				shutil.rmtree(os.path.join(out_path, 'images'))
				# os.remove(snap_path)

				### SAVING MESH
				if args.save_mesh:
					res = args.marching_cubes_res or 256
					testbed.compute_and_save_marching_cubes_mesh(os.path.join(out_path, "mesh.obj"), [res, res, res])

				del testbed
				gc.collect()
				torch.cuda.empty_cache()

			except Exception as e:
				shutil.rmtree(out_path)
				print("Exception occured", e)

		print(f"Object class finished - {num / len(classes) * 100:.2f}%")