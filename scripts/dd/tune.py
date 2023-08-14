import glob
import os
import yaml
import argparse
from copy import deepcopy
import re
from collections import defaultdict
import torch
from datetime import datetime
from queue import Queue
import shutil
import time
from concurrent.futures import ThreadPoolExecutor


def unwrap(dct):
    items = []
    for k, v in dct.items():
        if isinstance(v, dict):
            for subkey, subvalue in unwrap(v):
                items.append((f'{k}.{subkey}', subvalue))
        else:
            items.append((k, v))
    return items


def wrap(items):
    dct = dict()
    for k, v in items:
        parts = k.split('.')
        __dct = dct
        for part in parts[:-1]:
            if part not in __dct:
                __dct[part] = {}
            __dct = __dct[part]
        __dct[parts[-1]] = v

    return dct


def calc_iters(configs, min_cycles, max_cycles=5, min_models=1):
    # assume that we start with cycle of len 1 and multiply it by 2 each time
    # initial run
    total_iters, curr_iter = ((1 << min_cycles) - 1) * configs, 0
    configs = max(configs // 2, min_models)
    # successive halving
    while configs > min_models:
        total_iters += (1 << (min_cycles + curr_iter)) * configs
        configs, curr_iter = max(configs // 2, min_models), curr_iter + 1
    # train remained models until max cycles reached
    while min_cycles + curr_iter < max_cycles:
        total_iters += (1 << (min_cycles + curr_iter)) * min_models
        curr_iter += 1
    return total_iters


class MultiConfig:

    def __init__(self, cfg):
        self.base = dict(unwrap(cfg))
        self.variants = []
        pattern = re.compile('^choice\\(.*\\)$')
        for k, v in self.base.items():
            if isinstance(v, str) and pattern.fullmatch(v):
                values = yaml.safe_load(v[len('choice('):-len(')')])
                if len(self.variants) == 0:
                    for _ in range(len(values)):
                        self.variants.append({})
                assert len(values) == len(self.variants), \
                    f'Number of variants does not match for key {k}, current variants: {len(self.variants)}, provided variants: {len(values)}'
                for var, value in zip(self.variants, values):
                    var[k] = value

    def __len__(self):
        return len(self.variants)

    def __getitem__(self, idx):
        items = deepcopy(self.base)
        for k, v in self.variants[idx].items():
            items[k] = v
        return wrap(list(items.items()))


def run(cfg_path, cfg_id, model_name):
    device = device_queue.get()
    cmd = f'http_proxy= CUDA_VISIBLE_DEVICES={device} python {args.script} --config_path {cfg_path} > {args.logs}/{model_name}.log 2>&1'
    print(f'Running command:\n{cmd}')
    os.system(cmd)
    device_queue.put(device)
    return cfg_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default='scripts/dd/configs/upscaling.yaml')
    parser.add_argument("--script", type=str, default='scripts/dd/upscaling.py')
    parser.add_argument("--logs", type=str, default='logs')
    parser.add_argument("--tmp", type=str, default='tmp')
    parser.add_argument("--min_cycles", type=int, default=1)
    parser.add_argument("--max_cycles", type=int, default=3)
    parser.add_argument("--min_models", type=int, default=1)
    parser.add_argument("--step", type=int, default=1)
    args = parser.parse_args()

    # make all dirs
    for dir in [args.tmp, args.logs]:
        os.makedirs(dir, exist_ok=True)
    job_dir = os.path.join(args.tmp, datetime.now().strftime("%Y-%m-%d-%H-%M"))
    os.makedirs(job_dir, exist_ok=True)
    print(f'This job dir: {job_dir}')

    # cfg variants
    cfg = MultiConfig(yaml.load(open(args.config_path, 'r'), yaml.FullLoader))
    print(f'The total number of iterations: {calc_iters(len(cfg), args.min_cycles, args.max_cycles, args.min_models)}')

    # device queue for pool
    devices = torch.cuda.device_count()
    min_models = args.min_models
    device_queue = Queue(maxsize=devices)
    for idx in range(devices):
        device_queue.put(idx)

    pool = ThreadPoolExecutor(max_workers=devices)
    configs, current_iter, processed = list(range(len(cfg))), 0, 0
    checkpoints, metrics = [''] * len(cfg), [0] * len(cfg)
    step = args.step
    while args.min_cycles + current_iter <= args.max_cycles and processed < args.max_cycles:

        local_configs = [cfg[cfg_id] for cfg_id in configs]
        # schedule models run
        futures = []
        for cfg_id, local_cfg in zip(configs, local_configs):
            # specify config for current run
            model_name = f"{local_cfg['model']['name']}-{current_iter}-{cfg_id}"
            local_cfg['saving']['ckp_folder'] = f'{job_dir}/'
            local_cfg['saving']['log_folder'] = f'{job_dir}/'
            local_cfg['data']['out'] = None
            local_cfg['model']['name'] = model_name
            local_cfg['train']['pretrained_weights'] = checkpoints[cfg_id]

            # count epochs and cycles
            epochs = (1 << args.min_cycles) - 1 if current_iter == 0 \
                else (1 << args.min_cycles + current_iter) - (1 << (args.min_cycles + current_iter - step))
            local_cfg['train']['epochs'] = epochs
            local_cfg['train']['sched']['cycles'] = args.min_cycles if current_iter == 0 else step
            local_cfg['train']['sched']['start'] = 1
            local_cfg['train']['hard']['warmup'] = local_cfg['train']['hard']['warmup'] if current_iter == 0 else 0

            cfg_path = os.path.join(job_dir, f'{model_name}.yaml')
            with open(cfg_path, 'w') as f:
                yaml.dump(local_cfg, f, default_flow_style=False)

            futures.append(pool.submit(run, cfg_path, cfg_id, model_name))

        # wait for all models to finish
        while len(futures) > 0:
            # check whether some futures was already finished
            results = [future for future in futures if future.done()]
            if len(results) == 0:
                time.sleep(1)
                continue
            for future in results:
                futures.remove(future)
                cfg_id = future.result()
                model_name = local_configs[configs.index(cfg_id)]['model']['name']
                # find trained checkpoint
                checkpoint_paths = [os.path.basename(path)
                                    for path in glob.glob(f'{job_dir}/*.pth') if model_name in path and 'train' in path]
                if len(checkpoint_paths) == 0:
                    print(f'Not found checkpoint for {model_name}')
                    continue
                checkpoints[cfg_id] = checkpoint_paths[0]
                # load result of checkpoint
                state = torch.load(os.path.join(job_dir, checkpoints[cfg_id]), map_location='cpu')
                metrics[cfg_id] = state['val_psnr']
                del state

        processed = max(processed, args.min_cycles + current_iter)
        # take top 50%
        configs = sorted(configs, key=lambda cfg_id: metrics[cfg_id], reverse=True)
        configs = configs[:max(min_models, len(configs) // 2)]
        if len(configs) == min_models:
            step = args.max_cycles - (args.min_cycles + current_iter)
            current_iter = args.max_cycles - args.min_cycles

        print(f'Finished iters: {args.min_cycles + current_iter - step}')
        print(f'Top metrics: {[metrics[idx] for idx in configs]}')

        current_iter += step

    top_id = configs[0]
    top_ckp, top_cfg = checkpoints[top_id], cfg[top_id]
    shutil.copy(os.path.join(job_dir, top_ckp), f"{os.path.join(top_cfg['saving']['ckp_folder'])}")
    with open(f'{args.config_path.replace(".yaml", "_best.yaml")}', 'w') as f:
        yaml.dump(top_cfg, f, default_flow_style=False)

    pool.shutdown(wait=True)