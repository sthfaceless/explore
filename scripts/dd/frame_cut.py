import cv2
import os

import numpy as np
from tqdm import tqdm
from glob import glob
from concurrent.futures import ThreadPoolExecutor
from argparse import ArgumentParser
import imageio.v3 as iio
from imageio_ffmpeg import read_frames


# os.environ['IMAGEIO_FFMPEG_EXE'] = '/dsk1/anaconda3/envs/danil/bin/ffmpeg'
def process_game(game, videos_root, images_root, step=60):
    root = f'{videos_root}/{game}'
    lr_out = f'{images_root}/{game}/lr'
    hr_out = f'{images_root}/{game}/hr'
    os.makedirs(lr_out, exist_ok=True)
    os.makedirs(hr_out, exist_ok=True)
    for path in glob(f'{root}/*.mp4'):
        try:

            reader = read_frames(path)
            meta = reader.__next__()
            w, h = meta['size']

            for idx, frame in tqdm(enumerate(reader), desc=f'processing {os.path.basename(path)}'):
                if idx % step == 0:
                    img_name = f'{os.path.splitext(os.path.basename(path))[0]}_{idx}.png'
                    if not os.path.exists(f'{hr_out}/{img_name}'):
                        iio.imwrite(f'{hr_out}/{img_name}', np.frombuffer(frame, dtype=np.uint8).reshape(h, w, 3))

        except Exception as e:
            print(e)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--videos", type=str, default='/dsk1/danil/3d/nerf/data/games/video')
    parser.add_argument("--out", type=str, default='/dsk1/danil/3d/nerf/data/games/images')
    parser.add_argument("--step", type=int, default=60)
    args = parser.parse_args()

    VIDEOS_ROOT = args.videos
    IMAGES_ROOT = args.out
    games = [game for game in os.listdir(VIDEOS_ROOT) if os.path.isdir(f'{VIDEOS_ROOT}/{game}')]

    tasks = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for game in games:
            tasks.append(pool.submit(process_game, game=game, videos_root=VIDEOS_ROOT, images_root=IMAGES_ROOT,
                                     step=args.step))
        for task in tqdm(tasks, desc='games processed'):
            task.result()
