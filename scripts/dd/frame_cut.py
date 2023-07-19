import cv2
import os
from tqdm import tqdm
from glob import glob
from concurrent.futures import ThreadPoolExecutor
from argparse import ArgumentParser


def process_game(game, videos_root, images_root):
    root = f'{videos_root}/{game}'
    lr_out = f'{images_root}/{game}/lr'
    hr_out = f'{images_root}/{game}/hr'
    os.makedirs(lr_out, exist_ok=True)
    os.makedirs(hr_out, exist_ok=True)
    for path in glob(f'{root}/*.mp4'):
        cap = cv2.VideoCapture(path)
        try:
            video_fps = int(cap.get(cv2.CAP_PROP_FPS))
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            current_frame = 0
            step = video_fps

            pbar = tqdm(total=n_frames // step, desc=f'processing {os.path.basename(path)}')
            while cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = cap.read()
                if not ret:
                    break
                img_name = f'{os.path.basename(path)}_{current_frame}.png'
                if not os.path.exists(f'{hr_out}/{img_name}'):
                    cv2.imwrite(f'{hr_out}/{img_name}', frame)
                if not os.path.exists(f'{lr_out}/{img_name}'):
                    cv2.imwrite(f'{lr_out}/{img_name}',
                                cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2),
                                           interpolation=cv2.INTER_LINEAR))
                current_frame += step
                pbar.update(1)

            pbar.close()
            cap.release()
        except Exception as e:
            print(e)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--videos", type=str, default='/dsk1/danil/3d/nerf/data/games/video')
    parser.add_argument("--out", type=str, default='/dsk1/danil/3d/nerf/data/games/images')
    args = parser.parse_args()

    VIDEOS_ROOT = args.videos
    IMAGES_ROOT = args.out
    games = [game for game in os.listdir(VIDEOS_ROOT) if os.path.isdir(f'{VIDEOS_ROOT}/{game}')]

    tasks = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for game in games:
            tasks.append(pool.submit(process_game, game=game, videos_root=VIDEOS_ROOT, images_root=IMAGES_ROOT))
        for task in tqdm(tasks, desc='games processed'):
            task.result()
