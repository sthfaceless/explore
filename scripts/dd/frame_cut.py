import cv2
import os
from tqdm import tqdm

VIDEOS_ROOT = f'/dsk1/danil/3d/nerf/data/games/video'
IMAGES_ROOT = f'/dsk1/danil/3d/nerf/data/games/images'
files = os.listdir(VIDEOS_ROOT)

for file in tqdm(files, desc='video processed'):

    path = f'{VIDEOS_ROOT}/{file}'
    tokens = os.path.splitext(file)[0].split('_')
    prefix = tokens[0]
    bitrate = tokens[1] if len(tokens) > 1 else 'hr'

    out = f'{IMAGES_ROOT}/{prefix}/{bitrate}'
    os.makedirs(out, exist_ok=True)

    cap = cv2.VideoCapture(path)
    try:
        video_fps = int(cap.get(cv2.CAP_PROP_FPS))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        current_frame = 0
        step = video_fps

        pbar = tqdm(total=n_frames // step)
        while (cap.isOpened()):
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret:
                break
            img_out = f'{out}/{current_frame}.png'
            if not os.path.exists(img_out):
                cv2.imwrite(img_out, frame)
            current_frame += step
            pbar.update(1)

        pbar.close()
        cap.release()
    except Exception as e:
        print(e)
