import ffmpeg
import os

VIDEOS_ROOT = f'/dsk1/danil/3d/nerf/data/games/video'
files = os.listdir(VIDEOS_ROOT)

bitrates = [1, 2, 4]

for bitrate in bitrates:
    for file in files:
        tokens = os.path.splitext(file)[0].split('_')
        if len(tokens) > 1:
            continue
        path = f'{VIDEOS_ROOT}/{file}'
        out = path.replace(".mp4", f"_b{bitrate}.mp4")
        if os.path.exists(out):
            continue
        (
            ffmpeg
                .input(path)  # 2560x1440 original
                .filter('scale', w=1280, h=720, force_original_aspect_ratio='decrease', sws_flags="bicubic")
                .output(out, preset='ultrafast', movflags='faststart', pix_fmt='yuv420p', video_bitrate=str(bitrate * 1000000))
                .run()
        )