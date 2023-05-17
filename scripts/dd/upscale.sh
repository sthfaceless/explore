#!/usr/bin/env bash

DEVICE=$1
BATCH=$2
args=("$@")
games=("huawei" "dirt" "csgo" "genshin" "vrising" "warmsnow")
usualgames=("dirt" "csgo" "genshin" "vrising")

ROOT="/dsk1/danil/3d/nerf/data/games/video"

export CUDA_VISIBLE_DEVICES=${DEVICE}
export http_proxy=
export https_proxy=


for bitrate in "${args[@]:2}"; do

  paths=()
  for game in "${games[@]}"; do
    paths+=("${ROOT}/${game}_${bitrate}.mp4")
  done
  echo "${paths[@]}" "${bitrate}"

  /dsk1/anaconda3/envs/danil/bin/python scripts/dd/patch_upscale.py --tmp "tmp${DEVICE}" \
  --dataset /dsk1/danil/3d/nerf/data/games/images/csgo/hr /dsk1/danil/3d/nerf/data/games/images/dirt/hr \
   /dsk1/danil/3d/nerf/data/games/images/genshin/hr /dsk1/danil/3d/nerf/data/games/images/vrising/hr \
    /dsk1/danil/3d/nerf/data/games/images/warmsnow/hr \
    --cache_size 512 --full_batch_size 5 --batch_size "${BATCH}" \
     --in_channels 3 --dim 16 --tile 80 --tile_pad 4 --n_blocks 3 --disc_lr 0.001 --disc_w 0.01 --disc_dim 16 --disc_blocks 2 \
     --sample "checkpoints/upscaler_${bitrate}.ckpt" --video_suffix "upscaled" --videos "${paths[@]}" \
     --task_name "upscale video ${bitrate}" > "logs/upscale_video_${bitrate}.log" 2>&1

  paths=()
  for game in "${games[@]}"; do
    paths+=("${ROOT}/${game}.mp4")
    paths+=("${ROOT}/${game}_${bitrate}_upscaled.avi")
  done

  /dsk1/anaconda3/envs/danil/bin/python scripts/dd/calc_lpips.py --batch 8 --videos "${paths[@]}" > "logs/lpips_${bitrate}.log" 2>&1

  for game in "${games[@]}"; do
    ffmpeg-quality-metrics "${ROOT}/${game}_${bitrate}_upscaled.avi" "${ROOT}/${game}.mp4" --metrics psnr ssim vmaf --progress \
    > "logs/upscale_metrics_${game}_${bitrate}.log" 2>&1
  done

  for game in "${usualgames[@]}"; do
    rm "${ROOT}/${game}_${bitrate}_upscaled.avi"
  done
done