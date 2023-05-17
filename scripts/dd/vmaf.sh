#pip install meson
#sudo apt install -y nasm doxygen
#
#git clone https://github.com/Netflix/vmaf
#cd vmaf/libvmaf
#
#meson build --buildtype release
#ninja -vC build install
#
#cd ../../

args=("$@")

for video in "${args[@]}"; do
  ffmpeg -y -i "${video}" -pix_fmt yuv420p "${video%.*}.y4m"
done

for ((video_id=0;video_id < ${#args[@]};video_id+=2)); do
  vmaf --reference "${args[video_id]%.*}.y4m" --distorted "${args[video_id+1]%.*}.y4m" --feature psnr --feature float_ssim --output "metrics${video_id}.txt"
  tail -20 "metrics${video_id}.txt"
done

