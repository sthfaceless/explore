import sys, os
import torch
need_pytorch3d = False
try:
    import pytorch3d
except ModuleNotFoundError:
    need_pytorch3d = True
if need_pytorch3d:
    if torch.__version__.startswith("1.12.") and sys.platform.startswith("linux"):
        # We try to install PyTorch3D via a released wheel.
        pyt_version_str = torch.__version__.split("+")[0].replace(".", "")
        version_str = "".join([
            f"py3{sys.version_info.minor}_cu",
            torch.version.cuda.replace(".", ""),
            f"_pyt{pyt_version_str}"
        ])
        os.system("pip install fvcore iopath")
        os.system(f"pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html")
    else:
        # We try to install PyTorch3D from source.
        os.system("curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz")
        os.system("tar xzf 1.10.0.tar.gz")
        os.environ["CUB_HOME"] = os.getcwd() + "/cub-1.10.0"
        os.system("pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'")