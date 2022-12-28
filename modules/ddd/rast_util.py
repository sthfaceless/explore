from nvdiffrast import torch as nvt

from modules.ddd.util import *


class Rasterizer:

    def __init__(self, device):
        self.context = nvt.RasterizeCudaContext(device=device)

    def rasterize(self, vertices, faces, poses, projections, res=128):
        n_vertices, n_poses = len(vertices), len(poses)

        # add proj dim to vertices
        vertices = torch.cat([vertices, torch.ones_like(vertices[:, 0]).unsqueeze(1)], dim=1)

        # make vertices in coordinates of each camera
        vertices = vertices.unsqueeze(0).repeat(n_poses, 1, 1).view(-1, 4, 1)
        poses = poses.unsqueeze(1).repeat(1, n_vertices, 1, 1).view(-1, 4, 4)
        projections = projections.unsqueeze(1).repeat(1, n_vertices, 1, 1).view(-1, 4, 4)
        vertices = torch.matmul(projections, torch.matmul(poses, vertices)).view(n_poses, n_vertices, 4)

        out = nvt.rasterize(glctx=self.context, pos=vertices.float(), tri=faces.int(),
                            resolution=(res, res))[0]
        return out

    def interpolate(self, attr, rast, tri):
        out = nvt.interpolate(attr.float(), rast, tri.int())[0]
        return out
