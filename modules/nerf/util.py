import numpy as np
import torch


def build_nerf_weights(layers, shapes, raw):
    weights = {}
    offset = 0
    for layer, shape in zip(layers, shapes):
        size = 1
        for dim in shape:
            size *= dim
        w = raw[offset: offset + size].view(shape)
        weights[f'{layer}.0.weight'] = w[..., :-1]
        weights[f'{layer}.0.bias'] = w[..., -1]
        offset += size

    return weights


def get_image_coords(h, w, focal_x, focal_y, device=torch.device('cpu')):
    # returns coordinates of each pixel in camera coordinate system
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    x = (x + 0.5 - w / 2) / (focal_x * w)
    y = (y + 0.5 - h / 2) / (focal_y * h)
    dirs = torch.stack([
        x, -y, -torch.ones((h, w), device=device)
    ], dim=-1)  # h w 3
    return dirs


def get_rays(poses, pixel_coords):
    ray_o = poses[..., -1]  # b 3
    ray_d = torch.matmul(poses[..., :3], pixel_coords[..., None]).squeeze(-1)  # b 3 3
    return ray_o, ray_d


def positional_points_encoding(points, pe_powers, base=2):
    powers = base ** torch.arange(0, pe_powers // 2, device=points.device, dtype=torch.float32)
    x = torch.matmul(points.unsqueeze(-1), powers.unsqueeze(0)).view(*points.shape[:-1], 3 * pe_powers // 2)
    x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
    return x


def positional_encoding(ray_o, ray_d, dists, pe_powers):
    means = (dists[..., 1:] + dists[..., :-1]) / 2
    points = ray_o.unsqueeze(1) + ray_d.unsqueeze(1) * means.unsqueeze(-1)
    x = positional_points_encoding(points, pe_powers)
    return x  # (b spp 3 * l)


def conical_gaussians(ray_o, ray_d, dists, radius):
    t_mu = (dists[..., 1:] + dists[..., :-1]) / 2
    t_sigma = (dists[..., 1:] - dists[..., :-1]) / 2
    #
    mu_t = t_mu + (2 * t_mu * t_sigma ** 2) / (3 * t_mu ** 2 + t_sigma ** 2)  # (b spp)
    sigma_t = t_sigma ** 2 / 3 - 4 * t_sigma ** 4 * (12 * t_mu ** 2 - t_sigma ** 2) / (
            15 * (3 * t_mu ** 2 + t_sigma ** 2) ** 2)
    sigma_r = radius.unsqueeze(1) ** 2 * (
            t_mu ** 2 / 4 + 5 * t_sigma ** 2 / 12 - 4 * t_sigma ** 4 / (15 * (3 * t_mu ** 2 + t_sigma ** 2)))

    ray_o, ray_d = ray_o.unsqueeze(1), ray_d.unsqueeze(1)  # (b 1 3) to broadcast them to spp

    mu = ray_o + ray_d * mu_t.unsqueeze(-1)  # (b spp 3)
    sigma = sigma_t.unsqueeze(-1) * (ray_d * ray_d) \
            + sigma_r.unsqueeze(-1) * (1 - ray_d * ray_d / torch.norm(ray_d, p=2, dim=-1, keepdim=True))
    return mu, sigma


def encode_gaussians(mu, sigma, pe_powers):
    device = mu.device
    b, spp, _ = mu.shape
    P = 2 ** torch.arange(start=0, end=pe_powers // 2, device=device, dtype=torch.float32)  # (1 l/2)
    mu_encoded = torch.matmul(mu.unsqueeze(-1), P.unsqueeze(0)).view(b, spp, 3 * pe_powers // 2)

    P2 = 4 ** torch.arange(start=0, end=pe_powers // 2, device=device, dtype=torch.float32)
    sigma_encoded = torch.matmul(sigma.unsqueeze(-1), P2.unsqueeze(0)).view(b, spp, 3 * pe_powers // 2)

    x = torch.cat([
        torch.sin(mu_encoded) * torch.exp(-sigma_encoded / 2),
        torch.cos(mu_encoded) * torch.exp(-sigma_encoded / 2)
    ], dim=-1)  # (b spp 3*l)

    return x


def conical_encoding(ray_o, ray_d, dists, pe_powers, radius):
    # x (b spp+1 3)
    mu, sigma = conical_gaussians(ray_o, ray_d, dists, radius)
    x = encode_gaussians(mu, sigma, pe_powers)
    return x


def render_pixels(rgb, density, dists):
    # rgb (b spp 3)
    # density (b spp)
    # dists (b spp+1)
    b, spp = density.shape
    device = density.device
    weighted_intervals = (dists[:, 1:] - dists[:, :-1]) * density  # b spp
    cum_transmittance = torch.cat(
        [torch.ones(b, 1, device=device), torch.exp(-torch.cumsum(weighted_intervals, dim=1))[:, :-1]], dim=-1)  # b spp
    weights = cum_transmittance * (1 - torch.exp(-weighted_intervals))
    pixels = torch.sum(weights.unsqueeze(-1) * rgb, dim=1)  # b 3
    return pixels, weights, cum_transmittance


def sample_dists(near, far, spp):
    device = near.device
    b = near.shape[0]

    dist = near.unsqueeze(1) + (far - near).unsqueeze(1) \
           * torch.linspace(start=0, end=1, steps=spp + 1, device=device).unsqueeze(0)  # b spp+1
    dist += torch.rand(b, spp + 1, device=device) * ((far - near) / spp).unsqueeze(1)
    return dist


def dist_to_rays(ray_o, ray_d, dist):
    # (b 1 3) * (b spp+1 1) so it will copy d 3 times and ray_d, ray_o spp+1 times
    points = ray_o.unsquueze(1) + ray_d.unsqueeze(1) * dist.unsquueze(2)
    return points


def adaptive_sample_dists(near, far, spp, coarse_dists, weights, weight_noise=0.01):
    device = near.device
    b = near.shape[0]

    weights = weights.detach()

    # noise weights as in the paper
    w_pad = torch.cat([weights[..., :1], weights, weights[..., -1:]], dim=-1)
    w_mx = torch.maximum(w_pad[..., :-1], w_pad[..., 1:])
    weights = 1 / 2 * (w_mx[..., :-1] + w_mx[..., 1:]) + weight_noise

    # applying inverse transform sampling to sample from distribution on the ray
    pdf = weights / torch.sum(weights, dim=1, keepdim=True)
    cdf = torch.minimum(torch.tensor(1.0), torch.cumsum(pdf, dim=1))  # b spp
    cdf = torch.cat([torch.zeros(b, 1, device=device), cdf, torch.ones(b, 1, device=device)], dim=-1)  # b spp+2

    # make uniform samples in increasing order
    u = torch.rand(b, spp + 1, device=device)
    msk = u[:, :, None] >= cdf[:, None, :]  # (b spp+1 1) (b 1 spp+2)

    # get indexes of intervals by inverse transform sampling
    ints = torch.argmax(msk.type(torch.int32) * torch.arange(spp + 2, device=device).view(1, 1, -1),
                        dim=-1).float()  # b spp+1
    # add some noise to intervals
    ints += torch.rand(b, spp + 1, device=device)
    ints /= (spp + 1)

    dists = near.unsqueeze(1) + (far - near).unsqueeze(1) * ints
    dists = torch.cat([dists, coarse_dists], dim=-1)

    # sort them to be in increasing order
    dists, indices = torch.sort(dists, dim=-1)

    return dists


def render_rays(model, ray_o, ray_d, near, far, base_radius, spp, pe_powers, scene_scale=2.0, dens_threshold=1e-2,
                dgrid_res=64, dgrid=None, coarse_dists=None, ray_weights=None):
    # dists (b, spp+1)
    if ray_weights is not None:
        dists = adaptive_sample_dists(near, far, spp, coarse_dists, ray_weights)
    else:
        dists = sample_dists(near, far, spp)

    if dgrid is not None:

        nonzero_indexes = dgrid.important_intervals(dists, ray_o, ray_d)

        x = conical_encoding(ray_o, ray_d, dists, pe_powers, base_radius)
        b, spp, features = x.shape
        x = x.view(b * spp, features)

        x_imp = x[nonzero_indexes]
        rgb_imp, density_imp = model.forward(x_imp)

        rgb = torch.zeros(b * spp, 3, device=x.device, dtype=torch.float32)
        density = torch.zeros(b * spp, device=x.device, dtype=torch.float32)
        rgb[nonzero_indexes] = rgb_imp
        density[nonzero_indexes] = density_imp.squeeze(-1)

        pixels, weights, trans = render_pixels(rgb.view(b, spp, 3), density.view(b, spp), dists)
        return pixels, dists, weights, trans

    else:
        x = conical_encoding(ray_o, ray_d, dists, pe_powers, base_radius)
        b, spp, features = x.shape
        rgb, density = model.forward(x.view(b * spp, features))
        pixels, weights, trans = render_pixels(rgb.view(b, spp, 3), density.view(b, spp), dists)
        return pixels, dists, weights, trans


def latent_encoding(points, encodings, nums, cube_clip=1.0):
    # points (n, 3)
    n, _ = points.shape
    # encodings (b, features, dim, dim)
    b, features, dim, _ = encodings.shape
    features //= 3

    encodings = torch.permute(encodings, (0, 2, 3, 1))
    encodings = torch.tensor_split(encodings, 3, dim=-1)

    points = torch.tensor_split(points, 3, dim=1)
    outs = []
    for x1, x2, encoding in zip(points, points[1:] + points[:1], encodings):
        # x1, x2 = x1.squeeze(1), x2.squeeze(1)
        # we map all points inside cube into [0; dim-2]
        # then add 1 before for values lower -cube_clip and 1 after for values greater cube_clip
        x1 = torch.clip((x1 / cube_clip + 1) / 2 * (dim - 2) + 1, min=0, max=dim - 1).view(n)
        x2 = torch.clip((x2 / cube_clip + 1) / 2 * (dim - 2) + 1, min=0, max=dim - 1).view(n)
        x1_low, x1_high = torch.floor(x1), torch.ceil(x1)
        x2_low, x2_high = torch.floor(x2), torch.ceil(x2)

        # do bilinear interpolation of values on vertexes
        latents = (encoding[nums, x1_low.long(), x2_low.long()] * (x1 - x1_low).view(n, 1) +
                   encoding[nums, x1_high.long(), x2_low.long()] * (x1_high - x1).view(n, 1)) * (
                          x2 - x2_low).view(n, 1) + \
                  (encoding[nums, x1_low.long(), x2_high.long()] * (x1 - x1_low).view(n, 1) +
                   encoding[nums, x1_high.long(), x2_high.long()] * (x1_high - x1).view(n, 1)) * (
                          x2_high - x2).view(n, 1)
        outs.append(latents)

    return torch.cat(outs, dim=-1)


def render_latent_nerf(latents, model, w=128, h=128, focal=1.5, camera_distance=4.0,
                       near_val=4 - 3 ** (1 / 2), far_val=4 + 3 ** (1 / 2), batch_rays=8192):
    n_pixels = w * h
    dirs = get_image_coords(h, w, focal, focal).view(n_pixels, 3).type_as(latents)

    default_cams = get_default_cams()

    galleries = []
    for gallery_id in range(len(latents)):

        images = []
        for cam in default_cams:
            cam = torch.tensor(cam)[:3].type_as(latents)  # (3 4)
            cam[..., -1] *= camera_distance / 4.0
            pixels = []
            for batch_id in range(n_pixels // batch_rays + min(1, n_pixels % batch_rays)):
                l, r = batch_id * batch_rays, min(n_pixels, (batch_id + 1) * batch_rays)
                size = r - l
                poses = cam.view(1, 3, 4).repeat(size, 1, 1)
                pixel_coords = dirs[l:r]
                near = torch.ones(size).type_as(latents) * near_val
                far = torch.ones(size).type_as(latents) * far_val
                base_radius = torch.ones(size).type_as(latents) / (focal * w * 3 ** (1 / 2))
                with torch.no_grad():
                    out = model(latents=latents[gallery_id].unsqueeze(0), near=near, far=far, base_radius=base_radius,
                                poses=poses, pixel_coords=pixel_coords)
                rendered_pixels = (out['fine_pixels'] + 1) * 255 / 2
                pixels.append(rendered_pixels.cpu().detach().numpy().astype(np.uint8))
            images.append(np.concatenate(pixels, axis=0).reshape((h, w, 3)))
        gallery = np.array(images).reshape((2, 4, h, w, 3)).transpose(0, 2, 1, 3, 4).reshape((2 * h, 4 * w, 3))
        galleries.append(gallery)
    return galleries


def render_batch(batch, model, pe_powers, spp):
    ray_o, ray_d = get_rays(batch['poses'], batch['pixel_coords'])
    coarse_pixels, coarse_dists, weights, coarse_trans = render_rays(model=model,
                                                                     ray_o=ray_o, ray_d=ray_d,
                                                                     near=batch['near'], far=batch['far'],
                                                                     base_radius=batch['base_radius'],
                                                                     spp=spp, pe_powers=pe_powers)
    fine_pixels, _, _, fine_trans = render_rays(model=model,
                                                ray_o=ray_o, ray_d=ray_d,
                                                near=batch['near'], far=batch['far'], base_radius=batch['base_radius'],
                                                spp=spp, pe_powers=pe_powers,
                                                coarse_dists=coarse_dists, ray_weights=weights)
    return coarse_pixels, fine_pixels, torch.cat([coarse_trans, fine_trans], dim=1)


def render_gallery(model, batch_size, spp, pe_powers, w=128, h=128, focal=64.0, camera_distance=4.0,
                   dgrid=None, device=torch.device('cpu')):
    default_cams = get_default_cams()
    n_pixels = h * w
    dirs = get_image_coords(h, w, focal, focal, device=device).view(n_pixels, 3)
    images = []
    for cam in default_cams:
        cam = torch.tensor(cam, device=device)[:3]  # (3 4)
        cam[..., -1] *= camera_distance / 4.0
        pixels = []
        for batch_id in range(n_pixels // batch_size + min(1, n_pixels % batch_size)):
            l, r = batch_id * batch_size, min(n_pixels, (batch_id + 1) * batch_size)
            size = r - l
            batch = {
                'poses': cam.view(1, 3, 4).repeat(size, 1, 1),
                'pixel_coords': dirs[l:r],
                'near': torch.ones(size, device=device, dtype=torch.float32) * 2.0,
                'far': torch.ones(size, device=device, dtype=torch.float32) * 6.0,
                'base_radius': torch.ones(size, device=device, dtype=torch.float32) / (3 ** (1 / 2) * focal * w)
            }
            with torch.no_grad():
                _, fine_pixels, _ = render_batch(batch, model, pe_powers, spp)
                fine_pixels = (fine_pixels + 1) * 255 / 2
            pixels.append(fine_pixels.cpu().detach().numpy().astype(np.uint8))

        images.append(np.concatenate(pixels, axis=0).reshape((h, w, 3)))
    gallery = np.array(images).reshape((2, 4, h, w, 3)).transpose(0, 2, 1, 3, 4).reshape((2 * h, 4 * w, 3))
    return gallery


def get_default_cams():
    render_gallery._default_cams = [[[-1.0, -0.0, -0.0, -0.0],
                                     [0.0, 0.866, 0.5, 0.5],
                                     [-0.0, 0.5, -0.866, -0.866]],
                                    [[-1.0, -0.0, -0.0, -0.0],
                                     [0.0, 0.866, -0.5, -0.5],
                                     [-0.0, -0.5, -0.866, -0.866]],
                                    [[1.0, -0.0, 0.0, 0.0],
                                     [-0.0, 0.866, 0.5, 0.5],
                                     [-0.0, -0.5, 0.866, 0.866]],
                                    [[1.0, 0.0, 0.0, 0.0],
                                     [0.0, 0.866, -0.5, -0.5],
                                     [-0.0, 0.5, 0.866, 0.866]],
                                    [[0.0, 0.5, -0.866, -0.866],
                                     [-0.0, 0.866, 0.5, 0.5],
                                     [1.0, -0.0, 0.0, 0.0]],
                                    [[0.0, -0.5, -0.866, -0.866],
                                     [0.0, 0.866, -0.5, -0.5],
                                     [1.0, 0.0, 0.0, 0.0]],
                                    [[0.0, -0.5, 0.866, 0.866],
                                     [0.0, 0.866, 0.5, 0.5],
                                     [-1.0, -0.0, 0.0, 0.0]],
                                    [[0.0, 0.5, 0.866, 0.866],
                                     [-0.0, 0.866, -0.5, -0.5],
                                     [-1.0, 0.0, 0.0, 0.0]]]
    render_gallery.default_cams = [[[-1, 0, 0, 0], [0, -0.7341099977493286, 0.6790306568145752, 2.737260103225708],
                                    [0, 0.6790306568145752, 0.7341099381446838, 2.959291696548462], [0, 0, 0, 1]], [
                                       [-0.9048271179199219, 0.3107704222202301, -0.29104936122894287,
                                        -1.1732574701309204],
                                       [-0.4257793426513672, -0.6604207754135132, 0.6185113787651062,
                                        2.4932990074157715],
                                       [0, 0.6835686564445496, 0.7298862338066101, 2.942265510559082], [0, 0, 0, 1]],
                                   [[0.2486903965473175, 0.6220898032188416, -0.7423997521400452, -2.992709159851074],
                                    [-0.9685831069946289, 0.1597258448600769, -0.19061626493930817,
                                     -0.7683987617492676],
                                    [-7.450581485102248e-9, 0.7664802670478821, 0.6422678232192993, 2.589064359664917],
                                    [0, 0, 0, 1]],
                                   [[0.8090174198150635, 0.3284419775009155, -0.48745936155319214, -1.9650115966796875],
                                    [-0.5877846479415894, 0.45206233859062195, -0.670931339263916, -2.704610824584961],
                                    [0, 0.8293163180351257, 0.5587794184684753, 2.252511978149414], [0, 0, 0, 1]],
                                   [[0.535825788974762, -0.26195481419563293, 0.8026645183563232, 3.2356443405151367],
                                    [0.8443286418914795, 0.1662411242723465, -0.5093849897384644, -2.053396701812744],
                                    [0, 0.9506542682647705, 0.3102521598339081, 1.250666618347168], [0, 0, 0, 1]],
                                   [[-0.7289698123931885, -0.1044226884841919, 0.6765344142913818, 2.7271976470947266],
                                    [0.6845458149909973, -0.11119924485683441, 0.7204384803771973, 2.9041807651519775],
                                    [-7.450580596923828e-9, 0.9882968664169312, 0.15254297852516174,
                                     0.6149204969406128], [0, 0, 0, 1]], [
                                       [-0.9921149611473083, -0.015738578513264656, 0.12433910369873047,
                                        0.5012269616127014],
                                       [0.12533122301101685, -0.12458571791648865, 0.9842613935470581,
                                        3.967684507369995],
                                       [0, 0.9920839667320251, 0.12557590007781982, 0.5062126517295837], [0, 0, 0, 1]],
                                   [[0.30901575088500977, -0.2606591582298279, 0.9146398305892944, 3.6870310306549072],
                                    [0.951056957244873, 0.08469292521476746, -0.29718318581581116, -1.197983741760254],
                                    [0, 0.9617089033126831, 0.27407315373420715, 1.104824185371399], [0, 0, 0, 1]]]
    return render_gallery.default_cams
