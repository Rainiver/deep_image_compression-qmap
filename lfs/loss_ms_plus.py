import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import numpy as np
import spring.linklink as link


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def rprint(*args, **kwargs):
    if link.get_rank() == 0:
        print(*args, **kwargs)


def ssim(img1, img2,
         window_size=11, window=None, size_average=True, full=False, val_range=None,
         a=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    mu12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel)

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
    # add sigma1_sigma2
    # sigma1_sigma2 = torch.sqrt(sigma1_sq * sigma2_sq)
    sigma1 = torch.sqrt(sigma1_sq.abs())
    sigma2 = torch.sqrt(sigma2_sq.abs())
    sigma1_sigma2 = sigma1 * sigma2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = v1 / v2  # contrast sensitivity
    # add etc.

    # cs_norm = (2.0 * sigma1_sigma2 + C2) / (sigma1_sq + sigma2_sq + C2)
    cs_mse = (1 - (sigma1 - sigma2) ** 2) * (sigma12 + C2) / (sigma1_sigma2 + C2)
    # print("some val", sigma1_sigma2.mean(),
    #       sigma1.mean(),
    #       sigma2.mean(),
    #       mu12.mean(),
    #       ((sigma12 + C2) / (sigma1_sigma2 + C2)).mean(), flush=True)

    cs_ret = (a[0] * cs + a[1] * cs_mse) / (a[0] + a[1] + 1e-6)
    # print("c and s", link.get_rank(), a[0], a[1], cs.mean(), cs_mse.mean(), flush=True)

    ssim_map = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1) * cs_ret
    # mse and etc.
    ssim_map_mse = (1. - (mu1 - mu2) ** 2) * cs_ret
    ssim_map_dot = (mu12 + C1) / (mu1_sq + mu2_sq + C1) * cs_ret

    a = a[2:]
    ssim_ret = (a[0] * ssim_map + a[1] * ssim_map_mse + a[2] * ssim_map_dot) / (a[0] + a[1] + a[2] + 1e-6)
    # print("some val 2", ((mu1 - mu2) ** 2).mean(),
    #       ((mu12 + C1) / (mu1_sq + mu2_sq + C1)).mean(), flush=True)

    # print("ssim mix up", link.get_rank(), a[0], a[1], a[2],
    #       ssim_map.mean(), ssim_map_mse.mean(), ssim_map_dot.mean(),
    #       flush=True)

    # ret = ssim_map.mean()
    # add channel attention
    # post process
    a = a[3:]
    ret = a[0] * ssim_ret[:, 0, ...] + a[1] * ssim_ret[:, 1, ...] + a[2] * ssim_ret[:, 2, ...]
    # print("channel mix up", link.get_rank(), a[0], a[1], a[2], ssim_ret.size(), flush=True)
    ret = ret / (a[0] + a[1] + a[2] + 1e-6)
    ret = ret.mean()

    # a = [1., 0., 1., 0., 0., 1., 1., 1.]

    # print("ret ssim and cs", ret, cs_ret.mean(), flush=True)
    # assert 0
    return ret, cs_ret.mean()


def msssim(img1, img2, a, lambda_dis=1.):
    window_size = 11
    size_average = True
    val_range = None
    normalize = True  # !!!

    device = img1.device
    # add weights
    weights_a = a[:5]  # [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

    # if link.get_rank() == 0:
    #     print(weights_a, flush=True)

    weights_a = [i / (sum(weights_a) + 1e-6) for i in weights_a]
    a = a[5:]
    """
    [
    0.0448, 0.2856, 0.3001, 0.2363, 0.1333,
    1., 0., 1., 0., 0., 1., 1., 1.,
    1., 0., 1., 0., 0., 1., 1., 1.,
    1., 0., 1., 0., 0., 1., 1., 1.,
    1., 0., 1., 0., 0., 1., 1., 1.,
    1., 0., 1., 0., 0., 1., 1., 1.,
    ]
    """

    weights = torch.FloatTensor(weights_a).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        a_len_per_level = len(a) // levels
        a_level_i = a[a_len_per_level * _: a_len_per_level * (_ + 1)]

        sim, cs = ssim(img1, img2,
                       window_size=window_size,
                       size_average=size_average, full=True, val_range=val_range,
                       a=a_level_i)

        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1]) * pow2[-1]
    return output


def loss_msssim(img1, img2, a, lambda_dis=1.):
    return 1 - msssim(img1, img2, a, lambda_dis)
