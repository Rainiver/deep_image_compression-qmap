import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.SSIM_Loss import msssim
import math


def hybrid_mse_ssim(output, target, a, lambda_dis=1.):
    # msssim 1000.
    # mse 0.4
    # entropy 0.01 - 8 - 400.
    x_hat = output
    x = target
    loss_msssim = 1 - msssim(x_hat, x, normalize=True)
    loss_mse = nn.MSELoss(reduction='mean')(x_hat, x) * 65025
    loss_rmse = torch.sqrt(loss_mse)
    loss_l1 = nn.SmoothL1Loss(reduction='mean')(x_hat * 255, x * 255)

    lambda_msssim = a[0] * 1000
    lambda_mse = a[1] * 0.4
    lambda_rmse = a[2] * 0.4
    lambda_l1 = a[3] * 0.4

    loss = loss_msssim * lambda_msssim + \
           loss_mse * lambda_mse + \
           loss_rmse * lambda_rmse + \
           loss_l1 * lambda_l1

    # return loss / (lambda_msssim + lambda_mse + lambda_rmse + lambda_l1)

    # this is correspondence to lambda(old) * metric
    # usage: loss4 * lambda4 + this_func * 1
    return lambda_dis * loss / a.sum()


def hybrid_mse_ssim_2lambda(output, target, a, lambda_dis=1.):
    x_hat = output
    x = target

    loss_msssim = 1 - msssim(x_hat, x, normalize=True)
    loss_mse = nn.MSELoss(reduction='mean')(x_hat, x) * 65025

    lambda_msssim = a[0] * 1000
    lambda_mse = a[1] * 0.4

    loss = loss_msssim * lambda_msssim + \
           loss_mse * lambda_mse

    return lambda_dis * loss / a.sum()


def hybrid_mse_ssim_1lambda(output, target, a, lambda_dis=1.):
    x_hat = output
    x = target

    loss_msssim = 1 - msssim(x_hat, x, normalize=True)
    loss_mse = nn.MSELoss(reduction='mean')(x_hat, x) * 65025

    lambda_msssim = a[0] * 1000
    lambda_mse = (1 - a[0]) * 0.4

    loss = loss_msssim * lambda_msssim + \
           loss_mse * lambda_mse

    return lambda_dis * loss


def hybrid_mse_ssim_1lambda_exp_repara(output, target, a, lambda_dis=1., **kwargs):
    x_hat = output
    x = target

    loss_msssim = 1 - msssim(x_hat, x, normalize=True)
    loss_mse = nn.MSELoss(reduction='mean')(x_hat, x) * 65025

    base = kwargs["base"]
    if base == "e":
        base = math.e

    part0 = 1 / (1 + base ** a[0])
    part1 = 1 - part0
    lambda_msssim = part0 * 1000
    lambda_mse = part1 * 0.4

    loss = loss_msssim * lambda_msssim + \
           loss_mse * lambda_mse

    return lambda_dis * loss


def mse_plus_default_x(output, target, a, lambda_dis=1., **kwargs):
    x_hat = output * 255
    x = target * 255

    x_interval = kwargs['interval']
    loss = torch.zeros_like(x)
    for i in range(len(x_interval) - 1):
        l = x_interval[i]
        r = x_interval[i + 1]
        al = a[i]
        ar = a[i + 1]
        # we use symmetry loss for <0 and >0
        d = torch.abs(x - x_hat)
        k = (al - ar) / (l - r)
        # const is not matter, df/dd gives a piecewise linear grad
        f = 0.5 * k * d ** 2 - (k * l - al) * d
        mask = (l <= d) & (d < r)
        loss += f * mask

    # more safe
    loss /= sum(a)
    loss = loss.mean()
    return lambda_dis * loss


def mse_plus_no_norm(output, target, a, lambda_dis=1., **kwargs):
    # print("use no norm!!!", flush=True)

    x_hat = output * 255
    x = target * 255

    if 'a_l' in kwargs and 'a_r' in kwargs:
        _a_l = kwargs['a_l']
        _a_r = kwargs['a_r']
        a = a[_a_l: _a_r]

        # don't print
        # print(f'use a_l{_a_l} a_r{_a_r} and a{a} succeed!!!')

    x_interval = kwargs['interval']
    loss = torch.zeros_like(x)
    for i in range(len(x_interval) - 1):
        l = x_interval[i]
        r = x_interval[i + 1]
        al = a[i]
        ar = a[i + 1]
        # we use symmetry loss for <0 and >0
        d = torch.abs(x - x_hat)
        k = (al - ar) / (l - r)
        # const is not matter, df/dd gives a piecewise linear grad
        f = 0.5 * k * d ** 2 - (k * l - al) * d
        mask = (l <= d) & (d < r)
        loss += f * mask

    loss = loss.mean()
    return lambda_dis * loss


def mse_plus_no_norm_fix(output, target, a, lambda_dis=1., **kwargs):
    # print("use no norm!!!", flush=True)

    x_hat = output * 255
    x = target * 255

    if 'a_l' in kwargs and 'a_r' in kwargs:
        _a_l = kwargs['a_l']
        _a_r = kwargs['a_r']
        a = a[_a_l: _a_r]

        # don't print
        # print(f'use a_l{_a_l} a_r{_a_r} and a{a} succeed!!!')

    x_interval = kwargs['interval']
    loss = torch.zeros_like(x)
    for i in range(len(x_interval) - 1):
        l = x_interval[i]
        r = x_interval[i + 1]
        al = a[i]
        ar = a[i + 1]
        # we use symmetry loss for <0 and >0
        d = torch.abs(x - x_hat)
        k = (al - ar) / (l - r)
        # const is not matter, df/dd gives a piecewise linear grad
        f = 0.5 * k * d ** 2 - (k * l - al) * d
        mask = (l <= d) & (d < r)
        loss += torch.where(mask, f, torch.zeros_like(f))

    loss = loss.mean()
    return lambda_dis * loss * 2


def mse_plus_no_norm_fix_exp_repara(output, target, a, lambda_dis=1., **kwargs):
    base = kwargs["base"]
    if base == "e":
        base = math.e
    a = [loc_i * base ** a_i for a_i, loc_i in zip(a, kwargs["init_loc"])]
    return mse_plus_no_norm_fix(output, target, a, lambda_dis, **kwargs)
