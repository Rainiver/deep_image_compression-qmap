from scipy.interpolate import interp1d
import numpy as np
import torch


def reward_base_psnr(bpp, msssim, psnr):
    return psnr


def reward_base_ms(bpp, msssim, psnr):
    return msssim


def reward_rd(bpp, msssim, psnr, evals, **kwargs):
    # print(evals, "evals!!!", flush=True)
    # print(kwargs, "reward kwargs!!!", flush=True)
    return - (bpp + kwargs['lambda'] * evals['eval/' + kwargs['d']])
