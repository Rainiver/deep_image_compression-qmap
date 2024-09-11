from functools import partial
import random
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from nets.augment.augment_delta import DeltaAugment
from nets.augment.augment_sr import *


def get_low_freq(im, k=4):
    ret = nn.AvgPool2d(k, stride=k)(im.clone())
    ret = nn.AdaptiveAvgPool2d(im.shape[2:])(ret)
    assert ret.size() == im.size()
    return ret


def blend_partial(im, prob=0.5, alpha=0.6):
    return blend(get_low_freq(im), im, prob=prob, alpha=alpha)[1].clamp(min=0., max=1.)


def cutmixup_partial(im, mixup_prob=0.5, mixup_alpha=1.2, cutmix_prob=0.5, cutmix_alpha=0.7):
    return cutmixup(get_low_freq(im), im,
                    mixup_prob=mixup_prob, mixup_alpha=mixup_alpha,
                    cutmix_prob=cutmix_prob, cutmix_alpha=cutmix_alpha)[1]


def cutblur_partial(im, prob=0.5, alpha=0.7):
    return cutblur(get_low_freq(im), im, prob=prob, alpha=alpha)[1]


def rgb_partial(im, prob=0.5):
    return rgb(get_low_freq(im), im, prob=prob)[1]


import functools


def _warp_sr_augment(f):
    # return transforms.Compose([
    #     lambda im: im.unsqueeze(dim=0),
    #     f,
    #     lambda im: im.squeeze(dim=0),
    # ])
    @functools.wraps(f)
    def warp(x):
        return f(x.clone())

    return warp


def get_transform_old(tag, crop_size=None):
    # PILImage 2 PILImage
    if tag == 'ColorJitter':
        return transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
    elif tag == 'Flip':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
        ])
    elif tag == 'Crop':
        # 1000 * scale = 256
        return transforms.RandomResizedCrop(crop_size, scale=(0.256, 1.0), ratio=(1., 1.))
    elif tag == 'Erase':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomErasing(),
            transforms.ToPILImage(),
        ])
    else:
        raise ValueError('unsupported transform tag:{}'.format(tag))


def get_transform(tag, **kwargs):
    # tensor 2 tensor
    if tag == 'blend':
        return _warp_sr_augment(blend_partial)
    elif tag == 'cutmixup':
        return _warp_sr_augment(cutmixup_partial)
    elif tag == 'cutblur':
        return _warp_sr_augment(cutblur_partial)
    elif tag == 'rgb':
        return _warp_sr_augment(rgb_partial)
    elif tag == 'delta':
        return DeltaAugment(kwargs['que'])
    else:
        raise ValueError('unsupported transform tag:{}'.format(tag))


class Warp(nn.Module):
    def __init__(self, tfs):
        super(Warp, self).__init__()
        self.tfs = tfs

    def forward(self, x):
        for tf in self.tfs:
            # print(tf.__name__)
            x = tf(x)
        return x


"""
> Rethinking Data Augmentation for Image Super-resolution: A Comprehensive Analysis and a New Strategy
see: https://arxiv.org/abs/2004.00448

 - blend
 - cutmixup
 - cutblur
 - rgb

> x + k * (x - x_hat) to enhance origin image

 - delta
 
"""


def data_augment(cfg='NONE', **kwargs):
    if cfg == 'NONE':
        tfs = nn.Identity()
    elif cfg == 'delta':
        tfs = Warp([get_transform('delta', **kwargs)])
    else:
        assert isinstance(cfg, list)
        tfs = [get_transform(i, **kwargs) for i in cfg]
        tfs = Warp(tfs)

    return tfs


if __name__ == '__main__':
    import yaml
    import matplotlib.pyplot as plt
    import cv2
    from easydict import EasyDict as ed
    import numpy as np

    args = ed(yaml.load(open('t.yml', 'r')))
    args.crop_size = 1024

    t = data_augment(args.data_augment)
    # x = cv2.imread('lena_std.tif')
    # x = torch.Tensor(x)
    im1 = cv2.imread('kodim14.png')
    im2 = cv2.imread('kodim15.png')
    im3 = cv2.imread('kodim19.png')


    def pre(im):
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return torch.Tensor(im).permute((2, 0, 1)).unsqueeze(0)


    im1 = pre(im1)
    im2 = pre(im2)
    im3 = pre(im3)
    im4 = torch.cat((im1, im2, im3))

    print(im4.size())


    def show_numpy(tt):
        y = np.transpose(tt, (1, 2, 0))
        plt.imshow(y)
        plt.show()


    def show_torch(tt):
        y = torchvision.utils.make_grid(tt.int(), nrow=3)
        y = np.transpose(y, (1, 2, 0))
        plt.imshow(y)
        plt.show()


    show_torch(torch.cat((im4.clone(), *[t(im4.clone()) for i in range(10)])))
