import torch.nn as nn
import torch.nn.functional as F
import math


def initial(model, scale_factor=1.0, mode="FAN_IN"):
    if mode != "FAN_IN" and mode != "FAN_out":
        assert 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            if mode == "FAN_IN":
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(scale_factor / n))
            else:
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(scale_factor / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def _conv_layer(in_channels, out_channels, kernel, stride, padding, bias=True, bn=True):
    layers = []
    layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride,
                            padding=padding, bias=bias, padding_mode='reflection'))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2))
    return nn.Sequential(*layers)


def _deconv_layer(in_channels, out_channels, kernel, stride, padding, bias=True, bn=True):
    layers = []
    layers.append(nn.ConvTranspose2d(
        in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride,
        padding=padding, bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2))
    return nn.Sequential(*layers)


def _pool_layer(kernel, stride, padding=0, mode="Avg"):
    if mode != "Max" and mode != "Avg":
        assert 0
    if mode == "Max":
        return nn.Sequential(nn.MaxPool2d(kernel_size=kernel, stride=stride))
    else:
        return nn.Sequential(nn.AvgPool2d(kernel_size=kernel, stride=stride, padding=padding))
