import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms


# noinspection PyUnresolvedReferences
# torch.FloatTensor is usable actually
def yuv_to_rgb(input):
    """
    ret = []
    for i in range(input_colors.shape[0]):
        origin = np.transpose(input_colors[i].detach().cpu().numpy(), [1, 2, 0])
        ret.append(np.transpose(cv2.cvtColor(origin, cv2.COLOR_YUV2RGB), [2, 0, 1]))
    return torch.FloatTensor(ret)
    """
    x = input.transpose(1, 3)
    x = x.contiguous().view(-1, 3).float()
    mat = torch.tensor([[1., 1., 1.],
                        [0, -0.3455, 1.779],
                        [1.4075, -0.7169, 0]])
    bias = torch.tensor([0, -0.5, -0.5])
    if torch.cuda.is_available():
        mat = mat.cuda()
        bias = bias.cuda()
    temp = (x + bias).mm(mat)
    return temp.view(input.shape[0], input.shape[3], input.shape[2], 3).transpose(1, 3)


def yuv_to_bgr(input):
    """
    ret = []
    for i in range(input.shape[0]):
        origin = np.transpose(input[i].detach().cpu().numpy(), [1, 2, 0])
        ret.append(np.transpose(cv2.cvtColor(origin, cv2.COLOR_YUV2BGR), [2, 0, 1]))
    return torch.FloatTensor(ret)
    """
    x = input.transpose(1, 3)
    x = x.contiguous().view(-1, 3).float()
    mat = torch.tensor([[1., 1., 1.],
                        [1.779, -0.3455, 0],
                        [0, -0.7169, 1.4075]])
    bias = torch.tensor([0, -0.5, -0.5])
    if torch.cuda.is_available():
        mat = mat.cuda()
        bias = bias.cuda()
    temp = (x + bias).mm(mat)
    return temp.view(input.shape[0], input.shape[3], input.shape[2], 3).transpose(1, 3)


# noinspection PyUnresolvedReferences
# torch.FloatTensor is usable actually
def rgb_to_yuv(input):
    """
    ret = []
    for i in range(input_colors.shape[0]):
        origin = np.transpose(input_colors[i].detach().cpu().numpy(), [1, 2, 0])
        ret.append(np.transpose(cv2.cvtColor(origin, cv2.COLOR_RGB2YUV), [2, 0, 1]))
    return torch.FloatTensor(ret)
    """
    x = input.transpose(1, 3)
    x = x.contiguous().view(-1, 3).float()
    mat = torch.tensor([[0.299, -0.168735892, 0.5],
                        [0.587, -0.331264108, -0.418687589],
                        [0.114, 0.5, -0.081312411]])
    bias = torch.tensor([0, 0.5, 0.5])
    if torch.cuda.is_available():
        mat = mat.cuda()
        bias = bias.cuda()
    temp = x.mm(mat) + bias
    return temp.view(input.shape[0], input.shape[3], input.shape[2], 3).transpose(1, 3)


def bgr_to_yuv(input):
    """
    ret = []
    for i in range(input.shape[0]):
        origin = np.transpose(input[i].detach().cpu().numpy(), [1, 2, 0])
        ret.append(np.transpose(cv2.cvtColor(origin, cv2.COLOR_BGR2YUV), [2, 0, 1]))
    return torch.FloatTensor(ret)
    """
    x = input.transpose(1, 3)
    x = x.contiguous().view(-1, 3).float()
    mat = torch.tensor([[0.114, 0.5, -0.081312411],
                        [0.587, -0.331264108, -0.418687589],
                        [0.299, -0.168735892, 0.5]])
    bias = torch.tensor([0, 0.5, 0.5])
    if torch.cuda.is_available():
        mat = mat.cuda()
        bias = bias.cuda()
    temp = x.mm(mat) + bias
    return temp.view(input.shape[0], input.shape[3], input.shape[2], 3).transpose(1, 3)


def yuv_to_yuv420(yuv, method='drop'):
    '''
    yuv 444 to yuv420
    :param method: "drop" "avg"
    '''
    if method == 'drop':
        y = yuv[:, 0:1, :, :]
        u = yuv[:, 1:2, ::2, ::2]
        v = yuv[:, 2:3, 1::2, ::2]
    elif method == 'avg':
        downsample = nn.AvgPool2d((2, 2), 2)
        y = yuv[:, 0:1, :, :]
        u = downsample(yuv[:, 1:2, :, :])
        v = downsample(yuv[:, 2:3, :, :])
    else:
        raise KeyError

    return [y, u, v]


# noinspection PyUnresolvedReferences
# torch.cat is usable actually
def yuv420_to_yuv(input_colors, method='bilinear'):
    '''
    yuv420 to yuv444
    :param method: bilinear, nearest
    '''
    up = nn.Upsample(size=input_colors[0].shape[2:], mode=method)
    input_colors[1] = up(input_colors[1])
    input_colors[2] = up(input_colors[2])
    return torch.cat(list(input_colors), 1)


if __name__ == '__main__':
    x = torch.rand([4, 3, 256, 256])
    y = x.clone()
    x = yuv_to_yuv420(x, 'avg')
    print(x[0].shape, x[1].shape, x[2].shape)
    x = yuv420_to_yuv(x)
    print(x[0].shape, x[1].shape, x[2].shape)
    print((y - x).mean())
