import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self, k=0.001):
        super(Loss, self).__init__()
        self.k = k

    def forward(self, input):
        b, c, h, w = input.shape
        sum_h = self.get_size(input[:, :, 1:, :])
        sum_w = self.get_size(input[:, :, :, 1:])
        tv_h = torch.pow((input[:, :, 1:, :] - input[:, :, :h - 1, :]), 2).sum()
        tv_w = torch.pow((input[:, :, :, 1:] - input[:, :, :, :w - 1]), 2).sum()
        return self.k * 2 * (tv_h / sum_h + tv_w / sum_w) / b

    def get_size(self, x):
        return x.shape[1] * x.shape[2] * x.shape[3]


if __name__ == '__main__':
    x = torch.randn(1, 3, 111, 111)
    test = Loss()
    print(test(x))
