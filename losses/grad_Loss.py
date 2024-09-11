import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms
from .Lp_Loss import Loss as Lp


class TVLoss2(nn.Module):
    def __init__(self, loss=nn.L1Loss()):
        super(TVLoss2, self).__init__()
        self.criterion = loss

    def forward(self, x, y):
        h_x = x.size()[2]
        w_x = x.size()[3]
        h_tv = self.criterion((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), (y[:, :, 1:, :] - y[:, :, :h_x - 1, :]))
        w_tv = self.criterion((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), (y[:, :, :, 1:] - y[:, :, :, :w_x - 1]))
        return h_tv + w_tv


class Loss_Simple(nn.Module):
    def __init__(self):
        super(Loss_Simple, self).__init__()

    def grad(self, src):
        b, c, h, w = src.shape
        retx = src[:, :, 1:, :] - src[:, :, :h - 1, :]
        rety = src[:, :, :, 1:] - src[:, :, :, :w - 1]
        return retx, rety

    def forward(self, src, dst):
        """
        :param src:
        :param dst:
        :return: grad square error pre pixel
        """
        src_x_grad, src_y_grad = self.grad(src)
        dst_x_grad, dst_y_grad = self.grad(dst)
        retx = (src_x_grad - dst_x_grad) ** 2
        rety = (src_y_grad - dst_y_grad) ** 2

        b, c, h, w = src.shape
        retx = retx / 2 / b / c / h / w
        rety = rety / 2 / b / c / h / w
        return retx.sum() + rety.sum()


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def grey(self, input):
        return input[:, 0, :, :] * 0.299 + input[:, 1, :, :] * 0.587 + input[:, 2, :, :] * 0.114

    def gauss(self, input):
        conv1 = nn.Conv2d(input.shape[1], input.shape[1], 5, 1, 1)
        kernel = torch.tensor(
            [[2, 4, 5, 4, 2],
             [4, 9, 12, 9, 4],
             [5, 12, 15, 12, 5],
             [4, 8, 12, 9, 4],
             [2, 4, 5, 4, 2]]).float().unsqueeze(0).unsqueeze(0)
        kernel.requires_grad = False
        kernel = kernel.expand([input.shape[1], input.shape[1], -1, -1])
        conv1.weight.data = kernel
        conv1 = conv1.to(input.device)
        return conv1(input) / 159

    def grad(self, input):
        x = self.grey(input).unsqueeze(1)
        x = self.gauss(x)
        conv1 = nn.Conv2d(x.shape[1], x.shape[1], 3, 1, 1, bias=False)
        conv2 = nn.Conv2d(x.shape[1], x.shape[1], 3, 1, 1, bias=False)
        sobelx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0)
        sobely = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0)
        sobelx.requires_grad = False
        sobely.requires_grad = False
        sobelx = sobelx.expand([x.shape[1], x.shape[1], -1, -1])
        sobely = sobely.expand([x.shape[1], x.shape[1], -1, -1])
        conv1.weight.data = sobelx
        conv2.weight.data = sobely
        conv1 = conv1.to(input.device)
        conv2 = conv2.to(input.device)
        retx = conv1(x)
        rety = conv2(x)
        return (abs(retx) + abs(rety))

    def forward(self, input, target):
        l1 = Lp()
        ret = 0
        x = self.grad(input)
        y = self.grad(target)
        return l1(x, y)


if __name__ == '__main__':
    test = Loss()
    from dataset import Dataset

    data = Dataset(transform=transforms.Compose([transforms.ToTensor()]))
    input = data.__getitem__(4).unsqueeze(0)
    print(test(input, input))
    x = (255 * test.grad(input).squeeze(0))
    x = x.detach().numpy().astype(np.uint8)
    x = np.transpose(x, [1, 2, 0])
    cv2.imshow('test', x)
    cv2.waitKey(0)
