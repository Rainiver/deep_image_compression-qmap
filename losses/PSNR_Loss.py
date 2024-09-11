import torch
import torch.nn as nn
import numpy as np
import torchvision
import cv2


class Loss(nn.Module):
    def __init__(self, tp='float'):
        super(Loss, self).__init__()
        self.tp = tp

    def forward(self, input, target):
        if input.shape != target.shape:
            assert 0
        if len(input.shape) == 3:
            input = input.unsqueeze(0)
            target = target.unsqueeze(0)
        if self.tp == 'int':
            input = input / 255
            target = target / 255
        MSE = torch.zeros(1).long()
        MSE = ((target - input) ** 2).sum() / (input.shape[1] * input.shape[2] * input.shape[0] * input.shape[3])
        return 10 * torch.log10(1. / MSE)


if __name__ == '__main__':
    input = cv2.imread('./test.jpeg')
    # input = np.sum(input, 2)
    # input = (input) / 3 / 256
    input = input / 256
    test = Loss()
    x = input.copy()
    for i in range(0, 1):
        temp = x[i].copy()
        x[i] = x[i + 400]
        x[i + 400] = temp
    print(test(torch.from_numpy(input), torch.from_numpy(x)))
