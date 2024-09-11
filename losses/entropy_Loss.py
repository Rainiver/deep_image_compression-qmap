import torch
import torch.nn as nn
from functools import reduce


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, input):
        #num_ele = reduce(lambda a, b: a * b, input.shape, 1)
        return -(torch.log2(input)).sum() # / num_ele


if __name__ == '__main__':
    x = torch.randn(1, 3, 111, 111).clamp(0000.1, 0.9999)
    test = Loss()
    print(test(x))
