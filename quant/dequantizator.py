import torch
import torch.nn as nn


class Dequantizator_RT(nn.Module):
    def __init__(self, B=6):
        super(Dequantizator_RT, self).__init__()
        self.B = B

    def forward(self, input):
        return input


def dequantizators(tag, B):
    return Dequantizator_RT(B)


if __name__ == '__main__':
    x = torch.randn(2, 1, 2, 2)
    print(x)
    from quantizator import quantizators

    test = quantizators("RT", 6)
    test_ = dequantizators("RT", 6)
    print(x)
    print(test(x))
    print(test_(test(x)))
