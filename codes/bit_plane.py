import torch
import torch.nn as nn


class Bit_plane(nn.Module):
    def __init__(self, B=6):
        super(Bit_plane, self).__init__()
        self.B = B
        self.factor = (1 << B) - 1

    def forward(self, input):
        b, c, h, w = input.shape
        input = torch.round(input * self.factor).int()
        ret = torch.zeros([b, c, self.B, h, w]).int()
        for i in range(0, self.B):
            ret[:, :, i, :, :] = (input[:, :, :, :] >> i) & 1
        return ret


if __name__ == '__main__':
    from quant.quantizator import quantizators

    quan = quantizators("RT", 8, train=False)
    test = Bit_plane()
    x = torch.abs(torch.randn(1, 1, 2, 2)).clamp(0, 1)
    x = quan(x)
    print(x * 255)
    print(test(x))
