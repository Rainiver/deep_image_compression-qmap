import torch
import torch.nn as nn
import torch.nn.functional as F


class USM(nn.Module):
    def __init__(self, channel, kernel_type='norm', rate=0.1, learn=False):
        super(USM, self).__init__()
        self.channel = channel
        self.rate = nn.Parameter(torch.tensor([rate]), requires_grad=learn)
        if kernel_type == 'norm':
            self.k = nn.Parameter(torch.tensor([[[
                [0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0],
            ]]]).repeat(channel, 1, 1, 1).float(), requires_grad=False)
        elif kernel_type == 'special':
            self.k = nn.Parameter(torch.tensor([[[
                [-1, -1, -1],
                [-1, 8, -1],
                [-1, -1, -1],
            ]]]).repeat(channel, 1, 1, 1).float(), requires_grad=False)

    def forward(self, x):
        #print(self.k)
        #print("qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq")
        #print("x:",x.size())
        #print("x:",type(x))
        #print("c:",self.channel)
        #print("k:",type(self.k))
        highpass = F.conv2d(x, self.k, groups=self.channel, padding=1)
        x = x + self.rate * highpass
        return x


if __name__ == '__main__':
    usm = USM(channel=3, learn=False)
    x = torch.ones(1, 3, 2, 2)
    for i in range(x.shape[1]):
        x[0][i] *= i
    y = usm(x)
    # print(x, y)
    usm = USM(channel=1, learn=False)
    x = torch.tensor([[[
        [1, 2],
        [3, 4]
    ]]]).float()
    y = usm(x)
    print(x, y)
