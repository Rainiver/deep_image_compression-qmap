import torch
from torch import nn
from .layers import SignalConv2d

class ResBlock_Video(nn.Module):
    def __init__(self, n=192):
        super(ResBlock_Video, self).__init__()
        self.conv1 = SignalConv2d(in_channels=n, out_channels=n, kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = SignalConv2d(in_channels=n, out_channels=n, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x_ori = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = x + x_ori
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, downsample=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample

        self.blocks = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, downsample, (kernel_size - 1) // 2, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, 1, (kernel_size - 1) // 2, bias=False))

        if self.equal_channel and self.equal_space:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, downsample, bias=False)

        self.lrelu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.blocks(x)
        residual = self.shortcut(x)
        out += residual
        out = self.lrelu(out)
        return out

    @property
    def equal_channel(self):
        return self.in_channels == self.out_channels

    @property
    def equal_space(self):
        return self.downsample == 1

if __name__ == '__main__':
    input1 = torch.ones([4, 192, 64, 64])
    resblock = ResidualBlock(in_channels=192,
                             out_channels=256,
                             kernel_size=5,
                             downsample=4)
    output1 = resblock(input1)
    print(output1.shape)
