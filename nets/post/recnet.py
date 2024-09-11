import torch.nn as nn
from nets.resblock import ResidualBlock
from video.utils.misc_helper import decorator_input

class RecNet(nn.Module):
    '''
    A Foreground-background Parallel Compression with Residual Encoding for Surveillance Video
    reconstruction network to enhance quality
    '''
    def __init__(self, in_channels, out_channels, stage_widths=None, kernel_size=3, key_in=None, key_out=None):
        super().__init__()

        if not stage_widths:
            stage_widths = [64, 128, 256, 256, 256, 256, 128, 64]

        padding = (kernel_size - 1) // 2
        self.key_in = key_in
        self.key_out = key_out
        self.conv1 = nn.Conv2d(in_channels, stage_widths[0], kernel_size, 2, padding)
        self.conv2 = nn.Conv2d(stage_widths[0], stage_widths[1], kernel_size, 2, padding)
        self.conv3 = nn.Conv2d(stage_widths[1], stage_widths[2], kernel_size, 2, padding)
        self.resblock1 = ResidualBlock(stage_widths[2], stage_widths[3], kernel_size)
        self.resblock2 = ResidualBlock(stage_widths[3], stage_widths[4], kernel_size)
        self.dconv1 = nn.ConvTranspose2d(stage_widths[4], stage_widths[5], 4, 2, 1)
        self.dconv2 = nn.ConvTranspose2d(stage_widths[5], stage_widths[6], 4, 2, 1)
        self.dconv3 = nn.ConvTranspose2d(stage_widths[6], stage_widths[7], 4, 2, 1)
        self.conv4 = nn.Conv2d(stage_widths[7], out_channels, kernel_size, 1, padding)

    @decorator_input
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.resblock1(out)
        out = self.resblock2(out)
        out = self.dconv1(out)
        out = self.dconv2(out)
        out = self.dconv3(out)
        out = self.conv4(out)

        return out

class RecResNet(RecNet):

    @decorator_input
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.resblock1(out)
        out = self.resblock2(out)
        out = self.dconv1(out)
        out = self.dconv2(out)
        out = self.dconv3(out)
        out = self.conv4(out)

        return out + x