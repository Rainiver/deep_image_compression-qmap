import torch
import torch.nn as nn


class UNET(nn.Module):
    def __init__(self, in_channels=3):
        super(UNET, self).__init__()
        self.in_channels = in_channels
        self.build()

    def _conv_layer(self, in_channels, out_channels, kernel, stride, padding, bias=True, bn=False):
        layers = []
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride,
                                padding=padding, bias=bias, padding_mode='reflection'))
        if bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.0))
        return nn.Sequential(*layers)

    def _deconv_layer(self, in_channels, out_channels, kernel, stride, padding, bias=True, bn=False):
        layers = []
        layers.append(nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride,
            padding=padding, bias=bias))
        if bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.0))
        return nn.Sequential(*layers)

    def _pool_layer(self, kernel, stride, padding=0, mode="Max"):
        if mode != "Max" and mode != "Avg":
            assert 0
        if mode == "Max":
            return nn.Sequential(nn.MaxPool2d(kernel_size=kernel, stride=stride))
        else:
            return nn.Sequential(nn.AvgPool2d(kernel_size=kernel, stride=stride, padding=padding))

    def build_layers(self, in_channels, out_channels):
        layers = []
        layers.append(self._conv_layer(in_channels, out_channels, 3, 1, 1))
        layers.append(self._conv_layer(out_channels, out_channels, 3, 1, 1))
        return nn.Sequential(*layers)

    def build(self):
        self.f0 = []
        self.f1 = []
        self.g0 = []
        self.g1 = []
        down_channels = [self.in_channels, 16, 32, 64, 128, 256]
        up_channels = [256, 128, 64, 32, 16]
        for i in range(5):
            self.f0.append(self.build_layers(down_channels[i], down_channels[i + 1]))
            self.f1.append(self._pool_layer(2, 2))
        for i in range(4):
            self.g0.append(self._deconv_layer(up_channels[i], up_channels[i + 1], 4, 2, 1))
            self.g1.append(self.build_layers(up_channels[i], up_channels[i + 1]))

        self.G = self._conv_layer(up_channels[4], self.in_channels, 3, 1, 1)
        self.f0_list = nn.Sequential(*self.f0)
        self.f1_list = nn.Sequential(*self.f1)
        self.g0_list = nn.Sequential(*self.g0)
        self.g1_list = nn.Sequential(*self.g1)

    def forward(self, x):
        x_ori = x.clone()
        x_down = []
        x_down.append(x)
        for i in range(5):
            x_down.append(self.f0[i](x))
            x = self.f1[i](x_down[i + 1])
        x = x_down[5]
        for i in range(4):
            x = self.g0[i](x)
            x = torch.cat([x, x_down[4 - i]], 1)
            x = self.g1[i](x)
        return self.G(x) + x_ori


if __name__ == "__main__":
    test = UNET(3)
    x = torch.randn([2, 3, 256, 256])
    x = test(x)
    print(x.shape)
    torch.save(test.state_dict(), "test.pth")
