from collections import OrderedDict

import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, num_filters=32):
        super(ResidualBlock, self).__init__()
        self.layers = OrderedDict([
            ('conv1', nn.Conv2d(num_filters, num_filters, 3, padding=1)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(num_filters, num_filters, 3, padding=1)),
        ])
        self.layers = nn.Sequential(self.layers)

    def forward(self, x):
        return x + self.layers(x)


class EnhanceBlock(nn.Module):
    def __init__(self, num_residual_block=3, num_filters=32):
        super(EnhanceBlock, self).__init__()
        self.layers = OrderedDict()
        for i in range(num_residual_block):
            self.layers['residual_block_' + str(i)] = ResidualBlock(num_filters)
        self.layers = nn.Sequential(self.layers)

    def forward(self, x):
        return x + self.layers(x)


class EDIC(nn.Module):
    """
        > A Unified End-to-End Framework for Efficient Deep Image Compression
        see: https://arxiv.org/abs/1809.02736
    """

    def __init__(self, input_channels=3, num_enhance_block=3, num_filters=32):
        super(EDIC, self).__init__()
        self.layers = OrderedDict()
        self.layers['pre'] = nn.Conv2d(input_channels, num_filters, 3, padding=1)
        for i in range(num_enhance_block):
            self.layers['enhance_block_' + str(i)] = EnhanceBlock(num_filters=num_filters)
        self.layers['post'] = nn.Conv2d(num_filters, input_channels, 3, padding=1)
        self.layers = nn.Sequential(self.layers)

        self.to_caffe = False
        self.caffe_channels = input_channels

    def forward(self, x):
        return x + self.layers(x)


def enhance(tag: str, **kwargs):
    if tag == 'edic':
        return EDIC()
    elif tag == 'NONE':
        return nn.Identity()
    # elif tag is None:
    #     return nn.Identity()
    else:
        raise ValueError('unsupported enhance tag:{}'.format(tag))


if __name__ == '__main__':
    import torch
    import time

    x = torch.rand(5, 3, 256, 256)
    test = enhance('edic')
    print(test)
    t = time.time()
    y = test(x)
    print(time.time() - t)
    print(y.size())
