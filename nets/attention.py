import torch
import torch.nn as nn
import torch.nn.functional as F


# https://github.com/moskomule/senet.pytorch/tree/master/senet
class SELayer(nn.Module):
    """
        > A Unified End-to-End Framework for Efficient Deep Image Compression
        see: https://arxiv.org/abs/1809.02736
    """

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channel, channel // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channel //reduction, channel, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        self.to_caffe = False
        self.caffe_channels = channel

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Cheng20ResBlockAttention(nn.Module):
    """
    attention module used by:

    Learned Image Compression with Discretized Gaussian Mixture Likelihoods and Attention Modules
    (arXiv:2001.01568v3)

    also see: https://github.com/ZhengxueCheng/Learned-Image-Compression-with-GMM-and-Attention

    this is a simplified Non-Local block
    """
    def __init__(self, num_filters, conv_cls):
        super().__init__()
        self.num_filters = num_filters
        self.conv_cls = conv_cls
        self.trans_rbs = nn.ModuleList([
            self._build_res_block(),
            self._build_res_block(),
            self._build_res_block(),
        ])
        self.att_rbs = nn.ModuleList([
            self._build_res_block(),
            self._build_res_block(),
            self._build_res_block(),
        ])
        self.att_trans_conv = conv_cls(num_filters, num_filters, 1)

    def _build_res_block(self):
        conv_cls = self.conv_cls
        c = self.num_filters
        return nn.Sequential(
            conv_cls(c, c // 2, 1),
            nn.ReLU(inplace=True),
            conv_cls(c // 2, c // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            conv_cls(c // 2, c, 1),
        )

    def forward(self, x):
        att = x
        for rb in self.att_rbs:
            att = att + rb(att)
        att = self.att_trans_conv(att)
        att = torch.sigmoid(att)

        xt = x
        for rb in self.trans_rbs:
            xt = xt + rb(x)

        # xt *= att
        # xt += x
        xt = torch.addcmul(x, xt, att)

        return xt


def attention(tag: str, in_channels=256, out_channels=256, **kwargs):
    if tag == 'senet':
        return SELayer(in_channels)
    elif tag == 'NONE':
        return nn.Identity()
    else:
        raise ValueError('unsupported attention tag:{}'.format(tag))


if __name__ == '__main__':
    import torch
    import time

    # 256 / 2 ** 4 = 16
    x = torch.rand(5, 256, 16, 16)
    test = attention('senet')
    print(test)
    t = time.time()
    y = test(x)
    print(time.time() - t)
    print(y.size())
