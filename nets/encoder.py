from functools import partial
import torch.nn.functional as F

import numpy as np
import torch
import torch.nn as nn

try:
    from integer2.module import EMAAct, get_quant_conv, app

    assert app == 'DIC'
except:
    print("encoder load integer2 failed", flush=True)
    from integer.module import EMAAct, get_quant_conv
from nets.attention import Cheng20ResBlockAttention
from nets.base import PreparableMixin
from nets.layers import SignalConv2d, ConditionalConv2d
from nets.decoder_quant import GGQReLU
from video.utils.misc_helper import decorator_input
from .components import initial, _conv_layer, _deconv_layer
from .norm import GDN, GDNEfficient, L1GDNNoPow
from .resblock import ResBlock_Video, ResidualBlock
from quant.quantizator import STEQuant
from nets.layers import SFT, SFTResblk


class Encoder_RT(nn.Module):
    def __init__(self, in_channels=3, out_channels=30, M=6):
        super(Encoder_RT, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.M = M
        self.build(c=[16, 32, 64, 128, 256, 512])

    def build(self, c):
        self.d = []
        self.f = []
        self.g = []
        for i in range(0, self.M - 1):
            self.d.append(nn.Sequential(_conv_layer(3, 3, 3, 2, 1)))  # , self._pool_layer(2, 2, 0)))
        for i in range(0, 3):
            f1 = _conv_layer(3, c[i], 3, 1, 1)
            f2 = _conv_layer(c[i], c[i], 3, 2, 1)
            # f3 = self._pool_layer(2, 2, 0)
            self.f.append(nn.Sequential(f1, f2))
        for i in range(3, self.M):
            f1 = _conv_layer(3, c[i], 3, 1, 1)
            f2 = _conv_layer(c[i], c[i], 3, 1, 1)
            self.f.append(nn.Sequential(f1, f2))
        channels = self.out_channels // 6
        self.g.append(_conv_layer(c[0], channels, 5, 4, 1))
        self.g.append(_conv_layer(c[1], channels, 3, 2, 1))
        self.g.append(_conv_layer(c[2], channels, 3, 1, 1))
        self.g.append(_conv_layer(c[3], channels, 3, 1, 1))
        self.g.append(_deconv_layer(c[4], channels, 4, 2, 1))
        self.g.append(_deconv_layer(c[5], channels, 6, 4, 1))
        self.G = _conv_layer(self.out_channels, self.out_channels, 3, 1, 1)
        self.d_list = nn.Sequential(*self.d)
        self.f_list = nn.Sequential(*self.f)
        self.g_list = nn.Sequential(*self.g)

    def forward(self, x):
        pad = nn.ReflectionPad2d(32)
        x = pad(x)
        dx = []
        dx.append(x)
        fx = []
        gx = []
        for i in range(0, self.M - 1):
            dx.append(self.d[i](dx[i]))
        for i in range(0, self.M):
            fx.append(self.f[i](dx[i]))
            gx.append(self.g[i](fx[i]))
        return self.G(torch.cat(gx, 1))


class BaseGGEncoder(nn.Module, PreparableMixin):
    def __init__(self, in_channels=3, out_channels=192, stage_widths=None,
                 normalization='gdn', conv='conv', lambda4s=[1], key_in=None, key_out=None, **kwargs):
        super(BaseGGEncoder, self).__init__()
        self.in_channels = in_channels
        self.num_filters = out_channels
        self.caffe_channels = in_channels
        self.stage_widths = stage_widths
        self.key_in = key_in
        self.key_out = key_out

        self.lambda4s = lambda4s
        self.sampled_lambda = 1
        self.conv_name = conv
        self.kernel_size = kwargs.get('k', 5)  # only add to GG18's first layer

        dct_conv = kwargs.get("dct_conv", "NONE")
        if dct_conv == "4conv":

            dct_layers = [
                nn.Conv2d(3, 192, 8, 8, 0),
                nn.ReLU(),
                nn.ConvTranspose2d(192, 48, 4, 2, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(48, 12, 4, 2, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(12, 3, 4, 2, 1)
            ]
            self.dct_conv = nn.Sequential(*dct_layers)
        else:
            self.dct_conv = None

        if normalization == 'gdn':
            self.norm_layer = GDN
            self.activation_layer = lambda: None  # GDN do not need activation
        elif normalization == 'gdn_cenic':
            self.norm_layer = GDNEfficient
            self.activation_layer = lambda: None  # GDN do not need activation
        elif normalization == '1dn':
            self.norm_layer = partial(GDN, fast=True)
            self.activation_layer = lambda: None  # GDN do not need activation
        elif normalization == '1dn_no_pow':
            self.norm_layer = partial(L1GDNNoPow)
            self.activation_layer = lambda: None  # GDN do not need activation
        elif normalization == 'bn':
            self.norm_layer = nn.BatchNorm2d
            self.activation_layer = partial(nn.ReLU, inplace=True)
        elif normalization == 'in':
            self.norm_layer = nn.InstanceNorm2d
            self.activation_layer = partial(nn.ReLU, inplace=True)
        elif normalization == '1dn_quant':
            self.norm_layer = partial(GDN, fast=True, quant=True)
            self.activation_layer = lambda: None
        elif normalization == 'mixop_gdn':
            from nas.mixop import MixedOp_GDN
            self.norm_layer = MixedOp_GDN
            self.activation_layer = lambda: None
        else:
            raise NotImplemented('unsupported normalization {}'.format(normalization))

        if conv == 'conv':
            self.conv = nn.Conv2d
        elif conv == 'signal':
            self.conv = SignalConv2d
        elif conv == 'mixop_signal':
            from nas.mixop import MixedOp_Conv_Signal
            self.conv = MixedOp_Conv_Signal
        elif conv == 'mixop_origin':
            from nas.mixop import MixedOp_Conv_Origin
            self.conv = MixedOp_Conv_Origin
        else:
            raise NotImplemented('unsupported conv layer {}'.format(conv))

        self._layers = nn.ModuleList([])
        self.build()

    def build(self):
        raise NotImplementedError

    @decorator_input
    def forward(self, x):
        if self.dct_conv is not None:
            x = self.dct_conv(x)
        for layer in self._layers:
            x = layer(x)
        return x


class BaseGGEncoder_QUANT(BaseGGEncoder):
    def build(self):
        if self.conv_name == 'conv':
            self.conv = get_quant_conv(signal=False)
        elif self.conv_name == 'signal':
            self.conv = get_quant_conv(signal=True)
        else:
            raise NotImplemented('unsupported quant conv layer {}'.format(self.conv_name))

        self.build_quant()

    def build_quant(self):
        raise NotImplementedError


class BaseGGEncoder_VAR(BaseGGEncoder):
    def forward(self, x_):
        if isinstance(x_, dict):  # to support to_caffe
            x = x_['image']
            self.sampled_lambda = x_['sampled_lambda']
        else:
            x = x_

        lambdas = self.lambda4s[1:]

        bools = np.array(lambdas) == self.sampled_lambda
        one_hot = [int(b) for b in bools]
        one_hot = torch.Tensor(one_hot).to(x.device)
        one_hot = one_hot.unsqueeze(0)
        # one_hot = nn.Parameter(one_hot, requires_grad=False)

        for layer in self._layers:
            if isinstance(layer, ConditionalConv2d):
                x = layer(x, one_hot)
            else:
                x = layer(x)
        return x


class YEncoder_GG17(BaseGGEncoder):
    def build(self):
        conv1 = self.conv(
            self.in_channels, self.num_filters, (9, 9), stride=4, padding=4,
            bias=True, padding_mode="zeros")
        conv2 = self.conv(
            self.num_filters, self.num_filters, (5, 5), stride=2, padding=2,
            bias=True, padding_mode="zeros")
        conv3 = self.conv(
            self.num_filters, self.num_filters, (5, 5), stride=2, padding=2,
            bias=False, padding_mode="zeros")

        self._layers = nn.ModuleList([l for l in [
            conv1,
            self.norm_layer(self.num_filters),
            self.activation_layer(),
            conv2,
            self.norm_layer(self.num_filters),
            self.activation_layer(),
            conv3,
        ] if l])


class YEncoder_GG17_VAR(BaseGGEncoder_VAR):
    def build(self):
        lambdas_len = len(self.lambda4s) - 1

        conv1 = ConditionalConv2d(
            self.in_channels, self.num_filters, (9, 9), stride=4, padding=4,
            bias=True, padding_mode="zeros", conv=self.conv, lambdas_len=lambdas_len)
        conv2 = ConditionalConv2d(
            self.num_filters, self.num_filters, (5, 5), stride=2, padding=2,
            bias=True, padding_mode="zeros", conv=self.conv, lambdas_len=lambdas_len)
        conv3 = ConditionalConv2d(
            self.num_filters, self.num_filters, (5, 5), stride=2, padding=2,
            bias=False, padding_mode="zeros", conv=self.conv, lambdas_len=lambdas_len)

        self._layers = nn.ModuleList([l for l in [
            conv1,
            self.norm_layer(self.num_filters),
            self.activation_layer(),
            conv2,
            self.norm_layer(self.num_filters),
            self.activation_layer(),
            conv3,
        ] if l])


class YEncoder_GG18(BaseGGEncoder):
    def build(self):
        if not self.stage_widths:
            self.stage_widths = [self.num_filters] * 3
        k = self.kernel_size
        self._layers = nn.ModuleList([l for l in [
            self.conv(
                self.in_channels, self.stage_widths[0], (k, k), stride=2, padding=(k - 1) // 2,
                bias=True, padding_mode="zeros"),
            self.norm_layer(self.stage_widths[0]),
            self.activation_layer(),
            self.conv(
                self.stage_widths[0], self.stage_widths[1], (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros"),
            self.norm_layer(self.stage_widths[1]),
            self.activation_layer(),
            self.conv(
                self.stage_widths[1], self.stage_widths[2], (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros"),
            self.norm_layer(self.stage_widths[2]),
            self.activation_layer(),
            self.conv(
                self.stage_widths[2], self.num_filters, (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros")
        ] if l])


class YEncoder_QM(BaseGGEncoder):
    """
    Variable-rate deep image compression through spatially-adaptive feature transform
    """

    def build(self, prior_nc=64, sft_ks=3):
        if not self.stage_widths:
            self.stage_widths = [self.num_filters // 4, self.num_filters // 2, self.num_filters, self.num_filters,
                                 self.num_filters]
        sft1 = SFT(self.num_filters // 4, prior_nc, ks=sft_ks)
        sft2 = SFT(self.num_filters // 2, prior_nc, ks=sft_ks)
        sft3 = SFT(self.num_filters, prior_nc, ks=sft_ks)
        sft4 = SFT(self.num_filters, prior_nc, ks=sft_ks)

        k = self.kernel_size

        self._layer1 = nn.ModuleList([
            self.conv(
                self.in_channels, self.stage_widths[0], (k, k), stride=2, padding=(k - 1) // 2,
                bias=True, padding_mode="zeros"),
            self.norm_layer(self.stage_widths[0]),
            sft1,

            self.conv(
                self.stage_widths[0], self.stage_widths[1], (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros"),
            self.norm_layer(self.stage_widths[1]),
            sft2,

            self.conv(
                self.stage_widths[1], self.stage_widths[2], (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros"),
            self.norm_layer(self.stage_widths[2]),
            sft3,

            self.conv(
                self.stage_widths[2], self.stage_widths[3], (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros"),
            self.norm_layer(self.stage_widths[3]),
            sft4,

            self.conv(
                self.stage_widths[3], self.stage_widths[4], (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros"),
            SFTResblk(self.num_filters, prior_nc, ks=sft_ks),
            SFTResblk(self.num_filters, prior_nc, ks=sft_ks),

        ])

        self._layer2 = nn.ModuleList([
            self.conv(
                4, prior_nc * 4, (k, k), stride=1, padding=(k - 1) // 2,
                bias=True, padding_mode="zeros"),
            nn.LeakyReLU(0.1, True),
            self.conv(
                prior_nc * 4, prior_nc * 2, (k, k), stride=1, padding=(k - 1) // 2,
                bias=True, padding_mode="zeros"),
            nn.LeakyReLU(0.1, True),
            self.conv(
                prior_nc * 2, prior_nc, (k, k), stride=1, padding=(k - 1) // 2,
                bias=True, padding_mode="zeros"),

            self.conv(
                prior_nc, prior_nc, (k, k), stride=1, padding=(k - 1) // 2,
                bias=True, padding_mode="zeros"),
            nn.LeakyReLU(0.1, True),
            self.conv(
                prior_nc, prior_nc, (k, k), stride=1, padding=(k - 1) // 2,
                bias=True, padding_mode="zeros"),

            self.conv(
                prior_nc, prior_nc, (k, k), stride=1, padding=(k - 1) // 2,
                bias=True, padding_mode="zeros"),
            nn.LeakyReLU(0.1, True),
            self.conv(
                prior_nc, prior_nc, (k, k), stride=1, padding=(k - 1) // 2,
                bias=True, padding_mode="zeros"),

            self.conv(
                prior_nc, prior_nc, (k, k), stride=1, padding=(k - 1) // 2,
                bias=True, padding_mode="zeros"),
            nn.LeakyReLU(0.1, True),
            self.conv(
                prior_nc, prior_nc, (k, k), stride=1, padding=(k - 1) // 2,
                bias=True, padding_mode="zeros"),

            self.conv(
                prior_nc, prior_nc, (k, k), stride=1, padding=(k - 1) // 2,
                bias=True, padding_mode="zeros"),
            nn.LeakyReLU(0.1, True),
            self.conv(
                prior_nc, prior_nc, (k, k), stride=1, padding=(k - 1) // 2,
                bias=True, padding_mode="zeros"),

        ])

    def forward(self, x, qmap):
        m = torch.cat((qmap, x), 1)
        for i in range(5):
            m = self._layer2[i](m)
        x = self._layer1[0](x)
        x = self._layer1[1](x)
        x = self._layer1[2](x, m)

        x = self._layer1[3](x)
        x = self._layer1[4](x)
        for i in range(5, 8):
            m = self._layer2[i](m)
        x = self._layer1[5](x, m)

        x = self._layer1[6](x)
        x = self._layer1[7](x)
        for i in range(8, 11):
            m = self._layer2[i](m)
        x = self._layer1[8](x, m)

        x = self._layer1[9](x)
        x = self._layer1[10](x)
        for i in range(11, 14):
            m = self._layer2[i](m)
        x = self._layer1[11](x, m)

        x = self._layer1[12](x)
        for i in range(14, 17):
            m = self._layer2[i](m)
        x = self._layer1[13](x, m)
        y = self._layer1[14](x, m)

        return y


class YEncoderCheng20(BaseGGEncoder):
    """
    analysis transform used by:

    Learned Image Compression with Discretized Gaussian Mixture Likelihoods and Attention Modules
    (arXiv:2001.01568v3)

    also see: https://github.com/ZhengxueCheng/Learned-Image-Compression-with-GMM-and-Attention

    this is a simplified Non-Local block
    """

    class DownResBlock(nn.Module):
        def __init__(self, channel_in, num_filter, conv_cls, norm_cls, down_sample):
            super().__init__()
            if down_sample:
                layers = [
                    conv_cls(channel_in, num_filter, 3, stride=2, padding=1),
                    nn.LeakyReLU(inplace=True),
                    conv_cls(num_filter, num_filter, 3, padding=1),
                    norm_cls(num_filter),  # GDN here
                ]
                self.shortcut = conv_cls(channel_in, num_filter, 3, stride=2, padding=1)
            else:
                layers = [
                    conv_cls(channel_in, num_filter, 3, padding=1),
                    nn.LeakyReLU(inplace=True),
                    conv_cls(num_filter, num_filter, 3, padding=1),
                    nn.LeakyReLU(inplace=True),
                ]
                self.shortcut = nn.Identity()
            self.rb = nn.Sequential(*layers)

        def forward(self, x):
            shortcut = self.shortcut(x)
            trans = self.rb(x)
            return trans + shortcut

    def build(self):
        if not self.stage_widths:
            self.stage_widths = [self.num_filters] * 6
        self._layers = nn.ModuleList([
            self.DownResBlock(self.in_channels, self.stage_widths[0], self.conv, self.norm_layer, True),
            self.DownResBlock(self.stage_widths[0], self.stage_widths[1], self.conv, self.norm_layer, False),
            self.DownResBlock(self.stage_widths[1], self.stage_widths[2], self.conv, self.norm_layer, True),
            Cheng20ResBlockAttention(self.stage_widths[2], self.conv),

            self.DownResBlock(self.stage_widths[2], self.stage_widths[3], self.conv, self.norm_layer, False),
            self.DownResBlock(self.stage_widths[3], self.stage_widths[4], self.conv, self.norm_layer, True),
            self.DownResBlock(self.stage_widths[4], self.stage_widths[5], self.conv, self.norm_layer, False),
            self.conv(self.stage_widths[5], self.num_filters, 3, stride=2, padding=1),
            Cheng20ResBlockAttention(self.num_filters, self.conv),
        ])


class YEncoder_GG18_QUANT(BaseGGEncoder_QUANT):
    def build_quant(self):
        if not self.stage_widths:
            self.stage_widths = [self.num_filters] * 3
        self._layers = nn.ModuleList([l for l in [
            EMAAct(channel_num=self.in_channels),
            self.conv(
                self.in_channels, self.stage_widths[0], (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros"),
            EMAAct(channel_num=self.stage_widths[0]),
            self.norm_layer(self.stage_widths[0]),
            self.activation_layer(),
            EMAAct(channel_num=self.stage_widths[0]),
            self.conv(
                self.stage_widths[0], self.stage_widths[1], (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros"),
            EMAAct(channel_num=self.stage_widths[1]),
            self.norm_layer(self.stage_widths[1]),
            self.activation_layer(),
            EMAAct(channel_num=self.stage_widths[1]),
            self.conv(
                self.stage_widths[1], self.stage_widths[2], (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros"),
            EMAAct(channel_num=self.stage_widths[2]),
            self.norm_layer(self.stage_widths[2]),
            self.activation_layer(),
            EMAAct(channel_num=self.stage_widths[2]),
            self.conv(
                self.stage_widths[2], self.num_filters, (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros"),
            EMAAct(channel_num=self.num_filters)
        ] if l])


class ZEncoder_GG18(BaseGGEncoder):
    def build(self):
        if not self.stage_widths:
            self.stage_widths = [self.num_filters] * 2
        self._layers = nn.ModuleList([
            self.conv(
                self.in_channels, self.stage_widths[0], (3, 3), stride=1, padding=1,
                bias=True, padding_mode="zeros"),
            nn.ReLU(inplace=True),
            self.conv(
                self.stage_widths[0], self.stage_widths[1], (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros"),
            nn.ReLU(inplace=True),
            self.conv(
                self.stage_widths[1], self.num_filters, (5, 5), stride=2, padding=2,
                bias=False, padding_mode="zeros")
        ])


class ZEncoder_QM(BaseGGEncoder):
    def build(self, prior_nc=64, sft_ks=3):
        if not self.stage_widths:
            self.stage_widths = [self.num_filters] * 3
        k = self.kernel_size

        self._layer1 = nn.ModuleList([
            self.conv(
                self.in_channels, self.stage_widths[0], (3, 3), stride=1, padding=1,
                bias=True, padding_mode="zeros"),
            SFT(self.num_filters, prior_nc, ks=sft_ks),
            nn.LeakyReLU(inplace=True),

            self.conv(
                self.stage_widths[0], self.stage_widths[1], (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros"),
            SFT(self.num_filters, prior_nc, ks=sft_ks),
            nn.LeakyReLU(inplace=True),

            self.conv(
                self.stage_widths[1], self.num_filters, (5, 5), stride=2, padding=2,
                bias=False, padding_mode="zeros"),
            SFTResblk(self.num_filters, prior_nc, ks=sft_ks),
            SFTResblk(self.num_filters, prior_nc, ks=sft_ks),
        ])

        self._layer2 = nn.ModuleList([
            self.conv(
                self.stage_widths[0] + 1, prior_nc * 4, (k, k), stride=1, padding=(k - 1) // 2,
                bias=True, padding_mode="zeros"),
            nn.LeakyReLU(0.1, True),
            self.conv(
                prior_nc * 4, prior_nc * 2, (k, k), stride=1, padding=(k - 1) // 2,
                bias=True, padding_mode="zeros"),
            nn.LeakyReLU(0.1, True),
            self.conv(
                prior_nc * 2, prior_nc, (k, k), stride=1, padding=(k - 1) // 2,
                bias=True, padding_mode="zeros"),

            self.conv(
                prior_nc, prior_nc, (k, k), stride=1, padding=(k - 1) // 2,
                bias=True, padding_mode="zeros"),
            nn.LeakyReLU(0.1, True),
            self.conv(
                prior_nc, prior_nc, (k, k), stride=1, padding=(k - 1) // 2,
                bias=True, padding_mode="zeros"),

            self.conv(
                prior_nc, prior_nc, (k, k), stride=1, padding=(k - 1) // 2,
                bias=True, padding_mode="zeros"),
            nn.LeakyReLU(0.1, True),
            self.conv(
                prior_nc, prior_nc, (k, k), stride=1, padding=(k - 1) // 2,
                bias=True, padding_mode="zeros"),
        ])

    def forward(self, y, qmap):
        m = F.adaptive_avg_pool2d(qmap, y.size()[2:])
        m = torch.cat((m, y), 1)
        for i in range(5):
            m = self._layer2[i](m)
        y = self._layer1[0](y)
        y = self._layer1[1](y, m)
        y = self._layer1[2](y)
        y = self._layer1[3](y)
        for i in range(5, 8):
            m = self._layer2[i](m)
        y = self._layer1[4](y, m)
        y = self._layer1[5](y)
        y = self._layer1[6](y)
        for i in range(8, 11):
            m = self._layer2[i](m)
        y = self._layer1[7](y, m)
        z = self._layer1[8](y, m)

        return z


class ZEncoder_GG18Rep(nn.Module):

    def __init__(self, in_channels=3, out_channels=192, conv="conv", stage_widths=None, lambda4s=[1]):
        super(ZEncoder_GG18Rep, self).__init__()
        self.in_channels = in_channels
        self.num_filters = out_channels
        self.caffe_channels = in_channels

        # stage_widths for num_filters of each layer
        self.stage_widths = stage_widths

        self.lambda4s = lambda4s
        self.sampled_lambda = 1

        if conv == 'conv':
            self.conv = nn.Conv2d
        elif conv == 'signal':
            self.conv = SignalConv2d
        else:
            raise NotImplementedError

        self._layers = nn.ModuleList([])
        self.build()

    def build(self):
        if not self.stage_widths:
            self.stage_widths = [self.num_filters] * 2
        self.conv1 = self.conv(
            self.in_channels, self.stage_widths[0], (3, 3), stride=1, padding=1,
            bias=True, padding_mode="zeros")
        self.conv1_1x1 = self.conv(
            self.in_channels, self.stage_widths[0], (1, 1), stride=1, padding=0,
            bias=True, padding_mode="zeros")
        self.conv2 = self.conv(
            self.stage_widths[0], self.stage_widths[1], (5, 5), stride=2, padding=2,
            bias=True, padding_mode="zeros")
        self.conv2_1x1 = self.conv(
            self.stage_widths[0], self.stage_widths[1], (1, 1), stride=2, padding=0,
            bias=True, padding_mode="zeros")
        self.conv3 = self.conv(
            self.stage_widths[1], self.num_filters, (5, 5), stride=2, padding=2,
            bias=True, padding_mode="zeros")
        self.conv3_1x1 = self.conv(
            self.stage_widths[1], self.num_filters, (1, 1), stride=2, padding=0,
            bias=True, padding_mode="zeros")

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x) + self.conv1_1x1(x)
        x = self.relu(x)

        x = self.conv2(x) + self.conv2_1x1(x)
        x = self.relu(x)

        x = self.conv3(x) + self.conv3_1x1(x)
        return x


class ZEncoder_GG18Clip(ZEncoder_GG18):
    '''
    Wrap forward to support post clip
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip = GGQReLU()

    def forward(self, x):
        x = super().forward(x)
        return self.clip(x, 0, 255)


class ZEncoder_GG18_QUANT(BaseGGEncoder_QUANT):
    def build_quant(self):
        if not self.stage_widths:
            self.stage_widths = [self.num_filters] * 2
        self._layers = nn.ModuleList([
            EMAAct(channel_num=self.in_channels),
            self.conv(
                self.in_channels, self.stage_widths[0], (3, 3), stride=1, padding=1,
                bias=True, padding_mode="zeros"),
            nn.ReLU(inplace=True),
            EMAAct(channel_num=self.stage_widths[0]),
            self.conv(
                self.stage_widths[0], self.stage_widths[1], (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros"),
            nn.ReLU(inplace=True),
            EMAAct(channel_num=self.stage_widths[1]),
            self.conv(
                self.stage_widths[1], self.num_filters, (5, 5), stride=2, padding=2,
                bias=False, padding_mode="zeros"),
            EMAAct(channel_num=self.num_filters)
        ])


class ZEncoder_GG18C(BaseGGEncoder):
    def build(self):
        if not self.stage_widths:
            self.stage_widths = [self.num_filters] * 2
        self._layers = nn.ModuleList([
            self.conv(
                self.in_channels, self.stage_widths[0], (3, 3), stride=1, padding=1,
                bias=True, padding_mode="zeros"),
            nn.LeakyReLU(inplace=True),
            self.conv(
                self.stage_widths[0], self.stage_widths[1], (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros"),
            nn.LeakyReLU(inplace=True),
            self.conv(
                self.stage_widths[1], self.num_filters, (5, 5), stride=2, padding=2,
                bias=False, padding_mode="zeros")
        ])


class ZEncoderLinear(BaseGGEncoder):
    def build(self):
        m = self.in_channels
        n = self.num_filters
        d = (m - n) // 3
        l1 = m - d
        l2 = n + d
        self._layers = nn.ModuleList([
            self.conv(
                m, l1, (3, 3), stride=1, padding=1,
                bias=True, padding_mode="zeros"),
            nn.LeakyReLU(inplace=True),
            self.conv(
                l1, l2, (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros"),
            nn.LeakyReLU(inplace=True),
            self.conv(
                l2, n, (5, 5), stride=2, padding=2,
                bias=False, padding_mode="zeros")
        ])


class ZEncoderCheng20(BaseGGEncoder):
    def build(self):
        self._layers = nn.ModuleList([
            self.conv(self.in_channels, self.num_filters, (3, 3), padding=1),
            nn.LeakyReLU(inplace=True),
            self.conv(self.in_channels, self.num_filters, (3, 3), padding=1),
            nn.LeakyReLU(inplace=True),
            self.conv(self.in_channels, self.num_filters, (3, 3), padding=1, stride=2),
            nn.LeakyReLU(inplace=True),
            self.conv(self.in_channels, self.num_filters, (3, 3), padding=1),
            nn.LeakyReLU(inplace=True),
            self.conv(self.in_channels, self.num_filters, (3, 3), padding=1, stride=2, bias=False),
        ])


class YEncoder_GG18_VAR(BaseGGEncoder_VAR):
    def build(self):
        lambdas_len = len(self.lambda4s) - 1
        if not self.stage_widths:
            self.stage_widths = [self.num_filters] * 3
        self._layers = nn.ModuleList([l for l in [
            ConditionalConv2d(
                self.in_channels, self.stage_widths[0], (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros", conv=self.conv, lambdas_len=lambdas_len),
            self.norm_layer(self.stage_widths[0]),
            self.activation_layer(),
            ConditionalConv2d(
                self.stage_widths[0], self.stage_widths[1], (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros", conv=self.conv, lambdas_len=lambdas_len),
            self.norm_layer(self.stage_widths[1]),
            self.activation_layer(),
            ConditionalConv2d(
                self.stage_widths[1], self.stage_widths[2], (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros", conv=self.conv, lambdas_len=lambdas_len),
            self.norm_layer(self.stage_widths[2]),
            self.activation_layer(),
            ConditionalConv2d(
                self.stage_widths[2], self.num_filters, (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros", conv=self.conv, lambdas_len=lambdas_len)
        ] if l])


class ZEncoder_GG18_VAR(BaseGGEncoder_VAR):
    def build(self):
        lambdas_len = len(self.lambda4s) - 1
        if not self.stage_widths:
            self.stage_widths = [self.num_filters] * 2
        self._layers = nn.ModuleList([
            ConditionalConv2d(
                self.in_channels, self.stage_widths[0], (3, 3), stride=1, padding=1,
                bias=True, padding_mode="zeros", conv=self.conv, lambdas_len=lambdas_len),
            nn.ReLU(inplace=True),
            ConditionalConv2d(
                self.stage_widths[0], self.stage_widths[1], (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros", conv=self.conv, lambdas_len=lambdas_len),
            nn.ReLU(inplace=True),
            ConditionalConv2d(
                self.stage_widths[1], self.num_filters, (5, 5), stride=2, padding=2,
                bias=False, padding_mode="zeros", conv=self.conv, lambdas_len=lambdas_len)
        ])


class ZEncoder_GG18C_VAR(BaseGGEncoder_VAR):
    def build(self):
        lambdas_len = len(self.lambda4s) - 1
        if not self.stage_widths:
            self.stage_widths = [self.num_filters] * 2
        self._layers = nn.ModuleList([
            ConditionalConv2d(
                self.in_channels, self.stage_widths[0], (3, 3), stride=1, padding=1,
                bias=True, padding_mode="zeros", conv=self.conv, lambdas_len=lambdas_len),
            nn.LeakyReLU(inplace=True),
            ConditionalConv2d(
                self.stage_widths[0], self.stage_widths[1], (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros", conv=self.conv, lambdas_len=lambdas_len),
            nn.LeakyReLU(inplace=True),
            ConditionalConv2d(
                self.stage_widths[1], self.num_filters, (5, 5), stride=2, padding=2,
                bias=False, padding_mode="zeros", conv=self.conv, lambdas_len=lambdas_len)
        ])


class YEncoder_CEVideo(nn.Module):
    def __init__(self, in_channels=3, n=192, **kwargs):
        super(YEncoder_CEVideo, self).__init__()
        self.in_channels = in_channels
        self.n = n

        self.conv = SignalConv2d
        self.block = ResBlock_Video

        self._layers = nn.ModuleList([])
        self.build()

    def build(self):
        self._layers = nn.ModuleList([
            self.conv(self.in_channels, self.n, 5, 2, 2),
            self.block(self.n),
            self.block(self.n),
            self.block(self.n),
            self.conv(self.n, self.n, 5, 2, 2),
            self.conv(self.n, self.n, 5, 2, 2),
            self.block(self.n),
            self.block(self.n),
            self.block(self.n),
            self.conv(self.n, self.n, 5, 2, 2),
        ])

    def forward(self, x):
        for layer in self._layers:
            x = layer(x)
        return x


class ZEncoder_CEVideo(nn.Module):
    def __init__(self, n=192, **kwargs):
        super(ZEncoder_CEVideo, self).__init__()
        self.n = n

        self.conv = SignalConv2d
        self.block = ResBlock_Video

        self._layers = nn.ModuleList([])
        self.build()

    def build(self):
        self._layers = nn.ModuleList([
            self.conv(2 * self.n, self.n, 1, 1, 0),
            self.block(self.n),
            self.block(self.n),
            self.block(self.n),
            self.conv(self.n, self.n, 5, 2, 2),
            self.block(self.n),
            self.block(self.n),
            self.block(self.n),
            self.conv(self.n, self.n, 5, 2, 2),
        ])

    def forward(self, x):
        for layer in self._layers:
            x = layer(x)
        return x


class DCTYEncoderGG18(BaseGGEncoder):
    """DCT lossy compression.
    """

    def build(self):
        if not self.stage_widths:
            self.stage_widths = [self.num_filters] * 2
        k = self.kernel_size
        self._layers = nn.ModuleList([l for l in [
            self.conv(
                self.in_channels, self.stage_widths[0], (k, k), stride=2, padding=(k - 1) // 2,
                bias=True, padding_mode="zeros"),
            self.norm_layer(self.stage_widths[0]),
            self.activation_layer(),
            self.conv(
                self.stage_widths[0], self.stage_widths[1], (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros"),
            self.norm_layer(self.stage_widths[1]),
            self.activation_layer(),
            self.conv(
                self.stage_widths[1], self.num_filters, (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros")
        ] if l])


class Residual(nn.Module):
    def __init__(self, denorm):
        super(Residual, self).__init__()
        self.denrom = denorm
        self.round = STEQuant()

    def forward(self, x, x_hat):
        if self.denrom:
            x = self.round(x * 255.)
            x_hat = self.round(x_hat * 255)
        return x - x_hat


def encoders(tag: str = "RT", in_channels=3, out_channels=30, **kwargs):
    """
    get y-encoder
    :param tag: the type of encoder. ['RT', 'YGG17', 'YGG18', 'ZGG18', 'NONE']
    :param in_channels:
    :param M:
    :param out_channels:
    :return: if tag is 'NONE', return None. otherwise return a model
    """
    if tag == "RT":
        model = Encoder_RT(in_channels, out_channels, **kwargs)
        initial(model)

    elif tag == "YGG17":
        model = YEncoder_GG17(in_channels, out_channels, **kwargs)
    elif tag == "YGG18":
        model = YEncoder_GG18(in_channels, out_channels, **kwargs)
    elif tag == "ZGG18":
        model = ZEncoder_GG18(in_channels, out_channels, **kwargs)
    elif tag == "ZGG18C":
        model = ZEncoder_GG18C(in_channels, out_channels, **kwargs)
    elif tag == 'ZLinear':
        model = ZEncoderLinear(in_channels, out_channels, **kwargs)
    elif tag == "YGG17_VAR":
        model = YEncoder_GG17_VAR(in_channels, out_channels, **kwargs)
    elif tag == "YGG18_VAR":
        model = YEncoder_GG18_VAR(in_channels, out_channels, **kwargs)
    elif tag == "ZGG18_VAR":
        model = ZEncoder_GG18_VAR(in_channels, out_channels, **kwargs)
    elif tag == "ZGG18C_VAR":
        model = ZEncoder_GG18C_VAR(in_channels, out_channels, **kwargs)
    elif tag == "ZGG18Clip":
        model = ZEncoder_GG18Clip(in_channels, out_channels, **kwargs)

    elif tag == "YGG18_QUANT":
        model = YEncoder_GG18_QUANT(in_channels, out_channels, **kwargs)
    elif tag == "ZGG18_QUANT":
        model = ZEncoder_GG18_QUANT(in_channels, out_channels, **kwargs)

    elif tag == "Y_QM":
        model = YEncoder_QM(in_channels, out_channels, **kwargs)
    elif tag == "Z_QM":
        model = ZEncoder_QM(in_channels, out_channels, **kwargs)

    # DCT related encoders
    elif tag == "DCTYGG18":
        model = DCTYEncoderGG18(in_channels, out_channels, **kwargs)

    elif tag == 'NONE':
        return None
    else:
        raise NotImplementedError('encoder with tag ' + tag)

    return model


if __name__ == '__main__':
    x = torch.randn(2, 3, 1280, 1280 * 2)
    test = Encoder_RT(3)
    with torch.no_grad():
        print(test(x).shape)

    test = encoders()
    with torch.no_grad():
        print(test(x).shape)

    test = encoders('YGG17')
    test.eval()
    print(test._layers[0].training)
