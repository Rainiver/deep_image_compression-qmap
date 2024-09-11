from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    from integer2.module import EMAAct, NoBnConvTranspose2d, app
    assert app == 'DIC'
except:
    print("decoder load integer2 failed", flush=True)
    from integer.module import EMAAct, NoBnConvTranspose2d
from nets.attention import Cheng20ResBlockAttention
from nets.base import PreparableMixin
from video.utils.misc_helper import decorator_input
from .components import initial, _conv_layer, _deconv_layer
from .decoder_quant import QZDecoder_GG18, NoBpQZDecoder_GG18, QZDecoder_GG18Clip, QZDecoderUpsample
from .layers import SignalConvTranspose2d, ConditionalConv2d, Conv_Upsample, SignalConv2d
from .norm import GDN, L1GDNNoPow
from .resblock import ResBlock_Video
from nets.layers import SFT, SFTResblk

class Decoder_RT(nn.Module):
    def __init__(self, out_channels=30, M=6):
        super(Decoder_RT, self).__init__()
        self.in_channels = out_channels
        self.out_channels = out_channels
        self.M = M
        self.build(c=[16, 32, 64, 128, 256, 512])

    def build(self, c):
        self.G = _conv_layer(self.out_channels, self.out_channels, 3, 1, 1)
        self.g = []
        self.f = []
        self.d = []
        channels = self.out_channels // 6
        self.g.append(_deconv_layer(channels, c[0], 6, 4, 1))
        self.g.append(_deconv_layer(channels, c[1], 4, 2, 1))
        self.g.append(_conv_layer(channels, c[2], 3, 1, 1))
        self.g.append(_conv_layer(channels, c[3], 3, 1, 1))
        self.g.append(_conv_layer(channels, c[4], 3, 2, 1))
        self.g.append(_conv_layer(channels, c[5], 5, 4, 1))
        for i in range(0, 3):
            f1 = _deconv_layer(c[i], 3, 4, 2, 1)
            f2 = _conv_layer(c[i], c[i], 3, 1, 1)
            self.f.append(nn.Sequential(f2, f1))
        for i in range(3, self.M):
            f1 = _conv_layer(c[i], 3, 3, 1, 1)
            f2 = _conv_layer(c[i], c[i], 3, 1, 1)
            self.f.append(nn.Sequential(f2, f1))
        for i in range(0, self.M - 1):
            self.d.append(_deconv_layer(3, 3, 4, 2, 1))
        self.d_list = nn.Sequential(*self.d)
        self.f_list = nn.Sequential(*self.f)
        self.g_list = nn.Sequential(*self.g)

    def forward(self, y):
        y = self.G(y)
        gy = []
        dy = []
        for i in range(0, self.M):
            gy.append(self.g[i](y[:, i * self.out_channels // 6:(i + 1) * self.out_channels // 6, :, :]))
        for i in range(0, self.M):
            dy.append(self.f[i](gy[i]))
            temp = dy[i]
        for i in range(1, self.M):
            j = self.M - i
            dy[j - 1] = (dy[j - 1] + self.d[j - 1](dy[j])) / 2
        return dy[0][:, :, 32:-32, 32:-32]


class BaseGGDecoder(nn.Module, PreparableMixin):
    def __init__(self, in_channels=3, out_channels=192, stage_widths=None,
                 deconv="deconv",
                 normalization='gdn',
                 activation='relu',
                 lambda4s=[1], upsample_mode=None,
                 conv_first=True,
                 key_in=None, key_out=None,
                 **kwargs):
        super(BaseGGDecoder, self).__init__()
        self.in_channels = in_channels
        self.num_filters = out_channels
        self.stage_widths = stage_widths
        self.caffe_channels = out_channels
        self.key_in = key_in
        self.key_out = key_out

        self.lambda4s = lambda4s
        self.sampled_lambda = 1
        self.deconv_name = deconv

        self.kernel_size = kwargs.get('k', 4)  # only add to GG18's first layer

        dequant = kwargs.get('dequant', "NONE")
        if dequant == "3conv":
            dequant_layers = [
                nn.Conv2d(self.num_filters, self.num_filters, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(self.num_filters, self.num_filters, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(self.num_filters, self.num_filters, 3, 1, 1)
            ]
            self.dequant = nn.Sequential(*dequant_layers)
        else:
            self.dequant = None

        if deconv == 'deconv':
            self.deconv = nn.ConvTranspose2d
        elif deconv == 'signal':
            self.deconv = SignalConvTranspose2d
        elif deconv == 'conv_upsample':
            self.deconv = partial(Conv_Upsample, conv=nn.Conv2d, upsample_mode=upsample_mode, conv_first=conv_first)
        elif deconv == 'signal_upsample':
            self.deconv = partial(Conv_Upsample, conv=SignalConv2d, upsample_mode=upsample_mode, conv_first=conv_first)
        elif deconv == 'mixop_signal':
            from nas.mixop import MixedOp_DeConv_Signal
            self.deconv = MixedOp_DeConv_Signal
        elif deconv == 'mixop_origin':
            from nas.mixop import MixedOp_DeConv_Origin
            self.deconv = MixedOp_DeConv_Origin
        else:
            raise NotImplemented('unsupported deconv layer {}'.format(deconv))

        self.deconv_up = self.deconv
        if deconv not in ['conv_upsample', 'signal_upsample']:
            # it seems like deconv can't be partial
            class UpDeconvWrapper(self.deconv):
                def forward(self, x):
                    n, c, h, w = x.shape
                    factor = self.stride[0]

                    # enable up-sampling deconv with odd kernel_size
                    return super().forward(x, output_size=(n, c, factor * h, factor * w))

            UpDeconvWrapper.__name__ = self.deconv.__name__ + '_UpDeconvWrapper'
            self.deconv_up = UpDeconvWrapper

        if normalization == 'gdn':
            self.norm_layer = GDN
        elif normalization == '1dn':
            self.norm_layer = partial(GDN, fast=True)
        elif normalization == '1dn_no_pow':
            self.norm_layer = partial(L1GDNNoPow)
        elif normalization == '1dn_quant':
            self.norm_layer = partial(GDN, fast=True, quant=True)
        elif normalization == 'mixop_gdn':
            from nas.mixop import MixedOp_GDN
            self.norm_layer = MixedOp_GDN
        else:
            raise NotImplementedError

        if activation == 'relu':
            self.activation_layer = nn.ReLU
        elif activation == 'leaky-relu':
            self.activation_layer = nn.LeakyReLU
        else:
            raise ValueError(f'unsupported act: {activation}')

        self._layers = nn.ModuleList([])
        self.build()
        self.post = None # eat post part in y_decoder when converting to trt

    def build(self):
        raise NotImplementedError

    @decorator_input
    def forward(self, x):
        if self.dequant is not None:
            x = x + self.dequant(x)
        for layer in self._layers:
            x = layer(x)
        if self.post is not None:
            x = self.post(x)
        return x


class BaseGGDecoder_QUANT(BaseGGDecoder):
    def build(self):
        assert self.deconv_name in ['signal'], self.deconv_name
        self.deconv = NoBnConvTranspose2d

        self.build_quant()

    def build_quant(self):
        raise NotImplementedError


class YDecoder_GG18_QUANT(BaseGGDecoder_QUANT):
    def build_quant(self):
        if not self.stage_widths:
            self.stage_widths = [self.num_filters] * 3
        self._layers = nn.ModuleList([
            # the input is the dequantized latent y, and is in int 8
            self.deconv(
                self.num_filters, self.stage_widths[0], (4, 4), stride=2, padding=1,
                bias=True, padding_mode="zeros"),  # padding_mode here is ignored in the integer lib
            EMAAct(channel_num=self.stage_widths[0]),
            self.norm_layer(self.stage_widths[0], inverse=True),
            EMAAct(channel_num=self.stage_widths[0]),
            self.deconv(
                self.stage_widths[0], self.stage_widths[1], (4, 4), stride=2, padding=1,
                bias=True, padding_mode="zeros"),
            EMAAct(channel_num=self.stage_widths[1]),
            self.norm_layer(self.stage_widths[1], inverse=True),
            EMAAct(channel_num=self.stage_widths[1]),
            self.deconv(
                self.stage_widths[1], self.stage_widths[2], (4, 4), stride=2, padding=1,
                bias=True, padding_mode="zeros"),
            EMAAct(channel_num=self.stage_widths[2]),
            self.norm_layer(self.stage_widths[2], inverse=True),
            EMAAct(channel_num=self.stage_widths[2]),
            self.deconv(
                self.stage_widths[2], self.in_channels, (4, 4), stride=2, padding=1,
                bias=True, padding_mode="zeros"),
            EMAAct(channel_num=self.in_channels)
            # TODO: can we ommit this EMAAct here by manually setting the output_scale in trt and dsp?
        ])


class ZDecoder_GG18_QUANT(BaseGGDecoder_QUANT):
    '''trt coupled with floor division:
    in trt: 
        int8_weight, int8_input -> fp32_act
        rescaled_fp32_act = fp_32_act * [output_scale / (input_scale * weight_scale[i])]
        rescaled_fp32_act2 = rescaled_fp32_act + output_scale * fp32_bias
        int8_output = saturate_int8(round(rescaled_fp32_act2))
    so if we use real Integer arithemtic (use_Int=True), we have
        input_scale = 1
        weight_scale[i] = 1
    then rescaled_fp32_act2 = fp_32_act * [output_scale + output_scale * fp32_bias
                            = output_scale * (fp_32_act + fp32_bias)
    we can set the output_scale = 1, the mutiplier here must be implemented by a floor division
    to convert int32 to int8, so the fp32_bias has to be in int32 too
    so we can use int8 conv in trt if we can insert a floor division layer before saturate_int8
    but what will happen if we skip the floor division to saturate_int8(int_32) directly ?
    '''

    pass
    # raise NotImplementedError

    '''
    use_floor_division = True
    if use_floor_division:
        Act = ProAct
    else:
        Act = lambda: None

    def build_quant(self):
        assert self.deconv_name in ['signal'], self.deconv_name
        self.deconv = partial(NoBnConvTranspose2d, use_Int=True)
        self.EMAAct = partial(EMAAct, use_Int=True)

        if not self.stage_widths:
            self.stage_widths = [self.num_filters] * 2
        self._layers = nn.ModuleList([
            self.deconv(
                self.num_filters, self.stage_widths[0], (4, 4), stride=2, padding=1,
                bias=True, padding_mode="zeros"),
            nn.ReLU(),
            self.EMAAct(),
            self.deconv(
                self.stage_widths[0], self.stage_widths[1], (4, 4), stride=2, padding=1,
                bias=True, padding_mode="zeros"),
            nn.ReLU(),
            self.EMAAct(),
            self.deconv(
                self.stage_widths[1], self.in_channels, (3, 3), stride=1, padding=1,
                bias=True, padding_mode="zeros"),
            self.EMAAct(), 
            GGQReLU(0, 63) # will require a clamp layer in deployment
        ])
    '''


class BaseGGDecoder_VAR(BaseGGDecoder):
    def forward(self, x):
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


class YDecoder_GG17(BaseGGDecoder):
    def build(self):
        if not self.stage_widths:
            self.stage_widths = [self.num_filters] * 3
        deconv1 = self.deconv(
            self.num_filters, self.stage_widths[0], (4, 4), stride=2, padding=1,
            bias=True, padding_mode="zeros")
        deconv2 = self.deconv(
            self.stage_widths[0], self.stage_widths[1], (4, 4), stride=2, padding=1,
            bias=True, padding_mode="zeros")
        deconv3 = self.deconv(
            self.stage_widths[1], self.in_channels, (6, 6), stride=4, padding=1,
            bias=True, padding_mode="zeros")

        self._layers = nn.ModuleList([
            deconv1,
            self.norm_layer(self.stage_widths[0], inverse=True),
            deconv2,
            self.norm_layer(self.stage_widths[1], inverse=True),
            deconv3,
        ])


class YDecoder_GG17_VAR(BaseGGDecoder_VAR):
    def build(self):
        lambdas_len = len(self.lambda4s) - 1
        if not self.stage_widths:
            self.stage_widths = [self.num_filters] * 2
        deconv1 = ConditionalConv2d(
            self.num_filters, self.stage_widths[0], (4, 4), stride=2, padding=1,
            bias=True, padding_mode="zeros", conv=self.deconv, lambdas_len=lambdas_len)
        deconv2 = ConditionalConv2d(
            self.stage_widths[0], self.stage_widths[1], (4, 4), stride=2, padding=1,
            bias=True, padding_mode="zeros", conv=self.deconv, lambdas_len=lambdas_len)
        deconv3 = ConditionalConv2d(
            self.stage_widths[1], self.in_channels, (6, 6), stride=4, padding=1,
            bias=True, padding_mode="zeros", conv=self.deconv, lambdas_len=lambdas_len)

        self._layers = nn.ModuleList([
            deconv1,
            self.norm_layer(self.stage_widths[0], inverse=True),
            deconv2,
            self.norm_layer(self.stage_widths[1], inverse=True),
            deconv3,
        ])


class YDecoder_GG18(BaseGGDecoder):
    def build(self):
        if not self.stage_widths:
            self.stage_widths = [self.num_filters] * 3
        k = self.kernel_size
        self._layers = nn.ModuleList([
            self.deconv(
                self.num_filters, self.stage_widths[0], (4, 4), stride=2, padding=1,
                bias=True, padding_mode="zeros"),
            self.norm_layer(self.stage_widths[0], inverse=True),
            self.deconv(
                self.stage_widths[0], self.stage_widths[1], (4, 4), stride=2, padding=1,
                bias=True, padding_mode="zeros"),
            self.norm_layer(self.stage_widths[1], inverse=True),
            self.deconv(
                self.stage_widths[1], self.stage_widths[2], (4, 4), stride=2, padding=1,
                bias=True, padding_mode="zeros"),
            self.norm_layer(self.stage_widths[2], inverse=True),
            self.deconv(
                self.stage_widths[2], self.in_channels, (k, k), stride=2,
                padding=(k - 1) // 2,
                bias=True, padding_mode="zeros"),
            # self.deconv(
            #     self.stage_widths[2], self.in_channels, (k, k), stride=2, padding=(k - 2) // 2,
            #     bias=True, padding_mode="zeros")
        ])


class YDecoder_QM(BaseGGDecoder):
    # def upconv2d(self, in_channels, out_channels, kernel_size=5, stride=2):
    #     return nn.ConvTranspose2d(
    #         in_channels,
    #         out_channels,
    #         kernel_size=kernel_size,
    #         stride=stride,
    #         output_padding=stride - 1,
    #         padding=kernel_size // 2,
    #     )

    def build(self, prior_nc=64, sft_ks=3):
        if not self.stage_widths:
            self.stage_widths = [self.num_filters, self.num_filters, self.num_filters//2, self.num_filters//4]
        k = self.kernel_size

        sft1 = SFT(self.stage_widths[0], prior_nc, ks=sft_ks)
        sft2 = SFT(self.stage_widths[1], prior_nc, ks=sft_ks)
        sft3 = SFT(self.stage_widths[2], prior_nc, ks=sft_ks)
        sft4 = SFT(self.stage_widths[3], prior_nc, ks=sft_ks)

        self._layer1 = nn.ModuleList([
            SFTResblk(self.num_filters, prior_nc, ks=sft_ks),
            SFTResblk(self.num_filters, prior_nc, ks=sft_ks),

            self.deconv(
                self.num_filters, self.stage_widths[0], (4, 4), stride=2, padding=1,
                bias=True, padding_mode="zeros"),
            self.norm_layer(self.stage_widths[0], inverse=True),
            sft1,

            self.deconv(
                self.stage_widths[0], self.stage_widths[1], (4, 4), stride=2, padding=1,
                bias=True, padding_mode="zeros"),
            self.norm_layer(self.stage_widths[1], inverse=True),
            sft2,

            self.deconv(
                self.stage_widths[1], self.stage_widths[2], (4, 4), stride=2, padding=1,
                bias=True, padding_mode="zeros"),
            self.norm_layer(self.stage_widths[2], inverse=True),
            sft3,

            self.deconv(
                self.stage_widths[2], self.stage_widths[3], (4, 4), stride=2, padding=1,
                bias=True, padding_mode="zeros"),
            self.norm_layer(self.stage_widths[3], inverse=True),
            sft4,
            self.conv(
                self.stage_widths[3], self.in_channels, (k, k), stride=2,
                padding=(k - 1) // 2,
                bias=True, padding_mode="zeros"),

        ])

        self._layer2 = nn.ModuleList([

            self.conv(
                240, prior_nc * 4, (3, 3), stride=1, padding=1,
                bias=True, padding_mode="zeros"),
            nn.LeakyReLU(0.1, True),
            self.conv(
                prior_nc * 4, prior_nc * 2, (3, 3), stride=1, padding=1,
                bias=True, padding_mode="zeros"),
            nn.LeakyReLU(0.1, True),
            self.conv(
                prior_nc * 2, prior_nc, (3, 3), stride=1, padding=1,
                bias=True, padding_mode="zeros"),

            self.deconv(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            self.deconv(prior_nc, prior_nc, 3),

            self.deconv(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            self.deconv(prior_nc, prior_nc, 3),

            self.deconv(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            self.deconv(prior_nc, prior_nc, 3),

            self.deconv(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            self.deconv(prior_nc, prior_nc, 3),
        ])

    def forward(self, y, w):
        w = torch.cat((y, w), 1)
        for i in range(5):
            w = self._layer2[i](w)
        y = self._layer1[0](y, w)
        y = self._layer1[1](y, w)
        y = self._layer1[2](y)
        y = self._layer1[3](y)
        for i in range(5, 8):
            w = self._layer2[i](w)
        y = self._layer1[4](y, w)

        y = self._layer1[5](y)
        y = self._layer1[6](y)
        for i in range(8, 11):
            w = self._layer2[i](w)
        y = self._layer1[7](y, w)

        y = self._layer1[8](y)
        y = self._layer1[9](y)
        for i in range(11, 14):
            w = self._layer2[i](w)
        y = self._layer1[10](y,w)

        y = self._layer1[11](y)
        y = self._layer1[12](y)
        for i in range(14, 17):
            w = self._layer2[i](w)
        y = self._layer1[13](y, w)
        y = self._layer1[14](y)
        return y


class FDecoder_QM(BaseGGDecoder):
    # def UpConv2d(self, in_channels, out_channels, kernel_size=5, stride=2):
    #     return nn.ConvTranspose2d(
    #         in_channels,
    #         out_channels,
    #         kernel_size=kernel_size,
    #         stride=stride,
    #         output_padding=stride - 1,
    #         padding=kernel_size // 2,
    #     )

    def build(self, N=192):
        if not self.stage_widths:
            self.stage_widths = [self.num_filters, self.num_filters, self.num_filters // 2, self.num_filters // 4]
        k = self.kernel_size

        self._layers = nn.ModuleList([

            self.deconv(N, N // 2, 3),  # ConvTranspose2 逆卷积
            nn.LeakyReLU(0.1, True),
            self.deconv(N // 2, N // 4),
            nn.LeakyReLU(0.1, True),
            self.deconv(
                N // 4, N // 4, (3, 3), stride=1, padding=1,
                bias=True, padding_mode="zeros"),
        ])


class YDecoderGG18K5(BaseGGDecoder):
    def build(self):
        if not self.stage_widths:
            self.stage_widths = [self.num_filters] * 3
        self._layers = nn.ModuleList([
            self.deconv_up(
                self.num_filters, self.stage_widths[0], (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros"),
            self.norm_layer(self.stage_widths[0], inverse=True),
            self.deconv_up(
                self.stage_widths[0], self.stage_widths[1], (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros"),
            self.norm_layer(self.stage_widths[1], inverse=True),
            self.deconv_up(
                self.stage_widths[1], self.stage_widths[2], (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros"),
            self.norm_layer(self.stage_widths[2], inverse=True),
            self.deconv_up(
                self.stage_widths[2], self.in_channels, (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros"),
        ])


class YDecoderCheng20(BaseGGDecoder):
    """
    synthesis transform used by:

    Learned Image Compression with Discretized Gaussian Mixture Likelihoods and Attention Modules
    (arXiv:2001.01568v3)

    also see: https://github.com/ZhengxueCheng/Learned-Image-Compression-with-GMM-and-Attention

    this is a simplified Non-Local block
    """

    def __init__(self, *args, deconv='signal', pixel_shuffle='pytorch', **kwargs):
        if deconv == 'signal':
            self.conv = SignalConv2d
        elif deconv == 'deconv':
            self.conv = nn.Conv2d
        else:
            raise ValueError(f'deconv type {deconv} is not supported')

        if pixel_shuffle == 'pytorch':
            self.pixel_shuffle = self.PytorchSubPixelUp
        elif pixel_shuffle == 'naive':
            self.pixel_shuffle = self.SubPixelUp
        else:
            raise ValueError(f'pixel_shuffle: {pixel_shuffle}')

        super().__init__(*args, deconv=deconv, **kwargs)

    class SubPixelUp(nn.Module):
        def __init__(self, channel_in, channel_out, conv_cls):
            super().__init__()
            self.conv = conv_cls(channel_in, 4 * channel_out, 3, padding=1)

        def forward(self, x):
            x = self.conv(x)  # sub pixels
            n, c, h, w = x.shape
            assert c % 4 == 0
            x = x.reshape(n, c // 4, 2, 2, h, w).permute(0, 1, 4, 3, 5, 2)
            x = x.reshape(n, c // 4, 2 * h, 2 * w)
            return x

    class PytorchSubPixelUp(SubPixelUp):
        def forward(self, x):
            x = self.conv(x)
            return F.pixel_shuffle(x, 2)

    class UpResBlock(nn.Module):
        def __init__(self, channel_in, num_filter, conv_cls, norm_layer, up_layer, up_sample):
            super().__init__()
            if up_sample:
                layers = [
                    up_layer(channel_in, num_filter, conv_cls),
                    nn.LeakyReLU(inplace=True),
                    conv_cls(num_filter, num_filter, 3, padding=1),
                    norm_layer(num_filter, inverse=True),
                ]
                self.shortcut = up_layer(channel_in, num_filter, conv_cls)
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
            Cheng20ResBlockAttention(self.num_filters, self.conv),
            self.UpResBlock(self.num_filters, self.stage_widths[0], self.conv, self.norm_layer, self.pixel_shuffle, False),
            self.UpResBlock(self.stage_widths[0], self.stage_widths[1], self.conv, self.norm_layer, self.pixel_shuffle, True),
            self.UpResBlock(self.stage_widths[1], self.stage_widths[2], self.conv, self.norm_layer, self.pixel_shuffle, False),
            self.UpResBlock(self.stage_widths[2], self.stage_widths[3], self.conv, self.norm_layer, self.pixel_shuffle, True),
            Cheng20ResBlockAttention(self.stage_widths[3], self.conv),
            self.UpResBlock(self.stage_widths[3], self.stage_widths[4], self.conv, self.norm_layer, self.pixel_shuffle, False),
            self.UpResBlock(self.stage_widths[4], self.stage_widths[5], self.conv, self.norm_layer, self.pixel_shuffle, True),
            self.UpResBlock(self.stage_widths[5], self.num_filters, self.conv, self.norm_layer, self.pixel_shuffle, False),
            self.pixel_shuffle(self.num_filters, self.in_channels, self.conv)
        ])


class ZDecoder_GG18(BaseGGDecoder):
    def build(self):
        if not self.stage_widths:
            self.stage_widths = [self.num_filters] * 2
        self._layers = nn.ModuleList([
            self.deconv(
                self.num_filters, self.stage_widths[0], (4, 4), stride=2, padding=1,
                bias=True, padding_mode="zeros"),
            nn.ReLU(inplace=True),
            self.deconv(
                self.stage_widths[0], self.stage_widths[1], (4, 4), stride=2, padding=1,
                bias=True, padding_mode="zeros"),
            nn.ReLU(inplace=True),
            self.deconv(
                self.stage_widths[1], self.in_channels, (3, 3), stride=1, padding=1,
                bias=True, padding_mode="zeros"),
        ])


class ZDecoder_QM(BaseGGDecoder):
    def build(self):
        if not self.stage_widths:
            self.stage_widths = [self.num_filters, self.num_filters * 3//2, self.num_filters * 2]
        self._layers = nn.ModuleList([
            self.deconv(
                self.num_filters, self.stage_widths[0], (4, 4), stride=2, padding=1,
                bias=True, padding_mode="zeros"),
            nn.LeakyReLU(inplace=True),
            self.deconv(
                self.stage_widths[0], self.stage_widths[1], (4, 4), stride=2, padding=1,
                bias=True, padding_mode="zeros"),
            nn.LeakyReLU(inplace=True),
            self.deconv(
                self.stage_widths[1], self.stage_widths[2], (3, 3), stride=1, padding=1,
                bias=True, padding_mode="zeros"),
        ])



class ZDecoderGG18K5(BaseGGDecoder):
    def build(self):
        if not self.stage_widths:
            self.stage_widths = [self.num_filters] * 2
        self._layers = nn.ModuleList([
            self.deconv_up(
                self.num_filters, self.stage_widths[0], (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros"),
            nn.ReLU(inplace=True),
            self.deconv_up(
                self.stage_widths[0], self.stage_widths[1], (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros"),
            nn.ReLU(inplace=True),
            self.deconv(
                self.stage_widths[1], self.in_channels, (3, 3), stride=1, padding=1,
                bias=True, padding_mode="zeros"),
        ])


class ZDecoderLinearK5(BaseGGDecoder):
    """
    a general z decoder with linear in/de-crease stage widths
    """
    def build(self):
        m = self.in_channels
        n = self.num_filters
        d = (m-n) // 3
        l1 = n + d
        l2 = m - d
        self._layers = nn.ModuleList([
            self.deconv_up(
                n, l1, (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros"),
            self.activation_layer(inplace=True),
            self.deconv_up(
                l1, l2, (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros"),
            self.activation_layer(inplace=True),
            self.deconv(
                l2, m, (3, 3), stride=1, padding=1,
                bias=True, padding_mode="zeros"),
        ])


class TwoPathZDecoderLinearK5(BaseGGDecoder):
    """
    a general z decoder with linear in/de-crease stage widths
    """
    def build(self):
        m = self.in_channels
        assert m % 2 == 0
        half_m = m // 2
        n = self.num_filters
        d = (half_m - n) // 3
        l1 = n + d
        l2 = half_m - d
        self._layers = nn.ModuleList([nn.Sequential(
            self.deconv_up(
                n, l1, (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros"),
            self.activation_layer(inplace=True),
            self.deconv_up(
                l1, l2, (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros"),
            self.activation_layer(inplace=True),
            self.deconv(
                l2, half_m, (3, 3), stride=1, padding=1,
                bias=True, padding_mode="zeros")
        ) for _ in range(2)
        ])

    def forward(self, x):
        if self.dequant is not None:
            x = x + self.dequant(x)

        hyper = [l(x) for l in self._layers]
        x = torch.cat(hyper, 1)

        if self.post is not None:
            x = self.post(x)
        return x



class ZDecoder_GG18C(BaseGGDecoder):
    def build(self):
        self._layers = nn.ModuleList([
            self.deconv(
                self.num_filters, self.num_filters, (4, 4), stride=2, padding=1,
                bias=True, padding_mode="zeros"),
            nn.LeakyReLU(inplace=True),
            self.deconv(
                self.num_filters, int(self.num_filters * 1.5), (4, 4), stride=2, padding=1,
                bias=True, padding_mode="zeros"),
            nn.LeakyReLU(inplace=True),
            self.deconv(
                int(self.num_filters * 1.5), self.in_channels * 2, (3, 3), stride=1, padding=1,
                bias=True, padding_mode="zeros"),
        ])


class ZDecoderGG18CK5(BaseGGDecoder):
    def build(self):
        self._layers = nn.ModuleList([
            self.deconv_up(
                self.num_filters, self.num_filters, (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros"),
            nn.LeakyReLU(inplace=True),
            self.deconv_up(
                self.num_filters, int(self.num_filters * 1.5), (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros"),
            nn.LeakyReLU(inplace=True),
            self.deconv(
                int(self.num_filters * 1.5), self.in_channels * 2, (3, 3), stride=1, padding=1,
                bias=True, padding_mode="zeros"),
        ])


class ZDecoderCheng20(BaseGGDecoder):
    def build(self):
        c = self.num_filters
        self._layers = nn.ModuleList([
            self.deconv(c, c, (3, 3), padding=1),
            nn.LeakyReLU(inplace=True),
            self.deconv_up(c, c, (3, 3), padding=1, stride=2),
            nn.LeakyReLU(inplace=True),
            self.deconv(c, int(1.5 * c), (3, 3), padding=1),
            nn.LeakyReLU(inplace=True),
            self.deconv_up(int(1.5 * c), int(1.5 * c), (3, 3), padding=1, stride=2),
            nn.LeakyReLU(inplace=True),
            self.deconv(int(1.5 * c), int(2 * c), (3, 3), padding=1),
        ])


class YDecoder_GG18_VAR(BaseGGDecoder_VAR):
    def build(self):
        lambdas_len = len(self.lambda4s) - 1
        if not self.stage_widths:
            self.stage_widths = [self.num_filters] * 3
        self._layers = nn.ModuleList([
            ConditionalConv2d(
                self.num_filters, self.stage_widths[0], (4, 4), stride=2, padding=1,
                bias=True, padding_mode="zeros", conv=self.deconv, lambdas_len=lambdas_len),
            self.norm_layer(self.stage_widths[0], inverse=True),
            ConditionalConv2d(
                self.stage_widths[0], self.stage_widths[1], (4, 4), stride=2, padding=1,
                bias=True, padding_mode="zeros", conv=self.deconv, lambdas_len=lambdas_len),
            self.norm_layer(self.stage_widths[1], inverse=True),
            ConditionalConv2d(
                self.stage_widths[1], self.stage_widths[2], (4, 4), stride=2, padding=1,
                bias=True, padding_mode="zeros", conv=self.deconv, lambdas_len=lambdas_len),
            self.norm_layer(self.stage_widths[2], inverse=True),
            ConditionalConv2d(
                self.stage_widths[2], self.in_channels, (4, 4), stride=2, padding=1,
                bias=True, padding_mode="zeros", conv=self.deconv, lambdas_len=lambdas_len)
        ])


class ZDecoder_GG18_VAR(BaseGGDecoder_VAR):
    def build(self):
        lambdas_len = len(self.lambda4s) - 1

        self._layers = nn.ModuleList([
            ConditionalConv2d(
                self.num_filters, self.stage_widths[0], (4, 4), stride=2, padding=1,
                bias=True, padding_mode="zeros", conv=self.deconv, lambdas_len=lambdas_len),
            nn.ReLU(inplace=True),
            ConditionalConv2d(
                self.stage_widths[0], self.stage_widths[0], (4, 4), stride=2, padding=1,
                bias=True, padding_mode="zeros", conv=self.deconv, lambdas_len=lambdas_len),
            nn.ReLU(inplace=True),
            ConditionalConv2d(
                self.stage_widths[0], self.in_channels, (3, 3), stride=1, padding=1,
                bias=True, padding_mode="zeros", conv=self.deconv, lambdas_len=lambdas_len),
        ])


class ZDecoder_GG18C_VAR(BaseGGDecoder_VAR):
    def build(self):
        lambdas_len = len(self.lambda4s) - 1

        self._layers = nn.ModuleList([
            ConditionalConv2d(
                self.num_filters, self.stage_widths[0], (4, 4), stride=2, padding=1,
                bias=True, padding_mode="zeros", conv=self.deconv, lambdas_len=lambdas_len),
            nn.LeakyReLU(inplace=True),
            ConditionalConv2d(
                self.stage_widths[0], self.stage_widths[1], (4, 4), stride=2, padding=1,
                bias=True, padding_mode="zeros", conv=self.deconv, lambdas_len=lambdas_len),
            nn.LeakyReLU(inplace=True),
            ConditionalConv2d(
                self.stage_widths[1], self.in_channels, (3, 3), stride=1, padding=1,
                bias=True, padding_mode="zeros", conv=self.deconv, lambdas_len=lambdas_len),
        ])


class YUV_decoder(BaseGGDecoder):
    def build(self):
        self.layer = nn.Sequential(self.deconv(3, 3, (4, 4), stride=2, padding=1, bias=True, padding_mode="zeros"))

    def forward(self, x):
        return self.layer(x)


class YDecoder_CEVideo(nn.Module):
    def __init__(self, n=192, out_channels=3, **kwargs):
        super(YDecoder_CEVideo, self).__init__()
        self.out_channels = out_channels
        self.n = n

        self.deconv = SignalConvTranspose2d
        self.block = ResBlock_Video

        self._layers = nn.ModuleList([])
        self.build()

    def build(self):
        self._layers = nn.ModuleList([
            self.deconv(self.n, self.n, 5, 2, 2, output_padding=1),
            self.block(self.n),
            self.block(self.n),
            self.block(self.n),
            self.deconv(self.n, self.n, 5, 2, 2, output_padding=1),
            self.deconv(self.n, self.n, 5, 2, 2, output_padding=1),
            self.block(self.n),
            self.block(self.n),
            self.block(self.n),
            self.deconv(self.n, self.out_channels, 5, 2, 2, output_padding=1),
        ])

    def forward(self, x):
        for layer in self._layers:
            x = layer(x)
        return x


class ZDecoder_CEVideo(nn.Module):
    def __init__(self, n=192, m=80, out_channels=3, **kwargs):
        super(ZDecoder_CEVideo, self).__init__()
        self.out_channels = out_channels
        self.n = n
        self.m = m

        self.conv = SignalConv2d
        self.deconv = SignalConvTranspose2d
        self.block = ResBlock_Video

        self._layer1 = nn.ModuleList([
            self.deconv(self.n, self.n, 5, 2, 2, output_padding=1),
            self.block(self.n),
            self.deconv(self.n, self.n, 5, 2, 2, output_padding=1),
            self.block(self.n),
            # concat1
            self.deconv(self.n, self.n, 5, 2, 2, output_padding=1),
            GDN(n, inverse=True),
            # concat2
            self.deconv(self.n, self.n, 5, 2, 2, output_padding=1),
            GDN(n, inverse=True),
            # concat3
            self.deconv(self.n, self.n, 5, 2, 2, output_padding=1),
            GDN(n, inverse=True),
            # concat4
        ])

        self._layer2 = nn.ModuleList([
            # concat1
            self.deconv(self.n * 2, self.m, 5, 2, 2, output_padding=1),
            GDN(m, inverse=True),
            # concat2
            self.deconv(self.n + self.m, self.m, 5, 2, 2, output_padding=1),
            GDN(m, inverse=True),
            # concat3
            self.deconv(self.n + self.m, self.m, 5, 2, 2, output_padding=1),
            GDN(m, inverse=True),
            # concat4
            self.deconv(self.n + self.m, 5, 5, 2, 2, output_padding=1),
            self.conv(5, self.m, 5, 2, 2),
            GDN(m, inverse=False),
            self.conv(self.m, self.m, 5, 2, 2),
            GDN(m, inverse=False),
            self.conv(self.m, self.m, 5, 2, 2),
            GDN(m, inverse=False),
            self.conv(self.m, self.out_channels, 5, 2, 2),
        ])

    def forward(self, y, z):
        # input x[N, 2*C, H, W]
        # y = x[:, :self.n]
        # z = x[:, self.n:]
        for i in range(4):
            z = self._layer1[i](z)
        y = torch.cat((y, z), 1)
        y = self._layer2[0](y)
        y = self._layer2[1](y)

        z = self._layer1[4](z)
        z = self._layer1[5](z)
        y = torch.cat((y, z), 1)
        y = self._layer2[2](y)
        y = self._layer2[3](y)

        z = self._layer1[6](z)
        z = self._layer1[7](z)
        y = torch.cat((y, z), 1)
        y = self._layer2[4](y)
        y = self._layer2[5](y)

        z = self._layer1[8](z)
        z = self._layer1[9](z)
        y = torch.cat((y, z), 1)
        for i in range(6, 14):
            y = self._layer2[i](y)
        return y


class DCTYDecoderGG18(BaseGGDecoder):
    """DCT lossy compression.
    """
    def build(self):
        if not self.stage_widths:
            self.stage_widths = [self.num_filters] * 2
        k = self.kernel_size
        self._layers = nn.ModuleList([
            self.deconv(
                self.num_filters, self.stage_widths[0], (4, 4), stride=2, padding=1,
                bias=True, padding_mode="zeros"),
            self.norm_layer(self.stage_widths[0], inverse=True),
            self.deconv(
                self.stage_widths[0], self.stage_widths[1], (4, 4), stride=2, padding=1,
                bias=True, padding_mode="zeros"),
            self.norm_layer(self.stage_widths[1], inverse=True),
            self.deconv(
                self.stage_widths[1], self.in_channels, (k, k), stride=2,
                padding=(k - 1) // 2,
                bias=True, padding_mode="zeros"),
        ])


def decoders(tag="RT", out_channels=30, M=6, in_channels=3, **kwargs):
    """
    get y-decoder
    :param tag: the type of decoder. ['RT', 'YGG17', 'YGG18', 'ZGG18', 'NONE']
    :param in_channels:
    :param M:
    :param out_channels:
    :return: if tag is 'NONE', return None. otherwise return a model
    """
    if tag == "RT":
        model = Decoder_RT(out_channels, M)
        initial(model)

    elif tag == "YGG17":
        model = YDecoder_GG17(in_channels, out_channels, **kwargs)
    elif tag == "YGG18":
        model = YDecoder_GG18(in_channels, out_channels, **kwargs)
    elif tag == 'YGG18K5':
        model = YDecoderGG18K5(in_channels, out_channels, **kwargs)
    elif tag == "ZGG18":
        model = ZDecoder_GG18(in_channels, out_channels, **kwargs)
    elif tag == 'ZGG18K5':
        model = ZDecoderGG18K5(in_channels, out_channels, **kwargs)
    elif tag == "ZGG18C":
        model = ZDecoder_GG18C(in_channels, out_channels, **kwargs)
    elif tag == "ZGG18CK5":
        model = ZDecoderGG18CK5(in_channels, out_channels, **kwargs)
    elif tag == 'ZLinearK5':
        model = ZDecoderLinearK5(in_channels, out_channels, **kwargs)
    elif tag == 'TwoPathZLinearK5':
        model = TwoPathZDecoderLinearK5(in_channels, out_channels, **kwargs)
    elif tag == "YGG17_VAR":
        model = YDecoder_GG17_VAR(in_channels, out_channels, **kwargs)
    elif tag == "YGG18_VAR":
        model = YDecoder_GG18_VAR(in_channels, out_channels, **kwargs)
    elif tag == "ZGG18_VAR":
        model = ZDecoder_GG18_VAR(in_channels, out_channels, **kwargs)
    elif tag == "ZGG18C_VAR":
        model = ZDecoder_GG18C_VAR(in_channels, out_channels, **kwargs)

    elif tag == "YGG18_yuv":
        model = nn.Sequential(YDecoder_GG18(in_channels, out_channels, **kwargs))

    elif tag == "QZGG18":
        model = QZDecoder_GG18(in_channels, out_channels, **kwargs)
    elif tag == "NoBpQZGG18":
        model = NoBpQZDecoder_GG18(in_channels, out_channels, **kwargs)
    elif tag == "QZGG18Clip":
        model = QZDecoder_GG18Clip(in_channels, out_channels, **kwargs)
    elif tag == "QZUpsample":
        model = QZDecoderUpsample(in_channels, out_channels, **kwargs)

    elif tag == "YGG18_QUANT":
        model = YDecoder_GG18_QUANT(in_channels, out_channels, **kwargs)

    elif tag == "Y_QM":
        model = YDecoder_QM(in_channels, out_channels, **kwargs)
    elif tag == "Z_QM":
        model = ZDecoder_QM(in_channels, out_channels, **kwargs)
    elif tag == "Z_CON":
        model = FDecoder_QM(in_channels, out_channels, **kwargs)

    # DCT related decoders
    elif tag == "DCTYGG18":
        model = DCTYDecoderGG18(in_channels, out_channels, **kwargs)

    elif tag == 'NONE':
        return None
    else:
        raise NotImplementedError('decoder with tag ' + tag)

    return model


if __name__ == '__main__':
    x = torch.randn(2, 30, 160, 320)
    test = Decoder_RT()
    with torch.no_grad():
        print(test(x).shape)

    test = decoders()
    with torch.no_grad():
        print(test(x).shape)
