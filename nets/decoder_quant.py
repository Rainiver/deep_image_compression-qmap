import torch
import torch.nn as nn
import numpy as np
import math
from .layers_quant import QConvTranspose2d, QReLU, BpAct, NoBpAct, Round, QLeakyReLU, Clip, GGQReLU, GGBpAct, ProAct,\
    GGQConvTranspose2d, GGQUpDeconvWrapper, SymmetricActReparameterizer, AsymmetricActReparameterizer
from .norm import GDN, L1GDNNoPow

class QYDecoderGG18K5(nn.Module):
    Act = GGBpAct

    def __init__(self, in_channels=3, out_channels=192, normalization='gdn', deconv="deconv", stage_widths=None, lambda4s=[1]):
        super(QYDecoderGG18K5, self).__init__()
        self.in_channels = in_channels
        self.num_filters = out_channels
        self.caffe_channels = out_channels

        # stage_widths for num_filters of each layer
        self.stage_widths = stage_widths

        self.lambda4s = lambda4s
        self.sampled_lambda = 1

        if deconv == 'signal':
            self.deconv = GGQConvTranspose2d
        if deconv == 'deconv_old':
            self.deconv = QConvTranspose2d
        if deconv == 'deconv_up':
            self.deconv = GGQUpDeconvWrapper

        if normalization == 'gdn':
            self.norm_layer = GDN
            self.activation_layer = lambda: None  # GDN do not need activation
        elif normalization == '1dn':
            self.norm_layer = L1GDNNoPow
            self.activation_layer = lambda: None
        elif normalization == 'qgdn':
            raise NotImplementedError

        self._layers = nn.ModuleList([])
        self.build()

        self.debug = True
        self.quant = True

    def build(self):
        if not self.stage_widths:
            self.stage_widths = [self.num_filters] * 3
        self.deconv1 = self.deconv(
                self.num_filters, self.stage_widths[0], (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros")
        self.norm_layer1 = self.norm_layer(self.stage_widths[0], inverse=True)
        self.deconv2 = self.deconv(
                self.stage_widths[0], self.stage_widths[1], (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros")
        self.norm_layer2 = self.norm_layer(self.stage_widths[1], inverse=True)
        self.deconv3 = self.deconv(
                self.stage_widths[1], self.stage_widths[2], (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros")
        self.norm_layer3 = self.norm_layer(self.stage_widths[2], inverse=True)
        self.deconv4 = self.deconv(
                self.stage_widths[2], self.in_channels, (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros")

        # int32 signed accumulator -> int8 unsigned act
        # int32 unsigned c in GG19I is like the layer-wise ema mag
        self.round_func = Round.apply
        self.act_reparam = AsymmetricActReparameterizer.apply
        #self.act0 = BpAct(self.num_filters, 8)
        self.act1 = self.Act(self.stage_widths[0], 8)
        self.act2 = self.Act(self.stage_widths[1], 8)
        self.act3 = self.Act(self.stage_widths[2], 8)
        self.act4 = self.Act(self.in_channels, 8)

        self.relu = GGQReLU()

    def print_range(self, x, info):
        if self.debug and not self.training:
            print('{} max: {}'.format(info, x.max()), flush=True)
            print('{} min: {}'.format(info, x.min()), flush=True)
            # print('{} mode: {}'.format(info, x.mode()), flush=True)

    def forward(self, x):
        self.print_range(x, 'input')

        x = self.deconv1(x)
        self.print_range(x, 'deconv1')
        x = self.act1(x)
        self.print_range(x, 'act1')
        x = self.norm_layer1(x)
        self.print_range(x, 'igdn1')
        x = self.act_reparam(x, 8)

        x = self.deconv2(x)
        self.print_range(x, 'deconv2')
        x = self.act2(x)
        self.print_range(x, 'act2')
        x = self.norm_layer2(x)
        self.print_range(x, 'igdn2')
        x = self.act_reparam(x, 8)

        x = self.deconv3(x)
        self.print_range(x, 'deconv3')
        x = self.act3(x)
        self.print_range(x, 'act3')
        x = self.norm_layer3(x)
        self.print_range(x, 'igdn3')
        x = self.act_reparam(x, 8)

        x = self.deconv4(x)
        self.print_range(x, 'deconv4')
        x = self.act4(x)
        self.print_range(x, 'act4')
        x = self.relu(x, 0, 255)
        self.print_range(x, 'relu')

        x = x / 255.0

        self.print_range(self.deconv1.weight, 'conv1_w')
        self.print_range(self.deconv1.bias, 'conv1_b')
        self.print_range(self.deconv2.weight, 'conv2_w')
        self.print_range(self.deconv2.bias, 'conv2_b')
        self.print_range(self.deconv3.weight, 'conv3_w')
        self.print_range(self.deconv3.bias, 'conv3_b')
        self.print_range(self.deconv4.weight, 'conv4_w')
        self.print_range(self.deconv4.bias, 'conv4_b')
        self.print_range(self.act1.c, 'act1_c')
        self.print_range(self.act2.c, 'act2_c')
        self.print_range(self.act3.c, 'act3_c')
        self.print_range(self.act4.c, 'act4_c')

        return x


class QZDecoder_GG18(nn.Module):
    Act = GGBpAct

    def __init__(self, in_channels=3, out_channels=192, deconv="deconv", stage_widths=None, lambda4s=[1]):
        super(QZDecoder_GG18, self).__init__()
        self.in_channels = in_channels
        self.num_filters = out_channels
        self.caffe_channels = out_channels

        # stage_widths for num_filters of each layer
        self.stage_widths = stage_widths

        self.lambda4s = lambda4s
        self.sampled_lambda = 1

        if deconv == 'signal':
            self.deconv = GGQConvTranspose2d
        if deconv == 'deconv_old':
            self.deconv = QConvTranspose2d
        if deconv == 'deconv_up':
            self.deconv = GGQUpDeconvWrapper

        self._layers = nn.ModuleList([])
        self.build()

        self.debug = True
        self.quant = True

    def build(self):
        if not self.stage_widths:
            self.stage_widths = [self.num_filters] * 2
        self.deconv1 = self.deconv(
                self.num_filters, self.stage_widths[0], (4, 4), stride=2, padding=1,
                bias=True, padding_mode="zeros")
        self.deconv2 = self.deconv(
                self.stage_widths[0], self.stage_widths[1], (4, 4), stride=2, padding=1,
                bias=True, padding_mode="zeros")
        self.deconv3 = self.deconv(
                self.stage_widths[1], self.in_channels, (3, 3), stride=1, padding=1,
                bias=True, padding_mode="zeros")

        # int32 signed accumulator -> int8 unsigned act
        # int32 unsigned c in GG19I is like the layer-wise ema mag
        self.round_func = Round.apply
        #self.act0 = BpAct(self.num_filters, 8)
        self.act1 = self.Act(self.stage_widths[0], 8, dtype=torch.int32)
        self.act2 = self.Act(self.stage_widths[1], 8, dtype=torch.int32)
        self.act3 = self.Act(self.in_channels, 8, dtype=torch.int32)

        self.relu1 = GGQReLU()
        self.relu2 = GGQReLU()
        self.relu3 = GGQReLU()

    def print_range(self, x, info):
        if self.debug and not self.training:
            print('{} max: {}'.format(info, x.max()), flush=True)
            print('{} min: {}'.format(info, x.min()), flush=True)
            # print('{} mode: {}'.format(info, x.mode()), flush=True)

    def forward(self, x):
        self.print_range(x, 'input')

        x = self.deconv1(x)
        self.print_range(x, 'conv1')
        x = self.act1(x)
        self.print_range(x, 'act1')
        x = self.relu1(x, 0, 255)
        self.print_range(x, 'relu1')

        x = self.deconv2(x)
        self.print_range(x, 'conv2')
        x = self.act2(x)
        self.print_range(x, 'act2')
        x = self.relu2(x, 0, 255)
        self.print_range(x, 'relu2')

        x = self.deconv3(x)
        self.print_range(x, 'conv3')
        x = self.act3(x)
        self.print_range(x, 'act3')
        x = self.relu3(x, 0, 63)
        self.print_range(x, 'relu3_out')

        self.print_range(self.deconv1.weight, 'conv1_w')
        self.print_range(self.deconv1.bias, 'conv1_b')
        self.print_range(self.deconv2.weight, 'conv2_w')
        self.print_range(self.deconv2.bias, 'conv2_b')
        self.print_range(self.deconv3.weight, 'conv3_w')
        self.print_range(self.deconv3.bias, 'conv3_b')
        self.print_range(self.act1.c, 'act1_c')
        self.print_range(self.act2.c, 'act2_c')
        self.print_range(self.act3.c, 'act3_c')

        return x

class QZDecoder_GG18Clip(QZDecoder_GG18):
    '''
    Wrap forward to support pre clip
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.relu0 = GGQReLU()

    def forward(self, x):
        x = self.relu0(x, 0, 255)
        self.print_range(x, 'relu0')
        return super().forward(x)

class QZDecoderGG18K5(QZDecoder_GG18):
    """
    qzdecoder model for gg18q. kelnel size strictly consists with gg19i
    """
    
    def build(self):
        if not self.stage_widths:
            self.stage_widths = [self.num_filters] * 2
        self.deconv1 = self.deconv(
                self.num_filters, self.stage_widths[0], (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros")
        self.deconv2 = self.deconv(
                self.stage_widths[0], self.stage_widths[1], (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros")
        self.deconv3 = self.deconv(
                self.stage_widths[1], self.in_channels, (3, 3), stride=1, padding=1,
                bias=True, padding_mode="zeros")
                
        # int32 signed accumulator -> int8 unsigned act
        # int32 unsigned c in GG19I is like the layer-wise ema mag
        self.round_func = Round.apply
        #self.act0 = BpAct(self.num_filters, 8)
        self.act1 = self.Act(self.stage_widths[0], 8)
        self.act2 = self.Act(self.stage_widths[1], 8)
        self.act3 = self.Act(self.in_channels, 8)

        self.relu1 = GGQReLU()
        self.relu2 = GGQReLU()
        self.relu3 = GGQReLU()

class QZDecoderGG18C(QZDecoder_GG18):
    """
    qzdecoder model for gg18cq
    """

    def build(self):
        self.deconv1 = self.deconv(
                self.num_filters, self.num_filters, (4, 4), stride=2, padding=1,
                bias=True, padding_mode="zeros")
        self.deconv2 = self.deconv(
                self.num_filters, int(self.num_filters * 1.5), (4, 4), stride=2, padding=1,
                bias=True, padding_mode="zeros")
        self.deconv3 = self.deconv(
                int(self.num_filters * 1.5), self.in_channels * 2, (3, 3), stride=1, padding=1,
                bias=True, padding_mode="zeros")

        # int32 signed accumulator -> int8 unsigned act
        # int32 unsigned c in GG19I is like the layer-wise ema mag
        self.round_func = Round.apply
        #self.act0 = BpAct(self.num_filters, 8)
        self.act1 = self.Act(self.num_filters, 8)
        self.act2 = self.Act(int(self.num_filters * 1.5), 8)
        self.act3 = self.Act(self.in_channels * 2, 8)

        self.relu = GGQReLU()

    def forward(self, x):
        self.print_range(x, 'input')

        x = self.deconv1(x)
        self.print_range(x, 'conv1')
        x = self.act1(x)
        self.print_range(x, 'act1')
        x = self.relu(x, 0, 255)
        self.print_range(x, 'relu')

        x = self.deconv2(x)
        self.print_range(x, 'conv2')
        x = self.act2(x)
        self.print_range(x, 'act2')
        x = self.relu(x, 0, 255)
        self.print_range(x, 'relu')

        x = self.deconv3(x)
        self.print_range(x, 'conv3')
        x = self.act3(x)
        self.print_range(x, 'act3')
        x = self.relu(x, 0, 255)
        self.print_range(x, 'relu_out')

        self.print_range(self.deconv1.weight, 'conv1_w')
        self.print_range(self.deconv1.bias, 'conv1_b')
        self.print_range(self.deconv2.weight, 'conv2_w')
        self.print_range(self.deconv2.bias, 'conv2_b')
        self.print_range(self.deconv3.weight, 'conv3_w')
        self.print_range(self.deconv3.bias, 'conv3_b')
        self.print_range(self.act1.c, 'act1_c')
        self.print_range(self.act2.c, 'act2_c')
        self.print_range(self.act3.c, 'act3_c')

        return x




class NoBpQZDecoder_GG18(QZDecoder_GG18):
        Act = ProAct


class NoBpQZDecoder_GG18_old(nn.Module):
    def __init__(self, in_channels=3, out_channels=192, deconv="deconv", lambda4s=[1]):
        super(NoBpQZDecoder_GG18_old, self).__init__()
        self.in_channels = in_channels
        self.num_filters = out_channels
        self.caffe_channels = out_channels

        self.lambda4s = lambda4s
        self.sampled_lambda = 1

        if deconv == 'deconv':
            self.deconv = QConvTranspose2d

        self._layers = nn.ModuleList([])
        self.build()

        self.debug = True
        self.quant = True

    def build(self):
        self.deconv1 = self.deconv(
                self.num_filters, self.num_filters, (4, 4), stride=2, padding=1,
                bias=True, padding_mode="zeros")
        self.deconv2 = self.deconv(
                self.num_filters, self.num_filters, (4, 4), stride=2, padding=1,
                bias=True, padding_mode="zeros")
        self.deconv3 = self.deconv(
                self.num_filters, self.in_channels, (3, 3), stride=1, padding=1,
                bias=True, padding_mode="zeros")

        # int32 signed accumulator -> int8 unsigned act
        # int32 unsigned c in GG19I is like the layer-wise ema mag
        self.round_func = Round.apply
        #self.act0 = NoBpAct(self.num_filters, 8) # NoBp 8
        self.act1 = NoBpAct(self.num_filters, 8) # 8
        self.act2 = NoBpAct(self.num_filters, 8) # 8
        self.act3 = NoBpAct(self.in_channels, 6) # 6

        self.relu = QReLU()

    def print_range(self, x, info):
        if self.debug and not self.training:
            print('{} max: {}'.format(info, x.max()), flush=True)
            print('{} min: {}'.format(info, x.min()), flush=True)
            # print('{} mode: {}'.format(info, x.mode()), flush=True)

    def forward(self, x):
        # x = self.round_func(x, 8)
        self.print_range(x, 'input')
        #x = self.act0(x)
        #self.print_range(x, 'act0')
        #self.print_range(self.act0.x_min, 'x_min')
        #self.print_range(self.act0.x_max, 'x_max')
        #x = self.relu(x)
        #self.print_range(x, 'relu')

        x = self.deconv1(x)
        self.print_range(x, 'conv1')
        x = self.act1(x)
        self.print_range(x, 'act1')
        x = self.relu(x, 0, 255)
        self.print_range(x, 'relu')

        x = self.deconv2(x)
        self.print_range(x, 'conv2')
        x = self.act2(x)
        self.print_range(x, 'act2')
        x = self.relu(x, 0, 255)
        self.print_range(x, 'relu')

        x = self.deconv3(x)
        self.print_range(x, 'conv3')
        x = self.act3(x)
        self.print_range(x, 'act3')
        x = self.relu(x, 0, 63)
        self.print_range(x, 'relu_out')

        self.print_range(self.deconv1.weight, 'conv1_w')
        self.print_range(self.deconv1.bias, 'conv1_b')
        self.print_range(self.deconv2.weight, 'conv2_w')
        self.print_range(self.deconv2.bias, 'conv2_b')
        self.print_range(self.deconv3.weight, 'conv3_w')
        self.print_range(self.deconv3.bias, 'conv3_b')

        return x

class QZDecoderUpsample(nn.Module):
    Act = GGBpAct

    def __init__(self, in_channels=3, out_channels=192, conv="signal", upmode='bilinear', stage_widths=None, lambda4s=[1]):
        super(QZDecoderUpsample, self).__init__()
        self.in_channels = in_channels
        self.num_filters = out_channels
        self.caffe_channels = out_channels

        # stage_widths for num_filters of each layer
        self.stage_widths = stage_widths

        self.lambda4s = lambda4s
        self.sampled_lambda = 1

        if conv == 'signal':
            self.conv = GGQConv2d

        if upmode == 'pixelshuffle':
            self.upsample = nn.PixelShuffle(2)
            self.channel_scale = 2**2
        else:
            self.upsample = nn.Upsample(mode=upmode, scale_factor=2)
            self.channel_scale = 1

        self._layers = nn.ModuleList([])
        self.build()

        self.debug = True
        self.quant = True

    def build(self):
        if not self.stage_widths:
            self.stage_widths = [self.num_filters] * 2
        self.conv1 = nn.Sequential(self.conv(self.num_filters, self.stage_widths[0], (5, 5), stride=1, padding=2,
                                        bias=True, padding_mode="zeros"), self.upsample)
        self.conv2 = nn.Sequential(self.conv(
                self.stage_widths[0] // self.channel_scale, self.stage_widths[1], (5, 5), stride=1, padding=2,
                bias=True, padding_mode="zeros"), self.upsample)
        self.conv3 = self.conv(
                self.stage_widths[1] // self.channel_scale, self.in_channels, (3, 3), stride=1, padding=1,
                bias=True, padding_mode="zeros")

        # int32 signed accumulator -> int8 unsigned act
        # int32 unsigned c in GG19I is like the layer-wise ema mag
        self.round_func = Round.apply
        #self.act0 = BpAct(self.num_filters, 8)
        self.act1 = self.Act(self.stage_widths[0] // self.channel_scale, 8)
        self.act2 = self.Act(self.stage_widths[1] // self.channel_scale, 8)
        self.act3 = self.Act(self.in_channels, 8)

        self.relu = GGQReLU()

    def print_range(self, x, info):
        if self.debug and not self.training:
            print('{} max: {}'.format(info, x.max()), flush=True)
            print('{} min: {}'.format(info, x.min()), flush=True)
            # print('{} mode: {}'.format(info, x.mode()), flush=True)

    def forward(self, x):
        self.print_range(x, 'input')
        x = self.conv1(x)
        self.print_range(x, 'conv1')
        x = self.act1(x)
        self.print_range(x, 'act1')
        x = self.relu(x, 0, 255)
        self.print_range(x, 'relu')

        x = self.conv2(x)
        self.print_range(x, 'conv2')
        x = self.act2(x)
        self.print_range(x, 'act2')
        x = self.relu(x, 0, 255)
        self.print_range(x, 'relu')

        x = self.conv3(x)
        self.print_range(x, 'conv3')
        x = self.act3(x)
        self.print_range(x, 'act3')
        x = self.relu(x, 0, 63)
        self.print_range(x, 'relu_out')

        self.print_range(self.conv1[0].weight, 'conv1_w')
        self.print_range(self.conv1[0].bias, 'conv1_b')
        self.print_range(self.conv2[0].weight, 'conv2_w')
        self.print_range(self.conv2[0].bias, 'conv2_b')
        self.print_range(self.conv3.weight, 'conv3_w')
        self.print_range(self.conv3.bias, 'conv3_b')
        self.print_range(self.act1.c, 'act1_c')
        self.print_range(self.act2.c, 'act2_c')
        self.print_range(self.act3.c, 'act3_c')

        return x