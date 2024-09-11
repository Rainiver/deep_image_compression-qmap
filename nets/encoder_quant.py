import torch
import torch.nn as nn
import numpy as np
import math
from .layers_quant import QReLU, BpAct, NoBpAct, Round, QLeakyReLU, Clip, GGQReLU, GGBpAct, ProAct,\
    GGQConv2d, SymmetricActReparameterizer, AsymmetricActReparameterizer
from .norm import GDN, L1GDNNoPow

class QYEncoderGG18K5(nn.Module):
    Act = GGBpAct

    def __init__(self, in_channels=3, out_channels=192, normalization="gdn", conv="conv", stage_widths=None, lambda4s=[1]):
        super(QYEncoderGG18K5, self).__init__()
        self.in_channels = in_channels
        self.num_filters = out_channels
        self.caffe_channels = in_channels

        # stage_widths for num_filters of each layer
        self.stage_widths = stage_widths

        self.lambda4s = lambda4s
        self.sampled_lambda = 1

        if normalization == 'gdn':
            self.norm_layer = GDN
            self.activation_layer = lambda: None  # GDN do not need activation
        elif normalization == '1dn':
            self.norm_layer = L1GDNNoPow
            self.activation_layer = lambda: None
        elif normalization == 'qgdn':
            raise NotImplementedError

        if conv == 'conv':
            self.conv = GGQConv2d

        self._layers = nn.ModuleList([])
        self.build()

        self.debug = True
        self.quant = True

    def build(self):
        if not self.stage_widths:
            self.stage_widths = [self.num_filters] * 3
        self.conv1 = self.conv(
                self.in_channels, self.stage_widths[0], (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros")
        self.norm_layer1 = self.norm_layer(self.stage_widths[0])
        self.conv2 = self.conv(
                self.stage_widths[0], self.stage_widths[1], (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros")
        self.norm_layer2 = self.norm_layer(self.stage_widths[1])
        self.conv3 = self.conv(
                self.stage_widths[1], self.stage_widths[2], (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros")
        self.norm_layer3 = self.norm_layer(self.stage_widths[2])
        self.conv4 = self.conv(
                self.stage_widths[2], self.num_filters, (5, 5), stride=2, padding=2, 
                bias=True, padding_mode="zeros")

        # int32 signed accumulator -> int8 unsigned act
        # int32 unsigned c in GG19I is like the layer-wise ema mag
        self.round_func = Round.apply
        self.act_reparam = AsymmetricActReparameterizer.apply
        #self.act0 = BpAct(self.num_filters, 8)
        self.act1 = self.Act(self.stage_widths[0], 8)
        self.act2 = self.Act(self.stage_widths[1], 8)
        self.act3 = self.Act(self.stage_widths[2], 8)
        self.act4 = self.Act(self.num_filters, 8)

        self.relu = GGQReLU()

    def print_range(self, x, info):
        if self.debug and not self.training:
            print('{} max: {}'.format(info, x.max()), flush=True)
            print('{} min: {}'.format(info, x.min()), flush=True)
            # print('{} mode: {}'.format(info, x.mode()), flush=True)

    def forward(self, x):
        self.print_range(x, 'input')

        x = x * 255.0
        x = self.round_func(x, 0)
        
        x = self.conv1(x)
        self.print_range(x, 'conv1')
        x = self.act1(x)
        self.print_range(x, 'act1')
        x = self.norm_layer1(x)
        self.print_range(x, 'gdn1')
        x = self.act_reparam(x, 8)

        x = self.conv2(x)
        self.print_range(x, 'conv2')
        x = self.act2(x)
        self.print_range(x, 'act2')
        x = self.norm_layer2(x)
        self.print_range(x, 'gdn2')
        x = self.act_reparam(x, 8)

        x = self.conv3(x)
        self.print_range(x, 'conv3')
        x = self.act3(x)
        self.print_range(x, 'act3')
        x = self.norm_layer3(x)
        self.print_range(x, 'gdn3')
        x = self.act_reparam(x, 8)

        x = self.conv4(x)
        self.print_range(x, 'conv4')
        x = self.act4(x)
        self.print_range(x, 'act4')
        # x = self.relu(x)
        # self.print_range(x, 'relu')

        self.print_range(self.conv1.weight, 'conv1_w')
        self.print_range(self.conv1.bias, 'conv1_b')
        self.print_range(self.conv2.weight, 'conv2_w')
        self.print_range(self.conv2.bias, 'conv2_b')
        self.print_range(self.conv3.weight, 'conv3_w')
        self.print_range(self.conv3.bias, 'conv3_b')
        self.print_range(self.conv4.weight, 'conv4_w')
        self.print_range(self.conv4.bias, 'conv4_b')
        self.print_range(self.act1.c, 'act1_c')
        self.print_range(self.act2.c, 'act2_c')
        self.print_range(self.act3.c, 'act3_c')
        self.print_range(self.act4.c, 'act4_c')

        return x


class QZEncoderGG18K5(nn.Module):
    Act = GGBpAct

    def __init__(self, in_channels=3, out_channels=192, conv="conv", stage_widths=None, lambda4s=[1]):
        super(QZEncoderGG18K5, self).__init__()
        self.in_channels = in_channels
        self.num_filters = out_channels
        self.caffe_channels = in_channels

        # stage_widths for num_filters of each layer
        self.stage_widths = stage_widths

        self.lambda4s = lambda4s
        self.sampled_lambda = 1

        if conv == 'conv':
            self.conv = GGQConv2d

        self._layers = nn.ModuleList([])
        self.build()

        self.debug = True
        self.quant = True

    def build(self):
        if not self.stage_widths:
            self.stage_widths = [self.num_filters] * 2
        self.conv1 = self.conv(
                self.in_channels, self.stage_widths[0], (3, 3), stride=1, padding=1,
                bias=True, padding_mode="zeros")
        self.conv2 = self.conv(
                self.stage_widths[0], self.stage_widths[1], (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros")
        self.conv3 = self.conv(
                self.stage_widths[1], self.num_filters, (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros")

        # int32 signed accumulator -> int8 unsigned act
        # int32 unsigned c in GG19I is like the layer-wise ema mag
        self.round_func = Round.apply
        #self.act0 = BpAct(self.num_filters, 8)
        self.act1 = self.Act(self.stage_widths[0], 8)
        self.act2 = self.Act(self.stage_widths[1], 8)
        self.act3 = self.Act(self.num_filters, 8)

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
        # x = self.relu(x, 0, 255)
        # self.print_range(x, 'relu')

        self.print_range(self.conv1.weight, 'conv1_w')
        self.print_range(self.conv1.bias, 'conv1_b')
        self.print_range(self.conv2.weight, 'conv2_w')
        self.print_range(self.conv2.bias, 'conv2_b')
        self.print_range(self.conv3.weight, 'conv3_w')
        self.print_range(self.conv3.bias, 'conv3_b')
        self.print_range(self.act1.c, 'act1_c')
        self.print_range(self.act2.c, 'act2_c')
        self.print_range(self.act3.c, 'act3_c')

        return x