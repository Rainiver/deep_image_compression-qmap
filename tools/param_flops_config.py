from torch import nn
from thop import profile, clever_format
from thop.vision.basic_hooks import count_convNd, zero_ops
from nets.layers import SignalConv2d, SignalConvTranspose2d, RdftParameterizer
from nets.layers_quant import GGQConvTranspose2d, ScaleReparameterizer, BiasReparameterizer
from nets.norm import L1GDNNoPow, Power
from nets.encoder import YEncoder_GG18, ZEncoder_GG18
from nets.decoder import YDecoder_GG18
from nets.decoder_quant import QZDecoder_GG18, GGQReLU, GGBpAct, QZDecoder_GG18Clip

custom_ops = \
    {SignalConv2d: count_convNd,
     SignalConvTranspose2d: count_convNd,
     GGQConvTranspose2d: count_convNd,
     L1GDNNoPow: L1GDNNoPow.count_ops,
     Power: zero_ops,
     nn.Sequential: zero_ops,
     nn.ModuleList: zero_ops,
     YEncoder_GG18: zero_ops,
     YDecoder_GG18: zero_ops,
     ZEncoder_GG18: zero_ops,
     QZDecoder_GG18: zero_ops,
     QZDecoder_GG18Clip: zero_ops,
     GGQReLU: zero_ops,
     GGBpAct: GGBpAct.count_ops,
     RdftParameterizer: zero_ops,
     ScaleReparameterizer: zero_ops,
     BiasReparameterizer: zero_ops
     }
