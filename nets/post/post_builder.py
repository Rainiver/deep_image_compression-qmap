from torch import nn
from nets.grdn import Generator_one2many_gd_rir_old
from nets.post.enhance import EDIC
from nets.post.usm import USM
from nets.post.unet import UNET
from nets.post.recnet import RecNet
from nets.dct import IDCT


def post(tag, **kwargs):
    if tag == 'grdn':
        return Generator_one2many_gd_rir_old(input_channel=3)
    elif tag == 'usm':
        return USM(channel=3, **kwargs)
    elif tag == 'edic':
        return EDIC()
    elif tag == 'unet':
        return UNET(in_channels=3)
    elif tag == 'idct':
        return IDCT(block_size=8)
    elif tag == 'recnet':
        return RecNet(**kwargs)
    elif tag == 'NONE':
        return nn.Identity()  # post model is optional
    else:
        raise ValueError('unsupported post tag:{}'.format(tag))
