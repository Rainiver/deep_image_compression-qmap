from nets.dct import DCT
from nets.pre.yuv_pre import yuv_pre
from nets.dct import DCT
from nets.pre.change_format import change_format
from nets.pre.dct_coef_pre import DCTCoefPre
from torch import nn


def pre(tag, **kwargs):
    if tag == "NONE":
        return nn.Identity()
    elif tag == "YUV":
        return yuv_pre(**kwargs)
    elif tag == "format":
        return change_format(**kwargs)
    elif tag == "dct":
        return DCT(**kwargs)
    elif tag == "dct-coef":
        return DCTCoefPre(**kwargs)
    else:
        raise ValueError(f"Unsupported tage: {tag}")
