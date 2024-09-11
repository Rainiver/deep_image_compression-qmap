import torch
import torch.nn as nn

from nets.layers import SignalConvTranspose2d
from nets.norm import GDN
from utils import color_space


class YUV_pre(nn.Module):
    def __init__(self, input_mode='bgr', use_gdn=True):
        super(YUV_pre, self).__init__()
        self.input_mode = input_mode
        if self.input_mode == 'bilinear':
            self.uv_layer = nn.UpsamplingBilinear2d(scale_factor=2)
        else:
            if self.input_mode == 'deconv':
                layers = [nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=True), nn.BatchNorm2d(1), nn.ReLU()]
                self.uv_layer = nn.Sequential(*layers)
            elif self.input_mode == '2deconv':
                layers = [nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True), nn.BatchNorm2d(1), nn.ReLU()]
                self.uv_layer = nn.Sequential(*layers)
            else:
                raise ValueError('unsupported yuv input mode: {}'.format(input_mode))
            """
            layers = [
                SignalConvTranspose2d(2, 2, (4, 4), stride=2, padding=1, bias=True,
                                      padding_mode="zeros", groups=groups),
                GDN(2) if use_gdn else nn.Identity(),
                nn.ReLU() if not use_gdn else nn.Identity(),
            ]
            """

    def to_yuv420(self, x):
        # TODO support different upsampling mode on rgb/yuv/yuv420
        if self.input_mode == 'bgr':
            ret = color_space.yuv_to_yuv420(color_space.bgr_to_yuv(x))
        elif self.input_mode == 'rgb':
            ret = color_space.yuv_to_yuv420(color_space.rgb_to_yuv(x))
        elif self.input_mode == 'yuv':
            ret = color_space.yuv_to_yuv420(x)
        elif self.input_mode == 'yuv420':
            return x.cuda() if torch.cuda.is_available() else x
        else:
            # default bgr
            ret = color_space.yuv_to_yuv420(color_space.bgr_to_yuv(x))
        if torch.cuda.is_available():
            for i in range(len(ret)):
                ret[i] = ret[i].cuda()
        return ret

    def forward(self, x):
        y, u, v = self.to_yuv420(x)
        if self.input_mode != "2deconv":
            u, v = self.uv_layer(u), self.uv_layer(v)
            return torch.cat([y, u, v], 1)
        else:
            uv = self.uv_layer(torch.cat([u, v], 1))
            return torch.cat([y, uv], 1)


def yuv_pre(mode):
    if mode == "NONE":
        return nn.Identity()  # yuv-pre is optional
    else:
        return YUV_pre(mode)


if __name__ == '__main__':
    import cv2
    import numpy as np
    from torchvision import transforms

    img = cv2.imread('../../test.png')
    img = np.array(img)
    x = transforms.ToTensor()(img)
    x = x.unsqueeze(0)
    x = x[:, :, 0:x.shape[2] // 2 * 2, 0:x.shape[3] // 2 * 2]
    test = YUV_pre(input_mode='bilinear').cuda()
    print(test(x).shape)
    print((x - color_space.yuv_to_bgr(test(x))).mean())
