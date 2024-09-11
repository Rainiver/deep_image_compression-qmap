from torch import nn
from PIL import Image
import numpy as np
import torch
from random import randint


class change_format(nn.Module):
    def __init__(self, format, quality):
        super(change_format, self).__init__()
        self.format = format
        self.quality = quality

    def forward(self, x, rank):
        if self.quality == "NONE":
            quality = randint(30, 90)
        else:
            quality = self.quality
        shape = x.shape
        if self.format == "NONE":
            return x
        elif self.format == "JPEG":
            x_ori = x.clone()
            x = (x.transpose(0, 1) * 255).int().clamp(0, 255)
            x = x.reshape(x.shape[0], -1, x.shape[3])
            x = x.transpose(0, 1).transpose(1, 2).contiguous()
            x = x.cpu()
            x = x.numpy()
            x = Image.fromarray(np.uint8(x))
            b, g, r = x.split()
            x = Image.merge("RGB", (r, g, b))
            x.save("change_format" + str(rank) + "." + self.format, quality=self.quality)
            x = Image.open("change_format" + str(rank) + "." + self.format)
            r, g, b = x.split()
            x = Image.merge("RGB", (b, g, r))
            x = np.array(x)
            x = torch.from_numpy(x).reshape([shape[0], -1, x.shape[1], x.shape[2]])
            x = x.transpose(2, 3).transpose(1, 2).float() / 255
            # print(rank, (x_ori.cpu() - x).mean())
            return x.cuda() if torch.cuda.is_available() else x


if __name__ == '__main__':
    x = Image.open('test.png')
    x = np.array(x)
    x = torch.from_numpy(x)
    print(x.shape, x.max(), x.min())
    x = torch.randn([2, 3, 64, 64])
    test = change_format("JPEG", 75)
    x = test(x)
    print(x.shape)
