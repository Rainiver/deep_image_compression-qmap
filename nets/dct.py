import torch.nn as nn
import torch
import math
import os
from utils import color_space
from quant.quantizator import STEQuant, OrderQuant


class QUANT_TABLE():
    def __init__(self):
        table_path = "../jpeg/jpeg_decoder/quant_table.txt"
        if not os.path.exists(table_path):
            print("please compile and run get_QT.py first")
            assert 0
        f = open(table_path, "r")
        self.tables = [[]]
        zigzag_index = [
            0, 1, 5, 6, 14, 15, 27, 28,
            2, 4, 7, 13, 16, 26, 29, 42,
            3, 8, 12, 17, 25, 30, 41, 43,
            9, 11, 18, 24, 31, 40, 44, 53,
            10, 19, 23, 32, 39, 45, 52, 54,
            20, 22, 33, 38, 46, 51, 55, 60,
            21, 34, 37, 47, 50, 56, 59, 61,
            35, 36, 48, 49, 57, 58, 62, 63]
        lines = f.readlines()
        for i in range(100):
            table = torch.zeros([16, 8])
            if torch.cuda.is_available():
                table = table.cuda()
            table.requires_grad = False
            for j in range(16):
                line = lines[i * 16 + j].split(" ")
                for k in range(8):
                    table[j][k] = int(line[k])
            y_table = table[0:8, :].view(-1)[zigzag_index].view(8, 8)
            c_table = table[8:16, :].view(-1)[zigzag_index].view(8, 8)
            self.tables.append([y_table, c_table])
        f.close()

    def get_table(self, quality):
        return self.tables[quality]

class QUANT_TABLE2():
    '''
    Determining JPEG Image Standard Quality Factor from the Quantization Tables
    '''
    def __init__(self):
        self.y_table = torch.tensor(
            [[16, 11, 10, 16, 24, 40, 51, 61],
             [12, 12, 14, 19, 26, 58, 60, 55],
             [14, 13, 16, 24, 40, 57, 69, 56],
             [14, 17, 22, 29, 51, 87, 80, 62],
             [18, 22, 37, 56, 68, 109, 103, 77],
             [24, 35, 55, 64, 81, 104, 113, 92],
             [49, 64, 78, 87, 103, 121, 120, 101],
             [72, 92, 95, 98, 112, 100, 103, 99]], dtype=torch.float32)
        self.c_table = torch.full((8, 8), 99, dtype=torch.float32)
        self.c_table[:4, :4] = torch.tensor(([[17, 18, 24, 47],
                                             [18, 21, 26, 66],
                                             [24, 26, 56, 99],
                                             [47, 66, 99, 99]]), dtype=torch.float32)

        if torch.cuda.is_available():
            self.y_table = self.y_table.cuda()
            self.c_table = self.c_table.cuda()

    def get_table(self, quality):
        quality = min(max(quality, 1), 100)
        if quality < 50:
            s = 5000. / quality
        else:
            s = 200. - 2 * quality
        y_table = ((s * self.y_table + 50) / 100).floor().clamp(1, 255)
        c_table = ((s * self.c_table + 50) / 100).floor().clamp(1, 255)
        return [y_table, c_table]


class DCT_QUANT(nn.Module):
    def __init__(self, tag, quant_table):
        super(DCT_QUANT, self).__init__()
        self.tag = tag
        self.quant_table = quant_table

    def forward(self, x, quality, channel):
        if channel == "Y":
            table = self.quant_table.get_table(quality)[0]
        elif channel == "U" or channel == "V":
            table = self.quant_table.get_table(quality)[1]
        else:
            assert 0

        if self.tag == "round":
            quant = STEQuant()
        elif self.tag == "order":
            quant = OrderQuant()
        else:
            assert 0

        return quant(x / table)


class DCT_DEQUANT(nn.Module):
    def __init__(self, tag, quant_table):
        super(DCT_DEQUANT, self).__init__()
        self.tag = tag
        self.quant_table = quant_table

    def forward(self, x, quality, channel):
        if channel == "Y":
            table = self.quant_table.get_table(quality)[0]
        elif channel == "U" or channel == "V":
            table = self.quant_table.get_table(quality)[1]
        else:
            assert 0
        x = x * table
        return x


class DCT(nn.Module):
    def __init__(self, block_size=8, quantization="round", dct_norm=True, quality=75):
        super(DCT, self).__init__()
        self.quant_table = QUANT_TABLE2()
        self.block_size = block_size
        self.quantizator = DCT_QUANT(quantization, self.quant_table)
        self.dequantizator = DCT_DEQUANT(quantization, self.quant_table)
        self.mat = torch.zeros([block_size, block_size])
        self.imat = torch.zeros([block_size, block_size])
        if torch.cuda.is_available():
            self.mat = self.mat.cuda()
            self.imat = self.imat.cuda()
        self.mat.requires_grad = False
        self.imat.requires_grad = False
        self.dct_norm = dct_norm
        for i in range(block_size):
            for j in range(block_size):
                if i == 0:
                    # 1/sqrt(m)
                    self.mat[i, j] = 1. / math.sqrt(block_size)
                else:
                    # sqrt(2/m)*cos(pi(2j+1)i/2m)
                    self.mat[i, j] = math.sqrt(2. / block_size) * \
                                     math.cos((math.pi * (2 * j + 1) * i) / (2 * block_size))
        self.imat = self.mat.transpose(0, 1)
        self.quality = quality

    def dct_one_channel(self, x, channel, quality):
        # split to blocks
        blk2 = x.shape[2] // self.block_size
        blk3 = x.shape[3] // self.block_size
        x = torch.cat(x.split(self.block_size, 3), 1)
        x = torch.cat(x.split(self.block_size, 2), 1)
        # dct
        x = torch.matmul(self.mat, x)
        x = torch.matmul(x, self.imat)
        # quant
        x = self.quantizator(x, quality=quality, channel=channel)
        # reset shape
        x = torch.cat(x.split(blk3, 1), 2)
        x = torch.cat(x.split(1, 1), 3)
        return x

    def forward(self, x):
        if (x.shape[2] % self.block_size != 0) or (x.shape[3] % self.block_size != 0):
            assert 0

        # change to YUV
        x = color_space.bgr_to_yuv(x)
        Y, U, V = color_space.yuv_to_yuv420(x, 'avg')

        #[0, 1] -> [-128, 127]
        Y = (Y * 255) - 128
        U = (U * 255) - 128
        V = (V * 255) - 128

        # dct & quant
        Y = self.dct_one_channel(Y, "Y", self.quality)
        U = self.dct_one_channel(U, "U", self.quality)
        V = self.dct_one_channel(V, "V", self.quality)

        if self.dct_norm:
            Y /= 255
            U /= 255
            V /= 255
        return [Y, U, V]


class IDCT(DCT):
    def idct_one_channel(self, x, channel, quality):
        # split to blocks
        blk2 = x.shape[2] // self.block_size
        blk3 = x.shape[3] // self.block_size
        x = torch.cat(x.split(self.block_size, 3), 1)
        x = torch.cat(x.split(self.block_size, 2), 1)
        # dequant
        x = self.dequantizator(x, quality=quality, channel=channel)
        # idct
        x = torch.matmul(self.imat, x)
        x = torch.matmul(x, self.mat)
        # reset shape
        x = torch.cat(x.split(blk3, 1), 2)
        x = torch.cat(x.split(1, 1), 3)
        return x

    def forward(self, Y, U, V):
        if self.dct_norm:
            Y *= 255
            U *= 255
            V *= 255
        # split channels
        if (U.shape[2] % self.block_size != 0) or (U.shape[3] % self.block_size != 0):
            assert 0

        # dequant
        Y = self.idct_one_channel(Y, "Y", self.quality)
        U = self.idct_one_channel(U, "U", self.quality)
        V = self.idct_one_channel(V, "V", self.quality)

        ##[-128, 127] -> [0, 1]
        Y = (Y + 128) / 255
        U = (U + 128) / 255
        V = (V + 128) / 255

        # change to bgr
        x = color_space.yuv420_to_yuv([Y, U, V], 'bilinear')
        x = color_space.yuv_to_bgr(x)
        return x

class DCT_FLATTEN(nn.Module):
    '''
    flat via frequency
    [n, c, h, w] -> [n, c * 64, h // 8, w // 8]
    '''
    def __init__(self):
        super(DCT_FLATTEN, self).__init__()

    def flatten(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, h // 8, 8, w // 8, 8)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(n, c * 64, h // 8, w // 8)
        return x

    def forward(self, Y, U, V):
        Y = self.flatten(Y)
        U = self.flatten(U)
        V = self.flatten(V)

        return [Y, U, V]

class DCT_Net(nn.Module):
    '''
    pixel domain -> frequency domain via net
    '''
    def __init__(self, block_size=8, quantization="round", dct_norm=True, quality=75):
        super(DCT_Net, self).__init__()
        self.block_size = block_size
        self.dct_norm = dct_norm
        self.quality = quality
        self.quant_table = QUANT_TABLE2()
        self.quantizator = DCT_QUANT(quantization, self.quant_table)
        self.lumatrans = nn.Sequential(nn.Conv2d(1, 96, 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        nn.Conv2d(96, 96, 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        nn.Conv2d(96, 96, 3, 1, 1),
                                       nn.LeakyReLU(inplace=True),
                                        nn.Conv2d(96, 1, 3, 1, 1))
        self.chromatrans = nn.Sequential(nn.Conv2d(1, 96, 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        nn.Conv2d(96, 96, 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        nn.Conv2d(96, 96, 3, 1, 1),
                                       nn.LeakyReLU(inplace=True),
                                        nn.Conv2d(96, 1, 3, 1, 1))

    def block(self, x):
        '''
        [n, c, h, w] ->
        [n * h * w / (bls * bls), c, bls, bls]
        '''
        assert (not (x.shape[2] % self.block_size or
                     x.shape[3] % self.block_size != 0))
        # split to blocks
        x = torch.cat(x.split(self.block_size, 3), 0)
        x = torch.cat(x.split(self.block_size, 2), 0)
        return x

    def merge(self, x):
        '''
        [n * h * w / (bls * bls), c, bls, bls] ->
        [n, c, h, w]
        '''
        x = torch.cat(x.split((self.n * self.w) // self.block_size, 0), 2)
        x = torch.cat(x.split(self.n, 0), 3)
        return x

    def forward(self, x):
        if (x.shape[2] % self.block_size != 0) or (x.shape[3] % self.block_size != 0):
            assert 0

        # change to YUV
        x = color_space.bgr_to_yuv(x)
        Y, U, V = color_space.yuv_to_yuv420(x, 'avg')
        self.n, self.h, self.w = x.shape[0], x.shape[2], x.shape[3]

        #[0, 1] -> [-128, 127]
        Y = (Y * 255) - 128
        U = (U * 255) - 128
        V = (V * 255) - 128

        #block
        Y, U, V = list(map(self.block, [Y, U, V]))

        #net transform
        Y = self.lumatrans(Y)
        U = self.chromatrans(U)
        V = self.chromatrans(V)

        #quant
        Y = self.quantizator(Y, self.quality, 'Y')
        U = self.quantizator(U, self.quality, 'U')
        V = self.quantizator(V, self.quality, 'V')

        #merge
        Y, U, V = list(map(self.merge, [Y, U, V]))
        if self.dct_norm:
            Y /= 255
            U /= 255
            V /= 255

        return [Y, U, V]

class DCT_Param(DCT):
    '''
    replace constant dct metrix with learned param
    '''
    def __init__(self, **kwargs):
        super(DCT_Param, self).__init__(**kwargs)
        self.mat = nn.Parameter(self.mat, requires_grad=True)
        self.imat = self.mat.transpose(0, 1)



if __name__ == "__main__":
    import cv2
    import numpy
    from torchvision import transforms

    quality = 75
    qt = QUANT_TABLE()
    qt2 = QUANT_TABLE2()
    dct_flat = DCT_FLATTEN()
    print(qt.get_table(quality))
    print(qt2.get_table(quality))

    # dct = DCT(block_size=8, quality=quality)
    # idct = IDCT(block_size=8, quality=quality)
    #
    # a = cv2.imread("origin.png")
    # a = transforms.ToTensor()(a).unsqueeze(0).cuda()
    # a.requires_grad = True
    #
    #
    # b = dct(a)
    # c = idct(b).clamp(0, 1)
    # print((c - a).abs().mean())
    # print(c.requires_grad)
    # d=c.mean()
    # d.backward()
    #
    # f = dct_flat(b)
    # print(b.size(), f.size())
    # e = numpy.asarray(transforms.ToPILImage()(c.cpu().squeeze(0)))
    # cv2.imshow("test", numpy.array(e))
    # cv2.imwrite("test.png", e)
    # cv2.waitKey(0)
