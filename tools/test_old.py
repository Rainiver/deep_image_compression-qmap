import os

import cv2
import numpy as np
import torch
from torchvision import transforms

from losses import Lp_Loss
from losses import pytorch_ssim
from losses.PSNR_Loss import Loss as PSNR
from nets.decoder import decoders
from nets.encoder import encoders


def pad(x, base=32):
    b, c, h, w = x.shape
    H = (h // base + 1) * base
    W = (w // base + 1) * base
    y = torch.zeros([b, c, H, W])
    y[:, :, H - h:H, 0:w] = x.clone()
    y[:, :, 0:h, W - w:W] = x.clone()
    if x.dtype == torch.int:
        y = y.int()
    y[:, :, 0:h, 0:w] = x
    return y


def load(encoder, decoder, log_dir, epoch):
    if epoch < 0:
        # use the latest epoch
        checkpoints = os.listdir(log_dir)
        checkpoints = [f for f in checkpoints if f.startswith('encoder_epoch-') and f.endswith('.pth')]
        checkpoints = [int(f[14:-4]) for f in checkpoints]
        epoch = sorted(checkpoints)[-1]

    if encoder:
        pre_encoder = torch.load(os.path.join(log_dir, 'encoder_epoch-{}.pth'.format(epoch)), map_location='cpu')
        now_encoder = encoder.state_dict()
        pre_encoder = {k[7:]: v for k, v in pre_encoder.items()}
        now_encoder.update(pre_encoder)
        encoder.load_state_dict(now_encoder)

    if decoder:
        pre_decoder = torch.load(os.path.join(log_dir, 'decoder_epoch-{}.pth'.format(epoch)), map_location='cpu')
        now_decoder = decoder.state_dict()
        pre_decoder = {k[7:]: v for k, v in pre_decoder.items()}
        now_decoder.update(pre_decoder)
        decoder.load_state_dict(now_decoder)


s1 = torch.zeros(1)
s2 = torch.zeros(1)
num = 0

pic = []

encoder = None
decoder = None


def run(x):
    global pic, encoder, decoder, predictor
    from quantizator import quantizators
    from dequantizator import dequantizators
    print(x.shape)
    x.requires_grad = False
    quant = quantizators("RT", 6)
    dequant = dequantizators("RT", 6)
    with torch.no_grad():
        y = encoder(x)
        y = quant(y)
        p = y.squeeze(0).clone()
        p = torch.round(p * 255).int().cpu()
        p = np.array(p, dtype=np.uint8)
        p = p.reshape(y.shape[2] * 5, y.shape[3] * 2, 3)
        cv2.imwrite('test.jpg', p)
        y = dequant(y)
        x = decoder(y)
    return x


def main(img_path='../../3.png', base=32, log_dir='./logs/', epoch=1630):
    with torch.no_grad():
        global s1, s2, num, pic, encoder, decoder
        num += 1
        encoder = encoders("RT", out_channels=30)
        decoder = decoders("RT", out_channels=30)
        load(encoder, decoder, log_dir, epoch)
        encoder.eval()
        decoder.eval()
        img = cv2.imread(img_path)
        cv2.imwrite('origin.jpg', img)
        img = np.array(img)
        x = transforms.ToTensor()(img)
        x = x.unsqueeze(0)
        b, c, h, w = x.shape
        if x.shape[2] % base != 0 or x.shape[3] % base != 0:
            x = pad(x, base)
        b, c, H, W = x.shape
        x.requires_grad = False
        if torch.cuda.is_available():
            encoder = encoder.cuda()
            decoder = decoder.cuda()
            x = x.cuda()
        y = x.clone()[:, :, 0:h, 0:w]
        x = run(x)
        l1 = Lp_Loss.Loss(p=1)
        ss = pytorch_ssim.SSIM(window_size=11)
        psnr = PSNR()
        x = torch.clamp(x, 0, 1)
        x = x[:, :, 0:h, 0:w]
        s1 += ss(x, y)
        s2 += psnr(x, y)
        print(s1 / num, s2 / num)
        x = torch.round(x * 255).int().abs()
        x = x.detach().cpu().numpy()
        x = np.array(x, dtype=np.uint8)
        x = x.squeeze(0)
        x = np.swapaxes(x, 0, 2)
        x = np.swapaxes(x, 0, 1)
        cv2.imwrite('./output.png', x)


def test(data_dir="../../../dataset/compression/valid"):
    for maindir, subdir, file_name_list in os.walk(data_dir):
        for filename in file_name_list:
            print(filename)
            main(os.path.join(maindir, filename))


if __name__ == '__main__':
    main()
