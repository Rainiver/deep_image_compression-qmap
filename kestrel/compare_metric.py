from argparse import ArgumentParser
import cv2, os, sys
sys.path.append("..")
from torchvision import transforms
from losses.PSNR_Loss import Loss as PSNR
from losses.SSIM_Loss import msssim

parser = ArgumentParser()
parser.add_argument('arg1', type=str, help='origin image path')
parser.add_argument('arg2', type=str, help='recon image path')
parser.add_argument('arg3', type=str, help='compress bin path')
args = parser.parse_args()

oimg = transforms.ToTensor()(cv2.imread(args.arg1)).unsqueeze(0)
rimg = transforms.ToTensor()(cv2.imread(args.arg2)).unsqueeze(0)
assert oimg.shape == rimg.shape, "origin shape must be equal to recon !!"
ms_ssim = msssim(oimg, rimg)
psnr = PSNR()(oimg, rimg)
bits = os.path.getsize(args.arg3) * 8
_, c, h, w = oimg.size()
bpp = bits / (h * w)

print(" image: %s \n bpp: %f \n h * w * c: (%d * %d * %d) \n psnr: %f \n ms_ssim: %f \n" % (args.arg1, bpp, h, w, c, psnr, ms_ssim))


