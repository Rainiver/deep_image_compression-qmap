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

num = 0
L_origin = os.listdir(args.arg1)
L_recon = os.listdir(args.arg2)
L_diff = os.listdir(args.arg3)
L_origin.sort(key = lambda x : int(x.split('.')[0][-2:]))
L_recon.sort(key = lambda x : int(x.split('_')[0][-2:]))
L_diff.sort(key = lambda x : int(x.split('.')[0][-2:]))
L_ssim,L_psnr,L_bpp = [],[],[]
for origin_name,recon_name,diff_name in zip(L_origin,L_recon,L_diff):
    num += 1
    oimg = transforms.ToTensor()(cv2.imread(os.path.join(args.arg1,origin_name))).unsqueeze(0)
    rimg = transforms.ToTensor()(cv2.imread(os.path.join(args.arg2,recon_name))).unsqueeze(0)
    assert oimg.shape == rimg.shape, "origin shape must be equal to recon !!"
    ms_ssim = msssim(oimg, rimg)
    psnr = PSNR()(oimg, rimg)
    bits = os.path.getsize(os.path.join(args.arg3,diff_name)) * 8
    _, c, h, w = oimg.size()
    bpp = bits / (h * w)
    L_ssim.append(float(ms_ssim))
    L_psnr.append(float(psnr))
    L_bpp.append(float(bpp))
    print(" image: %s \n bpp: %f \n h * w * c: (%d * %d * %d) \n psnr: %f \n ms_ssim: %f \n" % (origin_name, bpp, h, w, c, psnr, ms_ssim))
print("averge_ms-ssim : ",sum(L_ssim) / num)
print("average_psnr : ",sum(L_psnr) / num)
print("average_bpp : ",sum(L_bpp) / num)


