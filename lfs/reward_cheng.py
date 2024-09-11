from scipy.interpolate import interp1d
import numpy as np
import torch

cheng_kodak = [
    0.098115338, 27.98974522,
    0.173912113, 29.82447267,
    0.299725056, 31.8515377,
    0.456783226, 33.87737664,
    0.679351191, 35.62697935,
    0.829, 36.654,
]

cheng_kodak_ms = [
    0.118258307, 13.312,
    0.271042242, 16.866,
    0.483132033, 19.787,
    0.842092671, 22.650,
]

bpp2msssim_db = cheng_kodak_ms
bpp2msssim_db.sort(key=lambda item: item[0])
bpp2msssim_db = np.array(bpp2msssim_db)
bpp2msssim_db_fn = interp1d(bpp2msssim_db[:, 0], bpp2msssim_db[:, 1], kind='cubic', fill_value="extrapolate")
msssim_db2bpp_fn = interp1d(bpp2msssim_db[:, 1], bpp2msssim_db[:, 0], kind='cubic', fill_value="extrapolate")

bpp2psnr = cheng_kodak
bpp2psnr.sort(key=lambda item: item[0])
bpp2psnr = np.array(bpp2psnr)
bpp2psnr_fn = interp1d(bpp2psnr[:, 0], bpp2psnr[:, 1], kind='cubic', fill_value="extrapolate")
psnr2bpp_fn = interp1d(bpp2psnr[:, 1], bpp2psnr[:, 0], kind='cubic', fill_value="extrapolate")


def reward_base_bpp(bpp, msssim, psnr):
    print("calc reward using cheng20 data", flush=True)
    bpp = np.array(bpp)
    msssim = np.array(msssim)
    psnr = np.array(psnr)

    bpp_psnr = psnr2bpp_fn(psnr)
    bpp_msssim = msssim_db2bpp_fn(-10 * np.log10(np.array(1 - msssim)))
    psnr_bpp_lower = bpp - bpp_psnr
    ssim_bpp_lower = bpp - bpp_msssim
    # psnr_bpp_lower = psnr_bpp_lower if psnr_bpp_lower > 0 else 0
    # ssim_bpp_lower = ssim_bpp_lower if ssim_bpp_lower > 0 else 0
    psnr_bpp_lower[psnr_bpp_lower < 0] = 0
    ssim_bpp_lower[ssim_bpp_lower < 0] = 0
    return - torch.from_numpy(psnr_bpp_lower + ssim_bpp_lower)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    b = np.linspace(0.1, 2.5, num=1000)
    # plt.plot(b, bpp2psnr_fn(b))
    # plt.plot(bpp2psnr[:, 0], bpp2psnr[:, 1], '--')
    # plt.show()
    # plt.plot(b, bpp2msssim_db_fn(b))
    # plt.plot(bpp2msssim_db[:, 0], bpp2msssim_db[:, 1], '--')
    # plt.show()
