from scipy.interpolate import interp1d
import numpy as np
import torch

gg18_kodak = [
    [0.9916744, 38.0582768, 1.0327098],
    [0.9886431, 36.6523088, 0.7818519],
    [0.9856153, 35.5762403, 0.63979],
    [0.9779055, 33.7737457, 0.4402618],
    [0.966106, 31.8614974, 0.2893219],
    [0.944394, 29.7620742, 0.1619424],
    [0.9145696, 28.1669612, 0.0957031],
    [0.8881734, 26.7903988, 0.0679075],
]

gg18_kodak_ms = [
    [0.9956765, 32.3579404, 1.0332286],
    [0.991324, 30.0418915, 0.5893563],
    [0.9837829, 28.5075152, 0.3475223],
    [0.9670103, 26.9859228, 0.1827045],
    [0.9041808, 23.5036151, 0.0622457],
    [0.8167177, 21.3982065, 0.032163],
]

bpp2msssim = [
    [i[2], i[0]] for i in gg18_kodak_ms
]
bpp2msssim.sort(key=lambda item: item[0])
bpp2msssim = np.array(bpp2msssim)
bpp2msssim_db = bpp2msssim
bpp2msssim_db[:, 1] = -10 * np.log10(np.array(1 - bpp2msssim[:, 1]))
bpp2msssim_fn = interp1d(bpp2msssim[:, 0], bpp2msssim[:, 1], kind='cubic', fill_value="extrapolate")
bpp2msssim_db_fn = interp1d(bpp2msssim_db[:, 0], bpp2msssim_db[:, 1], kind='quadratic', fill_value="extrapolate")
msssim_db2bpp_fn = interp1d(bpp2msssim_db[:, 1], bpp2msssim_db[:, 0], kind='quadratic', fill_value="extrapolate")

bpp2psnr = [
    [i[2], i[1]] for i in gg18_kodak
]
bpp2psnr.sort(key=lambda item: item[0])
bpp2psnr = np.array(bpp2psnr)
bpp2psnr_fn = interp1d(bpp2psnr[:, 0], bpp2psnr[:, 1], kind='quadratic', fill_value="extrapolate")
psnr2bpp_fn = interp1d(bpp2psnr[:, 1], bpp2psnr[:, 0], kind='quadratic', fill_value="extrapolate")


def reward_base_bpp(bpp, msssim, psnr):
    print("using reward: base cheng plus (pytorch)", flush=True)
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

    b = np.linspace(0.01, 2.5, num=1000)
    plt.plot(b, bpp2psnr_fn(b))
    plt.plot(bpp2psnr[:, 0], bpp2psnr[:, 1], '--')
    plt.show()
    plt.plot(b, bpp2msssim_db_fn(b))
    plt.plot(bpp2msssim_db[:, 0], bpp2msssim_db[:, 1], '--')
    plt.show()
