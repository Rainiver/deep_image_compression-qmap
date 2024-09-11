from scipy.interpolate import interp1d
import numpy as np
import torch

"""
gg18_kodak = [
    [0.9981746, 43.3785167, 2.9088652],
    [0.9930721, 38.8171684, 1.2980584],
    [0.9886457, 36.7519638, 0.9681057],
    [0.9767518, 33.3974810, 0.5148655],
    [0.9707273, 32.3531376, 0.4206450],
    [0.9541093, 30.4043438, 0.2738766],
    [0.9314400, 28.5909536, 0.1711570],
    [0.9008735, 26.7703106, 0.1054196],
    [0.8750564, 25.7816394, 0.0766576],
    [0.8406371, 24.3447708, 0.0475065],
]

gg18_kodak_ms = [
    [0.9993513, 38.103312, 5.9906091],
    [0.9990048, 37.482783, 3.0371077],
    [0.9976443, 34.0293475, 1.7674544],
    [0.9942121, 31.2603755, 0.9393972],
    [0.9910011, 29.7320684, 0.6754617],
    [0.9875482, 28.5941652, 0.52329],
    [0.984288, 28.2557509, 0.4310472],
    [0.9800979, 27.855115, 0.3511132],
    # [0.9669029, 26.7230347, 0.2758077], #
    [0.947397, 25.4720116, 0.1679009],
    [0.905872, 23.9697913, 0.0921453],
    [0.8612083, 22.1519262, 0.0599526],
]
"""

# [0.907527, 27.106351, 0.115239],
# [0.936307, 28.679134, 0.185698],
# [0.958691, 30.616753, 0.301804],
# [0.972416, 32.554935, 0.468972],
# [0.982478, 34.58096, 0.686378],
# [0.988344, 36.720366, 0.966864],
# [0.992647, 38.80796, 1.307441],
# [0.995267, 40.79492, 1.727503],

# [0.923012, 24.725623, 0.092031],
# [0.953873, 26.331231, 0.166818],
# [0.972845, 27.836886, 0.281376],
# [0.981048, 28.710904, 0.398321],
# [0.989613, 30.425433, 0.650651],
# [0.994012, 32.4625, 0.996335],
# [0.996171, 33.54685, 1.377538],
# [0.99802, 35.394737, 2.03033],

# use gg tecnick result...

gg18_kodak = [

    [0.940467, 29.234396, 0.105457],
    [0.957177, 30.862438, 0.154802],
    [0.96959, 32.571271, 0.22837],
    [0.977247, 34.146224, 0.330983],
    [0.983751, 35.867773, 0.471083],
    [0.988089, 37.42029, 0.655272],
    [0.991795, 39.192136, 0.905265],
    [0.994664, 40.975431, 1.262145],

]

gg18_kodak_ms = [

    [0.947396, 26.761386, 0.084147],
    [0.966016, 29.009099, 0.136033],
    [0.977739, 31.078285, 0.213825],
    [0.984355, 32.614011, 0.310643],
    [0.990337, 34.598202, 0.491282],
    [0.994525, 36.514694, 0.748023],
    [0.997132, 38.140798, 1.094067],
    [0.998906, 40.37175, 1.659701],

]
bpp2msssim = [
    [i[2], i[0]] for i in gg18_kodak_ms
]
bpp2msssim.sort(key=lambda item: item[0])
bpp2msssim = np.array(bpp2msssim)
bpp2msssim_db = bpp2msssim
bpp2msssim_db[:, 1] = -10 * np.log10(np.array(1 - bpp2msssim[:, 1]))
bpp2msssim_fn = interp1d(bpp2msssim[:, 0], bpp2msssim[:, 1], kind='cubic', fill_value="extrapolate")
bpp2msssim_db_fn = interp1d(bpp2msssim_db[:, 0], bpp2msssim_db[:, 1], kind='cubic', fill_value="extrapolate")
msssim_db2bpp_fn = interp1d(bpp2msssim_db[:, 1], bpp2msssim_db[:, 0], kind='cubic', fill_value="extrapolate")

bpp2psnr = [
    [i[2], i[1]] for i in gg18_kodak
]
bpp2psnr.sort(key=lambda item: item[0])
bpp2psnr = np.array(bpp2psnr)
bpp2psnr_fn = interp1d(bpp2psnr[:, 0], bpp2psnr[:, 1], kind='cubic', fill_value="extrapolate")
psnr2bpp_fn = interp1d(bpp2psnr[:, 1], bpp2psnr[:, 0], kind='cubic', fill_value="extrapolate")


def reward_base_bpp(bpp, msssim, psnr, *args, **kwargs):
    bpp = np.array(bpp)
    msssim = np.array(msssim)
    psnr = np.array(psnr)

    bpp_psnr = psnr2bpp_fn(psnr)
    bpp_msssim = msssim_db2bpp_fn(-10 * np.log10(np.array(1 - msssim)))
    psnr_bpp_lower = bpp - bpp_psnr  # minimize the bpp increase vs selected baseline
    ssim_bpp_lower = bpp - bpp_msssim
    # psnr_bpp_lower = psnr_bpp_lower if psnr_bpp_lower > 0 else 0
    # ssim_bpp_lower = ssim_bpp_lower if ssim_bpp_lower > 0 else 0
    psnr_bpp_lower[psnr_bpp_lower < 0] = 0
    ssim_bpp_lower[ssim_bpp_lower < 0] = 0
    return - torch.from_numpy(psnr_bpp_lower + ssim_bpp_lower)  # maximize


def reward_base_bpp_pow2(bpp, msssim, psnr, *args, **kwargs):
    bpp = np.array(bpp)
    msssim = np.array(msssim)
    psnr = np.array(psnr)

    bpp_psnr = psnr2bpp_fn(psnr)
    bpp_msssim = msssim_db2bpp_fn(-10 * np.log10(np.array(1 - msssim)))
    psnr_bpp_lower = bpp - bpp_psnr
    ssim_bpp_lower = bpp - bpp_msssim

    ans = - torch.from_numpy(psnr_bpp_lower ** 2 + ssim_bpp_lower ** 2)

    if 'req_more' in kwargs and kwargs['req_more']:
        return {
            'ans': ans,
            'psnr_bpp_lower': torch.from_numpy(psnr_bpp_lower),
            'ssim_bpp_lower': torch.from_numpy(ssim_bpp_lower)
        }
    else:
        return ans


def reward_base_bpp_pow2_ms(bpp, msssim, psnr, *args, **kwargs):
    bpp = np.array(bpp)
    msssim = np.array(msssim)
    psnr = np.array(psnr)

    bpp_psnr = psnr2bpp_fn(psnr)
    bpp_msssim = msssim_db2bpp_fn(-10 * np.log10(np.array(1 - msssim)))
    psnr_bpp_lower = bpp - bpp_psnr
    ssim_bpp_lower = bpp - bpp_msssim * 0.8
    ans = - torch.from_numpy(psnr_bpp_lower ** 2 + ssim_bpp_lower ** 2)

    if 'req_more' in kwargs and kwargs['req_more']:
        return {
            'ans': ans,
            'psnr_bpp_lower': torch.from_numpy(psnr_bpp_lower),
            'ssim_bpp_lower': torch.from_numpy(ssim_bpp_lower)
        }
    else:
        return ans



def reward_base_bpp_pow2_psnr(bpp, msssim, psnr, *args, **kwargs):
    bpp = np.array(bpp)
    msssim = np.array(msssim)
    psnr = np.array(psnr)

    bpp_psnr = psnr2bpp_fn(psnr)
    bpp_msssim = msssim_db2bpp_fn(-10 * np.log10(np.array(1 - msssim)))
    psnr_bpp_lower = bpp - bpp_psnr * 0.8
    ssim_bpp_lower = bpp - bpp_msssim

    ans = - torch.from_numpy(psnr_bpp_lower ** 2 + ssim_bpp_lower ** 2)

    if 'req_more' in kwargs and kwargs['req_more']:
        return {
            'ans': ans,
            'psnr_bpp_lower': torch.from_numpy(psnr_bpp_lower),
            'ssim_bpp_lower': torch.from_numpy(ssim_bpp_lower)
        }
    else:
        return ans


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    b = np.linspace(0.1, 2.5, num=1000)
    # plt.plot(b, bpp2psnr_fn(b))
    # plt.plot(bpp2psnr[:, 0], bpp2psnr[:, 1], '--')
    # plt.show()
    # plt.plot(b, bpp2msssim_db_fn(b))
    plt.plot(bpp2msssim_db[:, 0], bpp2msssim_db[:, 1], '--o')
    plt.show()
