from .SSIM_Loss import ssim
import torch

try:
    import spring.linklink as link
except:
    link = None


def _dssim(x, y):
    return 0.5 * (1 - ssim(x, y, window_size=8, size_average=False))  # n


def _update_dssim_baseline(new_val, avg_d):
    D = avg_d
    assert link, ImportError('to use dssim, linklink must be avaliable')
    link.allreduce(new_val)
    new_val /= link.get_world_size()
    D = 0.99 * D + 0.01 * new_val
    return D


def dssim(x, y, avg_d, update_avg_d=True):
    assert x.shape == y.shape
    n, c, h, w = x.shape
    assert h % 8 == 0
    assert w % 8 == 0
    nh = h // 8
    nw = w // 8
    x = x.reshape(n, c, nh, 8, nw, 8).permute(0, 2, 4, 1, 3, 5).reshape(-1, c, 8, 8)
    y = y.reshape(n, c, nh, 8, nw, 8).permute(0, 2, 4, 1, 3, 5).reshape(-1, c, 8, 8)

    with torch.no_grad():
        ds = _dssim(x, y)
        if update_avg_d:
            avg_d = _update_dssim_baseline(ds.mean(), avg_d)
        w = ds / avg_d
        w = w.detach()
    l1 = torch.nn.L1Loss(reduce=False)(x, y)  # n, c, 8, 8
    w = w.reshape(-1, 1, 1, 1)
    loss = torch.mean(l1 * w)
    return loss, avg_d


class DSSIMLoss(torch.nn.Module):
    """
    DSSIM loss proposed in:
    Improved Lossy Image Compression with Priming and Spatially Adaptive Bit Rates for Recurrent Networks
    """

    def __init__(self):
        super().__init__()
        self.register_buffer('avg_d', torch.ones(1))

    def forward(self, x, y):
        loss, self.avg_d = dssim(x, y, self.avg_d, update_avg_d=self.training)
        return loss
