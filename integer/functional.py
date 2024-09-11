import math
from typing import Union, Tuple
from distutils.version import LooseVersion

import torch
import torch.nn.functional as F
from torch.autograd import Function, Variable

# from utils.distributed_utils import get_dist_interface
try:
    import spring.linklink as link
except:
    link = None

TensorT = torch.Tensor
ByteTensorT = (torch.ByteTensor, torch.cuda.ByteTensor, )

_is_0_4 = LooseVersion("0.4.0") <= LooseVersion(torch.__version__)
__all__ = ["Round", "Sign", "Percentile", "QuantGradFunc", "BiasedQuantFunc",
           "SymmetricQuantFunc", "SyncStatFunc", "LowPrecisionFloatFunc"]

if _is_0_4:
    try:
        import low_precision_fp as lowp
    except ModuleNotFoundError:
        pass


class Round(Function):
    """ Round operation with identity gradient. """
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, df):
        return df.clone()


class Sign(Function):
    """ Sign operation with identity gradient. """
    @staticmethod
    def forward(ctx, x):
        return torch.sign(x)

    @staticmethod
    def backward(ctx, df):
        return df.clone()


def _le(lb, ub, for_act=False):
    """ assert lb <= ub for scalar, scalar tensor and multi-dim tensor """
    if torch.is_tensor(lb) and torch.is_tensor(ub) and \
            lb.dim() > 0 and ub.dim() > 0:
        if not for_act:
            assert torch.sum(lb <= ub) == lb.size(0), f"lb: {lb}, ub: {ub}"
        else:
            assert torch.sum(lb <= ub) == lb.size(1), f"lb: {lb}, ub: {ub}"
    else:
        assert lb <= ub, f"lb: {lb}, ub: {ub}"


def nudge_scalar_min_max(k, lb, ub, use_Int=False):
    _le(lb, ub)
    # torch >= 0.4.0 supports scalar tensor
    if torch.is_tensor(lb) and torch.is_tensor(ub):
        _abs = torch.abs
        _round = torch.round
        _max = torch.max
    else:
        _abs = abs
        _round = round
        _max = max

    ub = _max(ub, lb + 1e-10)
    quant_min = 0.
    quant_max = 2. ** k - 1.
    if not use_Int:
        scale = (ub - lb) / (quant_max - quant_min)

        if lb > 0:
            nudged_zero_point = 0.
            scale = ub / (quant_max - quant_min)
        elif ub < 0:
            nudged_zero_point = 2. ** k - 1.
            scale = _abs(lb) / (quant_max - quant_min)
        else:
            nudged_zero_point = _round(_abs(lb) / scale)

        nudged_min = (quant_min - nudged_zero_point) * scale
        nudged_max = (quant_max - nudged_zero_point) * scale

    else:
        assert 1 == 0 # TODO: make c int
        def scale_with_int_c(x, scale):
            c = scale[0]
            idxes = scale[1]

        scale = (ub - lb, quant_max - quant_min)

        if lb > 0:
            nudged_zero_point = 0.
            scale = (ub, quant_max - quant_min)
        elif ub < 0:
            nudged_zero_point = 2. ** k - 1.
            scale = (_abs(lb), quant_max - quant_min)
        else:
            nudged_zero_point = _round(_abs(lb) / scale)

        nudged_min = (quant_min - nudged_zero_point) * scale
        nudged_max = (quant_max - nudged_zero_point) * scale



    return nudged_min, nudged_max, nudged_zero_point, scale


def nudge_channel_min_max(k, lb, ub):
    _le(lb, ub)
    torch.max(ub, lb + 1e-10, out=ub)
    quant_min = torch.zeros_like(lb)
    quant_max = torch.ones_like(lb).fill_(2. ** k - 1.)
    scale = (ub - lb).div_(quant_max)

    nudged_zero_point = lb.abs().div_(scale).round_()
    # lb > 0
    lb_mask = lb > 0
    nudged_zero_point[lb_mask] = 0
    scale[lb_mask] = ub.div(quant_max)[lb_mask]
    # ub < 0
    ub_mask = ub < 0
    nudged_zero_point[ub_mask] = 2. ** k - 1.
    scale[ub_mask] = lb.abs().div_(quant_max)[ub_mask]

    nudged_min = quant_min.sub_(nudged_zero_point).mul_(scale)
    nudged_max = quant_max.sub_(nudged_zero_point).mul_(scale)

    return nudged_min, nudged_max, nudged_zero_point, scale


def layer_affine_quant_func_(x, lb, ub, scale, outlier_mask=None):
    """ Inplace quantize input variables via affine methods.

    This method is described in Google's `Integer-Only`_ paper, eq. (12).

    .. _Integer-Only:
        https://arxiv.org/abs/1712.05877

    Args:
        x: input tensor.
        lb: lower bound to clamp ``x``.
        ub: upper bound to clamp ``x``.
        scale: per-channel scaling factor.
        outlier_mask: indices of ``x < lb UNION ub < x``.
    """
    _le(lb, ub)

    if outlier_mask is not None:
        outlier_mask.zero_()
        outlier_mask.add_(x < lb)
        outlier_mask.add_(ub < x)
    torch.clamp(x, lb, ub, out=x)
    x.sub_(lb).div_(scale).round_()


def channel_affine_quant_func_(x, lb, ub, scale, outlier_mask=None, for_act=False):
    """ Inplace quantize input variables via affine methods.

    This method is described in Google's `Integer-Only`_ paper, eq. (12).

    .. _Integer-Only:
        https://arxiv.org/abs/1712.05877

    Args:
        x: input tensor.
        lb: lower bound to clamp ``x``.
        ub: upper bound to clamp ``x``.
        scale: per-channel scaling factor.
        outlier_mask: indices of ``x < lb UNION ub < x``.
    """
    _le(lb, ub, for_act)
    if not for_act:
        c = x.size(0)
        assert lb.size(0) == c and ub.size(0) == c
    else:
        c = x.size(1)
        assert lb.size(1) == c and ub.size(1) == c       

    if outlier_mask is not None:
        outlier_mask.zero_()
        outlier_mask.add_(x < lb)
        outlier_mask.add_(ub < x)
    torch.max(x, lb, out=x)
    torch.min(x, ub, out=x)
    x.sub_(lb).div_(scale).round_()


def layer_symmetric_quant_func_(x, k, x_mag, outlier_mask=None):
    assert 0 < x_mag, f"magnitude: {x_mag}"

    lb, ub = -x_mag, x_mag
    if outlier_mask is not None:
        outlier_mask.zero_()
        outlier_mask.add_(x < lb)
        outlier_mask.add_(ub < x)
    torch.clamp(x, lb, ub, out=x)
    # almost the same as GG19I eq(9)
    n = 2 ** (k - 1) - 1
    scale = ub / n
    x.div_(scale).round_()

    return scale


def channel_symmetric_quant_func_(x, k, x_mag, outlier_mask=None, for_act=False):
    c = x.size(0) if not for_act else x.size(1)
    x_mag += 1e-10
    assert torch.sum(0 < x_mag) == c, f"magnitude: {x_mag}"
    if not for_act:
        assert x_mag.size(0) == c
    else:
        assert x_mag.size(1) == c

    lb = x_mag.mul(-1)
    ub = x_mag

    if outlier_mask is not None:
        outlier_mask.zero_()
        outlier_mask.add_(x < lb)
        outlier_mask.add_(ub < x)
    torch.max(x, lb, out=x)
    torch.min(x, ub, out=x)
    n = 2 ** (k - 1) - 1
    scale = ub.div_(n)
    x.div_(scale).round_()

    return scale


def grad_quant_func_(dx, k):
    """ Inplace stochastic quantization for gradient """
    assert dx.data.min() <= dx.data.max(), \
        f"min: {dx.data.min()}, max: {dx.data.max()}"
    n = 2. ** k - 1.
    dx_view = (dx.size(0),) + (1,) * (dx.dim() - 1)
    dx_instance_view = dx.view(dx.size(0), -1)
    lb, _ = dx_instance_view.min(1)
    ub, _ = dx_instance_view.max(1)
    del dx_instance_view
    scale = ub.sub_(lb).div_(n).view(dx_view)
    noise = torch.zeros_like(dx)
    noise.data.uniform_(-0.5, 0.5)
    noise.mul_(scale)
    lb_view = lb.view(dx_view)
    dx.add_(noise).sub_(lb_view).div_(scale).round_().mul_(scale).add_(lb_view)


class BiasedQuantFunc(Function):
    @staticmethod
    def forward(ctx, x, k, grad_k, tail="clip", quant_mask=None, lb=None, ub=None,
                channel_wise=False, quant_grad=False, inplace=False, use_Int=False, for_act=False):
        assert tail in ("clip", "preserve", "log", "inq")
        if tail == "inq":
            assert quant_mask is not None
        if quant_grad:
            assert 0 < grad_k < 32, f"invalid gradient bit-width {grad_k}"
            assert not channel_wise, "gradient quantization doesn't compatible " \
                                     "with channel-wise quantization"
        ctx.tail = tail
        ctx.quant_grad = quant_grad
        ctx.grad_k = grad_k
        if inplace:
            ctx.mark_dirty(x)
        else:
            x = x.clone()
        if tail == "inq":
            y = x.clone()

        outlier_mask = torch.zeros_like(x, dtype=torch.uint8) if tail == "clip" else None

        if channel_wise and not for_act:
            c = x.size(0)  # only works for weights: (C_out, *)
            x_view = (c,) + (1,) * (x.dim() - 1)
            if lb is None or ub is None:
                x_channel_view = x.view(c, -1)
                lb, _ = x_channel_view.min(1)
                ub, _ = x_channel_view.max(1)
            else:
                assert lb.shape == (c,) and ub.shape == (c,)
            lb, ub, z_idx, scale = nudge_channel_min_max(k, lb, ub)
            lb = lb.view(x_view)
            ub = ub.view(x_view)
            scale = scale.view(x_view)
            z_idx = z_idx.view(x_view)
            channel_affine_quant_func_(x, lb, ub, scale, outlier_mask)
            y_quant = x.sub_(z_idx).mul_(scale)
        elif channel_wise and for_act:
            c = x.size(1)
            x_view = (1,) + (c,) + (1,) * (x.dim() - 2)
            assert lb.shape == (c,) and ub.shape == (c,)
            lb, ub, z_idx, scale = nudge_channel_min_max(k, lb, ub)
            lb = lb.view(x_view)
            ub = ub.view(x_view)
            scale = scale.view(x_view)
            z_idx = z_idx.view(x_view)
            channel_affine_quant_func_(x, lb, ub, scale, outlier_mask, for_act=True)
            y_quant = x.sub_(z_idx).mul_(scale)
        else:
            if lb is None or ub is None:
                lb, ub = x.min(), x.max()
            if not use_Int:
                lb, ub, z_idx, scale = nudge_scalar_min_max(k, lb, ub)
                layer_affine_quant_func_(x, lb, ub, scale, outlier_mask)
                y_quant = x.sub_(z_idx).mul_(scale)
            else:
                _le(lb, ub)
                ub = torch.max(ub, lb + 1e-10)
                quant_min = 0.
                quant_max = 2. ** k - 1.
                scale = (ub - lb) / (quant_max - quant_min)
                y_quant = x.sub_(lb).div_(scale).round_() + quant_min
                ctx.scale = scale
        ctx.use_Int = use_Int

        if tail == "clip":
            ctx.outlier_mask = outlier_mask
        elif tail == "log":
            ctx.ub = ub
            ctx.lb = lb
            ctx.x = x
        elif tail == "inq":
            y[quant_mask] = y_quant[quant_mask]
            ctx.quant_mask = quant_mask
            return y
        return y_quant

    @staticmethod
    def backward(ctx, df):
        dx = df.clone()

        if ctx.use_Int:
            dx = dx * ctx.scale

        if ctx.quant_grad:
            grad_quant_func_(dx, ctx.grad_k)

        tail = ctx.tail
        if tail == "clip":
            outlier_mask = ctx.outlier_mask
            dx.masked_fill_(outlier_mask, 0.)
        elif tail == "log":
            x = Variable(ctx.x)
            ub = ctx.ub
            lb = ctx.lb
            lower_mask = x.data < lb
            upper_mask = ub < x.data

            tau = ub - 1.  # definition in HWGQ
            dx[lower_mask] = 0
            dx[upper_mask] /= x[upper_mask] - tau
        elif tail == "inq":
            quant_mask = ctx.quant_mask
            dx.masked_fill_(quant_mask, 0.)

        return dx, None, None, None, None, None, None, None, None, None, None, None


class SymmetricQuantFunc(Function):
    @staticmethod
    def forward(ctx, x, k, grad_k, tail="clip", quant_mask=None, magnitude=None,
                channel_wise=False, quant_grad=False, inplace=False, use_Int=False, for_act=False):
        assert tail in ("clip", "preserve", "log", "inq")
        if tail == "inq":
            assert quant_mask is not None
        if quant_grad:
            assert 0 < grad_k < 32, f"invalid gradient bit-width {grad_k}"
            assert not channel_wise, "gradient quantization doesn't compatible " \
                                     "with channel-wise quantization"
        ctx.tail = tail
        ctx.quant_grad = quant_grad
        ctx.grad_k = grad_k
        if inplace:
            ctx.mark_dirty(x)
        else:
            x = x.clone()
        if tail == "inq":
            y = x.clone()

        outlier_mask = torch.zeros_like(x, dtype=torch.uint8) if tail == "clip" else None

        if channel_wise and not for_act:
            c = x.size(0)
            x_view = (c,) + (1,) * (x.dim() - 1)
            if magnitude is None:
                magnitude, _ = x.view(c, -1).abs().max(1)
            else:
                assert magnitude.shape == (c,)
            scale = channel_symmetric_quant_func_(x, k, magnitude.view(x_view), outlier_mask)
        elif channel_wise and for_act:
            c = x.size(1)
            x_view = (1,) + (c,) + (1,) * (x.dim() - 2)
            assert magnitude.shape == (c,)
            scale = channel_symmetric_quant_func_(x, k, magnitude.view(x_view), outlier_mask, for_act=True)
        else:
            if magnitude is None:
                magnitude = x.abs().max()
            # x here is the same as Q(H(h'/s(s))), scale is s and is a small float (the delta)
            scale = layer_symmetric_quant_func_(x, k, magnitude, outlier_mask)
        if not use_Int:
            y_quant = x.mul_(scale)
        else:
            y_quant = x
            ctx.scale = scale
        ctx.use_Int = use_Int

        if tail == "clip":
            ctx.outlier_mask = outlier_mask
        elif tail == "log":
            ctx.magnitude = magnitude
            ctx.x = x
        elif tail == "inq":
            y[quant_mask] = y_quant[quant_mask]
            ctx.quant_mask = quant_mask
            return y
        return y_quant

    @staticmethod
    def backward(ctx, df):
        dx = df.clone()
      
        if ctx.use_Int:
            dx = dx * ctx.scale

        if ctx.quant_grad:
            grad_quant_func_(dx, ctx.grad_k)

        tail = ctx.tail
        if tail == "clip":
            outlier_mask = ctx.outlier_mask
            dx.masked_fill_(outlier_mask, 0.)
        elif tail == "log":
            x = Variable(ctx.x)
            ub = ctx.magnitude
            lb = -ub
            lower_mask = x.data < lb
            upper_mask = ub < x.data

            tau = ub - 1.  # definition in HWGQ
            dx[lower_mask] = 0
            dx[upper_mask] /= x[upper_mask] - tau
        elif tail == "inq":
            quant_mask = ctx.quant_mask
            dx[quant_mask] = 0

        return dx, None, None, None, None, None, None, None, None, None, None


class LowPrecisionFloatFunc(Function):
    @staticmethod
    def forward(ctx, x, exp_k, mantissa_k):
        # TODO(Rundong) fix this stupid exp bit hack
        x = torch.ones_like(x) * (-2 ** exp_k) * (x < -2 ** exp_k).to(x.dtype) + \
            torch.ones_like(x) * (2 ** exp_k) * (x > 2 ** exp_k).to(x.dtype) + \
            x * ((2 ** (-exp_k) <= x.abs()).to(x.dtype) * (x.abs() <= 2 ** exp_k).to(x.dtype))
        x += torch.abs(x) * (2. ** -(mantissa_k + 1))
        if _is_0_4:
            exp_k = 8 if x.dtype is torch.float32 else 11
            return lowp.forward(x.contiguous(), exp_k, mantissa_k)
        else:
            mantissa_shift = 2 ** mantissa_k
            x = torch.round(x * mantissa_shift) / mantissa_shift
            return x

    @staticmethod
    def backward(ctx, df):
        return df.clone(), None, None


class QuantGradFunc(Function):
    @staticmethod
    def forward(ctx, x, k):
        """
        Args:
            x: input activation in forward pass.
            k: number of bits used to quantize backward gradient.
        """
        ctx.bit_g = k

        return x

    @staticmethod
    def backward(ctx, df):
        dx = dk = None
        k = ctx.bit_g

        if k == 32:
            dx = df.clone()
        else:
            dtype = torch.cuda.FloatTensor if df.is_cuda else torch.FloatTensor
            n = 0.5 / float(2 ** k - 1)
            bias = Variable(dtype(df.shape).uniform_(-n, n))
            df_max, _ = df.view(df.size(0), -1).max(1)
            df = df / df_max
            df = torch.clamp(df * 0.5 + 0.5 + bias, 0., 1.)
            df_fix = torch.round(df * (2. ** k - 1.)) / (2. ** k - 1.)
            dx = df_max * (df_fix - 0.5) * 2.

        return dx, dk


class UpsampleLike(Function):
    """ Interp lhs tensor to same shape with rhs """
    @staticmethod
    def forward(ctx, lhs, rhs):
        _, _, lh, lw = lhs.shape
        _, _, rh, rw = rhs.shape
        ctx.origin_shape = lh, lw
        return torch._C._nn.upsample_bilinear2d(lhs, (rh, rw), False)

    @staticmethod
    def backward(ctx, df):
        raise RuntimeError("inference only")

    def symbolic(g, lhs, rhs):
        """ create dummy onnx symbol """
        return g.op("UpsampleLike", lhs, rhs)


class SyncAffineFunc(Function):
    """ Affine function for Synchronized BN """
    @staticmethod
    def forward(ctx, x, alpha, beta):
        c = x.size(1)
        x_view = (1, c, 1, 1)
        ctx.affine = x, alpha
        ctx.c = c
        return alpha.view(x_view) * x + beta.view(x_view)

    @staticmethod
    def backward(ctx, df):
        x, alpha = ctx.affine
        c = ctx.c
        d_x = d_alpha = d_beta = None

        df_c_view = df.clone().transpose(0, 1).contiguous().view(c, -1)
        x_c_view = x.transpose(0, 1).contiguous().view(c, -1)
        d_beta = df_c_view.sum(-1)
        d_alpha = (df_c_view * x_c_view).sum(-1)
        d_x = (df_c_view * alpha.view(c, 1)).reshape(x.shape)

        # average gradient for globally-normalized activation
        try:
            d = link # get_dist_interface()
            world_size = d.get_world_size()
        except AssertionError:
            world_size = 1
        if world_size > 1:
            d_x /= world_size
            d = link # get_dist_interface()
            d.all_reduce(d_x, d.reduce_op.SUM)

        return d_x, d_alpha, d_beta


class SyncStatFunc(Function):
    """ Global mean / variance statistics for Synchronized BN.

    reference: https://arxiv.org/abs/1803.08904 Appendix A
    """
    @staticmethod
    def forward(ctx, x, world_size, unbiased=True):
        assert x.dim() == 4
        n, c, h, w = x.shape
        x = x.transpose(0, 1).reshape(c, -1)
        ctx.x = x
        ctx.input_shape = n, c, h, w
        ctx.unbiased = unbiased
        ctx.world_size = world_size

        x_sum = x.sum(-1)
        x_square_sum = (x ** 2).sum(-1)
        if world_size > 1:
            d = link # get_dist_interface()
            d.all_reduce(x_sum, d.reduce_op.SUM)
            d.all_reduce(x_square_sum, d.reduce_op.SUM)

        n *= world_size * h * w
        x_mean = x_sum / n
        x_var = x_square_sum / n - x_mean ** 2
        ctx.x_mean = x_mean
        if unbiased:
            x_var *= n / (n - 1)

        return x_mean, x_var

    @staticmethod
    def backward(ctx, df_mean, df_var):
        dx = d_world_size = d_unbiased = None
        x = Variable(ctx.x)
        x_mean = Variable(ctx.x_mean)
        c, n = x.shape
        unbiased = ctx.unbiased
        world_size = ctx.world_size
        n *= world_size

        if unbiased:
            df_var = df_var.clone() * (n / (n - 1))
        else:
            df_var = df_var.clone()
        d_x_sum = (df_mean.clone() - 2 * x_mean * df_var) / n
        d_x_square_sum = df_var / n
        if world_size > 1:
            d = link # get_dist_interface()
            d.all_reduce(d_x_sum.data, d.reduce_op.SUM)
            d.all_reduce(d_x_square_sum.data, d.reduce_op.SUM)

        dx = torch.zeros_like(x)
        dx += d_x_sum.unsqueeze_(-1).expand_as(x)
        dx += 2 * d_x_square_sum.unsqueeze_(-1).expand_as(x) * x
        n, c, h, w = ctx.input_shape
        dx = dx.view(c, n, h, w).transpose(0, 1).contiguous()

        return dx, d_world_size, d_unbiased


class Percentile(Function):
    @staticmethod
    def forward(ctx, x, p):
        assert 0. <= p <= 100., "invalid percentile: {p}"
        largest = p > 50.
        tail_portion = (100. - p) if largest else p
        index_r = x.numel() * tail_portion / 100.
        k = math.ceil(index_r)
        if k == 1:
            if largest:
                return x.max()
            else:
                return x.min()
        else:
            tails, _ = torch.topk(x.view(-1), k, largest=largest)
            i, j = tails[-2:]
            ri = index_r % 1.
            rj = 1. - ri
            return i.mul_(ri).add_(j.mul_(rj))

    @staticmethod
    def backward(ctx, df):
        raise NotImplementedError(f"forward only")
