import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from scipy import fftpack
from nets.layers import RdftParameterizer, SignalConvTranspose2d, SignalConv2d
from quant.quantizator import Round

try:
    from integer2.functional import SymmetricQuantFunc
    from integer2.module import magnitude_inq_mask
except:
    print("layers_quant load integer2 failed", flush=True)
    from integer.functional import SymmetricQuantFunc
    from integer.module import magnitude_inq_mask
from torch.autograd import Function
from torch import tensor
from nets.norm import LowerBound, lowerbound
import scipy.special
try:
    import spring.linklink as link
except:
    link = None


class RoundDiv(object):  # NOTICE: this is not a torch Function
    """
    round(a/b)

    Round-div operation.

    support floor-div term described in:
    Integer networks for data compression with latent-variable models
    """

    @staticmethod
    def forward(ctx, a, b, bit, use_floor_div=False):
        """

        :param ctx: func context
        :param a: operand a, divisor
        :param b: operand b, dividend
        :param bit: rounding bit, same as Round func
        :param use_floor_div: whether to use floor-div term
        :return: round(a/b)
        """
        if not use_floor_div:
            x = a / b
            return Round.apply(x, bit)
        else:
            # assume that a is non-neg and b is positive
            # using floor-div term
            # round(a/b) = (a + b//2) // b
            delta = torch.floor(b / 2)
            x = torch.floor((a + delta) / b)
            return x

    @classmethod
    def apply(cls, *args):
        return cls.forward(None, *args)


class Floor(Function):
    """ Floor operation with identity gradient"""
    @staticmethod
    def forward(ctx, x, bit):
        ctx.bit = bit
        return torch.floor(2 ** bit * x)

    @staticmethod
    def backward(ctx, df):
        return df.clone() * (2 ** ctx.bit), None


class Ceil(Function):
    """ Ceil operation with identity gradient, only for use_Int=True. """
    @staticmethod
    def forward(ctx, x, bit):
        ctx.bit = bit
        return torch.ceil(2 ** bit * x)

    @staticmethod
    def backward(ctx, df):
        return df.clone() * (2 ** ctx.bit), None


class Clip(Function):
    #@staticmethod
    def forward(ctx, inputs, lower_bound, upper_bound):
        ones = torch.ones(inputs.size(), device=inputs.device)
        lower_bound = lower_bound.to(device=inputs.device)
        upper_bound = upper_bound.to(device=inputs.device)
        b1 = ones * lower_bound
        b2 = ones * upper_bound
        ctx.save_for_backward(inputs, b1, b2)
        return torch.min(torch.max(inputs, b1), b2)

    #@staticmethod
    def backward(ctx, grad_output):
        inputs, b1, b2 = ctx.saved_tensors

        pass_through_1 = (inputs >= b1) & (inputs <= b2)
        pass_through_2 = (grad_output < 0) & (inputs < b1)
        pass_through_3 = (grad_output > 0) & (inputs > b2)

        pass_through = pass_through_1 | pass_through_2 | pass_through_3
        return pass_through.type(grad_output.dtype) * grad_output, None, None


class GGClip(Clip):
    def __init__(self, beta=4.):
        super().__init__()
        alpha = 1. / beta * scipy.special.gamma(1. / beta)
        self.beta = beta
        self.alpha = alpha
        self._alpha_pow_beta = np.power(alpha, beta)

    #@staticmethod
    def backward(ctx, grad_output):
        alpha, beta = ctx.alpha, ctx.beta

        # b1 is the lower bound while b2 is the upper
        inputs, b1, b2 = ctx.saved_tensors

        inner_term = (2. * inputs / (b2 - 1) - 1.).abs().pow(beta)
        partial = torch.exp(-ctx._alpha_pow_beta * inner_term)

        return partial * grad_output, None, None

    # @classmethod
    # def apply(cls, *args):
    #     return cls().forward(*args)

class QReLU(nn.Module):
    def __init__(self):
        super(QReLU, self).__init__()
        self.to_caffe = False
        self.relu = nn.ReLU()

    def forward(self, x, min, max):
        x = Clip()(x, tensor(min), tensor(max))
        return x


class GGQReLU(QReLU):
    def forward(self, x, min, max):
        if not self.to_caffe:
            #x = GGClip.apply(x, tensor(min), tensor(max))
            x = GGClip()(x, tensor(min), tensor(max))
        else:
            x = torch.clamp(x, min, max)
            # x_target_right = self.relu(x - self.min) + self.min
            # x_target_left = self.max - self.relu(self.max - x)
            # x = (x_target_left + x_target_right - x)/2
        return x


class GGClipBi(Clip):
    """
    双边clip   是整数的mu使用的（-127到127）
    """

    def __init__(self, beta=4.):
        super().__init__()
        alpha = 1. / beta * scipy.special.gamma(1. / beta)
        self.beta = beta
        self.alpha = alpha
        self._alpha_pow_beta = np.power(alpha, beta)

    @staticmethod
    def backward(ctx, grad_output):
        alpha, beta = ctx.alpha, ctx.beta

        # b1 is the lower bound while b2 is the upper
        inputs, b1, b2 = ctx.saved_tensors

        # 梯度函数相对于GGClip向左平移128
        inner_term = (2. * (inputs+128.) / (256 - 1) - 1.).abs().pow(beta)
        partial = torch.exp(-ctx._alpha_pow_beta * inner_term)

        return partial * grad_output, None, None


class GGQBiReLU(QReLU):
    """
    GGQBiReLU for from -127 to 127
    """
    def forward(self, x):
        if not self.to_caffe:
            x = GGClipBi()(x, self.min, self.max)
        else:
            x = torch.clamp(x, self.min.data[0], self.max.data[0])
            # x_target_right = self.relu(x - self.min) + self.min
            # x_target_left = self.max - self.relu(self.max - x)
            # x = (x_target_left + x_target_right - x)/2
        return x


class QLeakyReLU(nn.Module):
    def __init__(self, _min, _max, _threshold=0, _neg_slope=0.333):
        super().__init__()
        self.min = nn.Parameter(torch.Tensor([_min]), requires_grad=False)
        self.max = nn.Parameter(torch.Tensor([_max]), requires_grad=False)
        self.threshold = nn.Parameter(torch.Tensor([_threshold]), requires_grad=False)
        self.neg_slope = nn.Parameter(torch.Tensor([_neg_slope]), requires_grad=False)

    def forward(self, x):
        msk = (x > self.threshold).float()
        pos = Clip()(x, self.threshold, self.max)
        neg = Clip()(Round.apply(x * self.neg_slope, 8), self.min, self.threshold)
        return msk * pos + (1. - msk) * neg


class ScaleReparameterizer(nn.Module):
    def __init__(self, weight, bit=8, dtype=torch.int8):
        super().__init__()
        self.weight = weight
        self.scale = 1 << (bit-1)
        self.dtype_min = torch.iinfo(dtype).min
        self.dtype_max = torch.iinfo(dtype).max
        self.clip = GGQReLU()

    def forward(self):
        weight = self.weight.data  # [num_filter, num_input_filter, K, K]
        weight_reshape = weight.reshape(weight.shape[0], -1)  # squeeze for each filter
        maximum, _ = torch.max(weight_reshape, 1)
        minimum, _ = torch.min(weight_reshape, 1)
        scale = self.scale
        choice1 = -minimum / scale
        choice2 = maximum / scale
        choice3 = torch.ones_like(choice1) * 1e-20

        # s(h) = max{minimum/(-scale), maximum/(scale), eps}
        sh = torch.max(torch.max(choice1, choice2), choice3)
        sh = sh.reshape(weight.shape[0], *([1] * (weight.ndimension() - 1)))

        return self.clip(Round.apply(weight / sh, 0), self.dtype_min, self.dtype_max)

    @property
    def data(self):
        return self.forward()


class BiasReparameterizer(nn.Module):
    def __init__(self, bias, bit=8, dtype=torch.int32):
        super().__init__()
        self.bias = bias
        self.scale = 1 << (bit)
        self.dtype_min = torch.iinfo(dtype).min
        self.dtype_max = torch.iinfo(dtype).max
        self.clip = GGQReLU()

    def forward(self):
        if self.bias is None:
            return None
        return self.clip(Round.apply(self.scale*self.bias, 0), self.dtype_min, self.dtype_max)

    @property
    def data(self):
        return self.forward()


# scale, zero version
class SymmetricActReparameterizer(Function):
    @staticmethod
    def forward(ctx, x, bit):
        qmax = (1 << (bit - 1)) - 1
        qmin = -(1 << (bit - 1))
        scale = (x.max() - x.min()) / (qmax - qmin) 
        zero = torch.round(qmax - x.max() / scale)
        ctx.save_for_backward(scale)
        return torch.round(x / scale + zero)

    @staticmethod
    def backward(ctx, grad_output):
        scale = ctx.saved_tensors[0]
        return grad_output / scale.type(grad_output.dtype), None


class SymmetricActReparameterizer_v2(SymmetricActReparameterizer):
    @staticmethod
    def forward(ctx, x, bit):
        x_max = x.abs().max()
        thresh = (1 << (bit - 1)) - 1
        scale = x_max / thresh
        ctx.save_for_backward(scale)
        return torch.round(x / scale)


class AsymmetricActReparameterizer(SymmetricActReparameterizer):
    @staticmethod
    def forward(ctx, x, bit):
        qmax = (1 << bit) - 1
        qmin = 0
        scale = (x.max() - x.min()) / (qmax - qmin)
        zero = torch.round(qmax - x.max() / scale)
        ctx.save_for_backward(scale)
        return torch.round(x / scale + zero)


class GGQConvTranspose2d(SignalConvTranspose2d):
    def __init__(self, *args, **kwargs):
        self._init = False
        super().__init__(*args, **kwargs)
        b = self.bias
        self._init = True
        self._weight = ScaleReparameterizer(self._weight, dtype=torch.int8)
        self._bias = BiasReparameterizer(b, dtype=torch.int32)
        del self._parameters['bias']

        # self.to_caffe = False

    def prepare(self):
        self.w2 = self._weight.data
        self.w2 = nn.Parameter(self.w2)
        self.b2 = self._bias.data
        self.b2 = nn.Parameter(self.b2)

    def __getattr__(self, item):
        if item == 'weight' and self._init:
            if not self.to_caffe:
                return self._weight.data
            else:
                return self.w2
        elif item == 'bias' and self._init:
            if not self.to_caffe:
                return self._bias.data
            else:
                return self.b2
        else:
            return super().__getattr__(item)


class GGQUpDeconvWrapper(GGQConvTranspose2d):
    def forward(self, x):
        n, c, h, w = x.shape
        factor = self.stride[0]

        # enable up-sampling deconv with odd kernel_size
        return super().forward(x, output_size=(n, c, factor * h, factor * w))


class QConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs):
        self._super_inited = False
        super().__init__(*args, **kwargs)

        self.fw = self.weight
        self.fb = self.bias
        self._super_inited = True

        self.weight_bit = 8
        self.grad_bit = 32
        self.enable_quant = True
        self.enable_quant_grad = False
        self.quant_mode = "symmetric"
        self.tail = "preserve"
        self.inq_mode = "disable"
        self.channel_wise = False
        self.write_back_quant_weight = False
        self.scale_weight = False

        self.to_caffe = False
        self.save_int_ckpt = False

        assert self.quant_mode in ("biased", "symmetric", "dorefa")
        # how to handle the grad out of the quant range
        assert self.tail in ("clip", "preserve", "log")
        assert self.inq_mode in ("disable", "magnitude")

        if self.inq_mode == "disable":
            self.tail = self.tail
        else:
            # INQ is not compatible with ordinary gradient handling schemes
            self.tail = "inq"
        if self.inq_mode == "disable":
            self.register_buffer("quant_mask", None)
        else:
            self.toggle_inq_mask = True  # flag set by Scheduler
            self.quant_portion = 0.
            self.register_buffer("quant_mask", torch.zeros_like(self.weight, dtype=torch.uint8))

        if self.quant_mode == "symmetric":
            self.weight_func = SymmetricQuantFunc.apply

        self.round_func = Round.apply

    def __getattr__(self, item):
        if item == 'weight' and self._super_inited and not self.to_caffe and not self.save_int_ckpt and self.enable_quant:
            return self._get_weight()
        elif item == 'bias' and self._super_inited and not self.to_caffe and not self.save_int_ckpt and self.enable_quant:
            return self._get_bias()
        else:
            return super().__getattr__(item)

    def _get_bias(self):
        # bias is int32 signed, no need to nudge zero-point
        # before test: use not self.enable_quant to use loaded int ckpt, use self.enable_quant to use loaded float ckpt

        b = self.round_func(self.fb.clone(), self.weight_bit)
        return b

    def _get_weight(self):
        w = self.fw
        inplace = False
        use_Int = True
        w = self.weight_func(w, self.weight_bit, self.grad_bit,
                                         self.tail, self.quant_mask,
                                         None, self.channel_wise, self.enable_quant_grad,
                                         inplace, use_Int)
        return w

    def save_int(self):
        self.weight.data = self._get_weight().data
        self.bias.data = self._get_bias().data


class GGBpAct(nn.Module):
    def __init__(self, num_filters, bit, dtype=torch.int32):
        super().__init__()
        self.c_prime = nn.Parameter(torch.ones((1, num_filters, 1, 1)))
        # eps = 1e-18
        self.register_buffer('c_bound', torch.tensor(np.sqrt(1. + 1e-36), dtype=torch.float))
        self.scale = 1 << bit
        self.round_fn = Round.apply
        self.dtype_min = torch.iinfo(dtype).min
        self.dtype_max = torch.iinfo(dtype).max
        self.clip = GGQReLU()
        self.to_caffe = False

    def _reparam(self):
        # r(c') = max(c', c_bound))^2 - eps^2
        if not self.to_caffe:
            return LowerBound.apply(self.c_prime, self.c_bound) ** 2 - 1e-36
        else:
            return lowerbound(self.c_prime, self.c_bound) ** 2 - 1e-36

    @property
    def c(self):
        # c = Q(scale * r(c'))
        if not self.to_caffe:
            c = self.round_fn(self.scale * self._reparam(), 0)
        else:
            c = torch.round(self.scale * self._reparam())

        return self.clip(c, self.dtype_min, self.dtype_max)

    def prepare(self):
        self.qc = self.c.cpu()

    def forward(self, x):
        if not self.to_caffe:
            delta = Floor.apply(self.c / 2, 0)
            x = Floor.apply((x + delta) / self.c, 0)
        else:
            delta = torch.floor(self.qc / 2)
            x = torch.floor((x + delta) / self.qc)

        return x

    def count_ops(m: nn.Module, x: (torch.Tensor,), y: torch.Tensor):
        add_ops = 1
        div_ops = 1

        # N x Cout x H x W x  (add + div)
        total_ops = y.nelement() * (add_ops + div_ops)

        m.total_ops += torch.DoubleTensor([int(total_ops)])


class ProAct(nn.Module):
    '''
    直接逐通道将最大值缩放到255，而不是学习逐通道的c (1, num_filters, 1, 1)
    假设通道最大值为max,　则c = max(ceil(max/255), 1)
    '''
    def __init__(self, num_filters, bit):
        super().__init__()
        self.c_maxes = nn.Parameter(torch.zeros((1, num_filters, 1, 1)), requires_grad = False)
        self.register_buffer('c_bound', torch.tensor([1], dtype=torch.float))
        self.scale = 1 << bit
        self.round_fn = Round.apply
        self.num_filters = num_filters
        self.momentum = 0.9999
        self.to_caffe = False

    def _reparam(self, maxes):
        maxes_ = torch.ceil(maxes / 255.0)
        return lowerbound(maxes_, self.c_bound)

    def prepare(self):
        self.c = self._reparam(self.c_maxes)
        self.qc = self.c.cpu()

    def forward(self, x):
        if self.training:
            num = self.num_filters
            maxes = x.max(0)[0].max(1)[0].max(1)[0]
            maxes = maxes.view(1, num, 1, 1)
            self.c_maxes.data = torch.max(self.c_maxes.data, maxes)
            # self.c_maxes = self.momentum * self.c_maxes + (1 - self.momentum) * maxes

        if not self.to_caffe:
            self.c = self._reparam(self.c_maxes)
            return self.round_fn(x / self.c, 0)
        else:
            # TODO: return torch.round(x / self.qc)
            return x / self.qc


class BpAct(nn.Module):
    def __init__(self, num_filters, bit):
        self._super_inited = False
        super(BpAct, self).__init__()

        self.bit = bit
        self.round_func = Round.apply
        self.c = nn.Parameter(torch.ones(1, num_filters, 1, 1))
        self.fc = self.c

        self.to_caffe = False
        self.enable_quant = True
        self.save_int_ckpt = False

        self.register_buffer('c_min', torch.FloatTensor([1.]))
        self.register_buffer('reparam_offset', torch.FloatTensor([2**-18]))
        self.register_buffer('pedestal', self.reparam_offset**2)
        self.c_bound = (self.reparam_offset**2 + self.c_min)**.5
        self.c_bound = nn.Parameter(self.c_bound, requires_grad=False)

        self._super_inited = True

    def __getattr__(self, item):
        if item == 'c' and self._super_inited and not self.save_int_ckpt and self.enable_quant:
            return self._get_c()
        else:
            return super().__getattr__(item)

    def save_int(self):
        self.c.data = self._get_c().data

    def _get_c(self):
        rc = self.fc.clone()
        rc = LowerBound()(rc, self.c_bound)
        rc = rc**2 - self.pedestal
        c = self.round_func(rc, self.bit)
        return c

    def forward(self, x):
        x = x / self.c
        # replace round by floor div in sdk: Q(x/c) = (x+c//2)//c, where // is floor division
        # gg19i: Q(m/n) = (m + n//2)//n for m, n in ints
        # safely assume 0<=m<n
        # if m<n/2, left=0, m + n//2 < n/2 + n/2 = n, so that right=0=left
        # if n/2<=m<n, left=1, 1 >= right >= (n/2 + n//2)//n
        #     if n=2k, right >= (k + k)//n = 1, so that right=1=left
        #     if n=2k+1, m >= k+0.5, m >= k+1, right >= (k+1 + k)//n = 1, so that right=1=left
        return self.round_func(x, 0)


class NoBpAct(nn.Module):
    def __init__(self, num_filters, bit):
        # in sdk c is Ceil((x_max - x_min) / 2 ** k) if do not use x - self.x_min
        super(NoBpAct, self).__init__()

        self.bit = bit
        self.momentum = 0.9999
        self.round_func = Round.apply

        self.to_caffe = False
        self.enable_quant = True
        self.save_int_ckpt = False

        self.register_buffer("x_min", torch.Tensor(1).zero_())
        self.register_buffer("x_max", torch.Tensor(1).zero_())

        self.first = True

    def forward(self, x):
        if self.training:
            # self.x_min = torch.min(torch.Tensor([x.data.min()]).cuda(), self.x_min)
            # self.x_max = torch.max(torch.Tensor([x.data.max()]).cuda(), self.x_max)

            try:
                world_size = link.get_world_size()
            except AssertionError:
                world_size = 1
            if world_size > 1:
                x_min = torch.Tensor([x.data.min() / world_size]).cuda()
                x_max = torch.Tensor([x.data.max() / world_size]).cuda()
                link.allreduce(x_min)
                link.allreduce(x_max)

            if self.first:
                self.x_min, self.x_max = x_min, x_max
                self.first = False
            else:
                self.x_min = self.momentum * self.x_min + (1. - self.momentum) * x_min
                self.x_max = self.momentum * self.x_max + (1. - self.momentum) * x_max

        x = (x - self.x_min) / (self.x_max - self.x_min)
        return self.round_func(x, self.bit)


class GGQConv2d(SignalConv2d):
    def __init__(self, *args, **kwargs):
        self._init = False
        super().__init__(*args, **kwargs)
        b = self.bias
        self._init = True
        self._weight = ScaleReparameterizer(self._weight)
        self._bias = BiasReparameterizer(b)
        del self._parameters['bias']

        # self.to_caffe = False

    def prepare(self):
        self.w2 = self._weight.data
        self.w2 = nn.Parameter(self.w2)
        self.b2 = self._bias.data
        self.b2 = nn.Parameter(self.b2)

    def __getattr__(self, item):
        if item == 'weight' and self._init:
            if not self.to_caffe:
                return self._weight.data
            else:
                return self.w2
        elif item == 'bias' and self._init:
            if not self.to_caffe:
                return self._bias.data
            else:
                return self.b2
        else:
            return super().__getattr__(item)


class MaskedQConv2d(GGQConv2d):
    '''only use top-left part of the kernel
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _, _, k, k = self._weight().size()
        assert k % 2 == 1
        mask = torch.ones(k, k)
        b = k // 2
        for i in range(k):
            for j in range(k):
                if i > b or i == b and j >= b:
                    mask[i, j] = 0
        mask = mask.view(1, 1, k, k)
        self.mask = nn.Parameter(mask)
        self.mask.requires_grad = False

        self._super_inited = True

        self.to_caffe = False

    def __getattr__(self, item):
        if item == 'weight' and self._super_inited and not self.to_caffe:
            return self._weight() * self.mask
        else:
            return super().__getattr__(item)

    def prepare(self):
        self._parameters['weight'] = self._weight() * self.mask

