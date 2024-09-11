import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import functional
from video.utils.misc_helper import decorator_input, clock
import numpy as np

class Round(Function):
    """ Round operation with identity gradient, only for use_Int=True. """

    @staticmethod
    def forward(ctx, x, bit):
        ctx.bit = bit
        return torch.round(2 ** bit * x)

    @staticmethod
    def backward(ctx, df):
        return df.clone() * (2 ** ctx.bit), None

class MAPFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x
    
    @staticmethod
    def backward(ctx, grad):
        return grad


class Quantizator_MAP(nn.Module):
    """
    for integer network
    """
    def __init__(self, B=1, train=True):
        super(Quantizator_MAP, self).__init__()
        self.B = B
        self.training = train

    def forward(self, input):
        if self.training:
            return MAPFunc.apply(input)
        else:
            return torch.round(input)
            

class Quantizator_RT(nn.Module):
    """
    uniform noise quant-estimator
    """
    def __init__(self, B=1, train=True, key_in=None, key_out=None):
        super(Quantizator_RT, self).__init__()
        self.B = B
        self.training = train
        self.factor = (1 << self.B) - 1
        self.key_in = key_in
        self.key_out = key_out

    def forward(self, input):
        factor = self.factor
        if self.training:
            x = torch.rand_like(input)
            x -= 0.5
            if factor != 1:
                x /= factor
            return input + x
        else:
            return torch.round(input * factor) / factor


class Quantizator_HARD(nn.Module):
    """
    Soft then Hard: Rethinking the Quantization in Neural Image Compression
    https://arxiv.org/pdf/2104.05168.pdf
    Directly round, so that encoder keeps unchanged, only ex-post tune decoder with hard quantization
    """
    def __init__(self):
        super(Quantizator_HARD, self).__init__()

    def forward(self, input):
        return torch.round(input)


class Quantizator_MYRT(nn.Module):
    """a differentiable Quantizator. only used in visualization
    """
    def __init__(self, B=1, train=True):
        super(Quantizator_MYRT, self).__init__()
        self.B = B
        self.training = train

    def forward(self, input):
        if self.training:
            x = torch.rand_like(input)
            x -= 0.5
            factor = (1 << self.B) - 1
            x /= factor
            return input + x
        else:
            return Round.apply(input, 0)


class StraightThroughEstimatorFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad):
        return grad


class STEQuant(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return StraightThroughEstimatorFunc.apply(x)  # B = 1


class OrderQuant(nn.Module):
    def __init__(self):
        super(OrderQuant, self).__init__()
    def forward(self, x):
        return OrderEstimatorFunc.apply(x)


class OrderEstimatorFunc(torch.autograd.Function):
    '''
    base on Learning to Improve Image Compression without Changing the Standard Decoder
    '''
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        return grad * 3 * (x - torch.round(x))**2


class NoiseOrderQuant(nn.Module):
    def __init__(self, train=True):
        super(NoiseOrderQuant, self).__init__()
        self.training = train

    def forward(self, input):
        if self.training:
            return NoiseOrderEstimatorFunc.apply(input)
        else:
            return torch.round(input)


class NoiseOrderEstimatorFunc(torch.autograd.Function):
    """
    uniform noise + 3order, using x to backward
    """
    @staticmethod
    def forward(ctx, x):
        noise = torch.rand_like(x) - 0.5
        ctx.save_for_backward(x)
        return x + noise

    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        return grad * 3 * (x - torch.round(x))**2


class FrontierQuant(nn.Module):
    def __init__(self, train=True, c=5):
        super(FrontierQuant, self).__init__()
        self.training = train
        self.c = c

    def forward(self, input):
        if self.training:
            x = input
            for i in range(1, self.c+1):
                x = x + (-1)**i / np.pi / i * torch.sin(2*np.pi*i*input)
            return x
        else:
            return torch.round(input)


class Quantizator_UQ(nn.Module):
    '''based on Variable Rate Deep Image Compression With a Conditional Autoencoder
    '''

    def __init__(self, B=1, train=True):
        super(Quantizator_UQ, self).__init__()
        self.B = B
        self.training = train

    def forward(self, input):
        if self.training:
            u = torch.rand(1)
            x = torch.ones_like(input)
            x *= u
            x -= 0.5
            factor = (1 << self.B) - 1
            x /= factor
            return torch.round((input + x) * factor) / factor - x
        else:
            factor = (1 << self.B) - 1
            return torch.round(input * factor) / factor


class Quantizator_RT_MIXED(nn.Module):
    '''based on Variable Rate Deep Image Compression With a Conditional Autoencoder
    '''

    def __init__(self, delta=1, train=True):
        super(Quantizator_RT_MIXED, self).__init__()
        self.delta = delta
        self.training = train

    def forward(self, input):
        if self.training:
            x = torch.rand_like(input)
            x -= 0.5
            x *= self.delta
            return input + x
        else:
            return torch.round(input / self.delta) * self.delta


class Quantizator_UQ_MIXED(nn.Module):
    '''based on Variable Rate Deep Image Compression With a Conditional Autoencoder
    '''

    def __init__(self, delta=1, train=True):
        super(Quantizator_UQ_MIXED, self).__init__()
        self.delta = delta
        self.training = train

    def forward(self, input):
        if self.training:
            u = torch.rand(1, device=input.device)
            x = torch.ones_like(input)
            x *= u
            x -= 0.5
            x *= self.delta
            return torch.round((input + x) / self.delta) * self.delta - x
        else:
            return torch.round(input / self.delta) * self.delta


class Quantizator_SOFT(nn.Module):
    def __init__(self, B=1, train=True, sigma=-1., p=2.):
        super(Quantizator_SOFT, self).__init__()
        self.B = B
        self.training = train
        self.sigma = sigma
        self.p = p
        C = nn.Parameter(torch.arange(-512, 513).float())  # there are 1024 intervals
        self.C = C

    def get_Q(self, input, sigma, C=None):
        if C is None:
            C = self.C
        factor = (1 << self.B)
        s = input.shape
        x = (input * factor)
        x = x.view(-1)
        x = x.repeat(C.shape[0], 1).transpose(0, 1)

        # where sigma is negative
        # e ^ (sigma * d(x, c)^p)
        """
        theta = torch.exp((x - C).abs() ** self.p * sigma)
        sum = theta.sum(1).unsqueeze(1)
        theta = theta / sum
        """
        theta = torch.softmax((x - C).abs() ** self.p * sigma, 1)
        x = (C * theta).sum(1)
        x = (x / factor).reshape(s)
        return x

    def sample_C(self, input):
        x = torch.round(input * ((1 << self.B) - 1))
        Max, Min = x.max().item(), x.min().item()
        L = int(Max - Min + 1)
        l, r = 0, 1024
        for i in range(1, 1024):
            if self.C[i] <= Min:
                l = i - 1
            elif self.C[i] >= Max:
                r = i + 1
                break
        C = self.C[l:r + 1].unsqueeze(0).unsqueeze(0)
        offset = 512 - l
        ret = functional.interpolate(C, L, mode='linear', align_corners=True)
        return ret.squeeze(0).squeeze(0), offset

    def get_index(self, input):
        s = input.shape
        C, offset = self.sample_C(input)
        x = input.view(-1).repeat(C.shape[0], 1).transpose(0, 1)
        x = torch.abs(x - C)
        ret = torch.argmin(x, 1) - offset
        return ret.reshape(s).float()

    def forward(self, input, get_index=False):
        """
        print("\n\n>>>>")
        sum = 0
        for i in range(-512, 513):
            sum += self.C[i].item() - i
        print(sum)
        """
        C, offset = self.sample_C(input)
        if get_index == True:
            return self.get_index(input)
        x_hard = self.get_Q(input, -10000000, C=C).detach()  # hard
        x_soft = self.get_Q(input, -1, C=C)
        return (x_hard - x_soft.detach()) + x_soft


class Quantizator_DA(nn.Module):
    """
    https://github.com/mandt-lab/improving-inference-for-neural-image-compression/blob/main/danneal.py
    Deterministic Annealing
    """
    def __init__(self, train=True, gap=1000, c=0.001):
        super(Quantizator_DA, self).__init__()
        self.training = train
        self.gap = gap
        self.c = c

    @decorator_input
    def forward(self, input):
        if self.training:
            x_floor = torch.floor(input)
            x_ceil = torch.ceil(input)
            x_bds = torch.stack([x_floor, x_ceil], dim=-1)

            eps = 1e-5
            self.iteration = self.iteration // self.gap
            T = 0.5 * np.exp(-self.c * self.iteration)

            x_interval1 = torch.clamp(input - x_floor, -1 + eps, 1 - eps)
            x_atanh1 = torch.log((1 + x_interval1) / (1 - x_interval1)) / 2
            x_interval2 = torch.clamp(x_ceil - input, -1 + eps, 1 - eps)
            x_atanh2 = torch.log((1 + x_interval2) / (1 - x_interval2)) / 2

            self.rx_logits = torch.stack([-x_atanh1 / T, -x_atanh2 / T], dim=-1)
            self.rx = functional.softmax(self.rx_logits, dim=-1)
            
            x_tilde = torch.sum(x_bds * self.rx, dim=-1)
            return x_tilde
        else:
            return torch.round(input)


class Quantizator_DA_STE(nn.Module):
    """
    Deterministic Annealing
    forward: deterministic annealing
    backward: ste
    """
    def __init__(self, train=True, gap=1000, c=0.001):
        super(Quantizator_DA_STE, self).__init__()
        self.training = train
        self.gap = gap
        self.c = c

    @decorator_input
    def forward(self, input):
        if self.training:
            x_floor = Floor.apply(input, 0)
            x_ceil = Ceil.apply(input, 0)
            x_bds = torch.stack([x_floor, x_ceil], dim=-1)

            eps = 1e-5
            self.iteration = self.iteration // self.gap
            T = 0.5 * np.exp(-self.c * self.iteration)

            x_interval1 = torch.clamp(input - x_floor, -1 + eps, 1 - eps)
            x_atanh1 = torch.log((1 + x_interval1) / (1 - x_interval1)) / 2
            x_interval2 = torch.clamp(x_ceil - input, -1 + eps, 1 - eps)
            x_atanh2 = torch.log((1 + x_interval2) / (1 - x_interval2)) / 2

            self.rx_logits = torch.stack([-x_atanh1 / T, -x_atanh2 / T], dim=-1)
            self.rx = functional.softmax(self.rx_logits, dim=-1)
            
            x_tilde = torch.sum(x_bds * self.rx.detach(), dim=-1)
            return x_tilde
        else:
            return torch.round(input)


class Quantizator_SGA(nn.Module):
    """
    https://github.com/mandt-lab/improving-inference-for-neural-image-compression/blob/c9b5c1354a38e0bb505fc34c6c8f27170f62a75b/sga.py#L110
    Stochastic Gumbeling Annealing
    sample() has no grad, so we choose STE to backward. We can also try other estimate func.
    """
    def __init__(self, train=True, gap=1000, c=0.001):
        super(Quantizator_SGA, self).__init__()
        self.training = train
        self.gap = gap
        self.c = c

    @decorator_input
    def forward(self, input):
        if self.training:
            x_floor = Floor.apply(input, 0)
            x_ceil = Ceil.apply(input, 0)
            x_bds = torch.stack([x_floor, x_ceil], dim=-1)
            
            eps = 1e-5
            self.iteration = self.iteration // self.gap
            T = 0.5 * np.exp(-self.c * self.iteration)

            x_interval1 = torch.clamp(input - x_floor, -1 + eps, 1 - eps)
            x_atanh1 = torch.log((1 + x_interval1) / (1 - x_interval1)) / 2
            x_interval2 = torch.clamp(x_ceil - input, -1 + eps, 1 - eps)
            x_atanh2 = torch.log((1 + x_interval2) / (1 - x_interval2)) / 2
            
            self.rx_logits = torch.stack([-x_atanh1 / T, -x_atanh2 / T], dim=-1)
            self.rx = functional.softmax(self.rx_logits, dim=-1) # just for observation in tensorboard
            self.rx_dist = torch.distributions.RelaxedOneHotCategorical(T, logits=self.rx_logits)
            self.rx_sample = self.rx_dist.sample()
            
            x_tilde = torch.sum(x_bds * self.rx_sample.detach(), dim=-1)
            return x_tilde
        else:
            return torch.round(input)


def quantizators(tag="RT", B=1, train=True, gap=1000, c=0.001):
    if tag == "RT":
        return Quantizator_RT(B, train)
    elif tag == 'round':
        return STEQuant()
    elif tag == "UQ":
        return Quantizator_UQ(B, train)
    elif tag == "UQ_MIXED":
        # B represents delta here
        return Quantizator_UQ_MIXED(delta=B, train=train)
    elif tag == "RT_MIXED":
        # B represents delta here
        return Quantizator_RT_MIXED(delta=B, train=train)
    elif tag == "SOFT":
        return Quantizator_SOFT(B=B, train=train, sigma=-1., p=2.)
    elif tag == 'NONE':
        return None
    elif tag == "MYRT":
        return Quantizator_MYRT(B, train)
    elif tag == "order":
        return OrderQuant()
    elif tag == "noise_order":
        return NoiseOrderQuant(train=train)
    elif tag == "Frontier":
        return FrontierQuant(train=train, c=c)
    elif tag == "DA":
        return Quantizator_DA(train=train, gap=gap, c=c)
    elif tag == "DA_STE":
        return Quantizator_DA_STE(train=train, gap=gap, c=c)
    elif tag == "SGA":
        return Quantizator_SGA(train=train, gap=gap, c=c)
    elif tag == "MAP":
        return Quantizator_MAP(B=B, train=train)
    elif tag == "HARD":
        return Quantizator_HARD()
    else:
        raise NotImplementedError('Unsupported quant: {}'.format(tag))


if __name__ == '__main__':
    x = torch.randn(2, 1, 2, 2)
    if torch.cuda.is_available():
        x = x.cuda()
    print(x)
    test = Quantizator_RT()
    print(test(x))

    test = quantizators(tag="RT", B=6, train=True)
    print(test(x))
    test = quantizators(tag="RT", B=6, train=False)
    print(test(x))
    print(test(x))

    x = torch.randn(2, 1, 2, 2)
    if torch.cuda.is_available():
        x = x.cuda()
    print(x)
    test = Quantizator_UQ()
    print(test(x))

    x = torch.tensor([[1., 1.5, 200.], [-1., -1.7, -200.]])
    if torch.cuda.is_available():
        x = x.cuda()
    print(x)
    test = Quantizator_SOFT(sigma=-1)
    print(test(x))
    """
    print(test.get_Q(x, -1))
    print(test.get_Q(x, -10000000))
    """
    print(test.C.shape)
    x = test.sample_C(10)
    print(x)  # 257 points but 256 interval, sampled_C should be intergers
    print(test.get_index(torch.tensor([-400., -250., 600.])))
