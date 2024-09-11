import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
try:
    from integer2.module import EMAAct, get_quant_conv, app
    assert app == 'DIC'
except:
    print("norm load integer2 failed", flush=True)
    from integer.module import EMAAct, get_quant_conv

# TODO: delete
def abs_func(inputs):
    # b, c, h, w = inputs.size()
    # neg_t = torch.Tensor([-1])
    # neg_t = neg_t.repeat([b, c, h, w])
    # return F.relu(inputs) + F.relu(inputs * neg_t)
    return torch.sqrt(inputs * inputs)
    # return torch.abs(inputs)


def lowerbound(inputs, bound):
    return torch.max(inputs, bound)


class Power(nn.Module):
    def __init__(self, power):
        super(Power, self).__init__()
        self.power = nn.Parameter(torch.Tensor([power]), requires_grad=False)

    def forward(self, x):
        return torch.pow(x, self.power)


class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        ones = torch.ones_like(inputs)
        b = ones * bound
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors

        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


class LowerBound_v2(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        ones = torch.ones_like(inputs)
        b = ones * bound
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors

        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


LowerBound_fn = LowerBound_v2.apply


class GDN(nn.Module):
    r"""Generalized divisive normalization layer.
        ```
        y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]^2))
        ```
        where `i` and `j` run over channels. This implementation never sums across
        spatial dimensions. It is similar to local response normalization, but much
        more flexible, as `beta` and `gamma` are trainable parameters.

        The parameterization of beta and gamma as their square roots lets
        the training slow down when their values are close to zero, which is
        desirable as small values in the denominator can lead to a situation
        where gradient noise on beta/gamma leads to extreme amounts of noise in
        the GDN activations. However, without the offset, we would get zero
        gradients if any elements of beta or gamma were exactly zero, and thus
        the training could get stuck. To prevent this, we add this small
        constant. The default value was empirically determined as a good
        starting point. Making it bigger potentially leads to more gradient
        noise on the activations, making it too small may lead to numerical
        precision issues.

        Computationally Efficient Neural Image Compression:
        1DN:
                y[i] = x[i] / (beta[i] + sum_j(gamma[j, i] * abs(x[j])))
    """
    alpha = 2,
    epsilon = 0.5,

    def __init__(self,
                 ch=None,
                 inverse=False,
                 beta_min=1e-6,
                 gamma_init=.1,
                 reparam_offset=2 ** -18,
                 fast=False,
                 quant=False,
                 init_fn=None):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.register_buffer('beta_min', torch.FloatTensor([beta_min]))
        self.register_buffer('gamma_init', torch.FloatTensor([gamma_init]))
        self.register_buffer('reparam_offset', torch.FloatTensor([reparam_offset]))
        self.register_buffer('pedestal', self.reparam_offset ** 2)

        self.beta_bound = (self.beta_min + self.reparam_offset ** 2) ** .5
        self.gamma_bound = self.reparam_offset
        self.beta_bound = nn.Parameter(self.beta_bound, requires_grad=False)
        self.gamma_bound = nn.Parameter(self.gamma_bound, requires_grad=False)

        # Create beta param
        beta = torch.sqrt(torch.ones(ch) + self.pedestal)
        self.beta = nn.Parameter(beta)

        # Create gamma param
        eye = torch.eye(ch)
        g = self.gamma_init * eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)

        if init_fn is not None:
            ori_gamma_shape = gamma.shape
            gamma = init_fn(torch.zeros(ch, ch, 1, 1))
            self.gamma = nn.Parameter(gamma.reshape(ori_gamma_shape))
        else:
            self.gamma = nn.Parameter(gamma)

        self.to_caffe = False
        if self.alpha == 1:
            self.pow1 = torch.abs
            # self.pow1 = abs_func
        else:
            self.pow1 = Power(self.alpha[0])
        self.pow2 = Power(self.epsilon[0])
        self.pow3 = Power(-1.0)

        self.fast = fast
        self.quant = quant
        self.ch = ch
        if self.quant:
            assert self.fast == True
            quant_conv = get_quant_conv(signal=False)
            self.qconv = quant_conv(in_channels=ch, out_channels=ch, kernel_size=(1, 1))
            self.qact = EMAAct(channel_num=ch)
            self.beta = self.qconv.bias
            self.gamma = self.qconv.weight

    def prepare(self):
        ch = self.ch

        # Beta bound and reparam
        if not self.to_caffe:
            beta = LowerBound_fn(self.beta, self.beta_bound)
        else:
            beta = lowerbound(self.beta, self.beta_bound)
        beta = beta ** 2 - self.pedestal

        # Gamma bound and reparam
        if not self.to_caffe:
            gamma = LowerBound_fn(self.gamma, self.gamma_bound)
        else:
            gamma = lowerbound(self.gamma, self.gamma_bound)
        gamma = gamma ** 2 - self.pedestal
        gamma = gamma.view(ch, ch, 1, 1)

        self.beta2, self.gamma2 = beta, gamma

        # if self.quant:
        #    self.qconv.weight = self.gamma2
        #    self.qconv.bias = self.beta2

        if self.to_caffe:
            self.beta2 = nn.Parameter(self.beta2)
            self.gamma2 = nn.Parameter(self.gamma2)

    def forward(self, inputs):
        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size()
            inputs = inputs.view(bs, ch, d * w, h)

        _, ch, _, _ = inputs.size()

        if not self.to_caffe:
            self.prepare()

        # Norm pool calc
        if not self.fast:
            temp = self.pow1(inputs)
            norm_ = nn.functional.conv2d(temp, self.gamma2, self.beta2)
            norm_ = self.pow2(norm_)
        else:
            if not self.to_caffe:
                temp = torch.abs(inputs)
                # temp = abs_func(inputs)
            else:
                # temp = torch.abs(inputs)
                temp = self.pow1(inputs)
                temp = self.pow2(temp)
                # temp = torch.relu(inputs) + torch.relu(-inputs)

            if self.quant:
                norm_ = self.qconv(temp, self.gamma2, self.beta2)
                norm_ = self.qact(norm_)
            else:
                norm_ = nn.functional.conv2d(temp, self.gamma2, self.beta2)

        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            norm_ = self.pow3(norm_)
            outputs = inputs * norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
        return outputs


class GDNEfficient(GDN):
    alpha = 1
    epsilon = 1


class L1GDNNoPow(GDN):
    """
    L1 norm GDN without power-op, for speeding up

    see:
    Computationally Efficient Neural Image Compression

    $zi = xi / (beta_i + \\SUM_j{ gamma_ij |xj| } )$
    """
    alpha = 1,  # NOTICE: this is a tuple (1, )
    epsilon = 1,

    def forward(self, inputs):
        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size()
            inputs = inputs.view(bs, ch, d * w, h)

        _, ch, _, _ = inputs.size()

        if not self.to_caffe:
            self.prepare()  # re-parameterize gamma and beta

        # Norm pool calc
        temp = inputs.abs()
        # temp = abs_func(inputs)
        norm_ = nn.functional.conv2d(temp, self.gamma2, self.beta2)

        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
        return outputs

    def count_ops(m: nn.Module, x: (torch.Tensor,), y: torch.Tensor):
        x = x[0]

        kernel_ops = torch.zeros(m.gamma2.size()[2:]).numel()  # Kw x Kh
        bias_ops = 1
        mul_div_ops = 1

        # N x Cout x H x W x  (Cin x Kw x Kh + bias + mul_div)
        total_ops = y.nelement() * (x.size()[1] * kernel_ops + bias_ops + mul_div_ops)

        m.total_ops += torch.DoubleTensor([int(total_ops)])


if __name__ == '__main__':
    swish = Swish()
    if torch.cuda.is_available():
        swish = swish.cuda()
    for p in swish.parameters():
        print(p.name, p.data)
    print(swish.beta)

    x = torch.randn(2, 3)
    if torch.cuda.is_available():
        x = x.cuda()
    print(x)
    y = swish(x)
    print(y)
    t = y.sum()
    t.backward()

    print(swish)
    print(list(swish.parameters()))
    print(swish.beta.grad)
