import torch
import torch.nn as nn
from torch.nn import functional as F, Parameter
import numpy as np
from scipy import fftpack
from torch.autograd import Function
from nets.base import PreparableMixin
from quant.quantizator import quantizators


def get_irdft_matrix(shape, dtype=torch.float32):
    """Matrix for implementing kernel reparameterization with `torch.matmul`.

    This can be used to represent a kernel with the provided shape in the RDFT
    domain.

    Args:
      shape: Iterable of integers. Shape of kernel to apply this matrix to.
      dtype: `dtype` of returned matrix.

    Returns:
      `Tensor` of shape `(prod(shape), prod(shape))` and dtype `dtype`.
    """
    shape = tuple(int(s) for s in shape)
    # dtype = tf.as_dtype(dtype)
    # key = (tf.get_default_graph(), "irdft", shape, dtype.as_datatype_enum)
    # matrix = _matrix_cache.get(key)
    # if matrix is None:

    size = np.prod(shape)
    rank = len(shape)
    matrix = np.identity(size, dtype=np.float64).reshape((size,) + shape)
    for axis in range(rank):
        matrix = fftpack.rfft(matrix, axis=axis + 1)
        slices = (rank + 1) * [slice(None)]
        if shape[axis] % 2 == 1:
            slices[axis + 1] = slice(1, None)
        else:
            slices[axis + 1] = slice(1, -1)
        matrix[tuple(slices)] *= np.sqrt(2)
    matrix /= np.sqrt(size)
    matrix = np.reshape(matrix, (size, size))
    matrix = torch.tensor(
        matrix, dtype=dtype)
    # _matrix_cache[key] = matrix
    return matrix


class RdftParameterizer(torch.nn.Module, PreparableMixin):
    """
    Object encapsulating RDFT reparameterization.

    This uses the real-input discrete Fourier transform (RDFT) of a kernel as
    its parameterization. The inverse RDFT is applied to the variable to produce
    the parameter.

    see:
    https://github.com/tensorflow/compression/blob/master/tensorflow_compression/python/layers/parameterizers.py#L81

    also see:
    https://en.wikipedia.org/wiki/Discrete_Fourier_transform
    """

    def __init__(self, weights, init_fn=torch.nn.init.kaiming_normal_):
        super().__init__()
        # tf style kernel shape:
        #   [k_size, k_size, input_channels, output_channels]
        # torch style kernel shape:
        #   [input_channels, output_channels, k_size, k_size]
        ori_weights_size = weights.size()
        weights = weights.permute(2, 3, 0, 1)

        var_shape = weights.shape
        dtype = weights.dtype
        size = var_shape[0]

        for s in var_shape[1:-2]:
            size *= s

        rdft_shape = (size, var_shape[-2] * var_shape[-1])

        irdft_matrix = get_irdft_matrix(var_shape[:-2], dtype=dtype)
        irdft_matrix_t = irdft_matrix.t()

        # default = init_fn == torch.nn.init.kaiming_normal_
        # if default:
        #     init = torch.nn.init.kaiming_normal_(torch.zeros(rdft_shape, dtype=dtype))
        # else:
        #     # why not raise error when use 'None'?
        #     init = init_fn(torch.zeros(ori_weights_size, dtype=dtype))
        #     init = init.permute(2, 3, 0, 1).contiguous()
        #     init = init.reshape(rdft_shape).contiguous()

        # why not raise error when use 'None'?

        # (Dailan)2020/10/24 fix: below cause NAN error
        #                         when using cheng20 nets(or GMM?)
        #  amazing

        # init = init_fn(torch.zeros(ori_weights_size, dtype=dtype))
        # init = init.permute(2, 3, 0, 1).contiguous()
        # init = init.reshape(rdft_shape).contiguous()

        init = init_fn(torch.zeros(rdft_shape, dtype=dtype))

        init.requires_grad = True

        init = torch.reshape(init, (-1, rdft_shape[-1]))
        init = torch.matmul(irdft_matrix_t, init)  # tf.matmul, transpose_a=True
        init = torch.nn.Parameter(init)

        self.register_buffer('irdft_matrix', irdft_matrix)
        self.var_shape = var_shape
        self.rdft = init

        self.fixed = False

    def forward(self):
        if self.fixed:
            return self.rdft
        rdft, irdft_matrix = self.rdft, self.irdft_matrix
        var_shape = self.var_shape

        var = torch.matmul(irdft_matrix, rdft)
        var = torch.reshape(var, var_shape)

        # convert from tf style to torch style
        var = var.permute(2, 3, 0, 1)
        return var

    @property
    def data(self):
        return self.forward()

    def prepare(self):
        self.rdft = nn.Parameter(self.forward(), requires_grad=False)
        self.fixed = True


class SignalConv2d(torch.nn.Conv2d, PreparableMixin):
    def __init__(self, *args, **kwargs):
        self._super_inited = False
        if 'init_fn' in kwargs:
            init_fn = kwargs['init_fn']
            del kwargs['init_fn']
        else:
            init_fn = torch.nn.init.kaiming_normal_
        super().__init__(*args, **kwargs)
        w = self.weight
        self._super_inited = True
        rdft = RdftParameterizer(w, init_fn=init_fn)
        self._weight = rdft
        del self._parameters['weight']
        if self.bias is not None:
            self.bias = torch.nn.init.zeros_(self.bias)

        self.to_caffe = False

    def prepare(self):
        self.w2 = self._weight.forward()
        if self.to_caffe:
            self.w2 = nn.Parameter(self.w2)
        del self._weight
        self.to_caffe = True

    def __getattr__(self, item):
        if item == 'weight' and self._super_inited:
            if not self.to_caffe:
                return self._weight.forward()
            else:
                return self.w2
        else:
            return super().__getattr__(item)


class SignalConvTranspose2d(torch.nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs):
        self._super_inited = False
        if 'init_fn' in kwargs:
            init_fn = kwargs['init_fn']
            del kwargs['init_fn']
        else:
            init_fn = torch.nn.init.kaiming_normal_
        super().__init__(*args, **kwargs)
        w = self.weight
        self._super_inited = True
        rdft = RdftParameterizer(w, init_fn=init_fn)
        self._weight = rdft
        del self._parameters['weight']

        self.to_caffe = False

    def prepare(self):
        self.w2 = self._weight.forward()
        if self.to_caffe:
            self.w2 = nn.Parameter(self.w2)
        del self._weight
        self.to_caffe = True

    def __getattr__(self, item):
        if item == 'weight' and self._super_inited:
            if not self.to_caffe:
                return self._weight.forward()
            else:
                return self.w2
        else:
            return super().__getattr__(item)


class Conv_Upsample(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, kernel_size=None, stride=None, \
                 padding=None, bias=None, padding_mode=None, conv=None, upsample_mode=None, conv_first=True):
        super(Conv_Upsample, self).__init__()
        if upsample_mode == 'subpixel':
            out_channels = out_channels * stride ** 2
        self.conv = conv(in_channels, out_channels, kernel_size=5, stride=1, padding=2, \
                         bias=bias, padding_mode=padding_mode)

        assert upsample_mode in ['bilinear', 'nearest', 'subpixel']
        if upsample_mode == 'bilinear':
            self.upsample = nn.Upsample(scale_factor=stride, mode='bilinear')
        elif upsample_mode == 'nearest':
            self.upsample = nn.Upsample(scale_factor=stride, mode='nearest')
        elif upsample_mode == 'subpixel':
            self.upsample = nn.PixelShuffle(upscale_factor=stride)

        self.conv_first = conv_first
        if upsample_mode == 'subpixel':
            assert self.conv_first == True

    def forward(self, x):
        if self.conv_first:
            x = self.conv(x)
            x = self.upsample(x)
        else:
            x = self.upsample(x)
            x = self.conv(x)
        return x


class ConditionalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, \
                 padding, bias, padding_mode, conv, lambdas_len):
        super(ConditionalConv2d, self).__init__()
        self.conv = conv(in_channels, out_channels, kernel_size, stride=stride, padding=padding, \
                         bias=bias, padding_mode=padding_mode)
        self.fc1 = nn.Linear(lambdas_len, out_channels, bias=False)
        # self.fc2 = nn.Linear(lambdas_len, out_channels, bias=False)
        # self.fc1 = nn.Conv2d(lambdas_len, out_channels, (1, 1), stride=1, padding=0, bias=True, padding_mode="zeros")
        # self.fc2 = nn.Conv2d(lambdas_len, out_channels, (1, 1), stride=1, padding=0, bias=True, padding_mode="zeros")
        self.relu = nn.Softplus()
        self.num_filters = out_channels
        self.lambdas_len = lambdas_len

    def forward(self, x, one_hot):
        # n, c = one_hot.size()
        # one_hot = one_hot.view(n, c, n, n)
        x = self.conv(x)
        scale = self.relu(self.fc1(one_hot)).view(1, self.num_filters, 1, 1)
        # bias = self.fc2(one_hot).view(1, self.num_filters, 1, 1)
        x = x * scale
        # n, c, h, w = x.size()
        # bias =  
        # x = x + bias
        return x


class OLD_ConditionalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, \
                 padding, bias, padding_mode, conv, lambdas_len):
        super(OLD_ConditionalConv2d, self).__init__()
        self.conv = conv(in_channels, out_channels, kernel_size, stride=stride, padding=padding, \
                         bias=bias, padding_mode=padding_mode)
        self.fc1 = nn.Linear(lambdas_len, out_channels, bias=False)
        self.fc2 = nn.Linear(lambdas_len, out_channels, bias=False)
        self.softplus = nn.Softplus()
        self.num_filters = out_channels

    def forward(self, x, one_hot):
        x = self.conv(x)
        x = x * self.softplus(self.fc1(one_hot)).view(1, self.num_filters, 1, 1) + \
            self.fc2(one_hot).view(1, self.num_filters, 1, 1)
        return x


class MaskedConv2d(torch.nn.Conv2d):
    '''only use top-left part of the kernel
    '''

    def __init__(self, *args, **kwargs):
        self._super_inited = False
        super().__init__(*args, **kwargs)
        _, _, k, k = self.weight.size()
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

        self._weight = self.weight
        self._super_inited = True
        del self._parameters['weight']

        self.to_caffe = False

    def __getattr__(self, item):
        if item == 'weight' and self._super_inited and not self.to_caffe:
            return self._weight * self.mask
        else:
            return super().__getattr__(item)

    def prepare(self):
        self._parameters['weight'] = self._weight * self.mask


class MaskedSignalConv2d(torch.nn.Conv2d):
    def __init__(self, *args, **kwargs):
        self._super_inited = False
        super().__init__(*args, **kwargs)
        _, _, k, k = self.weight.size()
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

        w = self.weight
        self._super_inited = True
        rdft = RdftParameterizer(w)
        self._weight = rdft
        del self._parameters['weight']
        if self.bias is not None:
            self.bias = torch.nn.init.zeros_(self.bias)

        self.to_caffe = False

    def __getattr__(self, item):
        if item == 'weight' and self._super_inited and not self.to_caffe:
            return self.mask * self._weight.forward()
        else:
            return super().__getattr__(item)

    def prepare(self):
        self._parameters['weight'] = self.mask * self._weight.forward()


class BaseMaskedSignalConv2d(torch.nn.Conv2d):
    def __init__(self, *args, **kwargs):
        self._super_inited = False
        super().__init__(*args, **kwargs)
        _, _, k, k = self.weight.size()
        assert k % 2 == 1
        mask = self.get_mask(k)
        mask = mask.view(1, 1, k, k)
        self.mask = nn.Parameter(mask)
        self.mask.requires_grad = False

        w = self.weight
        self._super_inited = True
        rdft = RdftParameterizer(w)
        self._weight = rdft
        del self._parameters['weight']
        if self.bias is not None:
            self.bias = torch.nn.init.zeros_(self.bias)

        self.to_caffe = False

    def get_mask(self, size):
        raise NotImplementedError(self.__class__.__name__)

    def __getattr__(self, item):
        if item == 'weight' and self._super_inited and not self.to_caffe:
            return self.mask * self._weight.forward()
        else:
            return super().__getattr__(item)

    def prepare(self):
        self._parameters['weight'] = self.mask * self._weight.forward()


class CrossMaskedSignalConv2d(BaseMaskedSignalConv2d):
    def get_mask(self, size):
        assert size % 2 == 1
        row_odd = [0., 1.] * (size // 2) + [0.]  # 0 1 0 1 ... 0
        row_even = [1., 0.] * (size // 2) + [1.]  # 1 0 1 0 ... 1
        kernel = [row_odd, row_even] * (size // 2) + [row_odd]
        return torch.tensor(kernel)


class XMaskedSignalConv2d(BaseMaskedSignalConv2d):
    def get_mask(self, size):
        assert size in [3]
        return torch.tensor([[1., 0., 1.], [0., 0., 0.], [1., 0., 1.]])


class Knight1MaskedSignalConv2d(BaseMaskedSignalConv2d):
    def get_mask(self, size):
        assert size in [5]
        row_zero = [1., 0., 0., 0., 0.]
        row_one = [0., 0., 0., 1., 0.]
        row_two = [0., 1., 0., 0., 0.]
        row_three = [0., 0., 0., 0., 1.]
        row_four = [0., 0., 1., 0., 0.]
        kernel = [row_zero, row_one, row_two, row_three, row_four]
        return torch.tensor(kernel)


class Knight2MaskedSignalConv2d(BaseMaskedSignalConv2d):
    def get_mask(self, size):
        assert size in [5]
        row_zero = [1., 0., 0., 0., 1.]
        row_one = [0., 0., 1., 1., 0.]
        row_two = [1., 1., 0., 0., 0.]
        row_three = [0., 0., 0., 1., 1.]
        row_four = [0., 1., 1., 0., 0.]
        kernel = [row_zero, row_one, row_two, row_three, row_four]
        return torch.tensor(kernel)


class Knight3MaskedSignalConv2d(BaseMaskedSignalConv2d):
    def get_mask(self, size):
        assert size in [5]
        row_zero = [1., 0., 0., 1., 1.]
        row_one = [0., 1., 1., 1., 0.]
        row_two = [1., 1., 0., 0., 1.]
        row_three = [0., 0., 1., 1., 1.]
        row_four = [1., 1., 1., 0., 0.]
        kernel = [row_zero, row_one, row_two, row_three, row_four]
        return torch.tensor(kernel)


class Knight4MaskedSignalConv2d(BaseMaskedSignalConv2d):
    def get_mask(self, size):
        assert size in [5]
        row_zero = [1., 0., 1., 1., 1.]
        row_one = [1., 1., 1., 1., 0.]
        row_two = [1., 1., 0., 1., 1.]
        row_three = [0., 1., 1., 1., 1.]
        row_four = [1., 1., 1., 0., 1.]
        kernel = [row_zero, row_one, row_two, row_three, row_four]
        return torch.tensor(kernel)


class KnightMaskedSignalConv2d(BaseMaskedSignalConv2d):
    def get_mask(self, size):
        # fake mask here
        return torch.ones(size, size)


class Descale(nn.Module):
    """
        Scale x from [0,1] to [0,255] or vice versa if reverse=True
    """

    def forward(self, x, reverse=False):
        if not reverse:
            x = torch.round(x * 255)
        else:
            x = x / 255
        return x


class Normalize(nn.Module):
    def __init__(self, inverse_bin_width=256, **kwargs):
        # set bin_width to power of 2 can avoid floating error
        super().__init__()
        self.domain = inverse_bin_width

    def forward(self, x, reverse=False):
        if not reverse:
            x = (x - self.domain / 2) / self.domain
        else:
            x = x * self.domain + self.domain / 2
        return x


class Squeeze(nn.Module):
    def __init__(self):
        super().__init__()

    def space_to_depth(self, x):
        xs = x.size()
        # Pick off every second element
        x = x.view(xs[0], xs[1], xs[2] // 2, 2, xs[3] // 2, 2)
        # Transpose picked elements next to channels.
        x = x.permute((0, 1, 3, 5, 2, 4)).contiguous()
        # Combine with channels.
        x = x.view(xs[0], xs[1] * 4, xs[2] // 2, xs[3] // 2)
        return x

    def depth_to_space(self, x):
        xs = x.size()
        # Pick off elements from channels
        x = x.view(xs[0], xs[1] // 4, 2, 2, xs[2], xs[3])
        # Transpose picked elements next to HW dimensions.
        x = x.permute((0, 1, 4, 2, 5, 3)).contiguous()
        # Combine with HW dimensions.
        x = x.view(xs[0], xs[1] // 4, xs[2] * 2, xs[3] * 2)
        return x

    def forward(self, z, ldj, reverse=False):
        if not reverse:
            z = self.space_to_depth(z)
        else:
            z = self.depth_to_space(z)
        return z, ldj


class Permute(nn.Module):
    def __init__(self, n_channels):
        super().__init__()

        permutation = np.arange(n_channels, dtype='int')
        np.random.shuffle(permutation)

        permutation_inv = np.zeros(n_channels, dtype='int')
        permutation_inv[permutation] = np.arange(n_channels, dtype='int')

        self.permutation = torch.from_numpy(permutation)
        self.permutation_inv = torch.from_numpy(permutation_inv)

    def forward(self, z, ldj, reverse=False):
        if not reverse:
            z = z[:, self.permutation, :, :]
        else:
            z = z[:, self.permutation_inv, :, :]

        return z, ldj

    def InversePermute(self):
        inv_permute = Permute(len(self.permutation))
        inv_permute.permutation = self.permutation_inv
        inv_permute.permutation_inv = self.permutation
        return inv_permute


class Conv2dReLU(nn.Module):
    def __init__(
            self, n_inputs, n_outputs, kernel_size=3, stride=1, padding=0,
            bias=True):
        super().__init__()
        self.nn = nn.Conv2d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        h = self.nn(x)
        y = F.relu(h)
        return y


class Swish(nn.Module):
    def __init__(self, beta=1):
        super().__init__()
        self.beta = nn.Parameter(beta * torch.ones(1))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class Conv2dSwish(nn.Module):
    def __init__(
            self, n_inputs, n_outputs, kernel_size=3, stride=1, padding=0,
            bias=True):
        super().__init__()
        self.nn = nn.Conv2d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, bias=bias)
        if n_outputs % 3 == 0:
            n_groups = 3
        elif n_outputs % 2 == 0:
            n_groups = 2
        else:
            n_groups = 1
        self.gn = nn.GroupNorm(n_groups, n_outputs)
        self.swish = Swish()

    def forward(self, x):
        h = self.nn(x)
        h2 = self.gn(h)
        y = self.swish(h2)
        return y


class ResidualBlock(nn.Module):
    def __init__(self, n_channels, kernel, Conv2dAct):
        super().__init__()
        self.nn = torch.nn.Sequential(
            Conv2dAct(n_channels, n_channels, kernel, padding=1),
            torch.nn.Conv2d(n_channels, n_channels, kernel, padding=1),
        )

    def forward(self, x):
        h = self.nn(x)
        h = F.relu(h + x)
        return h


class DenseLayer(nn.Module):
    def __init__(self, n_inputs, growth, kernel, Conv2dAct):
        super().__init__()
        conv1x1 = Conv2dAct(
            n_inputs, n_inputs, kernel_size=1, stride=1,
            padding=0, bias=True)
        self.nn = torch.nn.Sequential(
            conv1x1,
            Conv2dAct(
                n_inputs, growth, kernel_size=kernel, stride=1,
                padding=1, bias=True),
        )

    def forward(self, x):
        h = self.nn(x)
        h = torch.cat([x, h], dim=1)
        return h


class DenseBlock(nn.Module):
    def __init__(
            self, n_inputs, n_outputs, kernel, Conv2dAct, densenet_depth, **kwargs):
        super().__init__()
        depth = densenet_depth

        future_growth = n_outputs - n_inputs

        layers = []

        for d in range(depth):
            growth = future_growth // (depth - d)

            layers.append(DenseLayer(n_inputs, growth, kernel, Conv2dAct))
            n_inputs += growth
            future_growth -= growth

        self.nn = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.nn(x)


class NN(nn.Module):
    # network used in coupling layer and factor-out layer
    def __init__(
            self, c_in, c_out, nn_type, n_channels, kernel=3, **kwargs):
        super().__init__()
        Conv2dAct = Conv2dReLU
        if nn_type == 'resnet':
            layers = [
                Conv2dAct(c_in, n_channels, kernel, padding=1),
                ResidualBlock(n_channels, kernel, Conv2dAct),
                ResidualBlock(n_channels, kernel, Conv2dAct)]
            layers += [
                torch.nn.Conv2d(n_channels, c_out, kernel, padding=1)
            ]

        elif nn_type == 'densenet':
            layers = [
                DenseBlock(
                    n_inputs=c_in,
                    n_outputs=n_channels + c_in,
                    kernel=kernel,
                    Conv2dAct=Conv2dAct, **kwargs)]
            layers += [
                torch.nn.Conv2d(n_channels + c_in, c_out, kernel, padding=1)
            ]
        elif nn_type == 'densenet++':
            layers = [
                DenseBlock(
                    n_inputs=c_in,
                    n_outputs=n_channels + c_in,
                    kernel=kernel,
                    Conv2dAct=Conv2dSwish, **kwargs)]
            layers += [
                torch.nn.Conv2d(n_channels + c_in, c_out, kernel, padding=1)
            ]
        else:
            raise ValueError

        self.nn = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.nn(x)


class SFT(nn.Module):
    def __init__(self, x_nc, prior_nc=1, ks=3, nhidden=128):
        super().__init__()
        pw = ks // 2

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(prior_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, x_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, x_nc, kernel_size=ks, padding=pw)

    def forward(self, x, qmap):
        qmap = F.adaptive_avg_pool2d(qmap, x.size()[2:])
        actv = self.mlp_shared(qmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = x * (1 + gamma) + beta

        return out


class SFTResblk(nn.Module):
    def __init__(self, x_nc, prior_nc, ks=3):
        super().__init__()
        self.conv_0 = nn.Conv2d(x_nc, x_nc, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(x_nc, x_nc, kernel_size=3, padding=1)

        self.norm_0 = SFT(x_nc, prior_nc, ks=ks)
        self.norm_1 = SFT(x_nc, prior_nc, ks=ks)

    def forward(self, x, qmap):
        dx = self.conv_0(self.actvn(self.norm_0(x, qmap)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, qmap)))
        out = x + dx

        return out

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class BackRound(nn.Module):
    def __init__(self, inverse_bin_width, round_approx, **kwargs):
        """
        BackRound is an approximation to Round that allows for Backpropagation.

        Approximate the round function using a sum of translated sigmoids.
        The temperature determines how well the round function is approximated,
        i.e., a lower temperature corresponds to a better approximation, at
        the cost of more vanishing gradients.

        BackRound supports the following settings:
        * By setting hard to True and temperature > 0.25, BackRound
          reduces to a round function with a straight through gradient
          estimator
        * When using 0 < temperature <= 0.25 and hard = True, the
          output in the forward pass is equivalent to a round function, but the
          gradient is approximated by the gradient of a sum of sigmoids.
        * When using hard = False, the output is not constrained to integers.
        * When temperature > 0.25 and hard = False, BackRound reduces to
          the identity function.

        Arguments
        ---------
        temperature: float
            Temperature used for stacked sigmoid approximated. If temperature
            is greater than 0.25, the approximation reduces to the identity
            function.
        hard: bool
            If hard is True, a (hard) round is applied before returning. The
            gradient for this is approximated using the straight-through
            estimator.
        """
        super().__init__()
        self.inverse_bin_width = inverse_bin_width
        self.round_approx = round_approx

        if round_approx == 'smooth':
            self.round = quantizators(tag='round')
        elif round_approx == 'stochastic':
            self.round = quantizators(tag='RT')
        else:
            raise ValueError

    def forward(self, x):
        if self.round_approx == 'smooth' or self.round_approx == 'stochastic':
            h = x * self.inverse_bin_width
            h = self.round(h)
            return h / self.inverse_bin_width
        else:
            raise ValueError


class SplitFactorCoupling(nn.Module):
    # Coupling layer used in IDF
    def __init__(self, c_in, factor, coupling_type, rezero, **kwargs):
        super().__init__()
        self.kernel = 3
        self.round = BackRound(**kwargs)
        self.split_idx = c_in - (c_in // factor)
        self.rezero = rezero
        if self.rezero:
            self.alpha = Parameter(torch.zeros(1))
        self.nn = NN(
            c_in=self.split_idx,
            c_out=c_in - self.split_idx,
            kernel=self.kernel,
            nn_type=coupling_type, **kwargs)

    def forward(self, z, ldj, reverse=False):
        z1 = z[:, :self.split_idx, :, :]
        z2 = z[:, self.split_idx:, :, :]

        t = self.nn(z1)
        if self.rezero:
            t *= self.alpha
        if self.round is not None:
            t = self.round(t)

        if not reverse:
            z2 = z2 + t
        else:
            z2 = z2 - t
        z = torch.cat([z1, z2], dim=1)
        return z, ldj


class SplitPrior(nn.Module):
    # Factor-out layer used in IDF
    def __init__(self, c_in, factor_out, splitprior_type, inverse_bin_width, rezero, **kwargs):
        super().__init__()

        self.split_idx = c_in - factor_out
        self.inverse_bin_width = inverse_bin_width
        self.input_channel = c_in
        self.rezero = rezero
        if self.rezero:
            self.gamma = Parameter(torch.zeros(1))
            self.delta = Parameter(torch.zeros(1))

        self.nn = NN(
            c_in=c_in - factor_out,
            c_out=factor_out * 2,
            nn_type=splitprior_type,
            **kwargs)

    def get_py(self, z):
        h = self.nn(z)
        mu = h[:, ::2, :, :]
        logs = h[:, 1::2, :, :]
        if self.rezero:
            mu *= self.gamma
            logs *= self.delta

        py = [mu, logs]
        return py

    def split(self, z):
        z1 = z[:, :self.split_idx, :, :]
        y = z[:, self.split_idx:, :, :]
        return z1, y

    def combine(self, z, y):
        result = torch.cat([z, y], dim=1)
        return result

    def forward(self, z, ldj):
        z, y = self.split(z)
        py = self.get_py(z)
        return py, y, z, ldj

    def sample_prior(self, mean, logscale, inverse_bin_width):
        y = torch.randn_like(mean)
        x = torch.exp(logscale) * y + mean
        x = torch.round(x * inverse_bin_width) / inverse_bin_width
        return x

    def inverse(self, z, ldj, y):
        # Sample if y is not given.
        if y is None:
            py = self.get_py(z)
            y = self.sample_prior(*py, self.inverse_bin_width)
        z = self.combine(z, y)
        return z, ldj

    def decode(self, z, ldj, states, decode_fn):
        py = self.get_py(z)
        states, y = decode_fn(states, py)
        return self.combine(z, y), ldj, states


class Prior(nn.Module):
    def __init__(self, channel, height, width, inverse_bin_width=256, n_mixtures=1, **kwargs):
        super().__init__()
        self.inverse_bin_width = inverse_bin_width
        self.n_mixtures = n_mixtures
        # SymmetricConditional module requires input size being N(C*K)HW
        self.size = (self.n_mixtures * channel, height, width)
        if self.n_mixtures == 1:
            self.mu = Parameter(torch.Tensor(channel, height, width))
            self.logs = Parameter(torch.Tensor(channel, height, width))
        elif self.n_mixtures > 1:
            self.mu = Parameter(torch.Tensor(self.n_mixtures, channel, height, width))
            self.logs = Parameter(torch.Tensor(self.n_mixtures, channel, height, width))
            self.pi_logit = Parameter(torch.Tensor(self.n_mixtures, channel, height, width))

        self.reset_parameters()

    def reset_parameters(self):
        self.mu.data.zero_()
        if self.n_mixtures > 1:
            self.pi_logit.data.zero_()
            for i in range(self.n_mixtures):
                self.mu.data[i] += i - (self.n_mixtures - 1) / 2.

        self.logs.data.zero_()

    def get_pz(self, n):
        if self.n_mixtures == 1:
            mu = self.mu.repeat(n, 1, 1, 1)
            # scaling scale
            logs = self.logs.repeat(n, 1, 1, 1)
            scale = torch.exp(logs)
            return mu, scale

        elif self.n_mixtures > 1:
            pi = F.softmax(self.pi_logit, dim=0)
            mu = self.mu.view(self.size).repeat(n, 1, 1, 1)
            logs = self.logs.view(self.size).repeat(n, 1, 1, 1)
            scale = torch.exp(logs)
            pi = pi.view(self.size).repeat(n, 1, 1, 1)
            return mu, scale, pi

    def forward(self, z, ldj):
        pz = self.get_pz(z.size(0))
        return pz, z, ldj

    def decode(self, states, decode_fn):
        pz = self.get_pz(n=len(states))
        states, z = decode_fn(states, pz)
        return states, z


if __name__ == '__main__':
    x = torch.randn(1, 1, 7, 7)
    print(x)
    test = MaskedConv2d(1, 1, (7, 7), stride=1, padding=0,
                        bias=True, padding_mode="zeros")
    with torch.no_grad():
        print(test(x))
        print(test.weight)
        print(test.mask * test.weight)
        
    test = SignalConv2d(1, 1, (7, 7), stride=1, padding=0,
                        bias=True, padding_mode="zeros")
    with torch.no_grad():
        test.train()
        print(test(x))
        test.eval()
        print(test(x))

    x = torch.randn(1, 1, 3, 3)
    print(x)
    test = CrossMaskedSignalConv2d(1, 1, (3, 3), stride=1, padding=0,
                                   bias=True, padding_mode="zeros")
    with torch.no_grad():
        test.train()
        print(test(x))
        test.eval()
        print(test(x))
        print(test.mask * test.weight)

    test = XMaskedSignalConv2d(1, 1, (3, 3), stride=1, padding=0,
                               bias=True, padding_mode="zeros")
    with torch.no_grad():
        test.train()
        print(test(x))
        test.eval()
        print(test(x))
        print(test.mask * test.weight)

