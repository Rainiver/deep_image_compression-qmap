import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
from nets.components import initial, _conv_layer
from nets.norm import LowerBound_fn
from video.utils.misc_helper import decorator_input, clock


class Fake_Entropy(nn.Module):
    def __init__(self, num_filters):
        super(Fake_Entropy, self).__init__()

        self._layers = nn.ModuleList([
            nn.Conv2d(
                num_filters, num_filters, (3, 3), stride=1, padding=1,
                bias=True, padding_mode="zeros"),
            nn.Sigmoid()
        ])

    def forward(self, x):
        for layer in self._layers:
            x = layer(x)
        return x.clamp(0.00001, 0.99999)


class SymmetricConditional(nn.Module):
    def __init__(self, tag, scale_bound=0.11, likelihood_bound=1e-9,  key_in=None, key_out=None, **kwargs):
        super(SymmetricConditional, self).__init__()
        self.tag = tag
        self.delta = kwargs.get('delta', 1.)
        scale_bound = scale_bound * self.delta
        self.register_buffer('scale_bound', torch.tensor(scale_bound, dtype=torch.float))
        self.register_buffer('likelihood_bound', torch.tensor(likelihood_bound, dtype=torch.float))
        # self.K = kwargs.get('K', 1)
        self.key_in = key_in
        self.key_out = key_out

    def standardized_cumulative_gaussian(self, x):
        half = 0.5
        const = -(2 ** -0.5)
        # Using the complementary error function maximizes numerical precision.
        logits = half * torch.erfc(const * x)
        return logits

    def standardized_cumulative_logistic(self, x):
        return F.sigmoid(x)

    def standardized_cumulative_laplacian(self, x):
        exp = torch.exp(-torch.abs(x))
        return torch.where(x > 0, 2 - exp, exp) / 2

    @decorator_input
    #@clock
    def forward(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, omega=None):

        # omega = torch.ones(mean.size())

        if omega is not None:
            # x : NCHW -> N 1 C H W
            # mean, std, omega : N(C*K)HW -> N K C H W
            assert mean.shape == std.shape == omega.shape
            assert mean.shape[1] % x.shape[1] == 0
            C = x.shape[1]
            K = mean.shape[1] // C
            size = [x.shape[0], K, C, x.shape[2], x.shape[3]]
            x = x.unsqueeze(dim=1)
            mean = mean.view(size)
            std = std.view(size)
            omega = omega.view(size)
            # print('x', x.size())
            # print('mean', mean.size())

        if mean is not None:
            x = x - mean
        assert std is not None

        x = torch.abs(x)

        scale = LowerBound_fn(std, self.scale_bound)

        if self.tag == "gaussian":
            standardized_cumulative = self.standardized_cumulative_gaussian
        elif self.tag == "logistic":
            standardized_cumulative = self.standardized_cumulative_logistic
        elif self.tag == "laplacian":
            standardized_cumulative = self.standardized_cumulative_laplacian
        else:
            raise ValueError('unsupported tag {}'.format(self.tag))

        upper = standardized_cumulative((0.5 * self.delta - x) / scale)
        lower = standardized_cumulative((- 0.5 * self.delta - x) / scale)
        likelihood = upper - lower
        likelihood = LowerBound_fn(likelihood, self.likelihood_bound)

        if omega is not None:
            likelihood = likelihood * omega
            likelihood = likelihood.sum(dim=1)

        return likelihood


class Entropy_Bottleneck(nn.Module):
    def __init__(self, dim=3, input_channels=256, filters=None, init_scale=10,
                 tail_mass=2 ** -8, likelihood_bound=1e-9,
                 range_coder_precision=16, delta=1.0, key_in=None, key_out=None):
        """
        The layer assumes that the input tensor follows the 'NCHW' format. The
        layer trains an independent probability density model for each channel, but
        assumes that across all other dimensions, the inputs are i.i.d. (independent
        and identically distributed).

        From :

        https://github.com/tensorflow/compression/blob/master/tensorflow_compression/python/layers/entropy_models.py

        https://arxiv.org/abs/1802.01436

        :param dim: dimension of each filter
        :param filters: An iterable of ints, giving the number of filters at each layer
               of the density model. Generally, the more filters and layers, the more
               expressive is the density model in terms of modeling more complicated
               distributions of the layer inputs. For details, refer to the paper
               referenced above. The default is `[3, 3, 3]`, which should be sufficient
               for most practical purposes.
        :param init_scale: Float. A scaling factor determining the initial width of the
               probability densities. This should be chosen big enough so that the
               range of values of the layer inputs roughly falls within the interval
               [`-init_scale`, `init_scale`] at the beginning of training.

        TODO: apply the extra arguments

        :param tail_mass: (unimplemented) Float, between 0 and 1. The bottleneck layer automatically
               determines the range of input values based on their frequency of
               occurrence. Values occurring in the tails of the distributions will not
               be encoded with range coding, but using a Golomb-like code. `tail_mass`
               determines the amount of probability mass in the tails which will be
               Golomb-coded. For example, the default value of `2 ** -8` means that on
               average, one 256th of all values will use the Golomb code.
        :param likelihood_bound: (unimplemented) Float. If positive, the returned likelihood values are
               ensured to be greater than or equal to this value. This prevents very
               large gradients with a typical entropy loss (defaults to 1e-9).
        :param range_coder_precision: (unimplemented) Integer, between 1 and 16. The precision of the
               range coder used for compression and decompression. This trades off
               computation speed with compression efficiency, where 16 is the slowest
               but most efficient setting. Choosing lower values may increase the
               average codelength slightly compared to the estimated entropies.
        """
        super(Entropy_Bottleneck, self).__init__()

        if filters is None:
            filters = [dim] * 3

        filters = list(filters)
        num_features = [1] + filters + [1]

        scale = init_scale ** (1 / (len(filters) + 1))
        matrices = []
        biases = []
        factors = []
        softplus_layers = []
        self.likelihood_bound = likelihood_bound
        self.channel_size = input_channels
        self.delta = delta
        self.key_in = key_in
        self.key_out = key_out

        for i in range(len(filters) + 1):
            init = np.log(np.expm1(1 / scale / num_features[i + 1]))
            mat_data = torch.full((input_channels, num_features[i + 1], num_features[i]),
                                  init,
                                  requires_grad=True)
            matrix = nn.Parameter(mat_data)
            softplus = nn.Softplus()
            matrices.append(matrix)
            softplus_layers.append(softplus)

            bias_data = torch.empty(input_channels, num_features[i + 1], 1) \
                .uniform_(-.5, .5)
            bias_data.requires_grad = True
            bias = nn.Parameter(bias_data)
            biases.append(bias)

            if i < len(filters):
                factor = nn.Parameter(torch.zeros(input_channels, num_features[i + 1], 1,
                                                  requires_grad=True))
                factors.append(factor)

        self._filters = filters
        self._matrices = nn.ParameterList(matrices)
        self._biases = nn.ParameterList(biases)
        self._factors = nn.ParameterList(factors)
        self._softplus_layers = nn.ModuleList(softplus_layers)

    def _logits_cumulative(self, x):
        logits = x
        for i in range(len(self._filters) + 1):
            matrix = self._matrices[i]
            softplus = self._softplus_layers[i]
            matrix = softplus(matrix)
            logits = torch.matmul(matrix, logits)

            bias = self._biases[i]
            logits += bias

            if i < len(self._factors):
                factor = self._factors[i]
                logits += torch.tanh(factor) * torch.tanh(logits)
        return logits

    @decorator_input
    #@clock
    def forward(self, x):
        assert len(x.shape) == 4  # N, C, W, H
        x = x.permute(1, 0, 2, 3)  # C, N, W, H
        shape = x.shape
        x = x.reshape(x.shape[0], 1, -1)  # C, 1, N*W*H

        inputs = x
        half = 0.5 * self.delta
        lower = self._logits_cumulative(inputs - half)
        upper = self._logits_cumulative(inputs + half)

        # flip the sign for a higher precious
        #
        # sigmoid(x):
        #            ___
        #           /
        #          |
        #      ___/
        #      ^
        #    the left tail, where x is negative and small while sigmoid(x) -> 0.
        #
        # (according to the original code) We can use the special rule below
        # to only compute differences in the left tail of the sigmoid.
        # This increases numerical stability: sigmoid(x) is 1 for large x,
        # 0 for small x. Subtracting two numbers close to 0 can be done
        # with much higher precision than subtracting two numbers close to 1.
        sign = (- torch.sign(lower + upper)).detach()
        upper *= sign
        lower *= sign
        likelihood = torch.abs(torch.sigmoid(upper) - torch.sigmoid(lower))

        x = likelihood
        x = x.reshape(*shape)
        x = x.permute(1, 0, 2, 3)  # N,C,W,H
        return x.clamp(min=self.likelihood_bound)


def entropy_models(tag, num_filters=None, **kwargs):
    if tag == "fake":
        model = Fake_Entropy(num_filters)
        initial(model)
    elif tag == "gaussian":
        model = SymmetricConditional("gaussian", **kwargs)
    elif tag == "logistic":
        model = SymmetricConditional("logistic", **kwargs)
    elif tag == "laplacian":
        model = SymmetricConditional("laplacian", **kwargs)

    elif tag == 'bottleneck':
        # note the num_filters parameter here is not the input channel number
        model = Entropy_Bottleneck(dim=3, input_channels=num_filters, **kwargs)
    elif tag == 'NONE':
        model = None
    else:
        raise NotImplementedError(tag)
    return model


if __name__ == '__main__':
    m = entropy_models("gaussian")

    C = 3
    K = 2
    H = 10
    W = 10
    N = 2
    x = torch.randn(N, C, H, W, requires_grad=True)
    mu = torch.randn(N, K * C, H, W, requires_grad=True)
    sigma = torch.randn(N, K * C, H, W, requires_grad=True)
    omega = torch.randn(N, K, C, H, W, requires_grad=True).softmax(dim=1).view([N, K * C, H, W])

    print(x.size())
    y = m(x, mu, sigma, omega)
    print(y.size())
    x = torch.randn(N, K * C, H, W, requires_grad=True)
    y = m(x, mu, sigma)
    print(y.size())
    # print(y)
    z = sum(sum(sum(sum(y))))
    z.backward()

    # x = torch.randn(2, 2, 2, 2, requires_grad=True)
    # m = entropy_models("fake", 2)
    # if torch.cuda.is_available():
    #     x = x.cuda()
    #     m = m.cuda()
    # print(x)
    # print(m(x).size())
    #
    # mean = None
    # std = torch.tensor(1.)
    #
    # m = entropy_models("gaussian")
    # if torch.cuda.is_available():
    #     m = m.cuda()
    #     std = std.cuda()
    #     mean = mean.cuda() if mean is not None else None
    # y = m(x, mean, std)
    # print(y.size())
    # print(y)
    # z = sum(sum(sum(sum(y))))
    # z.backward()
    #
    # m = entropy_models("logistic")
    # if torch.cuda.is_available():
    #     m = m.cuda()
    # y = m(x, mean, std)
    # print(y.size())
    # print(y)
    # z = sum(sum(sum(sum(y))))
    # z.backward()
    #
    # m = entropy_models("laplacian")
    # if torch.cuda.is_available():
    #     m = m.cuda()
    # y = m(x, mean, std)
    # print(y.size())
    # print(y)
    # z = sum(sum(sum(sum(y))))
    # z.backward()
    #
    # # bottleneck model
    # m = entropy_models("bottleneck", 2)
    # if torch.cuda.is_available():
    #     m = m.cuda()  # test on GPU
    # y = m(x)
    # print(y.size())
    # print(y)
    # z = sum(sum(sum(sum(y))))
    # z.backward()
    # print(list(m.parameters()))
