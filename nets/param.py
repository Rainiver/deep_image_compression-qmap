import torch
import torch.nn as nn
import numpy as np
import math
from nets.layers import SignalConv2d, ConditionalConv2d
from nets.layers_quant import GGQConvTranspose2d, GGQReLU, GGBpAct, GGQConv2d
from nets.layers_quant import GGQBiReLU


class BaseParam(nn.Module):
    def __init__(self, in_channels=3, out_channels=192, conv='signal', lambda4s=[1, 2], c_bit=None, **kwargs):
        super(BaseParam, self).__init__()
        self.in_channels = in_channels
        self.num_filters = out_channels
        self.caffe_channels = in_channels

        self.lambda4s = lambda4s
        self.sampled_lambda = 1

        if conv == 'conv':
            self.conv = nn.Conv2d
        elif conv == 'signal':
            self.conv = SignalConv2d
        else:
            raise NotImplemented('unsupported conv layer {}'.format(conv))

        # c_bit for K in c, default --> [8, 8, 8]
        if c_bit:
            self.c_bit = c_bit
        else:
            self.c_bit = [8, 8, 8]

        self._layers = nn.ModuleList([])
        self.build()

    def build(self):
        raise NotImplementedError

    def forward(self, x):
        for layer in self._layers:
            x = layer(x)
        return x


class BaseParam_VAR(BaseParam):
    def forward(self, x):
        lambdas = self.lambda4s[1:]

        bools = np.array(lambdas) == self.sampled_lambda
        one_hot = [int(b) for b in bools]
        one_hot = torch.Tensor(one_hot)
        one_hot.unsqueeze_(0)
        if torch.cuda.is_available():
            one_hot = one_hot.cuda()

        for layer in self._layers:
            if isinstance(layer, ConditionalConv2d):
                x = layer(x, one_hot)
            else:
                x = layer(x)
        return x


class Param_GG18C(BaseParam):
    def build(self):
        self._layers = nn.ModuleList([
            self.conv(
                self.in_channels, 640, (1, 1), stride=1, padding=0,
                bias=True, padding_mode="zeros"),
            nn.LeakyReLU(),
            self.conv(
                640, 512, (1, 1), stride=1, padding=0,
                bias=True, padding_mode="zeros"),
            nn.LeakyReLU(),
            self.conv(
                512, self.num_filters, (1, 1), stride=1, padding=0,
                bias=False, padding_mode="zeros")
        ])


class ParamLinear3x3(BaseParam):
    def build(self):
        d = (self.in_channels - self.num_filters) // 3
        c1 = self.in_channels - d
        c2 = self.in_channels - 2*d
        self._layers = nn.ModuleList([
            self.conv(
                self.in_channels, c1, (3, 3), stride=1, padding=1,
                bias=True, padding_mode="zeros"),
            nn.ReLU(),
            self.conv(
                c1, c2, (3, 3), stride=1, padding=1,
                bias=True, padding_mode="zeros"),
            nn.ReLU(),
            self.conv(
                c2, self.num_filters, (3, 3), stride=1, padding=1,
                bias=False, padding_mode="zeros")
        ])


class GG20CParamTransform5x5(BaseParam):
    def build(self):
        self._layers = nn.ModuleList([
            self.conv(
                self.in_channels, 224, (5, 5), stride=1, padding=2,
                bias=True, padding_mode="zeros"),
            nn.ReLU(inplace=True),
            self.conv(
                224, 128, (5, 5), stride=1, padding=2,
                bias=True, padding_mode="zeros"),
            nn.ReLU(inplace=True),
            self.conv(
                128, self.num_filters, (3, 3), stride=1, padding=1,
                bias=True, padding_mode="zeros")
        ])


class GG20CTwoPathParam(BaseParam):
    param_transform = GG20CParamTransform5x5

    def __init__(self, in_channels=3, out_channels=192, **kwargs):
        self.param_kwargs = kwargs
        super().__init__(in_channels, out_channels, **kwargs)

    def build(self):
        ci = self.in_channels
        co = self.num_filters
        assert co % 2 == 0
        assert ci % 2 == 0

        half_ci = ci // 2
        half_co = co // 2

        self._layers = nn.ModuleList([nn.Sequential(
            self.param_transform(half_ci, half_co, **self.param_kwargs)
        ) for _ in range(2)])

    def forward(self, x):
        hyper = torch.chunk(x, 2, 1)
        mu, sigma = (layer(h) for layer, h in zip(self._layers, hyper))
        return torch.cat([mu, sigma], 1)


class TwoPathInOutExpSigmaParamLinear1x1(BaseParam):
    def build(self):
        ci = self.in_channels
        co = self.num_filters
        assert co % 2 == 0
        assert ci % 2 == 0

        half_ci = ci // 2
        half_co = co // 2

        d = (half_ci - half_co) // 3
        c1 = half_ci - d
        c2 = half_ci - 2*d
        self._layers = nn.ModuleList([nn.Sequential(
            self.conv(
                half_ci, c1, (1, 1), stride=1, padding=0,
                bias=True, padding_mode="zeros"),
            nn.ReLU(),
            self.conv(
                c1, c2, (1, 1), stride=1, padding=0,
                bias=True, padding_mode="zeros"),
            nn.ReLU(),
            self.conv(
                c2, half_co, (1, 1), stride=1, padding=0,
                bias=False, padding_mode="zeros")
        ) for _ in range(2)])

    def forward(self, x):
        hyper = torch.chunk(x, 2, 1)
        mu, sigma = (layer(h) for layer, h in zip(self._layers, hyper))
        sigma = torch.exp(sigma)
        return torch.cat([mu, sigma], 1)


class ParamLinear1x1(BaseParam):
    def build(self):
        d = (self.in_channels - self.num_filters) // 3
        c1 = self.in_channels - d
        c2 = self.in_channels - 2*d
        self._layers = nn.ModuleList([
            self.conv(
                self.in_channels, c1, (1, 1), stride=1, padding=0,
                bias=True, padding_mode="zeros"),
            nn.ReLU(),
            self.conv(
                c1, c2, (1, 1), stride=1, padding=0,
                bias=True, padding_mode="zeros"),
            nn.ReLU(),
            self.conv(
                c2, self.num_filters, (1, 1), stride=1, padding=0,
                bias=False, padding_mode="zeros")
        ])


class ParamLinearExpSigma1x1(ParamLinear1x1):
    def forward(self, x):
        x = super().forward(x)
        mu, sigma = torch.split(x, self.num_filters // 2, 1)
        sigma = torch.exp(sigma)
        x = torch.cat([mu, sigma], 1)
        return x


class TwoPathParamLinearExpSigma1x1(ParamLinear1x1):
    def __init__(self, *args, c_ctx=392, **kwargs):
        self.c_ctx = c_ctx
        super().__init__(*args, **kwargs)

    def build(self):
        c_ctx = self.c_ctx
        cin = (self.in_channels - c_ctx) // 2 + c_ctx
        cout = self.num_filters // 2
        d = (cin - cout) // 3
        c1 = cin - d
        c2 = cin - 2 * d
        self._layers = nn.ModuleList([nn.Sequential(
            self.conv(
                cin, c1, (1, 1), stride=1, padding=0,
                bias=True, padding_mode="zeros"),
            nn.ReLU(),
            self.conv(
                c1, c2, (1, 1), stride=1, padding=0,
                bias=True, padding_mode="zeros"),
            nn.ReLU(),
            self.conv(
                c2, cout, (1, 1), stride=1, padding=0,
                bias=False, padding_mode="zeros")
        ) for _ in range(2)])

    def forward(self, x):
        c_ctx = self.c_ctx
        ctx = x[:, c_ctx:, ...]
        x = x[:, :c_ctx, ...]
        x = torch.split(x, x.shape[1] // 2, 1)
        mu, sigma = (l(torch.cat((s, ctx), 1)) for l, s in zip(self._layers, x))
        sigma = torch.exp(sigma)
        return torch.cat((mu, sigma), 1)


class TwoPathParamLinear1x1(ParamLinear1x1):
    def __init__(self, *args, c_ctx=392, **kwargs):
        self.c_ctx = c_ctx
        super().__init__(*args, **kwargs)

    def build(self):
        c_ctx = self.c_ctx
        cin = (self.in_channels - c_ctx) // 2 + c_ctx
        cout = self.num_filters // 2
        d = (cin - cout) // 3
        c1 = cin - d
        c2 = cin - 2 * d
        self._layers = nn.ModuleList([nn.Sequential(
            self.conv(
                cin, c1, (1, 1), stride=1, padding=0,
                bias=True, padding_mode="zeros"),
            nn.ReLU(),
            self.conv(
                c1, c2, (1, 1), stride=1, padding=0,
                bias=True, padding_mode="zeros"),
            nn.ReLU(),
            self.conv(
                c2, cout, (1, 1), stride=1, padding=0,
                bias=False, padding_mode="zeros")
        ) for _ in range(2)])

    def forward(self, x):
        c_ctx = self.c_ctx
        ctx = x[:, c_ctx:, ...]
        x = x[:, :c_ctx, ...]
        x = torch.split(x, x.shape[1] // 2, 1)
        mu, sigma = (l(torch.cat((s, ctx), 1)) for l, s in zip(self._layers, x))
        return torch.cat((mu, sigma), 1)


class Param_GG20C(BaseParam):
    """
    see: CHANNEL-WISE AUTOREGRESSIVE ENTROPY MODELS FOR LEARNED IMAGE COMPRESSION

    grouped parameter network for gg20c
    """

    def __init__(self, *args, n_groups=4, **kwargs):
        self.n_groups = n_groups
        self.args = args
        self.kwargs = kwargs
        self._sub_modules = None
        super().__init__(*args, **kwargs)

    def build(self):
        num_filter = self.in_channels
        c_out = num_filter * 2
        c_ctx = num_filter
        n_groups = self.n_groups
        c_hyper = num_filter * 2
        c_out_slice = c_out // n_groups
        c_ctx_slice = c_ctx // n_groups
        self._sub_modules = nn.ModuleList([
            Param_GG18C(c_hyper + c_ctx_slice * (i + 1), c_out_slice,
                        **self.kwargs)
            for i in range(self.n_groups)
        ])
        self.c_hyper = c_hyper
        self.c_ctx_slice = c_ctx_slice
        self.c_out_slice = c_out_slice

    def forward(self, x):
        c_hyper = self.c_hyper
        c_ctx_slice = self.c_ctx_slice
        c_out_slice = self.c_out_slice
        hyper = x[:, :c_hyper, ...]
        ctx = x[:, c_hyper:, ...]

        mu = []
        sigma = []

        assert c_hyper + self.n_groups * c_ctx_slice == x.shape[1]
        for i, sub in enumerate(self._sub_modules):
            i += 1
            slice = ctx[:, :i * c_ctx_slice, ...]
            out = sub(torch.cat([hyper, slice], 1))
            mu.append(out[:, :c_out_slice // 2, ...])
            sigma.append(out[:, c_out_slice // 2:, ...])

        x = torch.cat(mu + sigma, 1)
        assert x.shape[1] == self.n_groups * c_out_slice
        return x


class Param_GG20CNew(Param_GG20C):
    """
    see: CHANNEL-WISE AUTOREGRESSIVE ENTROPY MODELS FOR LEARNED IMAGE COMPRESSION

    grouped parameter network for gg20c
    """

    def build(self):
        num_filter = self.in_channels
        c_out = num_filter * 2
        c_ctx = num_filter
        n_groups = self.n_groups
        c_hyper = num_filter * 2
        c_out_slice = c_out // n_groups
        c_ctx_slice = c_ctx // n_groups
        self._sub_modules = nn.ModuleList([
            ParamLinear3x3(c_hyper + c_ctx_slice * (i + 1), c_out_slice,
                           **self.kwargs)
            for i in range(self.n_groups)
        ])
        self.c_hyper = c_hyper
        self.c_ctx_slice = c_ctx_slice
        self.c_out_slice = c_out_slice


class GG20CLRPIterator(nn.Module):
    param_transform = GG20CParamTransform5x5

    def __init__(self, y_channels, hyper_channels, n_groups=4):
        super().__init__()
        self.y_slice_channels = y_channels // n_groups
        self.hyper_channels = hyper_channels
        self.n_groups = n_groups
        self._sub_models = None
        self.build()

    def build(self):
        c_hyper = self.hyper_channels
        c_slice = self.y_slice_channels
        self._sub_models = nn.ModuleList([
            self.param_transform(c_hyper // 2 + c_slice, c_slice)
            for _ in range(self.n_groups)
        ])

    def forward(self, y_slices_iterator, hyper):
        mu = hyper[:, :hyper.shape[1] // 2, ...]
        for y_slice, sub in zip(y_slices_iterator, self._sub_models):
            out = sub(torch.cat([mu, y_slice], 1))
            out = 0.5 * torch.tanh(out)
            yield out + y_slice

    def extra_repr(self):
        return 'n_groups: {}'.format(self.n_groups)


class LRPIterator(nn.Module):
    def __init__(self, y_channels, hyper_channels, n_groups=4):
        super().__init__()
        self.y_slice_channels = y_channels // n_groups
        self.hyper_channels = hyper_channels
        self.n_groups = n_groups
        self._sub_models = None
        self.build()

    def build(self):
        c_hyper = self.hyper_channels
        c_slice = self.y_slice_channels
        self._sub_models = nn.ModuleList([
            ParamLinear3x3(c_hyper // 2 + c_slice, c_slice)
            for _ in range(self.n_groups)
        ])

    def forward(self, y_slices_iterator, hyper):
        mu = hyper[:, :hyper.shape[1] // 2, ...]
        for y_slice, sub in zip(y_slices_iterator, self._sub_models):
            out = sub(torch.cat([mu, y_slice], 1))
            yield out + y_slice

    def extra_repr(self):
        return 'n_groups: {}'.format(self.n_groups)


class LRPIterator1x1(LRPIterator):
    def build(self):
        c_hyper = self.hyper_channels
        c_slice = self.y_slice_channels
        self._sub_models = nn.ModuleList([
            ParamLinear1x1(c_hyper // 2 + c_slice, c_slice)
            for _ in range(self.n_groups)
        ])


class IdentityLRP(nn.Module):
    def forward(self, y_slices_iterator, hyper):
        return y_slices_iterator


class GG20CParamIterator(nn.Module):
    """
    see: CHANNEL-WISE AUTOREGRESSIVE ENTROPY MODELS FOR LEARNED IMAGE COMPRESSION

    grouped parameter network for gg20c
    """

    inner_param_cls = GG20CTwoPathParam

    def __init__(self, y_channels, hyper_channels, out_channels, n_groups=4, lrp='identity', **kwargs):
        super().__init__()

        self.n_groups = n_groups
        self.y_channels = y_channels
        self.hyper_channels = hyper_channels
        self.out_channels = out_channels
        self.kwargs = kwargs
        self.lrp = lrp

        self._sub_modules = None
        self.build()

    def build(self):
        c_hyper = self.hyper_channels
        c_y_slice = self.y_channels // self.n_groups
        c_out_slice = self.out_channels // self.n_groups
        self._sub_modules = nn.ModuleList([
            self.inner_param_cls(c_hyper + c_y_slice * i, c_out_slice,
                           **self.kwargs)
            for i in range(self.n_groups)
        ])

        if self.lrp == 'lrp':
            self.lrp = GG20CLRPIterator(self.y_channels, self.hyper_channels, self.n_groups)
        elif self.lrp == 'identity':
            self.lrp = IdentityLRP()
        else:
            raise ValueError('lrp type: ' + self.lrp)

    def forward(self, y_slices_iter, hyper):
        """

        :param y_slices_iter: an iterator, generate y slices
        :param hyper:
        :return:
        """
        assert hyper.shape[1] == self.hyper_channels

        sub_modules = self._sub_modules

        c_y_slice = self.y_channels // self.n_groups

        y_slices_iter = self.lrp(y_slices_iter, hyper)
        y_slices = hyper
        for y_next_slice, sub in zip(y_slices_iter, sub_modules):
            assert c_y_slice == y_next_slice.shape[1]
            out = sub(y_slices)
            yield out, y_next_slice
            y_slices = torch.cat([y_slices, y_next_slice], 1)


class ParamMultiDimension(Param_GG20C):
    """
    parameter network for multi-dimension context

    similar to gg20c, but take spatial context into account
    """

    def build(self):
        num_filter = self.in_channels
        c_out = num_filter * 2
        c_ctx = num_filter * 3
        n_groups = self.n_groups
        c_hyper = num_filter * 2
        c_out_slice = c_out // n_groups
        c_ctx_slice = c_ctx // n_groups
        self._sub_modules = nn.ModuleList([
            Param_GG18C(c_hyper + c_ctx_slice * (i + 1), c_out_slice,
                        **self.kwargs)
            for i in range(self.n_groups)
        ])
        self.c_hyper = c_hyper
        self.c_ctx_slice = c_ctx_slice
        self.c_out_slice = c_out_slice


class Param_GG18C_VAR(BaseParam_VAR):
    def build(self):
        lambdas_len = len(self.lambda4s) - 1

        conv1 = ConditionalConv2d(
            self.in_channels, 640, (1, 1), stride=1, padding=0,
            bias=True, padding_mode="zeros", conv=self.conv, lambdas_len=lambdas_len)
        conv2 = ConditionalConv2d(
            640, 512, (1, 1), stride=1, padding=0,
            bias=True, padding_mode="zeros", conv=self.conv, lambdas_len=lambdas_len)
        conv3 = ConditionalConv2d(
            512, self.num_filters, (1, 1), stride=1, padding=0,
            bias=False, padding_mode="zeros", conv=self.conv, lambdas_len=lambdas_len)

        self._layers = nn.ModuleList([l for l in [
            conv1,
            nn.LeakyReLU(),
            conv2,
            nn.LeakyReLU(),
            conv3,
        ] if l])


class Param_EDIC(BaseParam):
    """
    > A Unified End-to-End Framework for Efficient Deep Image Compression
    see: https://arxiv.org/abs/1809.02736
    """

    def build(self):
        assert self.in_channels % 2 == 0
        N = self.in_channels // 2
        self._layers = nn.ModuleList([
            self.conv(
                2 * N, 3 * N, (1, 1), stride=1, padding=0,
                bias=True, padding_mode="zeros"),
            nn.LeakyReLU(),
            self.conv(
                3 * N, 4 * N, (1, 1), stride=1, padding=0,
                bias=True, padding_mode="zeros"),
            nn.LeakyReLU(),
            self.conv(
                4 * N, self.num_filters, (1, 1), stride=1, padding=0,
                bias=False, padding_mode="zeros")
        ])


class ParamCheng20(BaseParam):
    """
    param module used by:

    Learned Image Compression with Discretized Gaussian Mixture Likelihoods and Attention Modules
    (arXiv:2001.01568v3)

    also see: https://github.com/ZhengxueCheng/Learned-Image-Compression-with-GMM-and-Attention

    this is a simplified Non-Local block
    """

    def build(self):
        assert self.in_channels % 2 == 0
        N = self.in_channels // 2
        self._layers = nn.ModuleList([
            self.conv(
                2 * N, 640, (1, 1), stride=1, padding=0,
                bias=True, padding_mode="zeros"),
            nn.LeakyReLU(inplace=True),
            self.conv(
                640, 640, (1, 1), stride=1, padding=0,
                bias=True, padding_mode="zeros"),
            nn.LeakyReLU(inplace=True),
            self.conv(
                640, self.num_filters, (1, 1), stride=1, padding=0,
                bias=False, padding_mode="zeros")
        ])


class ParamGG18CQ(BaseParam):
    """
    Q_paramnet model for gg18cq
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self):
        self.conv = GGQConv2d
        self._layers = nn.ModuleList([
            self.conv(
                self.in_channels, 640, (1, 1), stride=1, padding=0,
                bias=True, padding_mode="zeros"),
            GGBpAct(640, self.c_bit[0]),  # self.in_channels is large, try to  use larger bits 12
            GGQReLU(0, 255),
            self.conv(
                640, 512, (1, 1), stride=1, padding=0,
                bias=True, padding_mode="zeros"),
            GGBpAct(512, self.c_bit[1]),
            GGQReLU(0, 255),
            self.conv(
                512, self.num_filters, (1, 1), stride=1, padding=0,
                bias=False, padding_mode="zeros"),
            GGBpAct(self.num_filters, self.c_bit[2]),
        ])
        self.mu_out_act = GGQBiReLU(-127, 127)  # only used for -127 and 127
        self.sigma_out_act = GGQReLU(0, 63)  # 64 levels

    def forward(self, x):
        x = super().forward(x)
        c = self.num_filters // 2
        mu = x[:, :c, ...]
        sigma = x[:, c:, ...]

        assert mu.shape == sigma.shape
        mu = self.mu_out_act(mu)
        sigma = self.sigma_out_act(sigma)
        return torch.cat([mu, sigma], 1)


class ContextIterator(nn.Module):
    def __init__(self, y_channels, n_groups=8):
        super().__init__()
        self.y_slice_channels = y_channels // n_groups
        self.n_groups = n_groups
        self._sub_models = None
        self.build()

    def build(self):

        from nets.context import Context_GG18C

        c_slice = self.y_slice_channels
        self._sub_models = nn.ModuleList([
            Context_GG18C(c_slice, c_slice * 2)
            for _ in range(self.n_groups)
        ])

    def forward(self, y_slices_iterator):
        for y_slice, sub in zip(y_slices_iterator, self._sub_models):
            yield sub(y_slice)

class ContextIterator_two_path(nn.Module):
    def __init__(self, y_channels, n_groups=8,tag='zero'):
        super().__init__()
        self.y_slice_channels = y_channels // n_groups
        self.n_groups = n_groups
        self.tag = tag
        self._sub_models = None
        self.build()

    def build(self):

        from nets.context import context_models

        c_slice = self.y_slice_channels
        self._sub_models = nn.ModuleList([
            context_models(self.tag, c_slice, c_slice * 2)
            for _ in range(self.n_groups)
        ])

    def forward(self, y_slices_iterator):
        for y_slice, sub in zip(y_slices_iterator, self._sub_models):
            yield sub(y_slice)

class ParamGG20CIteratorCC(nn.Module):
    """DCT coef compression with context model(gg18c + gg20c) and inverse zigzag order

    see: CHANNEL-WISE AUTOREGRESSIVE ENTROPY MODELS FOR LEARNED IMAGE COMPRESSION

    grouped parameter network for gg20c
    """

    inner_param_cls = ParamLinear3x3

    def __init__(self, y_channels, hyper_channels, out_channels, n_groups=8, lrp='identity', **kwargs):
        super().__init__()

        self.n_groups = n_groups
        self.y_channels = y_channels
        self.hyper_channels = hyper_channels
        self.out_channels = out_channels
        self.kwargs = kwargs
        self.lrp = lrp

        self._channel_and_hyper_modules = None
        self._spatial_context_modules = None
        self._param_modules = None
        self.build()

    def build(self):
        c_hyper = self.hyper_channels
        c_y_slice = self.y_channels // self.n_groups
        c_out_slice = self.out_channels // self.n_groups
        self._channel_and_hyper_modules = nn.ModuleList([
            self.inner_param_cls(c_hyper + c_y_slice * i, c_out_slice, **self.kwargs)
            for i in range(self.n_groups)
        ])
        self._param_modules = nn.ModuleList([
            ParamLinear1x1(c_out_slice * (i + 2), c_out_slice, **self.kwargs)
            for i in range(self.n_groups)
        ])

        if self.lrp == 'lrp':
            self.lrp = LRPIterator(self.y_channels, self.hyper_channels, self.n_groups)
        elif self.lrp == 'identity':
            self.lrp = IdentityLRP()
        else:
            raise ValueError('lrp type: ' + self.lrp)

        self._spatial_context_modules = ContextIterator(self.y_channels, self.n_groups)

    def forward(self, y_slices_iter, hyper):
        """

        :param y_slices_iter: an iterator, generate y slices
        :param hyper:
        :return:
        """
        assert hyper.shape[1] == self.hyper_channels

        ch_hyper_modules = self._channel_and_hyper_modules
        param_modules = self._param_modules

        c_y_slice = self.y_channels // self.n_groups

        y_context_slices_iter = self._spatial_context_modules(y_slices_iter)
        y_slices_iter = self.lrp(y_slices_iter, hyper)
        y_slices = hyper
        ctx_all = None
        for y_next_slice, y_next_context_slice, ch_mdl, p_mdl in zip(y_slices_iter, y_context_slices_iter, ch_hyper_modules, param_modules):
            assert c_y_slice == y_next_slice.shape[1], \
                f"{c_y_slice} == {y_next_slice.shape[1]}"
            ch_ctx = ch_mdl(y_slices)
            if ctx_all is None:
                ctx_all = y_next_context_slice
            else:
                ctx_all = torch.cat([ctx_all, y_next_context_slice], 1)
            out = p_mdl(torch.cat([ctx_all, ch_ctx], 1))
            yield out, y_next_slice
            y_slices = torch.cat([y_slices, y_next_slice], 1)

class ParamGG20CIteratorCC_two_path(nn.Module):
    """DCT coef compression with context model(Two_Path_Share_Weight + gg20c) and inverse zigzag order

    see: CHANNEL-WISE AUTOREGRESSIVE ENTROPY MODELS FOR LEARNED IMAGE COMPRESSION

    grouped parameter network for gg20c
    """

    inner_param_cls = ParamLinear3x3

    def __init__(self, y_channels, hyper_channels, out_channels, n_groups=8, lrp='identity', **kwargs):
        super().__init__()

        self.n_groups = n_groups
        self.y_channels = y_channels
        self.hyper_channels = hyper_channels
        self.out_channels = out_channels
        self.kwargs = kwargs
        self.lrp = lrp

        self._channel_and_hyper_modules = None
        self._spatial_context_modules = None
        self._param_modules = None
        self.build()

    def build(self):
        c_hyper = self.hyper_channels
        c_y_slice = self.y_channels // self.n_groups
        c_out_slice = self.out_channels // self.n_groups
        self._channel_and_hyper_modules = nn.ModuleList([
            self.inner_param_cls(c_hyper + c_y_slice * i, c_out_slice, **self.kwargs)
            for i in range(self.n_groups)
        ])
        self._param_modules = nn.ModuleList([
            ParamLinear1x1(c_out_slice * (i + 2), c_out_slice, **self.kwargs)
            for i in range(self.n_groups)
        ])

        if self.lrp == 'lrp':
            self.lrp = LRPIterator(self.y_channels, self.hyper_channels, self.n_groups)
        elif self.lrp == 'identity':
            self.lrp = IdentityLRP()
        else:
            raise ValueError('lrp type: ' + self.lrp)
        print('_spatial_context_modules', flush=True)
        self._spatial_context_modules = nn.ModuleList([ContextIterator_two_path(self.y_channels, self.n_groups,tag=tag) for tag in ['checkerboard','zero']])


    def forward(self, y_slices_iter, hyper):
        """

        :param y_slices_iter: an iterator, generate y slices
        :param hyper:
        :return:
        """
        assert hyper.shape[1] == self.hyper_channels

        ch_hyper_modules = self._channel_and_hyper_modules
        param_modules = self._param_modules

        c_y_slice = self.y_channels // self.n_groups
        y_context_slices_iter0 = self._spatial_context_modules[0](y_slices_iter)   #context1
        y_context_slices_iter1 = self._spatial_context_modules[1](y_slices_iter)   #context2
        y_slices_iter = self.lrp(y_slices_iter, hyper)
        y_slices = hyper
        ctx_all0 = None
        ctx_all1 = None
        for y_next_slice, y_next_context_slice0,y_next_context_slice1, ch_mdl, p_mdl in zip(y_slices_iter, y_context_slices_iter0,y_context_slices_iter1, ch_hyper_modules, param_modules):
            assert c_y_slice == y_next_slice.shape[1], \
                f"{c_y_slice} == {y_next_slice.shape[1]}"
            ch_ctx = ch_mdl(y_slices)
            if ctx_all0 is None:
                ctx_all0 = y_next_context_slice0
            else:
                ctx_all0 = torch.cat([ctx_all0, y_next_context_slice0], 1)
            out0 = p_mdl(torch.cat([ctx_all0, ch_ctx], 1))
            if ctx_all1 is None:
                ctx_all1 = y_next_context_slice1
            else:
                ctx_all1 = torch.cat([ctx_all1, y_next_context_slice1], 1)
            out1 = p_mdl(torch.cat([ctx_all1, ch_ctx], 1))

            yield out0,out1, y_next_slice
            y_slices = torch.cat([y_slices, y_next_slice], 1)

def param_models(tag: str = 'GG18C', in_channels=192, out_channels=192, **kwargs):
    # out_channels = out_channels * 2 # mu and sigma, so multiply 2
    if tag == 'GG18C':
        model = Param_GG18C(in_channels, out_channels, **kwargs)
    elif tag == 'Linear3x3':
        model = ParamLinear3x3(in_channels, out_channels, **kwargs)
    elif tag == 'Linear1x1':
        model = ParamLinear1x1(in_channels, out_channels, **kwargs)
    elif tag == 'Linear1x1ExpSigma':
        model = ParamLinearExpSigma1x1(in_channels, out_channels, **kwargs)
    elif tag == 'TwoPathLinear1x1ExpSigma':
        model = TwoPathParamLinearExpSigma1x1(in_channels, out_channels, **kwargs)
    elif tag == 'TwoPathLinear1x1':
        model = TwoPathParamLinear1x1(in_channels, out_channels, **kwargs)
    elif tag == 'GG20C':
        model = Param_GG20C(in_channels, out_channels, **kwargs)
    elif tag == 'GG20CNew':
        model = Param_GG20CNew(in_channels, out_channels, **kwargs)
    elif tag == 'MD':
        model = ParamMultiDimension(in_channels, out_channels, **kwargs)
    elif tag == 'GG18C_VAR':
        model = Param_GG18C_VAR(in_channels, out_channels, **kwargs)
    elif tag == 'EDIC':
        model = Param_EDIC(in_channels, out_channels, **kwargs)
    elif tag == 'NONE':
        model = None
    else:
        raise NotImplementedError('param model with tag ' + tag)

    return model


if __name__ == '__main__':
    x = torch.rand(2, 512, 10, 10)
    test = param_models('EDIC', 512, 1536)
    # (2 * 256) --K = 2--> 256 * 3 * K = 1536
    # (2 * 256) --K = 1--> 256 * 3 * K = 768
    print(test(x).shape)

    # x = torch.randn(2, 3, 80, 80 * 2)
    # test = param_models('GG18C', 3, 10)
    # with torch.no_grad():
    #     print(test(x).shape)
    #
    # test = param_models('GG18C_VAR', 3, 10)
    # with torch.no_grad():
    #     print(test(x).shape)
