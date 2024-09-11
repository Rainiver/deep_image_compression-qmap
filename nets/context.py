import torch
import torch.nn as nn
import numpy as np
import math
from nets.layers import MaskedConv2d, MaskedSignalConv2d, ConditionalConv2d, \
    XMaskedSignalConv2d, CrossMaskedSignalConv2d, Knight1MaskedSignalConv2d, \
    Knight2MaskedSignalConv2d, Knight3MaskedSignalConv2d, Knight4MaskedSignalConv2d, \
    KnightMaskedSignalConv2d
from .layers_quant import GGQReLU, GGBpAct, MaskedQConv2d


class BaseContext(nn.Module):
    def __init__(self, in_channels=3, out_channels=192, conv='signal', lambda4s=[1, 2], **kwargs):
        super(BaseContext, self).__init__()
        self.in_channels = in_channels
        self.num_filters = out_channels
        self.caffe_channels = in_channels

        self.lambda4s = lambda4s
        self.sampled_lambda = 1

        if conv == 'conv':
            self.conv = MaskedConv2d
        elif conv == 'signal':
            self.conv = MaskedSignalConv2d
        else:
            raise NotImplemented('unsupported conv layer {}'.format(conv))

        self._layers = nn.ModuleList([])
        self.build()

    def build(self):
        raise NotImplementedError

    def forward(self, x):
        for layer in self._layers:
            x = layer(x)
        return x


class BaseContext_VAR(BaseContext):
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


class Context_GG18C(BaseContext):
    def build(self):
        self._layers = nn.ModuleList([
            self.conv(
                self.in_channels, self.num_filters, (5, 5), stride=1, padding=2,
                bias=True, padding_mode="zeros"),
        ])


class GG18CQContext(nn.Module):
    """
    quant context model for gg18cq

    """

    def __init__(self, in_channels=3, out_channels=192, conv='signal', lambda4s=[1, 2]):
        super().__init__()
        self.in_channels = in_channels
        self.num_filters = out_channels
        self.caffe_channels = in_channels

        self.lambda4s = lambda4s
        self.sampled_lambda = 1

        self.conv = MaskedQConv2d
        self.Act = GGBpAct
        self.relu = GGQReLU
        self._layers = nn.ModuleList([])
        self.build()

    def build(self):
        self._layers = nn.ModuleList([
            self.conv(
                self.in_channels, self.num_filters, (5, 5), stride=1, padding=2,
                bias=True, padding_mode="zeros"),
            self.Act(self.num_filters, 8),
            self.relu(0, 255)
        ])

    def forward(self, x):
        for layer in self._layers:
            x = layer(x)
        return x


class ContextGG20C(BaseContext):
    """
    context model for gg20c

    input: y_tilde in k slices [y1, y2, ..., yk]
    output: shifted y_tilde slices [0, y1, y2, ..., y(k-1)]

    for the k-th slice, ALL previous decoded slices are its channel-wise context
    """

    def __init__(self, *args, n_groups=4, **kwargs):
        self.n_groups = n_groups
        super().__init__(*args, **kwargs)

    def build(self):
        pass

    def forward(self, x):
        n_groups = self.n_groups
        shape_in = x.shape
        c = shape_in[1]
        zero = torch.zeros(shape_in[0], c // n_groups, shape_in[2], shape_in[3])
        zero = zero.to(x.device)
        x = x[:, :-c // n_groups, ...]
        x = torch.cat([zero, x], 1)
        assert x.shape == shape_in
        return x

    def extra_repr(self):
        return 'group: {}'.format(self.n_groups)


class Context_GG18C_VAR(BaseContext_VAR):
    def build(self):
        lambdas_len = len(self.lambda4s) - 1

        conv1 = ConditionalConv2d(
            self.in_channels, self.num_filters, (5, 5), stride=1, padding=2,
            bias=True, padding_mode="zeros", conv=self.conv, lambdas_len=lambdas_len)

        self._layers = nn.ModuleList([l for l in [
            conv1
        ] if l])


class CheckerboardContext(BaseContext):
    """
    checkerboard shaped context model
    """

    def __init__(self, *args, k_size=5, **kwargs):
        self.k_size = k_size
        super().__init__(*args, **kwargs)

    def build(self):
        k = self.k_size
        self._layers = nn.ModuleList([
            CrossMaskedSignalConv2d(
                self.in_channels, self.num_filters, (k, k), stride=1, padding=k // 2,
                bias=True, padding_mode="zeros"),
        ])


class CheckerboardSliceContext(BaseContext):
    """
    CheckerboardContext with grouped convolution
    """

    def __init__(self, *args, k_size=5, n_groups=4, **kwargs):
        self.k_size = k_size
        self.n_groups = n_groups
        super().__init__(*args, **kwargs)

    def build(self):
        k = self.k_size
        self._layers = nn.ModuleList([
            CrossMaskedSignalConv2d(
                self.in_channels, self.num_filters, (k, k), stride=1, padding=k // 2,
                groups=self.n_groups,
                bias=True, padding_mode="zeros"),
        ])

    def extra_repr(self):
        return 'n_groups: {}'.format(self.n_groups)


class KnightsPositionContext1(BaseContext):
    """
    Knight's Position on Checkerboard shaped context model
    """

    def __init__(self, *args, k_size=5, **kwargs):
        self.k_size = k_size
        super().__init__(*args, **kwargs)

    def build(self):
        k = self.k_size
        self._layers = nn.ModuleList([
            Knight1MaskedSignalConv2d(
                self.in_channels, self.num_filters, (k, k), stride=1, padding=k // 2,
                bias=True, padding_mode="zeros"),
        ])


class KnightsPositionContext2(BaseContext):
    """
    Knight's Position on Checkerboard shaped context model
    """

    def __init__(self, *args, k_size=5, **kwargs):
        self.k_size = k_size
        super().__init__(*args, **kwargs)

    def build(self):
        k = self.k_size
        self._layers = nn.ModuleList([
            Knight2MaskedSignalConv2d(
                self.in_channels, self.num_filters, (k, k), stride=1, padding=k // 2,
                bias=True, padding_mode="zeros"),
        ])


class KnightsPositionContext3(BaseContext):
    """
    Knight's Position on Checkerboard shaped context model
    """

    def __init__(self, *args, k_size=5, **kwargs):
        self.k_size = k_size
        super().__init__(*args, **kwargs)

    def build(self):
        k = self.k_size
        self._layers = nn.ModuleList([
            Knight3MaskedSignalConv2d(
                self.in_channels, self.num_filters, (k, k), stride=1, padding=k // 2,
                bias=True, padding_mode="zeros"),
        ])


class KnightsPositionContext4(BaseContext):
    """
    Knight's Position on Checkerboard shaped context model
    """

    def __init__(self, *args, k_size=5, **kwargs):
        self.k_size = k_size
        super().__init__(*args, **kwargs)

    def build(self):
        k = self.k_size
        self._layers = nn.ModuleList([
            Knight4MaskedSignalConv2d(
                self.in_channels, self.num_filters, (k, k), stride=1, padding=k // 2,
                bias=True, padding_mode="zeros"),
        ])


class KnightsPositionShareWeightContext(BaseContext):
    """
    Knight's Position Share-Weight on Checkerboard shaped context model
    """

    def __init__(self, *args, k_size=5, **kwargs):
        self.k_size = k_size
        super().__init__(*args, **kwargs)

    def build(self):
        k = self.k_size
        self._layers = nn.ModuleList([
            KnightMaskedSignalConv2d(
                self.in_channels, self.num_filters, (k, k), stride=1, padding=k // 2,
                bias=True, padding_mode="zeros"),
        ])


class ContextMD(nn.Module):
    """
    multi-dimension context model

    input: k slices of y_tildes [y1, y2, ..., yk]
    spatial context: CM_k(y_tilde) = [CM(y1), CM(y2), ..., CM(yk)]
    channel context: CM_GG20C(y_tilde) = [0, y1, y2, ..., y(k-1)]
    mixture output:  [(CM(y1), 0), (CM(y2), y1), ..., (CM(yk), y(k-1))]
    """

    def __init__(self, *args, n_groups=4, **kwargs):
        self.n_groups = n_groups
        super().__init__()
        self.ckbd = CheckerboardSliceContext(*args, n_groups=n_groups, **kwargs)
        self.gg20c = ContextGG20C(*args, n_groups=n_groups, **kwargs)

    def forward(self, x):
        n_groups = self.n_groups
        x1 = self.ckbd(x)
        c1 = x1.shape[1] // n_groups
        x2 = self.gg20c(x)
        c2 = x2.shape[1] // n_groups
        x1s = [x1[:, i * c1: i * c1 + c1, ...] for i in range(n_groups)]
        x2s = [x2[:, i * c2: i * c2 + c2, ...] for i in range(n_groups)]
        xs = []
        for i in range(n_groups):
            xs += [x1s[i], x2s[i]]
        return torch.cat(xs, 1)


class ZeroContext(BaseContext):
    """
    fake context model
    """

    def build(self):
        pass

    def forward(self, x):
        shape = list(x.shape)
        shape[1] = self.num_filters
        zeros = torch.zeros(shape)
        if torch.cuda.is_available():
            zeros = zeros.cuda()
        return zeros


class ContextGatedSum(nn.Module):
    def __init__(self):
        super().__init__()
        self._cached_masks = {'x': None, 'cross': None}
        self._cached_shape = None

    def _get_fwd_masks(self, x):
        shape = x.shape
        if shape != self._cached_shape:
            self._cached_shape = shape
            assert len(shape) == 4  # NCHW
            num_plain = shape[0] * shape[1]
            i = torch.arange(shape[3]).repeat(num_plain * shape[2]).reshape(shape)
            j = torch.arange(shape[2]).repeat(num_plain * shape[3]) \
                .reshape(*shape[:2], shape[3], shape[2]).permute(0, 1, 3, 2)

            x_mask = ((i % 2 + j % 2) == 2).float()
            cross_mask = ((i % 2 + j % 2) == 1).float()
            if torch.cuda.is_available():
                x_mask = x_mask.cuda()
                cross_mask = cross_mask.cuda()

            self._cached_masks = {'x': x_mask, 'cross': cross_mask}

        return self._cached_masks

    def forward(self, checkerboard, anchor, blank):
        assert checkerboard.shape == anchor.shape
        assert checkerboard.shape == blank.shape

        masks = self._get_fwd_masks(checkerboard)
        ckbd_msk = masks['cross']
        x_msk = masks['x']
        b_msk = (1. - ckbd_msk) * (1. - x_msk)

        assert ckbd_msk.shape == checkerboard.shape
        assert x_msk.shape == anchor.shape
        assert b_msk.shape == blank.shape

        return ckbd_msk * checkerboard + x_msk * anchor + b_msk * blank


class CheckerboardSpaceSplitMux(nn.Module):
    def forward(self, checkerboard, anchor, blank):
        assert checkerboard.shape == anchor.shape
        assert checkerboard.shape == blank.shape
        assert anchor is blank  # accept only 2 inputs

        n, c, h, w = checkerboard.shape

        assert h % 2 == 0
        assert w % 2 == 0

        # AC NA   0 1
        # NA AC   2 3

        na = checkerboard.reshape(n, c, h//2, 2, w//2, 2)
        ac = anchor.reshape(n, c, h//2, 2, w//2, 2)
        # not using slice-assign
        # it is hard to export slice-assign layer
        #
        # ac[:, :, 1:3, ...] = na[:, :, 1:3, ...]
        ac = torch.stack((ac[..., 0, :, 0],
                        na[..., 0, :, 1],
                        na[..., 1, :, 0],
                        ac[..., 1, :, 1]), 2)\
            .reshape(n, c*4, h//2, w//2)

        ac = torch.pixel_shuffle(ac, 2)
        return ac


class CheckerboardSliceAlignMux(nn.Module):
    def forward(self, checkerboard, anchor, blank):
        assert checkerboard.shape == anchor.shape
        assert checkerboard.shape == blank.shape
        assert anchor is blank  # accept only 2 inputs

        anchor = torch.clone(anchor)
        anchor[..., 0::2, 1::2] = checkerboard[..., 0::2, 1::2]
        anchor[..., 1::2, 0::2] = checkerboard[..., 1::2, 0::2]
        return anchor


class AlwaysFirstMux(nn.Module):
    def forward(self, c, a, b):
        return c + 0*a + 0*b


class AlwaysSecondMux(nn.Module):
    def forward(self, c, a, b):
        return 0*c + a + 0*b


class KnightsPositionContextGatedSum(nn.Module):
    def forward(self, context1, context2, context3, context4, context5):
        shape = context1.shape
        assert shape == context2.shape
        assert shape == context3.shape
        assert shape == context4.shape
        assert shape == context5.shape

        contexts = [context1, context2, context3, context4, context5]

        context1 = torch.clone(context1)
        for i in range(1, 5):
            for j, k in enumerate([0, 3, 1, 4, 2]):
                # i is context index
                # j is row index
                # k is collum offset
                # example for i=1:
                #   [0., 1., 0., 0., 0.]
                #   [0., 0., 0., 0., 1.]
                #   [0., 0., 1., 0., 0.]
                #   [1., 0., 0., 0., 0.]
                #   [0., 0., 1., 0., 0.]
                # which is consistent with (KnightsPositionMask3 - KnightsPositionMask2)
                # to get the entropy of context2(contexts[i])
                context1[..., j::5, (i+k) % 5::5] = contexts[i][..., j::5, (i+k) % 5::5]

        return context1


class BaseMask(nn.Module):
    def get_mask(self, shape):
        return None

    def forward(self, x):
        mask = self.get_mask(x.shape)
        if torch.cuda.is_available():
            mask = mask.cuda()
        if mask is not None:
            x = x * mask
        return x


class KnightsPositionMask(BaseMask):
    def __init__(self, mask_type=1):
        super().__init__()
        assert mask_type in range(1, 6)
        cnt = mask_type - 1
        self.row_unit = [1.] * cnt + [0.] * (5 - cnt)

    def get_mask(self, shape):
        h, w = shape[-2:]
        row_unit = self.row_unit
        row = row_unit * (w // 5) + row_unit[:w % 5]
        kernel_unit = [row[-i:] + row[:-i] for i in [0, 3, 1, 4, 2]]
        kernel = kernel_unit * (h // 5) + kernel_unit[:h % 5]
        return torch.tensor(kernel)


class KnightsPositionMask1(BaseMask):
    def get_mask(self, shape):
        n, c, h, w = shape
        row_unit_zero = [0., 0., 0., 0., 0.]
        row_unit_one = [0., 0., 0., 0., 0.]
        row_unit_two = [0., 0., 0., 0., 0.]
        row_unit_three = [0., 0., 0., 0., 0.]
        row_unit_four = [0., 0., 0., 0., 0.]
        row_zero = row_unit_zero * (w // 5) + row_unit_zero[:w % 5]
        row_one = row_unit_one * (w // 5) + row_unit_one[:w % 5]
        row_two = row_unit_two * (w // 5) + row_unit_two[:w % 5]
        row_three = row_unit_three * (w // 5) + row_unit_three[:w % 5]
        row_four = row_unit_four * (w // 5) + row_unit_four[:w % 5]
        kernel_unit = [row_zero, row_one, row_two, row_three, row_four]
        kernel = kernel_unit * (h // 5) + kernel_unit[:h % 5]
        return torch.tensor(kernel)


class KnightsPositionMask2(BaseMask):
    def get_mask(self, shape):
        n, c, h, w = shape
        row_unit_zero = [1., 0., 0., 0., 0.]
        row_unit_one = [0., 0., 0., 1., 0.]
        row_unit_two = [0., 1., 0., 0., 0.]
        row_unit_three = [0., 0., 0., 0., 1.]
        row_unit_four = [0., 0., 1., 0., 0.]
        row_zero = row_unit_zero * (w // 5) + row_unit_zero[:w % 5]
        row_one = row_unit_one * (w // 5) + row_unit_one[:w % 5]
        row_two = row_unit_two * (w // 5) + row_unit_two[:w % 5]
        row_three = row_unit_three * (w // 5) + row_unit_three[:w % 5]
        row_four = row_unit_four * (w // 5) + row_unit_four[:w % 5]
        kernel_unit = [row_zero, row_one, row_two, row_three, row_four]
        kernel = kernel_unit * (h // 5) + kernel_unit[:h % 5]
        return torch.tensor(kernel)


class KnightsPositionMask3(BaseMask):
    def get_mask(self, shape):
        n, c, h, w = shape
        row_unit_zero = [1., 1., 0., 0., 0.]
        row_unit_one = [0., 0., 0., 1., 1.]
        row_unit_two = [0., 1., 1., 0., 0.]
        row_unit_three = [1., 0., 0., 0., 1.]
        row_unit_four = [0., 0., 1., 1., 0.]
        row_zero = row_unit_zero * (w // 5) + row_unit_zero[:w % 5]
        row_one = row_unit_one * (w // 5) + row_unit_one[:w % 5]
        row_two = row_unit_two * (w // 5) + row_unit_two[:w % 5]
        row_three = row_unit_three * (w // 5) + row_unit_three[:w % 5]
        row_four = row_unit_four * (w // 5) + row_unit_four[:w % 5]
        kernel_unit = [row_zero, row_one, row_two, row_three, row_four]
        kernel = kernel_unit * (h // 5) + kernel_unit[:h % 5]
        return torch.tensor(kernel)


class KnightsPositionMask4(BaseMask):
    def get_mask(self, shape):
        n, c, h, w = shape
        row_unit_zero = [1., 1., 1., 0., 0.]
        row_unit_one = [1., 0., 0., 1., 1.]
        row_unit_two = [0., 1., 1., 1., 0.]
        row_unit_three = [1., 1., 0., 0., 1.]
        row_unit_four = [0., 0., 1., 1., 1.]
        row_zero = row_unit_zero * (w // 5) + row_unit_zero[:w % 5]
        row_one = row_unit_one * (w // 5) + row_unit_one[:w % 5]
        row_two = row_unit_two * (w // 5) + row_unit_two[:w % 5]
        row_three = row_unit_three * (w // 5) + row_unit_three[:w % 5]
        row_four = row_unit_four * (w // 5) + row_unit_four[:w % 5]
        kernel_unit = [row_zero, row_one, row_two, row_three, row_four]
        kernel = kernel_unit * (h // 5) + kernel_unit[:h % 5]
        return torch.tensor(kernel)


class KnightsPositionMask5(BaseMask):
    def get_mask(self, shape):
        n, c, h, w = shape
        row_unit_zero = [1., 1., 1., 1., 0.]
        row_unit_one = [1., 1., 0., 1., 1.]
        row_unit_two = [0., 1., 1., 1., 1.]
        row_unit_three = [1., 1., 1., 0., 1.]
        row_unit_four = [1., 0., 1., 1., 1.]
        row_zero = row_unit_zero * (w // 5) + row_unit_zero[:w % 5]
        row_one = row_unit_one * (w // 5) + row_unit_one[:w % 5]
        row_two = row_unit_two * (w // 5) + row_unit_two[:w % 5]
        row_three = row_unit_three * (w // 5) + row_unit_three[:w % 5]
        row_four = row_unit_four * (w // 5) + row_unit_four[:w % 5]
        kernel_unit = [row_zero, row_one, row_two, row_three, row_four]
        kernel = kernel_unit * (h // 5) + kernel_unit[:h % 5]
        return torch.tensor(kernel)


def gate_models(tag):
    if tag == 'checkerboard':
        return ContextGatedSum()
    elif tag == 'checkerboard-space-split':
        return CheckerboardSpaceSplitMux()
    elif tag == 'checkerboard-slice-assign':
        return CheckerboardSliceAlignMux()
    elif tag == 'always-first':
        return AlwaysFirstMux()
    elif tag == 'always-second':
        return AlwaysSecondMux()
    elif tag == 'knights-position':
        return KnightsPositionContextGatedSum()
    elif tag == 'NONE':
        return None
    else:
        raise NotImplementedError(tag)


def context_models(tag: str = 'GG18C', in_channels=192, out_channels=192, **kwargs):
    # out_channels = out_channels * 2 # mu and sigma, so multiply 2
    if tag == 'GG18C':
        model = Context_GG18C(in_channels, out_channels, **kwargs)
    elif tag == 'GG20C':
        model = ContextGG20C(in_channels, out_channels, **kwargs)
    elif tag == 'MD':
        model = ContextMD(in_channels, out_channels, **kwargs)
    elif tag == 'GG18C_VAR':
        model = Context_GG18C_VAR(in_channels, out_channels, **kwargs)
    elif tag == 'checkerboard':
        model = CheckerboardContext(in_channels, out_channels, **kwargs)
    elif tag == 'knights-position1':
        model = KnightsPositionContext1(in_channels, out_channels, **kwargs)
    elif tag == 'knights-position2':
        model = KnightsPositionContext2(in_channels, out_channels, **kwargs)
    elif tag == 'knights-position3':
        model = KnightsPositionContext3(in_channels, out_channels, **kwargs)
    elif tag == 'knights-position4':
        model = KnightsPositionContext4(in_channels, out_channels, **kwargs)
    elif tag == 'knights-position-share-weight':
        model = KnightsPositionShareWeightContext(in_channels, out_channels, **kwargs)
    elif tag == 'NONE':
        model = None
    elif tag == 'zero':
        model = ZeroContext(in_channels, out_channels, **kwargs)
    else:
        raise NotImplementedError('param model with tag ' + tag)

    return model


def mask_models(tag: str):
    # for knights-position share weight
    if tag == 'knights-position1':
        model = KnightsPositionMask(mask_type=1)
    elif tag == 'knights-position2':
        model = KnightsPositionMask(mask_type=2)
    elif tag == 'knights-position3':
        model = KnightsPositionMask(mask_type=3)
    elif tag == 'knights-position4':
        model = KnightsPositionMask(mask_type=4)
    elif tag == 'knights-position5':
        model = KnightsPositionMask(mask_type=5)
    else:
        raise NotImplementedError('mask with tag ' + tag)

    return model


if __name__ == '__main__':
    x = torch.randn(2, 3, 80, 80 * 2)

    test = context_models('GG18C', 3, 10, conv='conv')
    with torch.no_grad():
        print(test(x).shape)

    test = context_models('GG18C', 3, 10)
    with torch.no_grad():
        print(test(x).shape)

    test = context_models('GG18C_VAR', 3, 10)
    with torch.no_grad():
        print(test(x).shape)

    test = context_models('parallel', 3, 10)

    if torch.cuda.is_available():
        x = x.cuda()
        test.cuda()
    with torch.no_grad():
        out = test(x)
        assert out.shape[2:] == x.shape[2:]
        print(out)
        print(test(x).shape)

        print('test cache...')
        out = test(x)
        out = test(x)
        out = test(x)
        print('done.')

        x = torch.randn(2, 3, 91, 91 * 2)
        if torch.cuda.is_available():
            x = x.cuda()
        out = test(x)
        assert out.shape[2:] == x.shape[2:]
        print(out)
        print(test(x).shape)
