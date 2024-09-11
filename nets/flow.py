import torch
import torch.nn as nn
import numpy as np
from functools import reduce

from nets.context import context_models
from nets.layers import Normalize, Squeeze, Permute, SplitFactorCoupling, SplitPrior, Descale
from nets.param import param_models


# TODO: this should be replaced by DIC built-in entropy model
class Distributions():
    def discretized_logistic_cdf(self, x, mean, logscale, inverse_bin_width=256):
        scale = torch.exp(logscale)
        cdf = torch.sigmoid((x + 0.5 / inverse_bin_width - mean) / scale)
        return cdf

    def mixture_discretized_logistic_cdf(self, x, mean, logscale, pi, inverse_bin_width=256):
        # x : NCHW -> N 1 C H W
        # mean, scale, pi : N(C*K)HW -> N K C H W
        C = x.shape[1]
        K = mean.shape[1] // C
        size = [x.shape[0], K, C, x.shape[2], x.shape[3]]
        x = x.unsqueeze(dim=1)
        mean = mean.view(size)
        scale = torch.exp(logscale.view(size))
        pi = pi.view(size)

        cdfs = torch.sigmoid((x + 0.5 / inverse_bin_width - mean) / scale)
        cdf = torch.sum(cdfs * pi, dim=1)
        return cdf


distributions = Distributions()


# TODO: this should be replaced by DIC built-in entropy coder
class rANS():
    def __init__(self):
        self.rans_l = 1 << 31  # the lower bound of the normalisation interval
        self.tail_bits = (1 << 32) - 1

        self.x_init = (self.rans_l, ())

    def append(self, x, start, freq, precision):
        """Encodes a symbol with range [start, start + freq).  All frequencies are
        assumed to sum to "1 << precision", and the resulting bits get written to
        x."""
        if x[0] >= ((self.rans_l >> precision) << 32) * freq:
            x = (x[0] >> 32, (x[0] & self.tail_bits, x[1]))
        return ((x[0] // freq) << precision) + (x[0] % freq) + start, x[1]

    def pop(self, x_, precision):
        """Advances in the bit stream by "popping" a single symbol with range start
        "start" and frequency "freq"."""
        cf = x_[0] & ((1 << precision) - 1)

        def pop(start, freq):
            x = freq * (x_[0] >> precision) + cf - start, x_[1]
            return ((x[0] << 32) | x[1][0], x[1][1]) if x[0] < self.rans_l else x

        return cf, pop

    def append_symbol(self, statfun, precision):
        def append_(x, symbol):
            start, freq = statfun(symbol)
            return self.append(x, start, freq, precision)

        return append_

    def pop_symbol(self, statfun, precision):
        def pop_(x):
            cf, pop_fun = self.pop(x, precision)
            symbol, (start, freq) = statfun(cf)
            return pop_fun(start, freq), symbol

        return pop_

    def flatten(self, x):
        """Flatten a rans state x into a 1d numpy array."""
        out, x = [x[0] >> 32, x[0]], x[1]
        while x:
            x_head, x = x
            out.append(x_head)
        return np.asarray(out, dtype=np.uint32)

    def unflatten(self, arr):
        """Unflatten a 1d numpy array into a rans state."""
        return (int(arr[0]) << 32 | int(arr[1]),
                reduce(lambda tl, hd: (int(hd), tl), reversed(arr[2:]), ()))


rans = rANS()


# TODO: this should be replaced by DIC built-in entropy coder
class Coder():
    def __init__(self):
        self.precision = 24
        self.n_bins = 4096

    def cdf_fn(self, z, pz, inverse_bin_width):
        if len(pz) == 2:
            return distributions.discretized_logistic_cdf(
                z, *pz, inverse_bin_width=inverse_bin_width)
        elif len(pz) == 3:
            return distributions.mixture_discretized_logistic_cdf(
                z, *pz, inverse_bin_width=inverse_bin_width)

        raise ValueError

    def CDF_fn(self, pz, bin_width):
        # TODO:size of pz is N(K*C)HW when n_mixtures is greater than 1
        mean = pz[0] if len(pz) == 2 else pz[0][..., (pz[0].size(-1) - 1) // 2]
        MEAN = torch.round(mean / bin_width).long()

        bin_locations = torch.arange(-self.n_bins // 2, self.n_bins // 2)[None, None, None, None, :] + MEAN.cpu()[
            ..., None]
        bin_locations = bin_locations.float() * bin_width
        bin_locations = bin_locations.to(device=pz[0].device)

        pz = [param[:, :, :, :, None] for param in pz]
        cdf = self.cdf_fn(
            bin_locations - bin_width,
            pz,
            1. / bin_width).cpu().numpy()

        # Compute CDFs, reweigh to give all bins at least
        # 1 / (2^precision) probability.
        # CDF is equal to floor[cdf * (2^precision - n_bins)] + range(n_bins)
        CDFs = (cdf * ((1 << self.precision) - self.n_bins)).astype('int') \
               + np.arange(self.n_bins)

        return CDFs, MEAN

    def encode_sample(self,
                      z, pz, bin_width=1. / 256, state=None):
        if state is None:
            state = rans.x_init
        else:
            state = rans.unflatten(state)

        CDFs, MEAN = self.CDF_fn(pz, bin_width)

        # z is transformed to Z to match the indices for the CDFs array
        Z = torch.round(z / bin_width).long() + self.n_bins // 2 - MEAN
        Z = Z.cpu().numpy()

        if not ((np.sum(Z < 0) == 0 and np.sum(Z >= self.n_bins - 1) == 0)):
            print('Z out of allowed range of values, canceling compression')
            return None

        if np.sum(np.isnan(Z)):
            print('NaN encountered in Z, canceling compression')
            return None

        Z, CDFs = Z.reshape(-1), CDFs.reshape(-1, self.n_bins).copy()
        for symbol, cdf in zip(Z[::-1], CDFs[::-1]):
            statfun = self.statfun_encode(cdf)
            state = rans.append_symbol(statfun, self.precision)(state, symbol)

        state = rans.flatten(state)

        return state

    def decode_sample(self,
                      state, pz, bin_width=1. / 256):
        state = rans.unflatten(state)

        device = pz[0].device
        size = pz[0].size()[0:4]

        CDFs, MEAN = self.CDF_fn(pz, bin_width)

        CDFs = CDFs.reshape(-1, self.n_bins)
        result = np.zeros(len(CDFs), dtype=int)
        for i, cdf in enumerate(CDFs):
            statfun = self.statfun_decode(cdf)
            state, symbol = rans.pop_symbol(statfun, self.precision)(state)
            result[i] = symbol

        Z_flat = torch.from_numpy(result).to(device)
        Z = Z_flat.view(size) - self.n_bins // 2 + MEAN

        z = Z.float() * bin_width

        state = rans.flatten(state)

        return state, z

    def statfun_encode(self, CDF):
        def _statfun_encode(symbol):
            return CDF[symbol], CDF[symbol + 1] - CDF[symbol]

        return _statfun_encode

    def statfun_decode(self, CDF):
        def _statfun_decode(cf):
            # Search such that CDF[s] <= cf < CDF[s]
            s = np.searchsorted(CDF, cf, side='right')
            s = s - 1
            start, freq = self.statfun_encode(CDF)(s)
            return s, (start, freq)

        return _statfun_decode


coder = Coder()


def encode_patches(imgs, model, decode=True):
    batchsize, img_c, img_h, img_w = imgs.size()

    states = model.encode(imgs)

    state_sizes = []
    error = 0

    for b in range(batchsize):
        if states[b] is None:
            # Using escape bit ;)
            state_sizes += [8 * img_c * img_h * img_w + 1]

            # Error remains unchanged.
            print('Escaping, not encoding.')

        else:
            if decode:
                x_recon = model.decode([states[b]], b)

                error += torch.sum(
                    torch.abs(x_recon.int() - (imgs[b] * 255).int())).item()

            # Append state plus an escape bit
            state_sizes += [32 * len(states[b]) + 1]

    return np.mean(state_sizes) / img_c / img_w / img_h, error


class SplitPriorContext(SplitPrior):
    def __init__(self, c_in, factor_out, splitprior_type, inverse_bin_width, rezero, context, entropy_param, **kwargs):
        super().__init__(c_in, factor_out, splitprior_type, inverse_bin_width, rezero, **kwargs)
        self.context_model = context_models(context, factor_out, factor_out * 2, **kwargs)
        self.entropy_param = param_models(entropy_param, factor_out * 4, factor_out * 2, **kwargs)

    def get_py_context(self, y, z):
        prior = self.nn(z)
        context = self.context_model(y)

        inputs = torch.cat([prior, context], 1)
        h = self.entropy_param(inputs)

        mu = h[:, ::2, :, :]
        logs = h[:, 1::2, :, :]
        if self.rezero:
            mu *= self.gamma
            logs *= self.delta

        py = [mu, logs]
        return py

    def forward(self, z, ldj):
        z, y = self.split(z)
        py = self.get_py_context(y, z)
        return py, y, z, ldj


class IDF(nn.Module):
    def __init__(self, n_levels, n_flows, in_channels, splitfactor, splitprior_type, descale, **kwargs):
        super().__init__()
        layers = []
        self.descale = Descale() if descale else None
        self.normalize = Normalize(**kwargs)
        layers.append(Squeeze())
        in_channels *= 4

        if 'context' not in kwargs:
            split_prior = SplitPrior
        else:
            split_prior = SplitPriorContext

        for level in range(n_levels):
            for i in range(n_flows):
                perm_layer = Permute(in_channels)
                layers.append(perm_layer)
                layers.append(
                    SplitFactorCoupling(in_channels, splitfactor, **kwargs))
                if kwargs.get('inv_permute'):
                    layers.append(perm_layer.InversePermute())
            if level < n_levels - 1:
                if splitprior_type != 'none':
                    # Standard splitprior
                    factor_out = in_channels // 2
                    layers.append(split_prior(
                        in_channels, factor_out, splitprior_type, **kwargs))
                    in_channels = in_channels - factor_out
                layers.append(Squeeze())
                in_channels *= 4

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, z, pys=(), ys=(), reverse=False):
        """
        Evaluates the model as a whole, encodes and decodes. Note that the log
         det jacobian is zero for a plain VAE (without flows), and z_0 = z_k.
        """
        ldj = torch.zeros_like(z[:, 0, 0, 0])
        zs = ()
        if not reverse:
            if self.descale:
                z = self.descale(z)
            z = self.normalize(z)
            for l, layer in enumerate(self.layers):
                if isinstance(layer, (SplitPrior)):
                    py, y, z, ldj = layer(z, ldj)
                    pys += (py,)
                    ys += (y,)
                    zs += (z,)
                else:
                    z, ldj = layer(z, ldj)
        else:
            for l, layer in reversed(list(enumerate(self.layers))):
                if isinstance(layer, (SplitPrior)):
                    if len(ys) > 0:
                        z, ldj = layer.inverse(z, ldj, y=ys[-1])
                        # Pop last element
                        ys = ys[:-1]
                    else:
                        z, ldj = layer.inverse(z, ldj, y=None)

                else:
                    z, ldj = layer(z, ldj, reverse=True)
            z = self.normalize(z, reverse=True)
            if self.descale:
                z = self.descale(z, reverse=True)

        return z, pys, ys, zs

    def encode(self, x):
        # TODO:encoding and decoding should use DIC built-in entropy coder,
        #  rather than this independently implemented rANS
        batchsize = x.size(0)
        z, pz, pys, ys = self.forward(x)

        pjs = list(pys) + [pz]
        js = list(ys) + [z]
        states = []

        for b in range(batchsize):
            state = None
            for pj, j in zip(pjs, js):
                pj_b = [param[b:b + 1] for param in pj]
                j_b = j[b:b + 1]
                state = coder.encode_sample(
                    j_b, pj_b, state=state)
                if state is None:
                    break
            states.append(state)
        return states

    def decode(self, states, batch_num):
        # TODO:encoding and decoding should use DIC built-in entropy coder,
        #  rather than this independently implemented rANS
        def decode_fn(states, pj):
            states = list(states)
            j = []
            for b in range(len(states)):
                pj_b = [param[b:b + 1] for param in pj]
                states[b], j_b = coder.decode_sample(
                    states[b], pj_b)
                j.append(j_b)
            j = torch.cat(j, dim=0)
            return states, j

        states, z = self.prior.decode(states, decode_fn=decode_fn)

        ldj = torch.zeros_like(z[:, 0, 0, 0])
        for l, layer in reversed(list(enumerate(self.layers))):
            if isinstance(layer, SplitPrior):
                z, ldj, states = layer.decode(z, ldj, states, decode_fn)
            else:
                z, ldj = layer(z, ldj, reverse=True)

        x = self.normalize(z, reverse=True)
        x = x.to(dtype=torch.uint8)
        return x


def flows(tag: str, **kwargs):
    if tag == 'IDF':
        model = IDF(**kwargs)
    elif tag == 'NONE':
        model = None
    else:
        raise ValueError('unsupported flow tag:{}'.format(tag))

    return model
