import math
from typing import List
from torch.nn import functional as F
from typing import NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
from typing import List, Union
from utils.distributed_utils import get_rank


class configs():
    best_bpsp = float("inf")
    plot = ""
    log_likelihood = True
    collect_probs = False


def conv(in_channels: int,
         out_channels: int,
         kernel_size: int,
         bias: bool = True,
         rate: int = 1,
         stride: int = 1) -> nn.Conv2d:
    padding = kernel_size // 2 if rate == 1 else rate

    return nn.Conv2d(
        in_channels, out_channels, kernel_size, stride=stride, dilation=rate,
        padding=padding, bias=bias)


def get_act(act: str, n_feats: int = 0) -> nn.Module:
    """ param act: Name of activation used.
        n_feats: channel size.
        returns the respective activation module, or raise
            NotImplementedError if act is not implememted.
    """
    if act == "relu":
        return nn.ReLU(inplace=True)
    elif act == "prelu":
        return nn.PReLU(n_feats)
    elif act == "leaky_relu":
        return nn.LeakyReLU(inplace=False)
    elif act == "none":
        return nn.Identity()  # type: ignore
    raise NotImplementedError(f"{act} is not implemented")


class Upsampler(nn.Sequential):
    def __init__(self,
                 scale: int,
                 n_feats: int,
                 bn: bool = False,
                 act: str = "none",
                 bias: bool = True) -> None:
        m: List[nn.Module] = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                m.append(get_act(act))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            m.append(get_act(act))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class ResBlock(nn.Module):
    """ Implementation for ResNet block. """

    def __init__(self,
                 n_feats: int,
                 kernel_size: int,
                 act: str = "leaky_relu",
                 atrous: int = 1,
                 bn: bool = False) -> None:
        """ param n_feats: Channel size.
            param kernel_size: kernel size.
            param act: string of activation to use.
            param atrous: controls amount of dilation to use in final conv.
            param bn: Turns on batch norm.
        """
        super().__init__()

        m: List[nn.Module] = []
        _repr = []
        for i in range(2):
            atrous_rate = 1 if i == 0 else atrous
            conv_filter = conv(
                n_feats, n_feats, kernel_size, rate=atrous_rate, bias=True)
            m.append(conv_filter)
            _repr.append(f"Conv({n_feats}x{kernel_size}" +
                         (f";A*{atrous_rate})" if atrous_rate != 1 else "") +
                         ")")

            if bn:
                m.append(nn.BatchNorm2d(n_feats))
                _repr.append(f"BN({n_feats})")

            if i == 0:
                m.append(get_act(act))
                _repr.append("Act")
        self.body = nn.Sequential(*m)

        self._repr = "/".join(_repr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        res = self.body(x)
        res += x
        return res

    def __repr__(self) -> str:
        return f"ResBlock({self._repr})"


class EDSRDec(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 resblocks: int = 8,
                 kernel_size: int = 3,
                 tail: str = "none",
                 channel_attention: bool = False) -> None:
        super().__init__()
        self.head = conv(in_ch, out_ch, 3)
        m_body: List[nn.Module] = [
            ResBlock(out_ch, kernel_size) for _ in range(resblocks)]
        self.body = nn.Sequential(*m_body)

        self.tail: nn.Module
        if tail == "conv":
            self.tail = conv(out_ch, out_ch, 3)
        elif tail == "none":
            self.tail = nn.Identity()  # type: ignore

    def forward(self,  # type: ignore
                x: torch.Tensor,
                features_to_fuse: torch.Tensor = 0.,  # type: ignore
                ) -> torch.Tensor:
        """
        :param x: N C+1 H W
        :return: N C" H W
        """
        x = self.head(x)

        x = x + features_to_fuse

        x = self.body(x) + x
        x = self.tail(x)

        return x


_NUM_PARAMS_RGB = 4  # mu, sigma, pi, lambda
_NUM_PARAMS_OTHER = 3  # mu, sigma, pi
_LOG_SCALES_MIN = -7.


# quantizer
class quantizer():
    def to_sym(x, x_min, x_max, L):
        sym_range = x_max - x_min
        bin_size = sym_range / (L - 1)
        return x.clamp(x_min, x_max).sub(x_min).div(bin_size).round()

    def to_bn(S, x_min, x_max, L):
        sym_range = x_max - x_min
        bin_size = sym_range / (L - 1)
        return S.float().mul(bin_size).add(x_min)


quantizer = quantizer()


class CDFOut(NamedTuple):
    logit_probs_c_sm: torch.Tensor
    means_c: torch.Tensor
    log_scales_c: torch.Tensor
    K: int
    targets: torch.Tensor


def non_shared_get_Kp(K, C, num_params):
    """ Get Kp=number of channels to predict.
        See note where we define _NUM_PARAMS_RGB above """
    return num_params * C * K


def non_shared_get_K(Kp: int, C: int, num_params: int) -> int:
    """ Inverse of non_shared_get_Kp, get back K=number of mixtures """
    return Kp // (num_params * C)


# --------------------------------------------------------------------------------
class DiscretizedMixLogisticLoss(nn.Module):
    def __init__(self, rgb_scale: bool, x_min=0, x_max=255, L=256):
        """
        :param rgb_scale: Whether this is the loss for the RGB scale. In that case,
            use_coeffs=True
            _num_params=_NUM_PARAMS_RGB == 4, since we predict coefficients lambda. See note above.
        :param x_min: minimum value in targets x
        :param x_max: maximum value in targets x
        :param L: number of symbols
        """
        super(DiscretizedMixLogisticLoss, self).__init__()
        self.rgb_scale = rgb_scale
        self.x_min = x_min
        self.x_max = x_max
        self.L = L
        # whether to use coefficients lambda to weight
        # means depending on previously outputed means.
        self.use_coeffs = rgb_scale
        # P means number of different variables contained
        # in l, l means output of network
        self._num_params = (
            _NUM_PARAMS_RGB if rgb_scale else
            _NUM_PARAMS_OTHER)

        # NOTE: in contrast to the original code,
        # we use a sigmoid (instead of a tanh)
        # The optimizer seems to not care,
        # but it would probably be more principaled to use a tanh
        # Compare with L55 here:
        # https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py#L55
        self._nonshared_coeffs_act = torch.sigmoid

        # Adapted bounds for our case.
        self.bin_width = (x_max - x_min) / (L - 1)
        self.x_lower_bound = x_min + 0.001
        self.x_upper_bound = x_max - 0.001

        self._extra_repr = 'DMLL: x={}, L={}, coeffs={}, P={}, bin_width={}'.format(
            (self.x_min, self.x_max), self.L, self.use_coeffs, self._num_params, self.bin_width)

    def extra_repr(self):
        return self._extra_repr

    @staticmethod
    def to_per_pixel(entropy, C):
        N, H, W = entropy.shape
        return entropy.sum() / (N * C * H * W)  # NHW -> scalar

    def to_sym(self, x):
        return quantizer.to_sym(x, self.x_min, self.x_max, self.L)

    def to_bn(self, S):
        return quantizer.to_bn(S, self.x_min, self.x_max, self.L)

    def cdf_step_non_shared(self, l, targets, c_cur, C, x_c=None) -> CDFOut:
        assert c_cur < C

        # NKHW         NKHW     NKHW
        logit_probs_c, means_c, log_scales_c, K = self._extract_non_shared_c(
            c_cur, C, l, x_c)

        logit_probs_c_softmax = F.softmax(logit_probs_c, dim=1)  # NKHW, pi_k
        return CDFOut(
            logit_probs_c_softmax, means_c,
            log_scales_c, K, targets.to(l.device))

    def sample(self, l, C):
        return self._non_shared_sample(l, C)

    def log_cdf(self, lo, hi, means, log_scales):
        assert torch.all(lo <= hi), f"{lo[lo > hi]} > {hi[lo > hi]}"
        assert lo.min() >= self.x_min and hi.max() <= self.x_max, \
            '{},{} not in {},{}'.format(
                lo.min(), hi.max(), self.x_min, self.x_max)

        centered_lo = lo - means  # NCKHW
        centered_hi = hi - means

        # Calc cdf_delta
        # all of the following is NCKHW
        # <= exp(7), is exp(-sigma), inverse std. deviation, i.e., sigma'
        inv_stdv = torch.exp(-log_scales)
        # sigma' * (x - mu + 0.5)
        # S(sigma' * (x - mu - 1/255)) = 1 / (1 + exp(sigma' * (x - mu - 1/255))
        normalized_lo = inv_stdv * (
                centered_lo - self.bin_width / 2)  # sigma' * (x - mu - 1/255)
        lo_cond = (lo >= self.x_lower_bound).float()
        # log probability for edge case of 0
        cdf_lo = lo_cond * torch.sigmoid(normalized_lo)
        normalized_hi = inv_stdv * (centered_hi + self.bin_width / 2)
        hi_cond = (hi <= self.x_upper_bound).float()
        cdf_hi = hi_cond * torch.sigmoid(normalized_hi) + (1 - hi_cond)  # * 1.
        # S(sigma' * (x - mu + 1/255))
        # NCKT, cdf^k(c)
        cdf_delta = cdf_hi - cdf_lo
        log_cdf_delta = torch.log(torch.clamp(cdf_delta, min=1e-12))

        assert not torch.any(
            log_cdf_delta > 1e-6
        ), f"{log_cdf_delta[log_cdf_delta > 1e-6]}"
        return log_cdf_delta

    def forward(  # type: ignore
            self, x: torch.Tensor, l: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param x: labels, i.e., NCT, float
        :param l: predicted distribution, i.e., NKpT, see above
        :return: log-likelihood, as NT if shared, NCT if non_shared pis
        """
        assert x.min() >= self.x_min and x.max() <= self.x_max, \
            f'{x.min()},{x.max()} not in {self.x_min},{self.x_max}'

        # Extract ---
        #  NC1T     NCKT      NCKT  NCKT
        x, logit_pis, means, log_scales, _ = self._extract_non_shared(x, l)

        log_probs = self.log_cdf(x, x, means, log_scales)

        # combine with pi, NCKT, (-inf, 0]
        log_weights = F.log_softmax(logit_pis, dim=2)
        log_probs_weighted = log_weights + log_probs

        # final log(P), NCT
        nll = -torch.logsumexp(log_probs_weighted, dim=2)
        return nll

    def _extract_non_shared(self, x, l):
        """
        :param x: targets, NCT
        :param l: output of net, NKpT, see above
        :return:
            x NC1T,
            logit_probs NCKT (probabilites of scales, i.e., \pi_k)
            means NCKT,
            log_scales NCKT (variances),
            K (number of mixtures)
        """
        N, C, T = x.shape
        Kp = l.shape[1]

        K = non_shared_get_K(Kp, C, self._num_params)

        # we have, for each channel: K pi / K mu / K sigma / [K coeffs]
        # note that this only holds for C=3 as for other channels,
        # there would be more than 3*K coeffs
        # but non_shared only holds for the C=3 case
        l = l.reshape(N, self._num_params, C, K, T)

        logit_probs = l[:, 0, ...]  # NCKT

        means = l[:, 1, ...]  # NCKT
        log_scales = torch.clamp(
            l[:, 2, ...], min=_LOG_SCALES_MIN)  # NCKT, is >= -7
        x = x.reshape(N, C, 1, T)

        if self.use_coeffs:
            # Coefficients only supported for multiples of 3,
            # see note where we define
            # _NUM_PARAMS_RGB NCKHW, basically coeffs_g_r, coeffs_b_r, coeffs_b_g
            assert C == 3, C
            # Each NCKHW
            coeffs = self._nonshared_coeffs_act(l[:, 3, ...])
            # each NKHW
            coeffs_g_r = coeffs[:, 0, ...]
            coeffs_b_r = coeffs[:, 1, ...]
            coeffs_b_g = coeffs[:, 2, ...]
            # NCKHW
            means = torch.stack(
                (means[:, 0, ...],
                 means[:, 1, ...] + coeffs_g_r * x[:, 0, ...],
                 means[:, 2, ...] + coeffs_b_r * x[:, 0, ...]
                 + coeffs_b_g * x[:, 1, ...]),
                dim=1)

        means = torch.clamp(means, min=self.x_min, max=self.x_max)
        assert means.shape == (N, C, K, T), (means.shape, (N, C, K, T))
        return x, logit_probs, means, log_scales, K

    def _extract_non_shared_c(
            self, c: int, C: int, l: torch.Tensor,
            x: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Same as _extract_non_shared but only for c-th channel, used to get CDF
        """
        assert c < C, f'{c} >= {C}'

        N, Kp, H, W = l.shape
        K = non_shared_get_K(Kp, C, self._num_params)

        l = l.reshape(N, self._num_params, C, K, H, W)
        logit_probs_c = l[:, 0, c, ...]  # NKHW
        means_c = l[:, 1, c, ...]  # NKHW
        log_scales_c = torch.clamp(
            l[:, 2, c, ...], min=_LOG_SCALES_MIN)  # NKHW, is >= -7

        if self.use_coeffs and c != 0:
            # N C K H W, coeffs_g_r, coeffs_b_r, coeffs_b_g
            unscaled_coeffs = l[:, 3, ...]
            if c == 1:
                assert x is not None
                coeffs_g_r = self._nonshared_coeffs_act(
                    unscaled_coeffs[:, 0, ...])  # NKHW
                means_c += coeffs_g_r * x[:, 0, ...]
            elif c == 2:
                assert x is not None
                coeffs_b_r = self._nonshared_coeffs_act(
                    unscaled_coeffs[:, 1, ...])  # NKHW
                coeffs_b_g = self._nonshared_coeffs_act(
                    unscaled_coeffs[:, 2, ...])  # NKHW
                means_c += coeffs_b_r * x[:, 0, ...] + coeffs_b_g * x[:, 1, ...]

        #      NKHW           NKHW     NKHW
        return logit_probs_c, means_c, log_scales_c, K

    def _non_shared_sample(self, l, C):
        """ sample from model """
        N, Kp, H, W = l.shape
        K = non_shared_get_K(Kp, C, self._num_params)
        l = l.reshape(N, self._num_params, C, K, H, W)

        logit_probs = l[:, 0, ...]  # NCKHW

        # sample mixture indicator from softmax
        u = torch.zeros_like(logit_probs).uniform_(1e-5, 1. - 1e-5)  # NCKHW
        # argmax over K, results in NCHW,
        # specifies for each c: which of the K mixtures to take
        sel = torch.argmax(
            logit_probs - torch.log(-torch.log(u)),  # gumbel sampling
            dim=2)
        assert sel.shape == (N, C, H, W), (sel.shape, (N, C, H, W))

        sel = sel.unsqueeze(2)  # NC1HW

        means = torch.gather(l[:, 1, ...], 2, sel).squeeze(2)
        log_scales = torch.clamp(torch.gather(
            l[:, 2, ...], 2, sel).squeeze(2), min=_LOG_SCALES_MIN)

        # sample from the resulting logistic,
        # which now has essentially 1 mixture component only.
        # We use inverse transform sampling.
        # i.e. X~logistic; generate u ~ Unfirom; x = CDF^-1(u),
        #  where CDF^-1 for the logistic is CDF^-1(y) = \mu + \sigma * log(y / (1-y))
        u = torch.zeros_like(means).uniform_(1e-5, 1. - 1e-5)  # NCHW
        x = means + torch.exp(log_scales) * \
            (torch.log(u) - torch.log(1. - u))  # NCHW

        if self.use_coeffs:
            assert C == 3

            def clamp(x_):
                return torch.clamp(x_, 0, 255.)

            # Be careful about coefficients!
            # We need to use the correct selection mask, namely the one for the G and
            #  B channels, as we update the G and B means!
            # Doing torch.gather(l[:, 3, ...], 2, sel) would be completly
            #  wrong.
            coeffs = torch.sigmoid(l[:, 3, ...])
            sel_g, sel_b = sel[:, 1, ...], sel[:, 2, ...]
            coeffs_g_r = torch.gather(coeffs[:, 0, ...], 1, sel_g).squeeze(1)
            coeffs_b_r = torch.gather(coeffs[:, 1, ...], 1, sel_b).squeeze(1)
            coeffs_b_g = torch.gather(coeffs[:, 2, ...], 1, sel_b).squeeze(1)

            # Note: In theory, we should go step by step over the channels
            # and update means with previously sampled
            # xs. But because of the math above (x = means + ...),
            # we can just update the means here and it's all good.
            x0 = clamp(x[:, 0, ...])
            x1 = clamp(x[:, 1, ...] + coeffs_g_r * x0)
            x2 = clamp(x[:, 2, ...] + coeffs_b_r * x0 + coeffs_b_g * x1)
            x = torch.stack((x0, x1, x2), dim=1)
        return x


class AtrousProbabilityClassifier(nn.Module):
    def __init__(self,
                 in_ch: int,
                 C: int,
                 num_params: int,
                 K: int = 10,
                 kernel_size: int = 3,
                 atrous_rates_str: str = '1,2,4') -> None:
        super(AtrousProbabilityClassifier, self).__init__()

        Kp = non_shared_get_Kp(K, C, num_params)

        self.atrous = StackedAtrousConvs(atrous_rates_str, in_ch, Kp,
                                         kernel_size=kernel_size)
        self._repr = f'C={C}; K={K}; Kp={Kp}; rates={atrous_rates_str}'

    def __repr__(self) -> str:
        return f'AtrousProbabilityClassifier({self._repr})'

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """
        :param x: N C H W
        :return: N Kp H W
        """

        x = self.atrous(x)

        return x


class StackedAtrousConvs(nn.Module):
    def __init__(self,
                 atrous_rates_str: Union[str, int],
                 Cin: int,
                 Cout: int,
                 bias: bool = True,
                 kernel_size: int = 3) -> None:
        super(StackedAtrousConvs, self).__init__()
        atrous_rates = self._parse_atrous_rates_str(atrous_rates_str)
        self.atrous = nn.ModuleList(
            [conv(Cin, Cin, kernel_size, rate=rate)
             for rate in atrous_rates])
        self.lin = conv(len(atrous_rates) * Cin, Cout, 1, bias=bias)
        self._extra_repr = 'rates={}'.format(atrous_rates)

    @staticmethod
    def _parse_atrous_rates_str(atrous_rates_str: Union[str, int]) -> List[int]:
        # expected to either be an int or a comma-separated string 1,2,4
        if isinstance(atrous_rates_str, int):
            return [atrous_rates_str]
        else:
            return list(map(int, atrous_rates_str.split(',')))

    def extra_repr(self) -> str:
        return self._extra_repr

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = torch.cat([atrous(x)
                       for atrous in self.atrous], dim=1)  # type: ignore
        x = self.lin(x)
        return x
