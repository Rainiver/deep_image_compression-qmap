from collections import defaultdict
from typing import (DefaultDict, Generator, KeysView, List, NamedTuple,
                    Optional, Tuple)

import numpy as np
import torch
from torch import nn

from nets.multi_nets import *


# from nets.multi_nets import logistic_mixture as lm


def get_mask(x: torch.Tensor, scale: int, pattern: str):
    # positions denoted by 'x' should be 0
    mask_list = []
    mask_group_list = []

    _, _, h, w = x.size()
    assert h % 2 ** scale == 0 and w % 2 ** scale == 0
    if pattern == 'fixed-a':
        mask_list.append(x)
        for i in range(scale):
            mask_group = torch.zeros(h, w)
            mask_group[::2, ::2], mask_group[::2, 1::2], mask_group[1::2, 0::2], \
            mask_group[1::2, 1::2] = 1, 2, 3, 0
            mask_group_list.append(mask_group)
            mask = x[:, :, 1::2, 1::2]
            x = mask
            mask_list.append(mask)
            _, _, h, w = x.size()
    elif pattern == 'fixed-b':
        mask_list.append(x)
        for i in range(scale):
            mask_group = torch.zeros(h, w)
            mask_group[1::2, 1::2] = 0
            mask_group[1::2, 2::4] = 1
            mask_group[0::2, 2::4] = 2
            mask_group[0::2, 1::4] = 3
            mask_group[0::2, 3::4] = 4
            mask_group[0::2, 0::4] = 5
            mask_group[1::2, 0::4] = 6
            mask_group_list.append(mask_group)
            mask = x[:, :, 1::2, 1::2]
            x = mask
            mask_list.append(mask)
            _, _, h, w = x.size()
    else:
        raise ValueError

    return mask_list, mask_group_list


def stack_mask(mask, n):
    x = mask.repeat(1, n, 1, 1)
    return x


def downsample(x: torch.Tensor):
    return


class LogisticMixtureProbability(NamedTuple):
    name: str
    pixel_index: int
    probs: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor


Probs = Tuple[torch.Tensor, Optional[LogisticMixtureProbability], int]


class Bits:
    """
    Tracks bpsps from different parts of the pipeline for one forward pass.
    """

    def __init__(self) -> None:
        assert configs.collect_probs or configs.log_likelihood, (
            configs.collect_probs, configs.log_likelihood)
        self.key_to_bits: DefaultDict[
            str, torch.Tensor] = defaultdict(float)  # type: ignore
        self.key_to_sizes: DefaultDict[str, int] = defaultdict(int)
        self.probs: List[Probs] = []

    def add_with_size(
            self, key: str, nll_sum: torch.Tensor, size: int,
    ) -> None:
        if configs.log_likelihood:
            assert key not in self.key_to_bits, f"{key} already exists"
            # Divide by np.log(2) to convert from natural log to log base 2
            self.key_to_bits[key] = nll_sum / np.log(2)
            self.key_to_sizes[key] = size

    def add(self, key: str, nll: torch.Tensor) -> None:
        self.add_with_size(
            key, nll.sum(), np.prod(nll.size()))

    def add_lm(
            self, y_i: torch.Tensor,
            lm_probs: LogisticMixtureProbability,
            loss_fn: DiscretizedMixLogisticLoss) -> None:
        assert lm_probs.probs.shape[-1:] == y_i.shape[-1:], (
            lm_probs.probs.shape, y_i.shape)
        if configs.log_likelihood:
            nll = loss_fn(y_i, lm_probs.probs)
            # print(nll.size())
            self.add(lm_probs.name, nll)
        if configs.collect_probs:
            self.probs.append((y_i, lm_probs, -1))

    def add_uniform(
            self,
            key: str,
            y_i: torch.Tensor,
            levels: int = 256) -> None:
        if configs.log_likelihood:
            size = np.prod(y_i.size())
            nll_sum = np.log(levels) * size
            self.add_with_size(key, nll_sum, size)
        if configs.collect_probs:
            self.probs.append((y_i, None, levels))

    def get_bits(self, key: str) -> torch.Tensor:
        return self.key_to_bits[key]

    def get_size(self, key: str) -> int:
        return self.key_to_sizes[key]

    def get_keys(self) -> KeysView:
        return self.key_to_bits.keys()

    def get_self_bpsp(self, key: str) -> torch.Tensor:
        return self.key_to_bits[key] / self.key_to_sizes[key]

    def get_scaled_bpsp(self, key: str, inp_size: int) -> torch.Tensor:
        return self.key_to_bits[key] / inp_size

    def get_total_bits(self) -> torch.Tensor:
        return sum(self.key_to_bits.values())

    def get_total_bpsp(self, inp_size: int) -> torch.Tensor:
        return sum(self.key_to_bits.values()) / inp_size  # type: ignore

    def update(self, other: "Bits") -> "Bits":
        # Used by Compressor to aggregate bits from decoder.
        assert len(self.get_keys() & other.get_keys()) == 0, \
            f"{self.get_keys()} and {other.get_keys()} intersect."
        self.key_to_bits.update(other.key_to_bits)
        self.key_to_sizes.update(other.key_to_sizes)
        self.probs += other.probs
        return self

    def add_bits(self, other: "Bits") -> "Bits":
        keys = other.get_keys()
        assert keys == self.get_keys() or len(self.get_keys()) == 0, (
            f"{self.get_keys()} != {keys}")

        for key in keys:
            self.key_to_bits[key] += other.get_bits(key)
            self.key_to_sizes[key] += other.get_size(key)
            # Don't do anything with self.key_to_probs at the moment.
        return self


class PixDecoder(nn.Module):
    """ Super-resolution based decoder for pixel-based factorization. """

    def __init__(self, scale: int) -> None:
        super().__init__()
        self.loss_fn = DiscretizedMixLogisticLoss(rgb_scale=True)
        self.scale = scale

    def forward_probs(
            self,
            x: torch.Tensor,
            ctx: torch.Tensor,
            mask: torch.Tensor
    ) -> Generator[LogisticMixtureProbability, torch.Tensor,
                   Tuple[torch.Tensor, torch.Tensor]]:
        raise NotImplementedError

    def forward(self,  # type: ignore
                x: torch.Tensor,
                y: torch.Tensor,
                ctx: torch.Tensor,
                mask: torch.Tensor,
                ) -> Tuple[Bits, torch.Tensor]:
        bits = Bits()

        # Check y are filled with integers.
        if __debug__:
            not_int = y.long().float() != y
            assert not torch.any(not_int), y[not_int]

        # mode is used to key tensorboard loggings
        mode = "train" if self.training else "eval"

        _, _, x_h, x_w = x.size()
        if not isinstance(ctx, float):
            ctx = ctx[..., :x_h, :x_w]

        gen = self.forward_probs(x, ctx, mask)
        try:
            for i in range(int(mask.max())):
                if i == 0:
                    lm_probs = next(gen)
                else:
                    lm_probs = gen.send(y)
                mask_tmp = mask == i + 1
                bits.add_lm(y[..., mask_tmp], lm_probs, self.loss_fn)
            ctx = gen.send(y)
        except StopIteration as e:
            last_pixels, ctx = e.value
            last_slice = y_slices[-1]
            _, _, last_h, last_w = last_slice.size()
            last_pixels = last_pixels[..., : last_h, : last_w]
            assert torch.all(last_pixels == last_slice), (
                last_pixels[last_pixels != last_slice],
                last_slice[last_pixels != last_slice])

        return bits, ctx


class StrongPixDecoder(PixDecoder):
    def __init__(self, scale: int, num_groups: int, n_feats: int = 64, n_resblocks: int = 5, K: int = 5) -> None:
        super().__init__(scale)
        # Input: N 3 H W
        # Output: N C H W
        self.num_groups = num_groups
        self.rgb_decs = nn.ModuleList([
            EDSRDec(
                3 + 1, n_feats,
                resblocks=n_resblocks, tail="conv")
            for i in range(self.num_groups)
        ])
        self.mix_logits_prob_clf = nn.ModuleList([
            conv(n_feats, 3 * K * self.loss_fn._num_params, 3)
            for _ in range(self.num_groups)
        ])
        self.feat_convs = nn.ModuleList([
            conv(n_feats, n_feats, 3) for _ in range(self.num_groups)
        ])
        assert (len(self.rgb_decs) == len(self.mix_logits_prob_clf) ==
                len(self.feat_convs)), (
            f"{len(self.rgb_decs)}, "
            f"{len(self.mix_logits_prob_clf)}, {len(self.feat_convs)}"
        )

    def forward_probs(
            self,
            x: torch.Tensor,
            ctx: torch.Tensor,
            mask: torch.Tensor
    ) -> Generator[LogisticMixtureProbability, torch.Tensor,
                   Tuple[torch.Tensor, torch.Tensor]]:
        # mode is used to key tensorboard loggings
        mode = "train" if self.training else "eval"
        # x: N 3 H W, [0, 255]
        # mask: H W, [0,1]

        xy_normalized = x / 127.5 - 1

        # init mask, positions denoted by 'x' should be 1
        mask_tmp = torch.where(mask == 0, torch.tensor(1.).cuda(),
                               torch.tensor(0.).cuda()).repeat(x.shape[0], 1, 1, 1)

        for i, (rgb_dec, clf, feat_conv) in enumerate(
                zip(self.rgb_decs,  # type: ignore
                    self.mix_logits_prob_clf, self.feat_convs)):
            assert mask_tmp.max() > 0

            xy_normalized_cat = torch.cat((xy_normalized, mask_tmp), dim=1)
            z = rgb_dec(xy_normalized_cat, ctx)
            ctx = feat_conv(z)

            # probs: N Kp T, T is num of pixels in current group
            probs = clf(z)[..., mask == i + 1]

            lower = torch.zeros(x.size(), device=x.device)
            upper = torch.full(x.size(), 255., device=x.device)

            y = yield LogisticMixtureProbability(
                f"{mode}/{self.scale}_{i}", i, probs, lower, upper)

            y = y / 127.5 - 1
            # update mask
            mask_tmp += torch.where(mask == i + 1, torch.tensor(1.).cuda(), torch.tensor(0.).cuda()).repeat(x.shape[0],
                                                                                                            1, 1, 1)
            xy_normalized = stack_mask(mask_tmp, 3) * y + (1 - stack_mask(mask_tmp, 3)) * xy_normalized

        yield ctx


class Compressor(nn.Module):
    def __init__(self, pattern, num_groups, scale, n_feats=64, n_resblocks=5, K=5) -> None:
        super().__init__()
        assert scale >= 0

        self.scale = scale
        self.pattern = pattern
        self.num_groups = num_groups
        self.ctx_upsamplers = nn.ModuleList([nn.Identity(),
                                             *[Upsampler(scale=2, n_feats=n_feats)
                                               for _ in range(self.scale - 1)]
                                             ])
        self.decs = nn.ModuleList([
            StrongPixDecoder(i, self.num_groups, n_feats, n_resblocks, K) for i in range(self.scale)
        ])
        assert len(self.ctx_upsamplers) == len(self.decs), \
            f"{len(self.ctx_upsamplers)}, {len(self.decs)}"

    def forward(self,  # type: ignore
                x: torch.Tensor
                ) -> Bits:
        downsampled, masks = get_mask(x, self.scale, self.pattern)

        assert len(downsampled) - 1 == len(self.decs), (
            f"{len(downsampled) - 1}, {len(self.decs)}")

        mode = "train" if self.training else "eval"

        bits = Bits()
        bits.add_uniform(f"{mode}/codes_0", downsampled[-1])

        ctx = 0.
        for dec, ctx_upsampler, x, y, mask in zip(  # type: ignore
                self.decs, self.ctx_upsamplers,
                downsampled[::-1], downsampled[-2::-1], masks[::-1]):
            _, _, h, w = x.size()
            _, _, h_tar, w_tar = y.size()
            ctx = ctx_upsampler(ctx)
            x = nn.functional.interpolate(x, [h_tar, w_tar], mode='nearest')
            mask = mask.cuda()
            dec_bits, ctx = dec(x, y, ctx, mask)
            bits.update(dec_bits)
        return bits
