import math
import logging
from typing import Tuple, Union
from distutils.version import LooseVersion

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter
from torch.nn.modules.conv import Conv2d, ConvTranspose2d
from torch.nn.modules.linear import Linear
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors, \
    _take_tensors
from nets.layers import SignalConv2d, SignalConvTranspose2d

from .functional import *
# from utils.distributed_utils import get_dist_interface
try:
    import spring.linklink as link
except:
    link = None

ParamT = Union[Tuple[Parameter, float, float], Parameter]

_is_0_4 = LooseVersion("0.4.0") <= LooseVersion(torch.__version__)
__all__ = ["EMAAct", "EMABnConv2d", "NoBnConv2d", "NoBnConvTranspose2d", "QuantLinear"]


def magnitude_inq_mask(x, quant_portion):
    """ Split input tensor by magnitudes.

    Args:
        x: input tensor, must be real-valued;
        quant_portion: portion of quantized and fixed values in current phase;

    Returns:
        quant_mask (ByteTensor): elements to quantize and freeze training;

    """
    assert 0 < quant_portion < 1, f"invalid magnitude threshold for INQ: {quant_portion}"
    with torch.no_grad():
        if quant_portion == 1.:
            quant_mask = torch.ones_like(x, dtype=torch.uint8)
        else:
            x_magnitudes = x.abs()
            inq_count = int(x.numel() * quant_portion)
            sorted_val, _ = torch.topk(x_magnitudes.view(-1), inq_count)
            split_point = sorted_val[-1]
            quant_mask = split_point < x_magnitudes

    return quant_mask


def bit_plane_split(x, x_min, x_max, fix_k):
    raise NotImplementedError("TODO(Rundong)")


class EMAAct(Module):
    def __init__(self, activation_bit=8, grad_bit=32, lb=None, ub=None,
                 quant_mode="symmetric", tail="clip", inq_mode="disable",
                 channel_wise=False, channel_num=None,
                 enable_quant=False, enable_quant_grad=False,
                 stat_mode="EMA", momentum=0.9999, running_stat=False, percentile=99.9,
                 sync=True, use_batch_stat=False,
                 exp_k=None, frac_k=None, enable_low_fp=False,
                 inplace=False, is_end=False, use_Int=False):
        super(EMAAct, self).__init__()
        # TODO(Rundong): redesign INQ interface for activation
        # currently INQ is checked then ignored
        assert inq_mode in ("disable", "magnitude")
        assert tail in ("clip", "preserve", "log")
        assert quant_mode in ("symmetric", "biased", "dorefa")
        assert stat_mode in ("EMA", "quick", "magnitude")

        self.use_Int = use_Int
        self.activation_bit = activation_bit
        self.gradient_bit = grad_bit
        self.momentum = momentum
        self.enable_quant = enable_quant
        self.enable_quant_grad = enable_quant_grad
        self.running_stat = running_stat
        self.use_batch_stat = use_batch_stat
        self.inplace = inplace
        self.quant_mode = quant_mode
        self.channel_wise = channel_wise
        self.channel_num = channel_num
        self.tail = tail
        self.sync = sync
        self.stat_mode = stat_mode
        self.percentile = percentile
        self.enable_low_fp = enable_low_fp
        self.is_end = is_end

        # refactor: since we perform all-reduce outside of module, the first
        # forward iteration's ema-stat should be carefully handled
        self.stat_updated = False
        self.batch_updated = False
        self.async_called = False  # used for linklink async all_reduce

        self.lb = lb
        self.ub = ub
        self.exp_k = exp_k
        self.frac_k = frac_k
        self.conditional_init()
        self.to_caffe2q = False
    
    def conditional_init(self):
        if self.quant_mode == "symmetric":
            self.act_func = SymmetricQuantFunc.apply
        elif self.quant_mode in ("biased", "dorefa"):
            self.act_func = BiasedQuantFunc.apply

        if self.sync:
            d = link # get_dist_interface()
            try:
                world_size = d.get_world_size()
            except AssertionError:
                world_size = 1
            self.sync &= world_size > 1
        
        if self.enable_low_fp:
            self.fp_func = LowPrecisionFloatFunc.apply
            # self.exp_k = exp_k
            # self.frac_k = frac_k
        if self.channel_wise:
            channel_num = self.channel_num
            assert channel_num is not None
            self.register_buffer("stat_min", torch.Tensor(channel_num).zero_())
            self.register_buffer("stat_max", torch.Tensor(channel_num).zero_())
            self.register_buffer("batch_min", torch.Tensor(channel_num).zero_())
            self.register_buffer("batch_max", torch.Tensor(channel_num).zero_())
        else:
            self.register_buffer("stat_min", torch.tensor(0.))
            self.register_buffer("stat_max", torch.tensor(0.))
            self.register_buffer("batch_min", torch.tensor(0.))
            self.register_buffer("batch_max", torch.tensor(0.))
        if self.lb is not None and self.ub is not None:
            self.stat_min.fill_(self.lb)
            self.stat_max.fill_(self.ub)
            self.freeze_stat = True
        else:
            self.freeze_stat = False

    def __repr__(self):
        if self.channel_wise:
            stat_min = self.stat_min.min().item()
            stat_max = self.stat_max.max().item()
        else:
            stat_min, stat_max = self.stat_min.item(), self.stat_max.item()
        return f"{self.__class__.__name__}(activation_bit={self.activation_bit}, " \
               f"gradient_bit={self.gradient_bit}, " \
               f"channel_wise={self.channel_wise},\n\t" \
               f"enable_quant={self.enable_quant}, " \
               f"enable_quant_grad={self.enable_quant_grad}, " \
               f"enable_low_fp={self.enable_low_fp},\n\t" \
               f"running_stat={self.running_stat}, " \
               f"stat_min={stat_min}, stat_max={stat_max})"

    def _get_batch_stat(self, x):
        with torch.no_grad():
            if self.stat_mode == "quick":
                _percentile = Percentile.apply
                if self.channel_wise:
                    raise NotImplementedError(f"channel-wise percentile not implemented yet")
                x_min = _percentile(x, 100. - self.percentile)
                x_max = _percentile(x, self.percentile)
            else:
                if self.channel_wise:
                    x_min = x.min(0)[0].min(1)[0].min(1)[0] # x.min(1)
                    x_max = x.max(0)[0].max(1)[0].max(1)[0] # x.max(1)
                else:
                    x_min = x.min()
                    x_max = x.max()

            if self.batch_updated:
                # for the case that some modules are invoked more than once in
                # each forward pass, e.g. the modules in RPN subnet
                self.batch_min = torch.min(x_min, self.batch_min)
                self.batch_max = torch.max(x_max, self.batch_max)
            else:
                self.batch_min = x_min
                self.batch_max = x_max
        self.batch_updated = True

    def _update_running_stat(self):
        assert self.training, "activation stat should only be enabled during training"
        assert self.batch_updated, "batch stat not updated yet"
        assert not self.freeze_stat, "activation range is already set"
        assert not self.stat_updated, "running stats already updated"

        with torch.no_grad():
            if self.stat_mode == "EMA":
                self.stat_min.mul_(self.momentum).add_(1. - self.momentum, self.batch_min)
                self.stat_max.mul_(self.momentum).add_(1. - self.momentum, self.batch_max)
            else:
                torch.min(self.batch_min, self.stat_min, out=self.stat_min)
                torch.max(self.batch_max, self.stat_max, out=self.stat_max)

            self.stat_updated = True
            self.batch_updated = False

    def forward(self, x):
        if self.to_caffe2q:
            return x

        if self.enable_low_fp:
            x = self.fp_func(x, self.exp_k, self.frac_k)

        if self.training and (self.running_stat or self.use_batch_stat):
            self._get_batch_stat(x)
            if self.sync:
                self.stat_updated = False
            else:
                self.stat_updated = False
                self._update_running_stat()

        if self.enable_quant:
            if self.training and self.use_batch_stat:
                assert self.batch_updated, self.batch_updated
                x_min, x_max = self.batch_min, self.batch_max
                self.batch_updated = False
            else:
                # assert self.stat_updated
                x_min, x_max = self.stat_min, self.stat_max

            # torch.Function doesn't support keyword arguments
            quant_mask = None
            for_act = True
            if self.quant_mode == "symmetric":
                magnitude = torch.max(x_min.abs(), x_max.abs())
                return self.act_func(x, self.activation_bit, self.gradient_bit,
                                     self.tail, quant_mask, magnitude,
                                     self.channel_wise, self.enable_quant_grad,
                                     self.inplace, self.use_Int, for_act)
            elif self.quant_mode in ("biased", "dorefa"):
                return self.act_func(x, self.activation_bit, self.gradient_bit,
                                     self.tail, quant_mask, x_min, x_max,
                                     self.channel_wise, self.enable_quant_grad,
                                     self.inplace, self.use_Int, for_act)
        else:
            return x

    @staticmethod
    def reduce_stat_multi_gpu(model):
        """ PyTorch official flavored reduce, used in conjunction with
        `nn.parallel.DistributedDataParallel`
        """
        d = link # get_dist_interface()

        assert isinstance(model, nn.parallel.DistributedDataParallel)
        assert d.get_world_size() > 1 and len(model.device_ids) > 1, \
            "this method should only be used in multi-node-multi-device setting"

        def _reduce_chunks(devices_stat, reduce_op):
            # in most cases we only need call this once
            bucket_size = model.nccl_reduce_bucket_size // len(devices_stat)
            for devices_chunks in zip(*(_take_tensors(device_stat, bucket_size)
                                        for device_stat in devices_stat)):
                flatten_chunks = [_flatten_dense_tensors(device_chunks)
                                  for device_chunks in devices_chunks]
                d.all_reduce_multigpu(flatten_chunks, reduce_op)
                synced_devices_chunks = [_unflatten_dense_tensors(flatten_chunk,
                                                                  device_chunks)
                                         for flatten_chunk, device_chunks
                                         in zip(flatten_chunks, devices_chunks)]
                for device_chunks, device_synced_chunks in zip(devices_chunks,
                                                               synced_devices_chunks):
                    for chunk, synced in zip(device_chunks, device_synced_chunks):
                        chunk.copy_(synced)

        devices_min = [[] for _ in model.device_ids]
        devices_max = [[] for _ in model.device_ids]
        for device_id in model.device_ids:
            # HACK: this implementation may changes among pytorch versions
            for n, m in model._module_copies[device_id].named_modules():
                if isinstance(m, EMAAct):
                    if not m.sync:
                        return
                    assert m.batch_updated and not m.stat_updated, \
                        f"rank[{d.get_rank()}].device[{device_id}].{n} " \
                        f"not ready yet: batch_updated={m.batch_updated}, " \
                        f"stat_updated={m.stat_updated}"
                    devices_min[device_id].append(m.batch_min)
                    devices_max[device_id].append(m.batch_max)

        _reduce_chunks(devices_min, d.reduce_op.MIN)
        _reduce_chunks(devices_max, d.reduce_op.MAX)

        # only update device[0], since all buffers will be broadcast to
        # all other intra-node devices
        for m in model.module.modules():
            if isinstance(m, EMAAct):
                m._update_running_stat()

    @staticmethod
    def reduce_stat(model):
        assert not isinstance(model, nn.parallel.DistributedDataParallel)
        d = link # get_dist_interface()

        def _reduce(tensors, reduce_op):
            for tensors in _take_tensors(tensors, bucket_size):
                flatten_tensors = _flatten_dense_tensors(tensors)
                d.all_reduce(flatten_tensors, reduce_op)
                for tensor, synced in zip(tensors,
                                          _unflatten_dense_tensors(flatten_tensors, tensors)):
                    tensor.copy_(synced)

        MB = 1024 * 1024
        bucket_size = 256 * MB
        all_min = []
        all_max = []

        for m in model.modules():
            if isinstance(m, EMAAct):
                if not m.sync:
                    return
                all_min.append(m.batch_min)
                all_max.append(m.batch_max)

        _reduce(all_min, d.reduce_op.MIN)
        _reduce(all_max, d.reduce_op.MAX)

        for m in model.modules():
            if isinstance(m, EMAAct):
                m._update_running_stat()

    @staticmethod
    def get_async_reduce_hook(name, module):
        raise NotImplementedError(f"linklink doesn't support MIN or MAX all_reduce yet")
        # TODO(Rundong): activate this method once MIN MAX all_reduce available
        # assert isinstance(module, EMAAct)
        #
        # def _hook(*unused):
        #     d = get_dist_interface()
        #     d.all_reduce_async(f"{name}_min", module.batch_min, d.reduce_op.MIN)
        #
        # return _hook



def _get_quant_conv(signal=True):
    if signal == True:
        base = SignalConv2d
    else:
        base = nn.Conv2d
    class NoBnConv2d(base):
        def __init__(self, in_channels, out_channels, kernel_size,
                    stride=1, padding=0, dilation=1, groups=1, bias=True,
                    weight_bit=8, grad_bit=32,
                    quant_mode="symmetric", tail="preserve", inq_mode="disable",
                    channel_wise=True, enable_quant=False, enable_quant_grad=False,
                    write_back_quant_weight=False, padding_mode="zeros"):
            assert quant_mode in ("biased", "symmetric", "dorefa")
            assert tail in ("clip", "preserve", "log")
            assert inq_mode in ("disable", "magnitude")
            super(NoBnConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                            stride, padding, dilation, groups, bias)
            nn.init.xavier_normal_(self.weight.data)
            self.weight_bit = weight_bit
            self.grad_bit = grad_bit
            self.enable_quant = enable_quant
            self.enable_quant_grad = enable_quant_grad
            self.quant_mode = quant_mode
            self.inq_mode = inq_mode
            self.channel_wise = channel_wise
            self.write_back_quant_weight = write_back_quant_weight
            self.scale_weight = False
            self.quantize_done = False

            self.tail = tail
            self.conditional_init()
            self.to_caffe2q = False

        def conditional_init(self):
            if self.inq_mode == "disable":
                # self.tail = tail
                pass
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
            elif self.quant_mode == "biased":
                self.weight_func = BiasedQuantFunc.apply
            elif self.quant_mode == "dorefa":
                self.weight_func = SymmetricQuantFunc.apply
                self.scale_weight = True

        def __repr__(self):
            s = super(NoBnConv2d, self).__repr__()
            s += f",\n\tweight_bit={self.weight_bit}, grad_bit={self.grad_bit}, " \
                f"channel_wise={self.channel_wise}, inq_mode={self.inq_mode},\n\t" \
                f"enable_quant={self.enable_quant}, enable_quant_grad={self.enable_quant_grad})"
            return s

        def _get_weight(self, out_weight=None):
            if out_weight is not None:
                weight = out_weight
            else:
                weight = self.weight
            if not self.enable_quant or \
                    (self.write_back_quant_weight and not self.training and self.quantize_done):
                return weight
            else:
                if self.scale_weight:
                    w = torch.tanh(self.weight)
                    # w_magnitude = w.abs().max()
                    # w = w / w_magnitude
                    inplace = False  # TODO: WTF?
                else:
                    w = weight
                    inplace = False
                if self.enable_quant:
                    if self.inq_mode == "magnitude":
                        if self.toggle_inq_mask:
                            self.quant_mask = magnitude_inq_mask(w, self.quant_portion)
                            self.toggle_inq_mask = False
                    if self.quant_mode in ("symmetric", "dorefa"):
                        w = self.weight_func(w, self.weight_bit, self.grad_bit,
                                            self.tail, self.quant_mask,
                                            None, self.channel_wise, self.enable_quant_grad,
                                            inplace)
                    elif self.quant_mode == "biased":
                        w = self.weight_func(w, self.weight_bit, self.grad_bit,
                                            self.tail, self.quant_mask,
                                            None, None, self.channel_wise, self.enable_quant_grad,
                                            inplace)
                if self.write_back_quant_weight and not self.training:
                    self.weight.data = w.data
                    self.quantize_done = True
                return w

        def prepare2q(self):
            self.w2q = self.weight # self._get_weight()

        def forward(self, x, out_weight=None, out_bias=None):
            if not self.to_caffe2q:
                self.w2q = self._get_weight(out_weight)
                if out_bias is not None:
                    self.bias2q = out_bias
                else:
                    self.bias2q = self.bias

                return F.conv2d(x, self.w2q, self.bias2q, self.stride,
                                self.padding, dilation=self.dilation, groups=self.groups)
            else:
                if out_weight is not None and out_bias is not None:
                    return F.conv2d(x, out_weight, out_bias, self.stride,
                                self.padding, dilation=self.dilation, groups=self.groups)
                else:
                    return F.conv2d(x, self.w2q, self.bias, self.stride,
                                self.padding, dilation=self.dilation, groups=self.groups)

        def dump(self):
            use_bias = self.bias is not None
            dump_model = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size,
                                self.stride, self.padding, self.dilation, self.groups,
                                use_bias)
            w = self._get_weight()
            dump_model.weight.data = w.data
            if use_bias:
                dump_model.bias.data = self.bias.data

            return dump_model
        
    return NoBnConv2d


M_NoBnSignalConv2d = _get_quant_conv(signal=True)
M_NoBnConv2d = _get_quant_conv(signal=False)
def get_quant_conv(signal=True):
    if signal == True:
        return M_NoBnSignalConv2d
    else:
        return M_NoBnConv2d


class NoBnConvTranspose2d(SignalConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 weight_bit=8, grad_bit=32,
                 quant_mode="symmetric", tail="preserve", inq_mode="disable",
                 channel_wise=True, enable_quant=False, enable_quant_grad=False,
                 write_back_quant_weight=False, padding_mode="zeros", use_Int=False):
        assert quant_mode in ("biased", "symmetric", "dorefa")
        assert tail in ("clip", "preserve", "log")
        assert inq_mode in ("disable", "magnitude")
        super(NoBnConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size,
                                         stride, padding, dilation, groups, bias)
        nn.init.xavier_normal_(self.weight.data)
        self.weight_bit = weight_bit
        self.grad_bit = grad_bit
        self.enable_quant = enable_quant
        self.enable_quant_grad = enable_quant_grad
        self.quant_mode = quant_mode
        self.inq_mode = inq_mode
        self.channel_wise = channel_wise
        self.write_back_quant_weight = write_back_quant_weight
        self.scale_weight = False
        self.quantize_done = False

        self.tail = tail
        self.conditional_init()
        self.to_caffe2q = False

    def conditional_init(self):
        '''
        can be called by scheduler to modify online
        '''
        if self.inq_mode == "disable":
            # self.tail = tail
            pass
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
        elif self.quant_mode == "biased":
            self.weight_func = BiasedQuantFunc.apply
        elif self.quant_mode == "dorefa":
            self.weight_func = SymmetricQuantFunc.apply
            self.scale_weight = True

    def __repr__(self):
        s = super(NoBnConvTranspose2d, self).__repr__()
        s += f",\n\tweight_bit={self.weight_bit}, grad_bit={self.grad_bit}, " \
             f"channel_wise={self.channel_wise}, inq_mode={self.inq_mode},\n\t" \
             f"enable_quant={self.enable_quant}, enable_quant_grad={self.enable_quant_grad})"
        return s

    def _get_weight(self):
        if not self.enable_quant or \
                (self.write_back_quant_weight and not self.training and self.quantize_done):
            return self.weight
        else:
            if self.scale_weight:
                w = torch.tanh(self.weight)
                # w_magnitude = w.abs().max()
                # w = w / w_magnitude
                inplace = False  # TODO: WTF?
            else:
                w = self.weight
                inplace = False
            if self.enable_quant:
                if self.inq_mode == "magnitude":
                    if self.toggle_inq_mask:
                        self.quant_mask = magnitude_inq_mask(w, self.quant_portion)
                        self.toggle_inq_mask = False
                if self.quant_mode in ("symmetric", "dorefa"):
                    w = self.weight_func(w, self.weight_bit, self.grad_bit,
                                         self.tail, self.quant_mask,
                                         None, self.channel_wise, self.enable_quant_grad,
                                         inplace)
                elif self.quant_mode == "biased":
                    w = self.weight_func(w, self.weight_bit, self.grad_bit,
                                         self.tail, self.quant_mask,
                                         None, None, self.channel_wise, self.enable_quant_grad,
                                         inplace)
            if self.write_back_quant_weight and not self.training:
                self.weight.data = w.data
                self.quantize_done = True
            return w

    def prepare2q(self):
        self.w2q = self.weight # self._get_weight()

    def forward(self, x):
        if not self.to_caffe2q:
            self.w2q = self._get_weight()
        return F.conv_transpose2d(x, self.w2q, self.bias, self.stride,
                        self.padding, dilation=self.dilation, groups=self.groups)

    def dump(self):
        use_bias = self.bias is not None
        dump_model = nn.ConvTranspose2d(self.in_channels, self.out_channels, self.kernel_size,
                               self.stride, self.padding, dilation=self.dilation, groups=self.groups,
                               bias=use_bias)
        w = self._get_weight()
        dump_model.weight.data = w.data
        if use_bias:
            dump_model.bias.data = self.bias.data

        return dump_model


class EMABnConv2d(Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 eps=1e-5, momentum=0.9, affine=True, sync=True,
                 weight_bit=8, grad_bit=32,
                 quant_mode="biased", tail="preserve",
                 channel_wise=True, inq_mode="disable",
                 enable_quant=False, enable_quant_grad=False,
                 write_back_quant_weight=False):
        assert quant_mode in ("biased", "symmetric", "dorefa")
        assert tail in ("clip", "preserve", "log")
        assert inq_mode in ("disable", "magnitude", "bit-plane")
        super(EMABnConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)
        self.weight_bit = weight_bit
        self.grad_bit = grad_bit
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.sync = sync
        self.quant_mode = quant_mode
        self.channel_wise = channel_wise
        self.enable_quant = enable_quant
        self.enable_quant_grad = enable_quant_grad
        self.inq_mode = inq_mode
        self.write_back_quant_weight = write_back_quant_weight
        self.scale_weight = False
        self.quantize_done = False
        self.freeze_bn = False

        if self.sync:
            d = link # get_dist_interface()
            try:
                world_size = d.get_world_size()
            except AssertionError:
                world_size = 1
            self.sync &= world_size > 1
            self.world_size = world_size

        if inq_mode == "disable":
            self.tail = tail
        else:
            # INQ is not compatible with ordinary gradient handling schemes
            self.tail = "inq"
        if inq_mode == "disable":
            self.register_buffer("quant_mask", None)
        else:
            self.toggle_inq_mask = True  # flag set by Scheduler
            self.quant_portion = 0.
            self.register_buffer("quant_mask",
                                 torch.zeros_like(self.weight, dtype=torch.uint8))

        if quant_mode == "symmetric":
            self.weight_func = SymmetricQuantFunc.apply
        elif quant_mode == "biased":
            self.weight_func = BiasedQuantFunc.apply
        elif quant_mode == "dorefa":
            self.weight_func = SymmetricQuantFunc.apply
            self.scale_weight = True

        self.register_buffer("running_mean", torch.Tensor(out_channels))
        self.register_buffer("running_var", torch.Tensor(out_channels))
        if self.affine:
            self.alpha = Parameter(torch.Tensor(out_channels))
            self.beta = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("alpha", None)
            self.register_parameter("beta", None)

        # for diagnosis
        self.register_buffer("batch_mean", torch.Tensor(out_channels))
        self.register_buffer("batch_var", torch.Tensor(out_channels))
        self.diagnosis = False

        self._reset_parameters()

    def __repr__(self):
        conv_s = super(EMABnConv2d, self).__repr__()
        bn_s = f"BN:(eps={self.eps}, affine={self.affine}, sync={self.sync}, " \
               f"freeze={self.freeze_bn})"
        s = f"{self.__class__.__name__}:(Conv:{conv_s}\n\t{bn_s}\n" \
            f"\tweight_bit={self.weight_bit}, grad_bit={self.grad_bit}, " \
            f"enable_quant={self.enable_quant}, enable_quant_grad={self.enable_quant_grad}, " \
            f"channel_wise={self.channel_wise})"

        return s

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.weight.data)
        self.running_mean.zero_()
        self.running_var.fill_(1.)
        if self.affine:
            self.alpha.data.fill_(1.)
            self.beta.data.zero_()

    def _quantize_weight(self, w):
        assert self.enable_quant
        inplace = False  # TODO: WTF?
        if self.inq_mode == "magnitude":
            if self.toggle_inq_mask:
                self.quant_mask = magnitude_inq_mask(w, self.quant_portion)
                self.toggle_inq_mask = False

        if self.quant_mode == "symmetric":
            w = self.weight_func(w, self.weight_bit, self.grad_bit,
                                 self.tail, self.quant_mask,
                                 None, self.channel_wise, self.enable_quant_grad,
                                 inplace)
        elif self.quant_mode == "dorefa":
            w = torch.tanh(w)
            # w_magnitude = w.abs().max()
            # w = w / w_magnitude
            w = self.weight_func(w, self.weight_bit, self.grad_bit,
                                 self.tail, self.quant_mask,
                                 None, self.channel_wise, self.enable_quant_grad,
                                 inplace)
        elif self.quant_mode == "biased":
            w = self.weight_func(w, self.weight_bit, self.grad_bit,
                                 self.tail, self.quant_mask,
                                 None, None, self.channel_wise, self.enable_quant_grad,
                                 inplace)
        return w

    def _fold_bn(self, w, mean, safe_std):
        w_view = (self.out_channels, 1, 1, 1)
        if self.affine:
            weight = w * (self.alpha / safe_std).view(w_view)
            beta = self.beta - self.alpha * mean / safe_std
            if self.bias is not None:
                bias = self.alpha * self.bias / safe_std + beta
            else:
                bias = beta
        else:
            weight = w / safe_std.view(w_view)
            beta = -mean / safe_std
            if self.bias is not None:
                bias = self.bias / safe_std + beta
            else:
                bias = beta
        return weight, bias

    def _stable_fold_bn(self, w, batch_mean, batch_std, ema_mean, ema_std):
        w_view = (self.out_channels, 1, 1, 1)
        if self.freeze_bn:
            mean, std = ema_mean, ema_std
        else:
            mean, std = batch_mean, batch_std
        alpha, beta = self.alpha, self.beta
        if self.affine:
            weight = w * (alpha / ema_std).view(w_view)
            bias = beta - alpha * mean / std
            if self.bias is not None:
                bias = alpha * self.bias / std + bias
        else:
            weight = w / ema_std.view(w_view)
            bias = -mean / std
            if self.bias is not None:
                bias = self.bias / std + bias
        return weight, bias

    def forward(self, x):
        # TODO(Rundong): make freeze_bn more elegant
        weight = self.weight
        if self.enable_quant:
            if self.training:
                if self.freeze_bn:
                    mean = self.running_mean
                    std = torch.sqrt(self.running_var + self.eps)
                    weight, bias = self._fold_bn(weight, mean, std)
                    weight = self._quantize_weight(weight)
                    return F.conv2d(x, weight, bias,
                                    self.stride, self.padding, self.dilation, self.groups)
                else:
                    y = F.conv2d(x, weight, self.bias,
                                 self.stride, self.padding, self.dilation, self.groups)
                    if self.sync:
                        stat_func = SyncStatFunc.apply
                        batch_mean, batch_var = stat_func(y, self.world_size)
                    else:
                        y_data = y.transpose(0, 1).reshape(self.out_channels, -1)
                        batch_mean = y_data.mean(1)
                        batch_var = y_data.var(1)
                    with torch.no_grad():
                        self.running_mean.mul_(self.momentum).add_(1. - self.momentum, batch_mean)
                        self.running_var.mul_(self.momentum).add_(1. - self.momentum, batch_var)
                        self.batch_mean.copy_(batch_mean)
                        self.batch_var.copy_(batch_var)
                    batch_std = torch.sqrt(batch_var + self.eps)
                    weight, bias = self._fold_bn(weight, batch_mean, batch_std)
                    weight = self._quantize_weight(weight)
                    return F.conv2d(x, weight, bias,
                                    self.stride, self.padding, self.dilation, self.groups)
            else:
                if self.write_back_quant_weight and self.quantize_done:
                    weight = self.weight
                    bias = self.bias
                else:
                    ema_mean = self.running_mean
                    ema_std = torch.sqrt(self.running_var + self.eps)
                    weight, bias = self._fold_bn(weight, ema_mean, ema_std)
                    weight = self._quantize_weight(weight)
                    if self.write_back_quant_weight:
                        self.weight.data = weight.data
                        if self.bias is None:
                            self.bias = Parameter(bias.data)
                        else:
                            self.bias.data = bias.data
                        self.quantize_done = True
                return F.conv2d(x, weight, bias, self.stride,
                                self.padding, self.dilation, self.groups)
        else:
            if self.freeze_bn or not self.training:
                use_batch_stat = False
            else:
                use_batch_stat = True

            conv = F.conv2d(x, weight, self.bias,
                            self.stride, self.padding, self.dilation, self.groups)
            bn = F.batch_norm(conv, self.running_mean, self.running_var,
                              self.alpha, self.beta, use_batch_stat,
                              1. - self.momentum, self.eps)
            return bn

    def dump(self):
        w = self._get_weight()
        running_mean = self.running_mean
        safe_std = torch.sqrt(self.running_var + self.eps)
        weight, bias = self._fold_bn(w, running_mean, safe_std)
        dump_model = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size,
                               self.stride, self.padding, self.dilation, self.groups,
                               True)
        if self.enable_quant:
            weight = self.weight_func(weight, self.weight_bit)
        dump_model.weight.data = weight.data
        dump_model.bias.data = bias.data

        return dump_model


class QuantLinear(Linear):
    def __init__(self, in_features, out_features, bias=True,
                 enable_quant=False, enable_quant_grad=False,
                 weight_bit=8, grad_bit=32,
                 quant_mode="biased", tail="preserve",
                 channel_wise=True, inq_mode="disable",
                 write_back_quant_weight=False):

        assert quant_mode in ("biased", "symmetric", "dorefa")
        assert tail in ("clip", "preserve", "log")
        assert inq_mode in ("disable", "magnitude", "bit-plane")
        super(QuantLinear, self).__init__(in_features, out_features, bias)

        self.weight_bit = weight_bit
        self.grad_bit = grad_bit
        self.quant_mode = quant_mode
        self.inq_mode = inq_mode
        self.enable_quant = enable_quant
        self.enable_quant_grad = enable_quant_grad
        self.channel_wise = channel_wise
        self.write_back_quant_weight = write_back_quant_weight
        self.quantize_done = False

        self.scale_weight = False
        if quant_mode == "symmetric":
            self.weight_func = SymmetricQuantFunc.apply
        elif quant_mode == "biased":
            self.weight_func = BiasedQuantFunc.apply
        elif quant_mode == "dorefa":
            self.weight_func = SymmetricQuantFunc.apply
            self.scale_weight = True

        if inq_mode == "disable":
            self.tail = tail
        else:
            # INQ is not compatible with ordinary gradient handling schemes
            self.tail = "inq"
        if inq_mode == "disable":
            self.register_buffer("quant_mask", None)
        else:
            self.toggle_inq_mask = True  # flag set by Scheduler
            self.quant_portion = 0.
            self.register_buffer("quant_mask", torch.zeros_like(self.weight, dtype=torch.uint8))

    def __repr__(self):
        s = super(QuantLinear, self).__repr__()
        s += f"\n\tweight_bit={self.weight_bit}, grad_bit={self.grad_bit}, " \
             f"channel_wise={self.channel_wise},"
        s += f"\n\tenable_quant={self.enable_quant}, enable_quant_grad={self.enable_quant_grad}"
        return s

    def _get_weight(self):
        if not self.enable_quant or \
                (self.write_back_quant_weight and not self.training and self.quantize_done):
            return self.weight

        else:
            if self.scale_weight:
                w = torch.tanh(self.weight)
                w_magnitude = w.abs().max()
                w = w / w_magnitude
                inplace = True
            else:
                w = self.weight
                inplace = False
            if self.enable_quant:
                if self.inq_mode == "magnitude":
                    if self.toggle_inq_mask:
                        self.quant_mask = magnitude_inq_mask(w, self.quant_portion)
                        self.toggle_inq_mask = False
                if self.quant_mode in ("symmetric", "dorefa"):
                    w = self.weight_func(w, self.weight_bit, self.grad_bit,
                                         self.tail, self.quant_mask,
                                         None, self.channel_wise, self.enable_quant_grad,
                                         inplace)
                elif self.quant_mode == "biased":
                    w = self.weight_func(w, self.weight_bit, self.grad_bit,
                                         self.tail, self.quant_mask,
                                         None, None, self.channel_wise, self.enable_quant_grad,
                                         inplace)
            if self.write_back_quant_weight and not self.training:
                self.weight.data = w.data
                self.quantize_done = True
            return w

    def forward(self, x):
        w = self._get_weight()
        return F.linear(x, w, self.bias)
