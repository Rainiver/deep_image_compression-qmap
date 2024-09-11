import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.layers import SignalConv2d, SignalConvTranspose2d
from nets.norm import GDN
from utils.initializer import initialize
import sys
from functools import partial
import math

try:
    import spring.linklink as link
except ImportError as e:
    from mock import Mock

    sys.modules['linklink'] = Mock()
    sys.modules['linklink.nn'] = Mock()
    import spring.linklink as link

try:
    from springnas_lite.base_mixop import MixedOp
except ImportError as e:
    raise e


def init_conv(*args, **kwargs):
    op_name_list = ['signal'] * 3
    op_args_list = [[args, kwargs]] * 3
    op_plus_info_list = []
    for init_type in ['normal', 'msra', 'xavier']:
        op_plus_info_list.append({'init': init_type})
    return op_args_list, op_name_list, op_plus_info_list


def init_gdn(*args, **kwargs):
    op_name_list = ['normal'] * 6
    op_args_list = [[args, kwargs]] * 6
    op_plus_info_list = []
    for init_type in [
        'default', 'kaiming', 'kaiming_uniform', 'xavier', 'xavier_uniform', 'NONE'
    ]:
        op_plus_info_list.append({'init': init_type})
    return op_args_list, op_name_list, op_plus_info_list


def init_conv_rdft(*args, **kwargs):
    op_name_list = ['signal'] * 5
    op_args_list = [[args, kwargs]] * 5
    op_plus_info_list = []
    for init_type in [
        'default', 'kaiming', 'kaiming_uniform', 'xavier', 'xavier_uniform'
    ]:
        op_plus_info_list.append({'init': init_type})
    return op_args_list, op_name_list, op_plus_info_list


def init_conv_origin(*args, **kwargs):
    op_name_list = ['origin'] * 5
    op_args_list = [[args, kwargs]] * 5
    op_plus_info_list = []
    for init_type in [
        'default', 'kaiming', 'kaiming_uniform', 'xavier', 'xavier_uniform'
    ]:
        op_plus_info_list.append({'init': init_type})
    return op_args_list, op_name_list, op_plus_info_list


def init_helper(init):
    if init == 'default':
        # only use for Conv2d.weight
        def torch_default_init(x):
            assert x.dim() == 4
            c_in = x.size(1)
            kernel_size = x.size(2) * x.size(3)
            k = 1. / c_in / kernel_size
            return torch.nn.init.uniform_(x, -math.sqrt(k), math.sqrt(k))

        init_fn = torch_default_init
    elif init == 'kaiming':
        init_fn = torch.nn.init.kaiming_normal_
    elif init == 'kaiming_uniform':
        init_fn = torch.nn.init.kaiming_uniform_
    elif init == 'xavier':
        init_fn = torch.nn.init.xavier_normal_
    elif init == 'xavier_uniform':
        init_fn = torch.nn.init.xavier_uniform_
    elif init == 'NONE':
        init_fn = None
    else:
        raise NotImplementedError

    return init_fn


class MixedOp_OneShot(MixedOp):
    def __init__(self, *args, **kwargs):
        self.op_args_list, self.op_name_list, self.op_plus_info_list = self.op_helper(*args, **kwargs)
        # args use kwargs form
        if MixedOp is not object:
            super(MixedOp_OneShot, self).__init__(self.op_args_list, self.op_name_list, None)
        self._ops = nn.ModuleList()

        # mystery attribute in nas-lite
        self.arch_parameters = nn.Parameter(torch.ones(self.num_ops), requires_grad=True)

    def op_helper(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, x, path_choosen=-1):
        if self.num_ops == 1:
            path = 0
        elif path_choosen >= 0:
            path = path_choosen
        else:
            probs = F.softmax(self.arch_parameters, dim=0)
            path = torch.multinomial(probs, 1)
            if self.training:
                link.broadcast(path, root=0)
            path = path[0].item()

        x = self._ops[path](x)
        return x


class MixedOp_Conv(MixedOp_OneShot):
    def __init__(self, *args, **kwargs):
        super(MixedOp_Conv, self).__init__(*args, **kwargs)
        for op_args, op_name, op_plus_info in zip(self.op_args_list, self.op_name_list, self.op_plus_info_list):
            init_fn = torch.nn.init.kaiming_normal_
            if 'init' in op_plus_info:
                init = op_plus_info['init']
                init_fn = init_helper(init)

            if op_name == 'signal':
                conv = SignalConv2d
                op = conv(init_fn=init_fn, *op_args[0], **op_args[1])
            else:
                raise TypeError(f'Not supported op name {op_name}')

            self._ops.append(op)


class MixedOp_Conv_Signal(MixedOp_Conv):
    def op_helper(self, *args, **kwargs):
        return init_conv_rdft(*args, **kwargs)


class MixedOp_DeConv(MixedOp_OneShot):
    def __init__(self, *args, **kwargs):
        super(MixedOp_DeConv, self).__init__(*args, **kwargs)

        for op_args, op_name, op_plus_info in zip(self.op_args_list, self.op_name_list, self.op_plus_info_list):

            # choose init
            init_fn = torch.nn.init.kaiming_normal_
            if 'init' in op_plus_info:
                init = op_plus_info['init']
                init_fn = init_helper(init)

            # construct and init
            if op_name == 'signal':
                deconv = SignalConvTranspose2d
                op = deconv(init_fn=init_fn, *op_args[0], **op_args[1])
            elif op_name == 'origin':
                deconv = nn.ConvTranspose2d
                op = deconv(*op_args[0], **op_args[1])
                init_fn(op.weight.data)
            else:
                raise TypeError(f'Not supported op name {op_name}')

            self._ops.append(op)


class MixedOp_DeConv_Signal(MixedOp_DeConv):
    def op_helper(self, *args, **kwargs):
        return init_conv_rdft(*args, **kwargs)


class MixedOp_DeConv_Origin(MixedOp_DeConv):
    def op_helper(self, *args, **kwargs):
        return init_conv_origin(*args, **kwargs)


class MixedOp_GDN(MixedOp_OneShot):
    def __init__(self, *args, **kwargs):
        super(MixedOp_GDN, self).__init__(*args, **kwargs)

        for op_args, op_name, op_plus_info in zip(self.op_args_list, self.op_name_list, self.op_plus_info_list):
            if op_name == 'normal':
                gdn = GDN
            else:
                raise TypeError(f'Not supported op name {op_name}')

            init_fn = torch.nn.init.kaiming_normal_
            if 'init' in op_plus_info:
                init = op_plus_info['init']
                init_fn = init_helper(init)
            op = GDN(init_fn=init_fn, *op_args[0], **op_args[1])
            self._ops.append(op)

    def op_helper(self, *args, **kwargs):
        return init_gdn(*args, **kwargs)
