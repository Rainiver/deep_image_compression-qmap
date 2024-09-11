import torch
import torch.nn as nn
from collections import Iterable
from nets.norm import GDN, Power
from nets.layers import RdftParameterizer, SignalConv2d, SignalConvTranspose2d
from torch.nn.modules.conv import _ConvNd

from nets.layers_quant import ProAct, GGBpAct

try:
    from integer2 import module as qm
except:
    print("flops_helper load integer2 failed", flush=True)
    from integer import module as qm
M_NoBnConv2d = qm.get_quant_conv(False)
M_NoBnSignalConv2d = qm.get_quant_conv(True)
M_NoBnConvTranspose2d = qm.NoBnConvTranspose2d
M_EMAAct = qm.EMAAct


debug = False

def to_device(input, device="cuda", dtype=None):
    """Transfer data between devidces"""

    if 'image' in input:
        input['image'] = input['image'].to(dtype=dtype)

    def transfer(x):
        if torch.is_tensor(x):
            return x.to(device=device)
        elif isinstance(x, list) and torch.is_tensor(x[0]):
            return [_.to(device=device) for _ in x]
        return x
    return {k: transfer(v) for k, v in input.items()}


def clever_format(nums, format="%.2f"):
    if not isinstance(nums, Iterable):
        nums = [nums]
    clever_nums = []

    for num in nums:
        num = int(num)
        if num > 1e12:
            clever_nums.append(format % (num / 1e12) + "T")
        elif num > 1e9:
            clever_nums.append(format % (num / 1e9) + "G")
        elif num > 1e6:
            clever_nums.append(format % (num / 1e6) + "M")
        elif num > 1e3:
            clever_nums.append(format % (num / 1e3) + "K")
        else:
            clever_nums.append(format % num + "B")

    clever_nums = clever_nums[0] if len(clever_nums) == 1 else (*clever_nums,)

    return clever_nums


def flops_cal(model, input_shape):
    inputs = {
        'image': torch.randn(1, input_shape[0], input_shape[1], input_shape[2])
    }
    flops, params = profile(model, inputs=to_device(inputs, 'cpu'))
    flops_str, params_str = clever_format([flops, params], "%.3f")
    flops = flops / 1e6
    params = flops / 1e6
    return flops, params, flops_str, params_str


def profile(model, inputs, verbose=True):
    handler_collection = []

    def add_hooks(m):
        if len(list(m.children())) > 0:
            return

        m.register_buffer('total_ops', torch.zeros(1))
        m.register_buffer('total_params', torch.zeros(1))

        m_type = type(m)
        fn = None
        if m_type in register_hooks:
            fn = register_hooks[m_type]

        if fn is None:
            if verbose:
                print("No implemented counting method for ", m)
        else:
            handler = m.register_forward_hook(fn)
            handler_collection.append(handler)

    # original_device = model.parameters().__next__().device
    training = model.training

    model.eval()
    model.apply(add_hooks)

    with torch.no_grad():
        model(inputs['image'])

    total_ops = 0
    total_params = 0
    for n, m in model.named_modules():
        if debug:
            print('named_modules: {}'.format(n), flush=True)
            print(list(m.children()), flush=True)
        if len(list(m.children())) > 0:  # skip for non-leaf module
            continue
        m_total_ops = m.total_ops
        m_total_params = m.total_params
        print('module {} flops {} params {}'.format(m, m_total_ops, m_total_params), flush=True)
        total_ops += m_total_ops
        total_params += m_total_params

    total_ops = total_ops.item()
    total_params = total_params.item()

    # reset model to original status
    model.train(training)
    for handler in handler_collection:
        handler.remove()

    return total_ops, total_params


multiply_adds = 1


def count_zero(m, x, y):
    m.total_ops = torch.Tensor([0])
    m.total_params = torch.Tensor([0])


def count_error(m, x, y):
    m.total_ops = torch.Tensor([0])
    m.total_params = torch.Tensor([0])
    assert 0, 'should rm this module in converting mode'


def count_convNd(m: _ConvNd, x: (torch.Tensor,), y: torch.Tensor):
    x = x[0]

    kernel_ops = torch.zeros(m.weight.size()[2:]).numel()  # Kw x Kh
    bias_ops = 1 if m.bias is not None else 0

    # N x Cout x H x W x  (Cin x Kw x Kh + bias)
    total_ops = y.nelement() * (m.in_channels // m.groups * kernel_ops + bias_ops)

    m.total_ops += torch.Tensor([int(total_ops)])


def count_conv2d(m, x, y):
    cin = m.in_channels
    cout = m.out_channels
    kh, kw = m.kernel_size
    out_h = y.size(2)
    out_w = y.size(3)

    kernel_ops = multiply_adds * kh * kw
    bias_ops = 1 if m.bias is not None else 0

    output_elements = out_w * out_h * cout
    total_ops = output_elements * kernel_ops * cin // m.groups + bias_ops * output_elements
    m.total_ops = torch.Tensor([int(total_ops)])

    total_params = kh * kw * cin * cout // m.groups + bias_ops * cout
    m.total_params = torch.Tensor([int(total_params)])


def count_bn(m, x, y):
    x = x[0]
    c_out = y.size(1)
    nelements = x.numel()
    # subtract, divide, gamma, beta
    total_ops = 4 * nelements

    m.total_ops = torch.Tensor([int(total_ops)])
    m.total_params = torch.Tensor([int(c_out) * 2])


def count_relu(m, x, y):
    x = x[0]
    nelements = x.numel()
    total_ops = nelements

    m.total_ops = torch.Tensor([int(total_ops)])


def count_softmax(m, x, y):
    x = x[0]
    batch_size, nfeatures = x.size()
    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)

    m.total_ops = torch.Tensor([int(total_ops)])


def count_avgpool(m, x, y):
    total_add = torch.prod(torch.Tensor([m.kernel_size]))
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops = torch.Tensor([int(total_ops)])


def count_adap_avgpool(m, x, y):
    kernel = torch.Tensor([*(x[0].shape[2:])]) // torch.Tensor(list((m.output_size,))).squeeze()
    total_add = torch.prod(kernel)
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops = torch.Tensor([int(total_ops)])


def count_linear(m, x, y):
    # per output element
    total_mul = m.in_features
    total_add = m.in_features - 1
    num_elements = y.numel()
    total_ops = (total_mul + total_add) * num_elements

    m.total_ops = torch.Tensor([int(total_ops)])
    m.total_params = torch.Tensor([m.in_features * m.out_features])


'''
more hooks:
https://github.com/Lyken17/pytorch-OpCounter/blob/master/thop/count_hooks.py
'''
register_hooks = {
    nn.Conv2d: count_conv2d,
    nn.BatchNorm2d: count_zero,
    nn.ReLU: count_zero,
    nn.ReLU6: count_zero,
    nn.LeakyReLU: count_zero,
    nn.AvgPool2d: count_zero,
    nn.AdaptiveAvgPool2d: count_zero,
    nn.Linear: count_linear,
    nn.Dropout: count_zero,
    nn.MaxPool2d: count_zero,
    nn.CrossEntropyLoss: count_zero,
    Power: count_zero,
    GDN: count_zero,
    RdftParameterizer: count_error,
    nn.ConvTranspose2d: count_conv2d,
    SignalConv2d: count_conv2d,
    SignalConvTranspose2d: count_conv2d,
    nn.ConvTranspose2d: count_conv2d,
    nn.Upsample: count_zero,
    M_NoBnConv2d: count_conv2d,
    M_NoBnSignalConv2d: count_conv2d,
    M_NoBnConvTranspose2d: count_conv2d,
    M_EMAAct: count_zero,
    GGBpAct: count_zero,
    ProAct: count_zero
}
