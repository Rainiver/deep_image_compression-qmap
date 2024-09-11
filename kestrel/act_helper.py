import os
import pickle
import torch
import torch.nn as nn
from functools import partial
from nets.layers import SignalConv2d, SignalConvTranspose2d
from nets.layers_quant import ProAct, GGBpAct, GGQReLU, GGQConvTranspose2d, QConvTranspose2d
try:
    from integer2 import module as qm
except:
    print("act_helper load integer2 failed", flush=True)
    from integer import module as qm
M_NoBnConv2d = qm.get_quant_conv(False)
M_NoBnSignalConv2d = qm.get_quant_conv(True)
M_NoBnConvTranspose2d = qm.NoBnConvTranspose2d
M_EMAAct = qm.EMAAct


def extract_act(model, log_dir):
    act_dir = os.path.join(log_dir, 'acts')
    if not os.path.exists(act_dir):
        os.mkdir(act_dir)

    def save_act(m, x, y, name=None):
        x = x[0]

        act_file_name = os.path.join(act_dir, name + '.pk')
        with open(act_file_name, "wb") as f:
            pickle.dump(y.cpu().numpy(), f, protocol=2)
        act_file_name = os.path.join(act_dir, name + '_in.pk')
        with open(act_file_name, "wb") as f:
            pickle.dump(x.cpu().numpy(), f, protocol=2)

    def _add_hooks(m, n):
        m_type = type(m)
        fn = None
        if m_type in register_hooks:
            fn = partial(save_act, name=n)

        if fn is None:
            pass
            # print("Not save act for  ", n)
        else:
            print("Save act for name {} with type {} ".format(n, m_type))
            handler = m.register_forward_hook(fn)

    for n, m in model.named_modules():
        _add_hooks(m, n)


register_hooks = {
    nn.Conv2d: None,
    nn.ConvTranspose2d: None,
    SignalConv2d: None,
    SignalConvTranspose2d: None,
    nn.ConvTranspose2d: None,
    M_NoBnConv2d: None,
    M_NoBnSignalConv2d: None,
    M_NoBnConvTranspose2d: None,
    M_EMAAct: None,
    GGBpAct: None,
    ProAct: None,
    GGQReLU: None,
    GGQConvTranspose2d: None,
    QConvTranspose2d:None,
}
