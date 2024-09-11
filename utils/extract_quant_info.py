import torch
try:
    from integer2 import module as qm
except:
    print("extract_quant load integer2 failed", flush=True)
    from integer import module as qm
from collections import OrderedDict


M_NoBnConv2d = qm.get_quant_conv(False)
M_NoBnSignalConv2d = qm.get_quant_conv(True)
M_NoBnConvTranspose2d = qm.NoBnConvTranspose2d
M_EMAAct = qm.EMAAct

quant_type = (M_NoBnConv2d, M_NoBnSignalConv2d, M_NoBnConvTranspose2d)
stat_type = M_EMAAct

def tensor_to_list(lb, ub):
    c = lb.size(0)
    res = []
    for i in range(c):
        res.append((lb[i].item(), ub[i].item()))
    return res

def extract_quant_info(model):
    '''extract quant params, in the same order as in onnx (caffe),
       
       we expect:
        final_quant_params:
            [
                {
                    'name': xxx.
                    'output_encoding': [{'min':x, 'max':x}], 
                    'weight_encoding': [{'min':x, 'max':x}]
                }
            ]
       However, there may exist caffe layer which is not from a nn.Module object, like abs, cat, element wise, and sometimes
       we need to merge the output_encoding from the following EMAAct layer into those layers. Two solutions:
       1. change all the layers in torch, all use nn.Module (not sure
       2. modify quant_info_to_proto, where we ignore this kind of layer or merge the output_encoding into them on demand
       We use 2.

       The output is two sequence of quant info only for layers with qparams, including 1st_level_layer_name, qparams, layer_type in
       the same order as in onnx (caffe).
       Value[0] in layer_names is the key in quant_params. 
       The layer_type sequence is (act), weight, ..., act.

       Mapping between caffe layers and qparams
       for type Convolution, Deconvolution: (act), weight, act
       for type Eltwise with operation PROD: act; this is caused by 1dn
       for type Abs: None
       others such as relu, cat, not met yet
    '''
    quant_params = OrderedDict()
    layer_names = OrderedDict()
    gather_handles = []
    forward_layer_names = OrderedDict()

    def gather_act_range_hook(m, x, y):
        """Gather activation ranges from EMAAct, in forward sequence"""
        assert isinstance(m, stat_type)

        forward_layer_names[id(m)] = layer_names[id(m)]

        if m.channel_wise == False:
            lb = m.stat_min.item()
            ub = m.stat_max.item()
            if m.quant_mode == 'symmetric':
                magnitude = max(abs(lb), abs(ub))
                quant_params[layer_names[id(m)][0]] = [(-magnitude, magnitude)]
            elif m.quant_mode == 'biased':
                quant_params[layer_names[id(m)][0]] = [(lb, ub)]
            else:
                raise NotImplemented('unsupported quant_mode {}'.format(m.quant_mode))
        else:
            lb = m.stat_min
            ub = m.stat_max
            if m.quant_mode == 'symmetric':
                magnitude = torch.max(lb.abs(), ub.abs())
                quant_params[layer_names[id(m)][0]] = tensor_to_list(-magnitude, magnitude)
            elif m.quant_mode == 'biased':
                quant_params[layer_names[id(m)][0]] = tensor_to_list(lb, ub)
            else:
                raise NotImplemented('unsupported quant_mode {}'.format(m.quant_mode))

    def gather_weight_range_hook(m, x, y):
        assert isinstance(m, quant_type)

        forward_layer_names[id(m)] = layer_names[id(m)]

        if m.channel_wise == False:
            if m.quant_mode == 'symmetric':
                magnitude = m.w2q.abs().max().item()
                quant_params[layer_names[id(m)][0]] = [(-magnitude, magnitude)]
            elif self.quant_mode == 'biased':
                lb = m.w2q.min().item()
                ub = m.w2q.max().item()
                quant_params[layer_names[id(m)][0]] = [(lb, ub)]
            else:
                raise NotImplemented('unsupported quant_mode {}'.format(m.quant_mode))
        else:
            if m.quant_mode == 'symmetric':
                c = m.w2q.size(0)
                magnitude, _ = m.w2q.contiguous().view(c, -1).abs().max(1)
                quant_params[layer_names[id(m)][0]] = tensor_to_list(-magnitude, magnitude)
            elif self.quant_mode == 'biased':
                c = m.w2q.size(0)  # only works for weights: (C_out, *)
                x_channel_view = m.w2q.contiguous().view(c, -1)
                lb, _ = x_channel_view.min(1)
                ub, _ = x_channel_view.max(1)
                quant_params[layer_names[id(m)][0]] = tensor_to_list(lb, ub)
            else:
                raise NotImplemented('unsupported quant_mode {}'.format(m.quant_mode))

    def gather_other_hook(m, x, y):
        layer_names[id(m)] = [n, m]

    for n, m in model.named_modules():
        if isinstance(m, stat_type):
            layer_names[id(m)] = [n, 'act']
            h = m.register_forward_hook(gather_act_range_hook)
            gather_handles.append(h)
        elif isinstance(m, quant_type):
            layer_names[id(m)] = [n, 'weight']
            h = m.register_forward_hook(gather_weight_range_hook)
            gather_handles.append(h)
        else:
            pass
            # h = m.register_forward_hook(gather_other_hook)
            # gather_handles.append(h)

    with torch.no_grad():
        dummy_x = torch.randn(1, model.caffe_channels, 256, 256)
        origin_y = model(dummy_x)

    for h in gather_handles:
        h.remove()

    print('forward_layer_names is {}'.format(forward_layer_names), flush=True)
    print('quant_params_keys is {}'.format(quant_params.keys()), flush=True)
    return quant_params, forward_layer_names
