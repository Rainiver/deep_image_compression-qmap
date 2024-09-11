# -*- coding: utf-8 -*-

from google.protobuf import text_format
from utils import ppl_caffe_pb2


def param_to_caffe_node(param, type_str):
    quantize_param = ppl_caffe_pb2.QuantizeParameter()
    quantize_param.type = type_str
    quantize_param.range_min = param[0] #['min']
    quantize_param.range_max = param[1] #['max']
    #quantize_param.step = param['scale']
    #quantize_param.zero_point = param['zero_point']
    return quantize_param


def quant_info_to_proto(quant_info, layer_names, src_proto, dst_proto):
    with open(src_proto) as f:
        net = ppl_caffe_pb2.NetParameter()
        text_format.Merge(f.read(), net)
    
    layer_names_keys = list(layer_names)
    def _info(j):
        name_layer_type = layer_names[layer_names_keys[j]]
        name = name_layer_type[0]
        layer_type = name_layer_type[1]
        param = quant_info[name]
        return layer_type, param

    j = 0
    for i, layer in enumerate(net.layer):
        if layer.type in ['Convolution', 'Deconvolution']:
            layer_type, param = _info(j)
            if i==0 and j==0:
                if layer_type == 'act':
                    c = len(param)
                    for k in range(c):
                        qparam = layer.quantize_param.add()
                        qparam.CopyFrom(param_to_caffe_node(param[k], 'bottom'))
                    j += 1
                    layer_type, param = _info(j)

            assert layer_type == 'weight'
            c = len(param)
            for k in range(c):
                caffe_node = param_to_caffe_node(param[k], 'filter')
                layer.convolution_param.quantize_param.CopyFrom(caffe_node)
            j += 1
            layer_type, param = _info(j)

            assert layer_type == 'act'
            c = len(param)
            for k in range(c):
                qparam = layer.quantize_param.add()
                qparam.CopyFrom(param_to_caffe_node(param[k], 'top'))
            j += 1
            continue

        if layer.type in ['Eltwise']:
            layer_type, param = _info(j)
            assert layer_type == 'act'
            c = len(param)
            for k in range(c):
                qparam = layer.quantize_param.add()
                qparam.CopyFrom(param_to_caffe_node(param[k], 'top'))
            j += 1
            continue

        print('no qparams layer.type: {}'.format(layer.type), flush=True)         

    assert j == len(layer_names_keys)

    '''
    no_data_layer = False

    for i, param in enumerate(quant_info):
        layer_idx = i - 1 if no_data_layer else i
        if i == 0:
            net.layer[layer_idx + 1].quantize_param.extend([param_to_caffe_node(param['output_encoding'], 'bottom')])
        else:
            assert unicode(net.layer[layer_idx].name) == unicode(param['name']), \
                "not match at layer {}: quant_param: {} vs. prototxt: {}" \
                .format(i, param['name'], net.layer[layer_idx].name)
            layer = net.layer[layer_idx]
            layer.quantize_param.extend([param_to_caffe_node(param['output_encoding'], 'top')])

            if 'weight_encoding' in param:
                caffe_node = param_to_caffe_node(param['weight_encoding'], 'filter')
                if layer.type == 'InnerProduct':
                    layer.inner_product_param.quantize_param.extend([caffe_node])
                elif layer.type == 'Convolution':
                    layer.convolution_param.quantize_param.extend([caffe_node])
                else:
                    raise ValueError("Invalid layer {} with quant param {}".format(layer, param))
    '''
    with open(dst_proto, 'w') as f:
        f.write(str(net))
