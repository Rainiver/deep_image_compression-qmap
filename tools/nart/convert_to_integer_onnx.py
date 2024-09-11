import argparse, os
import numpy as np
from onnx.checker import check_model
from onnx import TensorProto
from onnx import load, save, numpy_helper, helper
parser = argparse.ArgumentParser()
parser.add_argument('-path', required=True, help='path of onnx')
args = parser.parse_args()

'''
change z_decoder.onnx(float32) to z_decoder.onnx(int32)
align with "INTEGER NETWORKS FOR DATA COMPRESSION WITH LATENT-VARIABLE MODELS"(gg19i)
'''
if __name__ == "__main__":
    path = args.path
    name = os.path.basename(path).split('.')[0]
    assert 'z_decoder' in name, 'only support change z_decoder.onnx'

    onnx_model = load(path)
    graph = onnx_model.graph
    nodes, inputs, outputs, initials = graph.node, graph.input, graph.output, graph.initializer

    #reverse to delete node
    for i in range(len(nodes) - 1, -1, -1):
        #cast constant
        if nodes[i].op_type == 'Constant':
            tensor = nodes[i].attribute[0].t
            data = numpy_helper.to_array(tensor)
            tensor = numpy_helper.from_array(data.astype(np.int32))
            attr = helper.make_attribute('value', tensor)
            nodes[i].attribute.remove(nodes[i].attribute[0])
            nodes[i].attribute.insert(0, attr)

        #remove Floor because integer Div can guarantee
        if nodes[i].op_type == 'Floor':
            nodes[i-1].output.remove(nodes[i-1].output[0])
            nodes[i-1].output.insert(0, nodes[i].output[0])
            nodes.remove(nodes[i])

        #remove Constant Cast op, because it do nothing
        if nodes[i].op_type == 'Cast':
            #find previous node
            for previous in nodes:
                if previous.output == nodes[i].input:
                    break
            previous.output.remove(previous.output[0])
            previous.output.insert(0, nodes[i].output[0])
            nodes.remove(nodes[i])

    #add op
    for i in range(len(nodes) - 1, 0, -1):
        #append Cast after Clip op
        if nodes[i].op_type == 'Clip':
            #find next node
            for next in nodes:
                next_in = next.input[0] if len(next.input) else next.input
                if next_in == nodes[i].output[0]:
                    break
            for out in outputs:
                if out.name == nodes[i].output[0]:
                    next_in = out.name
                    break
            cast_node = helper.make_node('Cast', inputs=['cast_%d_in' % i], outputs=[next_in])
            attr = helper.make_attribute("to", TensorProto.UINT8)
            cast_node.attribute.insert(0, attr)
            nodes[i].output.remove(nodes[i].output[0])
            nodes[i].output.insert(0, cast_node.input[0])
            nodes.insert(i + 1, cast_node)


    #cast weight and bias
    n = len(initials)
    for i in range(n):
        name = initials[i].name
        if 'deconv' in name:
            if 'w' in name:
                dtype = np.int8
            if 'b' in name:
                dtype = np.int32
            data = numpy_helper.to_array(initials[i])
            tensor = numpy_helper.from_array(data.astype(dtype))
            tensor.name = name
            initials.remove(initials[i])
            initials.insert(i, tensor)

    #cast input and output
    for input in inputs:
        input.type.tensor_type.elem_type = TensorProto.INT32
    for output in outputs:
        output.type.tensor_type.elem_type = TensorProto.UINT8

    check_model(onnx_model)
    save(onnx_model, path.replace('.onnx', '_int.onnx'))
    print('converting z_decoder_int')





