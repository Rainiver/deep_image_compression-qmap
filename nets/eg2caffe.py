import torch
import torch.nn as nn
from torch.nn import functional as F
import spring.nart.tools.pytorch as pytorch
# import nart.python.spring.nart.tools.pytorch as pytorch
import os
import numpy as np


class ConditionalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, \
        padding, bias, padding_mode, conv, lambdas_len):
        super(ConditionalConv2d, self).__init__()
        self.conv = conv(in_channels, out_channels, kernel_size, stride=stride, padding=padding, \
            bias=bias, padding_mode=padding_mode)
        self.fc1 = nn.Linear(lambdas_len, out_channels, bias=False)
        self.fc2 = nn.Linear(lambdas_len, out_channels, bias=False)
        # self.fc1 = nn.Conv2d(lambdas_len, out_channels, (1, 1), stride=1, padding=0, bias=True, padding_mode="zeros")
        # self.fc2 = nn.Conv2d(lambdas_len, out_channels, (1, 1), stride=1, padding=0, bias=True, padding_mode="zeros")
        self.relu = nn.ReLU()
        self.num_filters = out_channels
        self.lambdas_len = lambdas_len

    def forward(self, x, one_hot):
        # n, c = one_hot.size()
        # one_hot = one_hot.view(n, c, n, n)
        x = self.conv(x)
        scale = self.relu(self.fc1(one_hot)).view(1, self.num_filters, 1, 1)
        bias = self.fc2(one_hot).view(1, self.num_filters, 1, 1)
        x = x * scale
        # n, c, h, w = x.size()
        # bias =  
        # x = x + bias
        return x


class Net(nn.Module):
    def __init__(self, in_channels=3, out_channels=192, lambda4s=[0,1,2,3,4,5]):
        super(Net, self).__init__()
        self.in_channels = in_channels
        self.num_filters = out_channels
        self.caffe_channels = in_channels

        self.lambda4s = lambda4s
        self.sampled_lambda = 1

        self.norm_layer = None
        self.activation_layer = nn.ReLU

        self.conv = nn.Conv2d

        self._layers = nn.ModuleList([])
        self.build()

    def build(self):
        lambdas_len = len(self.lambda4s) - 1
        # self.one_hot = nn.Parameter(torch.zeros(1, lambdas_len), requires_grad=False)
        # lambdas = self.lambda4s[1:]
        # self.lambdas = nn.Parameter(torch.Tensor([np.array(lambdas)]), requires_grad=False)

        self._layers = nn.ModuleList([l for l in [
            ConditionalConv2d(
                self.in_channels, self.num_filters, (5, 5), stride=2, padding=2,
                bias=True, padding_mode="zeros", conv=self.conv, lambdas_len=lambdas_len),
            self.activation_layer()]])

    def forward(self, x, one_hot):

        for layer in self._layers:
            if isinstance(layer, ConditionalConv2d):
                x = layer(x, one_hot)
            else:
                x = layer(x)
        return x

 
if __name__ == '__main__':
    mode = 'dict' # 'dict'

    m = Net()
    print(m, flush=True)
    x = torch.randn(2, 3, 2, 2)
    sampled_lambda = torch.Tensor([2])
    one_hot = torch.Tensor([0, 1, 0, 0, 0])
    one_hot.unsqueeze_(0)

    input = {'image': x, 'one_hot': one_hot}

    if mode is not 'dict':
        pass
    else:
        print(m(x, one_hot).size(), flush=True)

        with torch.no_grad():
            m.cpu()
            m.eval()

            model_folder = 'caffe'
            name ='eg'
            if not os.path.exists(model_folder):
                os.mkdir(model_folder)
            path = os.path.join(model_folder, name)
            with pytorch.convert_mode():
                pytorch.convert(
                    m, [(m.caffe_channels, 256, 256), (1,5)],
                    filename=path,
                    input_names=["data", "one_hot"],
                    output_names=["out"],
                    verbose=True
                )
            caffe_model = name + '.caffemodel'
            cmd = f"cd {model_folder} && python -m nart_tools.caffe.convert -a {caffe_model}"
            print(cmd, flush=True)
            os.system(cmd)

