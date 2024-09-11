import torch
import torch.nn as nn
from torch.nn import functional as F
import spring.nart.tools.pytorch as pytorch
# import nart.python.spring.nart.tools.pytorch as pytorch
import os
import numpy as np


'''
主要两个op，torch.clamp和torch.round(m/n)
torch.clamp→clip in kestrel_mixnet and kestrel_ppl （无法暂时用几个relu代替，因为x=x+1这种暂不支持）
torch.round(m/n)->onnx ?->rounding division in kestrel_mixnet and kestrel_ppl
'''

class Net(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, use_round=False):
        super(Net, self).__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv2d(3, 3, 3, 1)
        self.c = nn.Parameter(torch.ones((1, out_channels, 1, 1)))
        self.min = nn.Parameter(torch.Tensor([0]), requires_grad = False)
        self.max = nn.Parameter(torch.Tensor([255]), requires_grad = False)

    def forward(self, x):
        x = self.conv(x)

        if use_round:
            x = torch.floor((x + torch.floor(self.c / 2.0)) / self.c)  #torch.round(x / self.c)
        else:
            x = x / self.c
        
        x = torch.clamp(x, self.min.data[0], self.max.data[0])
        return x

 
if __name__ == '__main__':
    use_round = True

    m = Net(use_round=use_round)
    print(m, flush=True)

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
                m, [(m.in_channels, 256, 256)],
                filename=path,
                input_names=["data"],
                output_names=["out"],
                verbose=True
            )
        caffe_model = name + '.caffemodel'
        cmd = f"cd {model_folder} && python -m nart_tools.caffe.convert -a {caffe_model}"
        print(cmd, flush=True)
        os.system(cmd)

