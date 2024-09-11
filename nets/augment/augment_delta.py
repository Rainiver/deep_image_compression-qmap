import torch
import torch.nn as nn
import numpy as np
import collections


class DeltaAugment(nn.Module):
    def __init__(self, que: collections.deque, prob=0.5):
        super(DeltaAugment, self).__init__()
        self.que = que
        self.prob = prob

    def forward(self, x: torch.Tensor):
        # old_x = x.clone()
        flag = torch.BoolTensor(x.size()[0])
        for i in range(x.size()[0]):
            if np.random.rand(1) < self.prob and len(self.que):
                x[i] = self.que.pop()
                flag[i] = True
        # print('new: ', x.sum(), ' old: ', old_x.sum(), ' flag: ', flag)
        return {'x': x.contiguous(), 'tmp/x_collect_flag': flag}
