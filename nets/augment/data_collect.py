import torch.nn as nn
import collections
import numpy as np

"""

> collect x + k * (x - x_hat) which was used to enhance origin image

 - delta
 
"""


class DeltaCollect(nn.Module):
    def __init__(self, que: collections.deque, prob=0.5):
        super(DeltaCollect, self).__init__()
        self.que = que
        self.prob = prob

    def forward(self, x, x_hat, flag):
        # print('collecting')
        aug = x + 2 * (x - x_hat)  # x_hat Low Freq
        for i in range(x.size()[0]):
            if not flag[i] and np.random.rand(1) < self.prob:
                self.que.append(aug[i].clone().contiguous().detach())
        return


class FakeCollect(nn.Module):
    def __init__(self):
        super(FakeCollect, self).__init__()

    def forward(self, *args, **kwargs):
        return


def data_collect(cfg='NONE', **kwargs):
    if cfg == 'NONE':
        return FakeCollect()
    if cfg == 'delta':
        return DeltaCollect(kwargs['que'])
    else:
        raise ValueError('unsupported post tag:{}'.format(str(cfg)))
