from torch.utils.data import Sampler
import random


class RangeSample(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class MySubsetRandomSampler(Sampler):
    # https://www.cnpython.com/qa/125853 MT19937
    # https://www.cnblogs.com/xianbin7/p/10720638.html randperm

    def __init__(self, indices):
        self.indices = indices
        random.seed(13331)

    def __iter__(self):
        random.shuffle(self.indices)
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)
