from argparse import ArgumentParser
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import struct


parser = ArgumentParser()
parser.add_argument('act1', type=str, help='.pk from torch or .bin from kestrel')
parser.add_argument('act2', type=str, help='.pk from torch or .bin from kestrel')
args = parser.parse_args()

res = []
for path in [args.act1, args.act2]:

    if path.endswith(".pk"):
        file = open(path, 'rb')
        act = pickle.load(file)
        print(act.shape)

    elif path.endswith(".bin"):
        file = open(path, 'rb')
        N = file.read(4)
        N_V = struct.unpack("i", N)[0]
        C = file.read(4)
        C_V = struct.unpack("i", C)[0]
        W = file.read(4)
        W_V = struct.unpack("i", W)[0]
        H = file.read(4)
        H_V = struct.unpack("i", H)[0]
        print((N_V, C_V, H_V, W_V))
        act = np.zeros((N_V, C_V, H_V, W_V))
        for i in range(N_V):
            for j in range(C_V):
                for m in range(H_V):
                    for n in range(W_V):
                        temp = file.read(4)
                        act[i, j, m, n] = struct.unpack("f", temp)[0]
    else:
        raise NameError("Only support *.bin or *.pk !")
    res.append(act)

act1, act2 = res
assert act1.shape == act2.shape

feature1 = torch.Tensor(act1) #[:, :, :20, :20]) # for align without 64 padding
feature2 = torch.Tensor(act2) #[:, :, :20, :20])


feature1 = feature1.contiguous().view(feature1.shape[0], -1)#将特征转换为N*(C*W*H)，即两维
feature2 = feature2.contiguous().view(feature2.shape[0], -1)

print("Equal: ", (feature2 == feature1).all())
flat1 = feature1.view(-1)
flat2 = feature2.view(-1)
for i in range(flat1.shape[0]):
    if flat1[i] != flat2[i]:
        print("i: %d feature1: %.4f feature2: %.4f" % (i, flat1[i], flat2[i]))

print((abs(feature1 - feature2)).sum() / (feature1.shape[0] * feature1.shape[1]))
print((abs(feature1 - feature2)/feature1).sum() / (feature1.shape[0] * feature1.shape[1]))

feature1 = F.normalize(feature1)  #F.normalize只能处理两维的数据，L2归一化
feature2 = F.normalize(feature2)
distance = feature1.mm(feature2.t())#计算余弦相似度
print(distance)
