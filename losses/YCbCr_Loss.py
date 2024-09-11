import torch
import torch.nn as nn


# from https://en.wikipedia.org/wiki/YUV
def rgb_to_yuv(x: torch.Tensor):
    shape = x.shape
    x = x.permute(0, 2, 3, 1).reshape(-1, 3)
    mat = torch.tensor([[66., 129., 25.],
                        [-38., -74., 122.],
                        [112., -94., -18.]], requires_grad=False)
    bias = torch.tensor([16 / 255, 128 / 255, 128 / 255], requires_grad=False)
    mat = mat.to(x.device)
    bias = bias.to(x.device)
    temp = x.mm(mat.permute(1, 0)) / 2. ** 8 + bias
    return temp.reshape(shape[0], shape[2], shape[3], shape[1]).permute(0, 3, 1, 2)


def rgb_to_yuv1(x: torch.Tensor):
    shape = x.shape
    x = x.permute(0, 2, 3, 1).reshape(-1, 3)
    mat = torch.tensor([[0.256789, -0.148223, 0.439215],
                        [0.504129, -0.290992, -0.367789],
                        [0.097906, 0.439215, -0.071426]], requires_grad=False)
    bias = torch.tensor([16 / 255, 128 / 255, 128 / 255], requires_grad=False)
    temp = x.mm(mat) + bias
    return temp.reshape(shape[0], shape[2], shape[3], shape[1]).permute(0, 3, 1, 2)


"""
see : https://github.com/tensorflow/compression/tree/master/results/image_compression
"""


class Loss(nn.Module):
    def __init__(self, tag="YUV"):
        r"""
        :param tag:
            YUV:  Y:U:V = 6:1:1
            Y:  Y:U:V = 6:0:0
        """
        super(Loss, self).__init__()
        if tag == "Y":
            self.w = [1., 0., 0.]
        elif tag == "YUV":
            self.w = [6. / 8, 1. / 8, 1. / 8]
        else:
            raise ValueError('unsupported post tag:{}'.format(tag))

    def forward(self, input, target):
        input = rgb_to_yuv(input)
        target = rgb_to_yuv(target)
        loss_fn = nn.MSELoss()
        loss = torch.tensor([0.]).to(input.device)
        for i in range(input.size()[1]):
            loss += self.w[i] * loss_fn(input[:, i, ...], target[:, i, ...])

        return loss


if __name__ == '__main__':
    test = Loss()
    x = torch.ones(2, 3, 10, 10)
    y = torch.zeros(2, 3, 10, 10)
    # print(test(x, y))

    # print(rgb_to_yuv(x)[0][0][0])

    from utils.color_space import rgb_to_yuv as std

    x = torch.rand(4, 3, 1000, 1000)
    # print("0", (rgb_to_yuv(x) - std(x)).mean())
    # print("1", (rgb_to_yuv1(x) - std(x)).mean())

    # 0 tensor(0.0053)
    # 1 tensor(-0.0013)
