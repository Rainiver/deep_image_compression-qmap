import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms


def incorrect_png(data_dir="../../../dataset/compression/valid"):
    for maindir, subdir, file_name_list in os.walk(data_dir):
        n = len(file_name_list)
        i = 0
        for filename in file_name_list:
            # print(filename)
            p = os.path.join(maindir, filename)
            temp = cv2.imread(p)
            cv2.imwrite(p, temp)
            i += 1
            print(i, "/", n)


def add_noise(image, BIT=8):
    noise = np.random.uniform(0, 1, image.shape)
    noise -= 0.5
    factor = (1 << BIT) - 1
    noise /= factor
    return image + noise


def downsampling(image, factor=2):
    image = transforms.ToTensor()(image)
    layer = nn.AvgPool2d(factor, factor)
    image = layer(image)
    image = np.transpose(image.numpy(), [1, 2, 0])
    return image


def jpg2png(data_dir="/mnt/lustre/share/fe/face_data/zhengyaoyan_data/compression/train"):
    for maindir, subdir, file_name_list in os.walk(data_dir):
        n = len(file_name_list)
        i = 0
        for filename in file_name_list:
            # print(filename)
            p = os.path.join(maindir, filename)
            temp = cv2.imread(p)
            temp = add_noise(temp, 8)
            temp = downsampling(temp)
            cv2.imwrite(p[0:-5] + ".png", temp)
            i += 1
            print(i, "/", n)


def count(pixels=2000000, data_dir="/mnt/lustre/share/zhengyaoyan/compression/train"):
    cnt = 0
    for maindir, subdir, file_name_list in os.walk(data_dir):
        n = len(file_name_list)
        i = 0
        for filename in file_name_list:
            # print(filename)
            p = os.path.join(maindir, filename)
            temp = cv2.imread(p)
            nn = temp.shape[0] * temp.shape[1]
            if nn > pixels:
                cnt += 1
            i += 1
            print(i, "/", n)
    print('', '', cnt)


if __name__ == '__main__':
    count(200000, "../../../dataset/compression/val")
