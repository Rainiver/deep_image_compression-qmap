import os
import numpy as np
import cv2

path = "/mnt/lustre/share/images/train"
path_ = "../../../dataset/compression/val"


def entropy(image=None):
    c, h, w = image.shape
    n = c * h * w
    if n == 0:
        return 0
    ret = 0.
    for i in range(0, 256):
        tot = sum(sum(sum(image == i)))
        if tot > 0:
            ret += -np.log2(1. * tot / n) * tot / n
    return ret


a = []
b = []
if __name__ == '__main__':
    for maindir, subdir, file_name_list in os.walk(path):
        i = 0
        for filename in file_name_list:
            i += 1
            print(i, "/", len(file_name_list))
            image = cv2.imread(os.path.join(maindir, filename))
            #x = entropy(image)
            print(maindir)
            if image.shape[0]*image.shape[1]<1000000:
                continue
            else:
                 print('added')
                 pp='cp '+os.path.join(maindir, filename)+' '+'/mnt/lustre/share/zhengyaoyan/dataset/compression/train/'+filename
                 os.system(pp)
            a.append((filename, image.shape[0] * image.shape[1]))
    a.sort(key=lambda x: x[1])
    file = open("../dataset_choosen.txt", 'w')
    a = np.array(a)
    for i in range(0, len(a)):
        print(a[i][0], file=file)
