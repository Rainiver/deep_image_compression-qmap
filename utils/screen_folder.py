import cv2
import numpy as np
import os

target_dir = '/mnt/lustre/share/fe/face_data/codec/compression/codec_img_png'

i = 0
for name in os.listdir(target_dir):
    i += 1
    path = os.path.join(target_dir, name)
    img = cv2.imread(path)
    print(img.shape, flush=True)

    size = os.path.getsize(path)
    print(str(size*8) + 'bit', flush=True)

print(i, flush=True)
