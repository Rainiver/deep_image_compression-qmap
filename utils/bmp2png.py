from PIL import Image
import numpy as np
import os

debug = False
if debug:
    bmp_path = '/mnt/lustre/share/fe/face_data/codec/codec_img/37_qcom517_IMG20191114205256.bmp'
    img_bmp = Image.open(bmp_path)

    img_bmp.save('test.png')
    img_png = Image.open('test.png')

    dif = np.array(img_bmp) - np.array(img_png)
    print(dif.max())
    print(dif.min())


bmp_dir = '/mnt/lustre/share/fe/face_data/codec/codec_img'
png_dir = 'codec_img_png'

if not os.path.exists(png_dir):
    os.mkdir(png_dir)

for bmp_name in os.listdir(bmp_dir):
    name = bmp_name.rstrip('.bmp')
    png_name = name + '.png'
    bmp_path = os.path.join(bmp_dir, bmp_name)
    png_path = os.path.join(png_dir, png_name)
    Image.open(bmp_path).save(png_path)
