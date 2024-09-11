import numpy as np
import cv2
import argparse
import os

parser = argparse.ArgumentParser(description='Convert images to binaries for quantitile.')
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-l', '--image_list', dest='image_list', type=str, help='list for images to be processed.')
parser.add_argument('-o', '--output_dir', dest='output_dir', type=str, default='./output_videos', help='folder for video output.')
args = parser.parse_args()

def main():
    with open(args.image_list, 'r')as fp:
        lines = [line.strip() for line in fp.readlines()]
        for item in lines:
            image_list.append(item)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    image_cnt = 0
    for image in image_list:
        output_bin = os.path.join(args.output_dir, str(image_cnt) + ".bin")
        img = cv2.imread(image) / 255
        bin_str = img.astype('f').tostring()
        with open(output_bin, 'wb')as fp:
            fp.write(bin_str)
        image_cnt += 1
        
if __name__ == "__main__":
    main()