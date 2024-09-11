from PIL import Image
import os
import argparse
import sys
import subprocess
import re
from tqdm import tqdm

DOC = 'Png lossless conversion YUV'
def arg_parser():
	parser = argparse.ArgumentParser(description = DOC)
	parser.add_argument('-pi','--png_path',type = str,required = True,\
		help = 'Enter the png root path')
	parser.add_argument('-op','--yuv_path',type = str,required = True,\
		help = 'Enter the convert yuv root path')
	
	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit(1)
	args = parser.parse_args()
	return args

def convert(png_path,yuv_path):
	if not os.path.exists(yuv_path):
		os.makedirs(yuv_path)
	l = os.listdir(png_path)
	for png_name in tqdm(l):
		img_name = os.path.join(png_path,png_name)
		img_original = Image.open(img_name)
		img_width = img_original.width  
		img_height = img_original.height
		print(img_width)
		print(img_height)
		yuv_name = os.path.join(yuv_path,png_name.split('.')[0]) + '.yuv'
		#图片无损转yuv
		p = subprocess.Popen("ffmpeg -y -s " + str(img_width) + \
			'x' + str(img_height) + " -pix_fmt yuv444p -i " + \
			img_name + ' ' + yuv_name,shell = True)
		
		


if __name__ == '__main__':
	args = arg_parser()
	convert(args.png_path, args.yuv_path)