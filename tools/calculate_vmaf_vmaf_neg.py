from PIL import Image
import os
import argparse
import sys
import subprocess
import re

#计算vmaf和vmaf_neg暂不支持同级目录下不同size的图片

DOC = 'Vmaf and vmaf neg were calculated'
def arg_parser():
	parser = argparse.ArgumentParser(description = DOC)
	parser.add_argument('-pi','--vmaf_path',type = str,required = True,\
		help = 'Enter the vmaf root path')
	parser.add_argument('-op','--original_yuv_dir_path',type = str,required = True,\
		help = 'Enter the vmaf root path')
	parser.add_argument('-mp','--contrast_yuv_path',type = str, required = True,\
		help = 'Enter the match yuv path')
	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit(1)
	args = parser.parse_args()
	return args

def calculate_vmaf(l_original,l_match,vmaf_path,path_original,path_match):
	os.chdir(vmaf_path)
	for i,j in zip(l_original,l_match):
		img_original = Image.open(os.path.join(path_original,i))
		img_width = img_original.width 
		img_height = img_original.height
		# 在clone之后的vmaf根目录下执行下面的命令
		p = subprocess.Popen("PYTHONPATH=python ./python/vmaf/script/run_vmaf.py \
			yuv444p {} {} {}{} {}{} \
			--out-fmt json >> match.txt".format(img_width,img_height,path_original,\
				i,path_match,j),shell=True)
		if p.poll() == 0:
			continue
		else:
			p.wait()
	l = []
	print(os.getcwd())
	with open(os.path.join(os.getcwd(),'match.txt'),'r') as f:
		context = f.readlines()
		for con in context:
			print(con)
			match_str = re.findall(r'"VMAF_score": \d+\.\d+',con)
			if match_str:
				l.append(float(match_str[0].split()[-1]))
	l_score = []
	num = 0
	for index in range(0,len(l),2):
		l_score.append(l[index])
		num += 1

	print(l_score)
	print("average vmaf :",sum(l_score) / num)


def calculate_vmaf_neg(l_original,l_match,vmaf_path,path_original,path_match):
	os.chdir(os.path.join(vmaf_path,'libvmaf'))
	l_neg = []
	num = 0
	for i,j in zip(l_original,l_match):
		img_original = Image.open(os.path.join(path_original,i))
		img_width = img_original.width  
		img_height = img_original.height
		#在libvmaf目录下执行下面的命令
		p = subprocess.Popen("./build/tools/vmaf_rc --reference {}{} --distorted {}{} \
			--width {} --height {} --pixel_format 444 --bitdepth 8 \
			--model path=../model/vmaf_v0.6.1.pkl --feature float_vif=vif_enhn_gain_limit=1.0 \
			--feature float_adm=adm_enhn_gain_limit=1.0 \
			--output match.txt".format(img_width.img_height,path_original,i,path_match,j),shell=True)
		p.wait()
		with open('match.txt','r') as fr:
			fr_list = fr.readlines()
			for line in fr_list:
				match = re.findall(r'vmaf="\d+\.\d+"',line.strip())
				if match:
					vmaf_score = re.findall(r'\d+\.\d+',match[0])
					l_neg.append(float(vmaf_score[0]))
		num += 1
		if p.poll() == 0:			
			continue
		else:
			p.wait()
	print(l_neg)
	print("average vmaf_neg :",sum(l_neg) / num)


def main(vmaf_path,path_original,path_match):

	l_original = os.listdir(path_original)
	l_match = os.listdir(path_match)
	l_original.sort()
	l_match.sort()
	calculate_vmaf(l_original,l_match,vmaf_path,path_original,path_match)
	calculate_vmaf_neg(l_original,l_match,vmaf_path,path_original,path_match)


if __name__ == '__main__':
	args = arg_parser()
	main(args.vmaf_path, args.original_yuv_path,args.contrast_yuv_path)


	