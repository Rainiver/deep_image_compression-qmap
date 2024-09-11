from utils.bjontegaard_metric import *
import csv
import argparse
import sys

DOC = 'Calculate BD rate and BD psnr'

bpg_psnr_k_444 = [52.90655517578125, 52.1257209777832, 48.7451171875, 45.95499801635742, 42.74228286743164,
                      39.566505432128906, 36.887779235839844, 32.87386703491211, 29.803916931152344, 27.142967224121094,
                      24.445253372192383]

bpg_bpp_k_444 = [9.682867262098526, 8.554784986707899, 5.582457648383247, 3.6544469197591156, 2.3169623480902772,
                     1.4546186659071179, 0.9667510986328125, 0.46473269992404515, 0.22347598605685767,
                     0.09685940212673612, 0.033114115397135414]

jpeg_psnr_k = [21.474653244018555, 26.672494888305664, 29.146554946899414, 30.494787216186523, 31.425643920898438,
                   32.1790885925293, 32.917991638183594, 33.93019485473633, 34.54098129272461, 35.39570617675781,
                   38.09836959838867, 45.199951171875]

jpeg_bpp_k = [0.1731050279405382, 0.32661437988281244, 0.5083855523003472, 0.6601265801323783, 0.786026848687066,
                  0.906000773111979, 1.0372119479709203, 1.2398461235894098, 1.3688269721137154, 1.5718070136176214,
                  2.350199381510416, 6.765453762478299]



def arg_parser():
	parser = argparse.ArgumentParser(description = DOC)
	parser.add_argument('-pi','--path_in',type = str, required = True,\
		help = 'Enter the CSV address of the first model')
	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit(1)
	args = parser.parse_args()
	return args

def main(path):
	with open(path,'r') as csvfile:
		reader = csv.reader(csvfile)
		data = [row for row in reader]
		psnr = [float(row[1]) for row in data]
		bpp = [float(row[2]) for row in data]

	print('BD-PSNR:   ',BD_PSNR(bpg_bpp_k_444,bpg_psnr_k_444,bpp,psnr))
	print('BD-RATE:   ',BD_RATE(bpg_bpp_k_444,bpg_psnr_k_444,bpp,psnr))

if __name__ == '__main__':
	args = arg_parser()
	main(args.path_in)