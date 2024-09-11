import os
import re
import sys
import argparse

DOC = 'auto_test_four_run_type'
def arg_parser():
    parser = argparse.ArgumentParser(description = DOC)
    parser.add_argument('-md','--mode',choices = ['merge_tee','test','merge_test_tee','debug'],
        default = 'psnr',type = str,required = True, help = 'Enter the four mode')
    parser.add_argument('-pi','--input_root_dir',type = str, required = True,
        default = '/mnt/lustre/share/hedailan/gg18-ms-curve/',
        help = 'auto_train root dir_path_name') 
    #/mnt/lustre/share/fe/wangjian/deep_image_compression/deep/Deep_Image_Compression/experiments/auto_Train_ms
    parser.add_argument('-ow','--whether_overwrite',type = str,
        default = 'True',help = 'whether overwrite files in result dir, if False, will skip') 
    parser.add_argument('-df','--debug',type = str,
        default = False,help = 'whether debug')
    parser.add_argument('-po','--save_path_pattern',type = str,
        default = '../result/GG_18_ours256_ms_{}.csv',help = 'CSV output save directory')
    parser.add_argument('--partition', '-P', type=str, default='spring_scheduler')


    # 如果result中有同名csv文件则跳过不测试
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

# shared models
# redirect original test logs here, as we may not have write access in models_dir
# set to models_dir if merge train's last val results from tee.log directly
# otherwise, will merge auto_Test's tee_test.log
# when both exist (should not happen normally), will merge tee_test.log instead of tee.log

def csv_save(filename, data):
    # data is list of list
    file = open(filename, 'w')
    for i in range(len(data)):
        s = str(data[i]).replace('[', '').replace(']', '').replace(' ', '')
        s += '\n'
        file.write(s)
    file.close()

def log_parser(log_dir):
    '''
    target is one line, last matched:

    mean: [eval/y_bpp:0.0003255, eval/z_bpp:0.0004476, eval/bpp:0.0007731, \
        eval/z_entropy_loss:0.3372452, eval/y_entropy_loss:0.0000079, eval/msssim:nan, \
            eval/mse:17563.3062337, eval/tv:0.0000019, eval/entropy_total:0.3372531, eval/total:360.1375058, eval/psnr:5.9390656]
    '''
    link_pattern = 'finalized!'

    pattern = 'mean: \[(.*)eval/bpp:(\d+\.?\d*),(.*)eval/msssim:(\d+\.?\d*),(.*)eval/psnr:(\d+\.?\d*).*'
    res2 = None
    res2_link = None
    with open(log_dir) as f:
        for line in f:
            res = re.search(pattern, line)
            res_link = re.search(link_pattern, line)
            if res is not None:
                res2 = res
            if res_link is not None:
                res2_link = res_link
    if res2_link is None:
        return None
    else:
        print(res2_link)
    if res2 is None:
        return None
    return [float(res2.group(4)), float(res2.group(6)), float(res2.group(2))]

def judge_mode(mode,input_dir):
    test_datasets = {
    'valid': '/mnt/lustre/share/fe/codec_list/valid.txt',
    'validset': '/mnt/lustre/share/fe/codec_list/validset.txt',
    'kodak': '/mnt/lustre/share/fe/codec_list/kodak.txt',
    'spark_1080p': '/mnt/lustre/share/fe/codec_list/spark_29af1347f4187f7bcfca242c0a8d7680.mp4_png.txt',
    'west_1080p': '/mnt/lustre/share/fe/codec_list/XBSJ.S02E01.1080p.mp4_png.txt',
    'poi_1080p': '/mnt/lustre/share/fe/codec_list/Person.of.Interest.S01E23.2011.1080p.Blu-ray.x265.mkv_png.txt',
    'Forever': '/mnt/lustre/share/fe/codec_list/Forever.Young_png.txt',
    'Fast_2160p_10bit': '/mnt/lustre/share/fe/codec_list/Fast.and.Furious_png.txt',
    'codec_img_png': '/mnt/lustre/share/fe/codec_list/codec_img_png.txt',
    'tecnick':'/mnt/lustre/share/fe/codec_list/tecnic_RGB_OR_1200x1200.txt'
}
    if mode == 'merge_tee':
        merge_val_tee_log = True  # if True, merge train's val results from tee.log for kodak
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)
        test_datasets = {'kodak': '/mnt/lustre/share/fe/codec_list/kodak.txt'}
    if mode == 'test':
        merge_val_tee_log = False
    if mode == 'merge_test_tee':
        merge_val_tee_log = False
    return merge_val_tee_log,test_datasets


def mode_test(test_models,value,temp_d_dir):

    for idx, model in enumerate(test_models):
        if debug and idx >= max_model:
            break
        model_name = model.split('/')[-1]
        log_dir = os.path.join(temp_d_dir, model_name, 'tee_test.log')

        # 如果第一次跑遇到某些实验不能merge，删掉tee_test.log，再跑一次就行了，已经跑过的不会重复跑
        if os.path.exists(log_dir):
            print('skip test {}'.format(log_dir))
            continue

        # 用models_dir里的模型，但是所有结果都输出到temp_d_dir/model_name下面
        root = model
        save_dir = os.path.join(temp_d_dir, model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        test_class = value[:-4].split('/')[-1]
        print('testing {} in {}'.format(test_class, save_dir))
        os.system(f'sh auto_test_helper.sh {partition} {root} {save_dir} {max_testing} {value}')

def mode_merge(test_models,temp_d_dir,merge_val_tee_log,save_path):
    log_name = 'tee.log' if merge_val_tee_log else 'tee_test.log'
    results = []
    for idx, model in enumerate(test_models):
        model_name = model.split('/')[-1]
        log_dir = os.path.join(temp_d_dir, model_name,log_name)
        ms_ssim_psnr_bpp = log_parser(log_dir)
        if ms_ssim_psnr_bpp is None:
            print('parser error {}'.format(log_dir))
            continue
        results.append(ms_ssim_psnr_bpp)
    results.sort(key = lambda x: -x[-1])
    csv_save(save_path, results)

def main(mode,input_dir,overwrite,debug,save_csv):
    test_models = os.listdir(input_dir)
    test_models = [os.path.join(input_dir, item) for item in test_models]
    merge_val_tee_log,test_datasets = judge_mode(mode,input_dir)
    idx_data = 0
    if debug:
        max_testing = 5
        max_model = 2
        max_data = 2
    for key, value in test_datasets.items():
        if debug and idx_data >= max_data:
            break
        idx_data += 1
        # for save merged results
        save_path = save_csv.format(key)
        if not overwrite:
            if os.path.exists(save_path):
                print('skip {} {}'.format(mode, save_path))
                continue
        if mode != "merge_tee":
            temp_dir = "auto_Test"
            temp_d_dir = os.path.join(temp_dir, key)
        else:
            temp_d_dir = input_dir
        if not os.path.exists(temp_d_dir) and mode == 'test':
            os.makedirs(temp_d_dir)
        if mode == 'test':
            mode_test(test_models,value,temp_d_dir)
        elif mode == 'merge_tee':
            mode_merge(test_models,temp_d_dir,merge_val_tee_log,save_path)
        elif mode == 'merge_test_tee':
            # mode_test(test_models,value,temp_d_dir)
            mode_merge(test_models,temp_d_dir,merge_val_tee_log,save_path)

if __name__ == '__main__':
    args = arg_parser()
    partition = args.partition
    debug = False
    max_testing = 100
    main(args.mode, args.input_root_dir,args.whether_overwrite,args.debug,args.save_path_pattern)
