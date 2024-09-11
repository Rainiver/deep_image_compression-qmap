import argparse
import os
import cv2
import warnings
from typing import Union, List
from tqdm import tqdm
from PIL import Image
import numpy as np
from utils.converts import WebpEncoder, WebpDecoder, BpgEncoder, BpgDecoder, \
    Jp2Encoder, Jp2Decoder, HeifEncoder, HeifDecoder, JpegEncoder, JpegCopyDecoder, JpegLeptonEncoder, JpegLeptonDecoder, \
    VtmEncoder, VtmDecoder, JpegTurboEncoder, JpegTurboDecoder, JpegXlEncoder, JpegXlDecoder, CmixEncoder, CmixDecoder
from utils.losses import ssim, msssim, psnr

DEFAULT_DATA_DIR = '/mnt/lustre/share/fe/face_data/codec/compression/kodak'

parser = argparse.ArgumentParser(description='implementation of codec test')
parser.add_argument('--mode', default='jpeg',
                    choices=['webp', 'bpg', 'jp2', 'heif', 'jpeg_align', 'jpeg', 'jpeg_lepton', 'vtm', 'jpeg_turbo', 'jpeg_xl', 'cmix'],
                    help='the mode of the test')
parser.add_argument('--data_dir', default=DEFAULT_DATA_DIR,
                    help='the path of tested pictures')
parser.add_argument('--keep_image', choices=['true', 'false'], default='false',
                    help='whether to keep image')
parser.add_argument('-v', '--verbose', action='store_true', default=False,
                    help='print all the run log in process')
parser.add_argument('--test_only', action='store_true', default=False)
parser.add_argument('--encode_only', action='store_true', default=False)
arguments = parser.parse_args()


def average(list_: List[Union[float, int]]) -> float:
    """
    Get average value of a list consists of float or int.
    :param list_: list of float or int
    :return: average value of the list (float format)
    """
    return sum(list_) / len(list_)


CODEC_INFO = {
    "webp": (WebpEncoder, WebpDecoder, "webp", "png"),
    "bpg": (BpgEncoder, BpgDecoder, "bpg", "png"),
    "jp2": (Jp2Encoder, Jp2Decoder, "jp2", "png"),
    "heif": (HeifEncoder, HeifDecoder, "heic", "png"),
    "jpeg": (JpegEncoder, JpegCopyDecoder, "jpeg", "jpeg"),
    "jpeg_align": (JpegEncoder, JpegCopyDecoder, "jpeg", "jpeg"),
    "jpeg_lepton": (JpegLeptonEncoder, JpegLeptonDecoder, "lep", "jpeg"),
    "jpeg_turbo": (JpegTurboEncoder, JpegTurboDecoder, "jpeg", "bmp"),
    "vtm": (VtmEncoder, VtmDecoder, "266", "png"),
    "jpeg_xl": (JpegXlEncoder, JpegXlDecoder, "jxl", "jpeg"),
    "cmix": (CmixEncoder, CmixDecoder, "cmix", "jpeg"),
}


def test_codec(data_dir=DEFAULT_DATA_DIR, quality=50, mode='bpg', keep_image=False, verbose=False):
    """
    Ranges for quality:
    - webp: 0-100
    - typical 80
    - bpg: 0-51, default 29, also many other options
    :param data_dir: directory for image dataset
    :param quality: quality for encoding (should be an integer, default s 50)
    :param mode: encode mode (default is bpg)
    :param keep_image: keep image or not (default is False)
    :param verbose: print verbose information
    :return: average s1, s2, s3, s4 and time for encoding and decoding
    """
    print('testing: ' + data_dir)
    print('testing: ' + mode)

    index = 0
    ssims, psnrs, msssims, bpps = [], [], [], []
    encode_times, decode_times = [], []
    for maindir, subdir, filenames in os.walk(data_dir):
        filename_count = len(filenames)
        for filename in filenames:
            index += 1
            original_file = os.path.normpath(os.path.join(maindir, filename))
            h, w, _ = cv2.imread(original_file).shape
            name, _ = os.path.splitext(filename)

            assert not (test_only and encode_only), 'cannot turn on both test_only and encode_only'
            assert mode in CODEC_INFO.keys(), TypeError("Unknown image type")

            encoder_class, decoder_class, encode_ext, decode_ext = CODEC_INFO[mode]
            if keep_image:
                encoded_filename = "output_encoded_{}_{}_{}x{}.{}".format(name, quality, h, w, encode_ext)
                decoded_filename = "output_decoded_{}_{}_{}x{}.{}".format(name, quality, h, w, decode_ext)
            else:
                encoded_filename = "output_encoded_{}x{}.{}".format(h, w, encode_ext)
                decoded_filename = "output_decoded_{}x{}.{}".format(h, w, decode_ext)

            if not test_only:
                encoded_file, (_, _bpp), _duration = encoder_class(quality=quality, verbose=verbose).encode(
                    original_file, encoded_filename)
                bpps.append(_bpp)
                print("calculated bpp: {bpp}".format(bpp=_bpp))
                encode_times.append(_duration)

                decoded_file, _, _duration = decoder_class(verbose=verbose).decode(encoded_file, decoded_filename)
                decode_times.append(_duration)

            if encode_only:
                continue

            # saved image need rotate
            if mode in ['bpg']:
                _original = cv2.imread(original_file)
                _decoded = cv2.imread(decoded_filename)
                if _original.shape[:2] == _decoded.shape[:2][::-1] and \
                        _original.shape[:2] != _decoded.shape[:2]:  # we may meet square images
                    _decoded = cv2.rotate(_decoded, cv2.ROTATE_90_CLOCKWISE)
                cv2.imwrite(decoded_filename, _decoded)
                print(f'rotated bpg, input shape:{_original.shape}, output shape:{_decoded}')

            print('reading new file {}'.format(decoded_file))
            ssims.append(ssim(original_file, decoded_file))
            psnrs.append(psnr(original_file, decoded_file))
            msssims.append(msssim(original_file, decoded_file))
            warnings.warn("The original output metrics were cumulative average, now they are replaced "
                          "by individual test record's metric")
            print("ssim: {}, psnr: {}, msssim: {}, bpp: {}".
                  format(ssims[-1], psnrs[-1], msssims[-1], bpps[-1]))
            print("{}/{}".format(index, filename_count), flush=True)

            if not keep_image:
                if os.path.exists(encoded_file):
                    os.remove(encoded_file)
                if os.path.exists(decoded_file):
                    os.remove(decoded_file)
                os.system('rm -rf *.yuv')

    return average(ssims), average(psnrs), average(msssims), average(bpps), (
        average(encode_times), average(decode_times))


# debug
def psnr_func(original_file, decoded_file, norm=False):
    img_ori = Image.open(original_file)
    img_ori = np.asarray(img_ori, dtype=np.float32)
    img_dec = Image.open(decoded_file)
    img_dec = np.asarray(img_dec, dtype=np.float32)
    if norm:
        img_ori, img_dec = img_ori / 255., img_dec / 255.
        mse = np.square(img_ori - img_dec).mean()
        _psnr = 10. * np.log10(1. / mse)
        return _psnr
    mse = np.square(img_ori - img_dec).mean()
    _psnr = 10. * np.log10(255. ** 2 / mse)
    return _psnr


if __name__ == '__main__':
    _data_dir = arguments.data_dir
    _mode = arguments.mode
    _keep_image = arguments.keep_image == 'true'
    _verbose = arguments.verbose
    test_only = arguments.test_only
    encode_only = arguments.encode_only

    if _mode == 'jpeg':
        qualities = [1, 10, 20, 30, 40, 50, 60, 70, 75, 80, 90, 100]
    elif _mode == 'jpeg_lepton':
        qualities = [1, 10, 20, 30, 40, 50, 60, 70, 75, 80, 90, 100]
    elif _mode == 'webp':
        qualities = [1, 10, 20, 30, 40, 50, 60, 70, 75, 80, 90, 100]
    elif _mode == 'bpg':
        qualities = [1, 5, 10, 15, 20, 25, 29, 35, 40, 45, 51]
    elif _mode == 'heif':
        _mode = 'heif'
        os.environ['LD_LIBRARY_PATH'] = '/mnt/lustre/share/fe/wangliangzhou/codec/usr/local/lib:' + os.environ[
            'LD_LIBRARY_PATH']
        qualities = [1, 10, 20, 30, 40, 50, 60, 70, 75, 80, 90, 100]
    elif _mode == "jp2":
        _mode = "jp2"
        qualities = [None]
    elif _mode == 'vtm':
        qualities = [5, 10, 15, 20, 25, 29, 35, 40, 45]

    # 按照b点对应的bpp，找精确的c点，并保留a点和c点所有的图片，用于和b点的模型做对比
    # 如plot中按照ms-ssim来align的结果如下：
    # jpeg default is Q 75 bpp 1.3688269721137154 psnr 34.54098129272461 ms-ssim 0.9866936206817627
    # better id 5 bpp 0.5210088 ms-ssim 0.9874656
    # ratio is 2.6272626721731287
    # c id upper 3 Q 30 bpp 0.6601265801323783 ms-ssim 0.9631773829460144
    # c id lower 2 Q 20 bpp 0.5083855523003472 ms-ssim 0.9456934928894043
    # 测Q 20-30之间所有的结果
    elif _mode == 'jpeg_align':
        # 最后一个75对应默认质量的a点
        qualities = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 75]
        # 保存所有jpeg
        _keep_image = True
        target_bpp = 0.5210088
    elif _mode == 'jpeg_turbo':
        qualities = [1, 10, 20, 30, 40, 50, 60, 70, 75, 80, 90, 100]
    elif _mode == 'jpeg_xl':
        qualities = [1, 10, 20, 30, 40, 50, 60, 70, 75, 80, 90, 100]
    elif _mode == 'cmix':
        qualities = [1, 10, 20, 30, 40, 50, 60, 70, 75, 80, 90, 100]
    else:
        raise TypeError("Unknown mode {mode}.".format(mode=repr(_mode)))

    ssim_list, psnr_list, msssim_list, bpp_list = [], [], [], []
    speed_time_dict = {}
    for _quality in tqdm(qualities):
        en_de_time = []
        ssim_, psnr_, msssim_, bpp_, (encode_time, decode_time) = test_codec(
            data_dir=_data_dir, quality=_quality, mode=_mode,
            keep_image=_keep_image, verbose=_verbose,
        )
        ssim_list.append(ssim_)
        psnr_list.append(psnr_)
        msssim_list.append(msssim_)
        bpp_list.append(bpp_)
        speed_time_dict[str(_quality)] = (encode_time, decode_time)

    print('batch processing:')
    print('ssim:', ssim_list, flush=True)
    print('psnr', psnr_list, flush=True)
    print('msssim:', msssim_list, flush=True)
    print('bpp:', bpp_list, flush=True)
    print('speed_time :', speed_time_dict, flush=True)

    # 找到bpp和target_bpp最接近的（恰好大于）
    if _mode == 'jpeg_align':
        idx = 0
        for b in bpp_list:
            if b < target_bpp:
                idx += 1
            else:
                break
        print('precise c id {} Q {} bpp {} psnr {} ms-ssim {}'.format(
            idx, qualities[idx], bpp_list[idx], psnr_list[idx], msssim_list[idx]))
