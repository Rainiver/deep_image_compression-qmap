import argparse
import os

from tqdm import tqdm

from utils.converts import WebpEncoder, BpgEncoder, Jp2Encoder, HeifEncoder, JpegEncoder, \
    WebpDecoder, BpgDecoder, Jp2Decoder, HeifDecoder, JpegCopyDecoder

MODES_FOR_ENABLING_QUALITY = {'bpg_420': 51, 'bpg_444': 51, 'jpeg': 100}
COMPRESS_ADAPTERS = {
    'webp': (WebpEncoder, WebpDecoder, 'webp', 'png'),
    'bpg_420': (lambda quality, verbose, **kwargs: BpgEncoder(
        cfmt=420, quality=quality, verbose=verbose, **kwargs),
                BpgDecoder, 'bpg', 'png'),
    'bpg_444': (lambda quality, verbose, **kwargs: BpgEncoder(
        cfmt=444, quality=quality, verbose=verbose, **kwargs),
                BpgDecoder, 'bpg', 'png'),
    'jp2': (Jp2Encoder, Jp2Decoder, 'jp2', 'png'),
    'heif': (HeifEncoder, HeifDecoder, 'heif', 'png'),
    'jpeg': (JpegEncoder, JpegCopyDecoder, 'jpeg', 'jpeg')
}
ACCEPTABLE_SOURCE_IMAGE_EXTENSIONS = ['.png', '.jpeg', '.jpg', '.bmp']

parser = argparse.ArgumentParser(description='batch compress multiple images')
parser.add_argument('-m', '--mode', required=True, default='jpeg',
                    choices=COMPRESS_ADAPTERS.keys(),
                    help='the mode of the test')
parser.add_argument('-q', '--quality', default=None, help='quality of the compression (bpg and jpeg only)')
parser.add_argument('-s', '--src_dir', required=True, help='the source path of the original pictures')
parser.add_argument('-d', '--dst_dir', required=True, help='the destination path of the compressed images')
parser.add_argument('-v', '--verbose', action='store_true', default=False,
                    help='print all the run log in process')
parser.add_argument('-b', '--transform_back', action='store_true', default=False,
                    help='transform picture back to normal format or not')
arguments = parser.parse_args()

if __name__ == "__main__":
    src_dir = os.path.abspath(arguments.src_dir)
    dst_dir = os.path.abspath(arguments.dst_dir)
    verbose = arguments.verbose
    mode = arguments.mode
    transform_back = arguments.transform_back

    if mode in MODES_FOR_ENABLING_QUALITY.keys():
        quality = int(arguments.quality or MODES_FOR_ENABLING_QUALITY[mode])
    else:
        quality = None

    all_files = [os.path.normpath(os.path.join(src_dir, filename)) for filename in os.listdir(src_dir)]
    available_files = [filename for filename in all_files if os.access(filename, os.R_OK)]
    image_files = [os.path.split(filename)[1] for filename in available_files if
                   os.path.splitext(filename)[1].lower() in ACCEPTABLE_SOURCE_IMAGE_EXTENSIONS]

    os.makedirs(dst_dir, exist_ok=True)
    encoder_class, decoder_class, encoded_extension, decoded_extension = COMPRESS_ADAPTERS[mode]
    if transform_back:
        transformer = encoder_class(quality=quality, verbose=verbose) + decoder_class(verbose=verbose)
        extension = decoded_extension
    else:
        transformer = encoder_class(quality=quality, verbose=verbose)
        extension = encoded_extension

    bpps = []
    for filename in tqdm(image_files):
        print()

        short_name, _ = os.path.splitext(filename)
        src_path = os.path.normpath(os.path.join(src_dir, filename))
        dst_path = os.path.normpath(
            os.path.join(dst_dir, "{name}.{ext}".format(name=short_name, ext=extension)))

        print(src_path, dst_path)
        ret = transformer(src_path, dst_path)

        if transform_back:
            encode_ret, decode_ret = ret
        else:
            encode_ret, decode_ret = ret, None

        if encode_ret:
            encoded_file, (_, _bpp), _duration = encode_ret
            print("compressed bpp: {bpp}, {duration} second(s) cost.".format(bpp=_bpp, duration=_duration))
            bpps.append(_bpp)

        if decode_ret:
            decoded_file, _, _duration = decode_ret
            print("decompressed, {duration} second(s) cost.".format(duration=_duration))

        print()

    print("Average bpp:", sum(bpps) / len(bpps))
