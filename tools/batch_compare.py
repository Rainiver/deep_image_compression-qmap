import argparse
import codecs
import json
import os
import sys
from typing import Mapping

from tqdm import tqdm

from utils.losses import msssim, ssim, psnr

ACCEPTABLE_IMAGE_EXTENSIONS = ['.jpg', '.png', '.bmp', '.jpeg']
LOSS_FUNCTION_MAP = {
    "msssim": msssim,
    "ssim": ssim,
    "psnr": psnr,
}
DEFAULT_LOSS_FUNCTIONS = ["msssim", "ssim", "psnr"]
assert len(set(DEFAULT_LOSS_FUNCTIONS) & set(LOSS_FUNCTION_MAP.keys())) == len(set(DEFAULT_LOSS_FUNCTIONS))


def load_images(dir_: str) -> Mapping[str, str]:
    dir_ = os.path.abspath(dir_)
    files = [os.path.normpath(os.path.join(dir_, file)) for file in os.listdir(dir_)]
    accessible_files = [file for file in files if
                        os.access(file, os.R_OK) and os.path.isfile(file)]
    filtered_files = [file for file in accessible_files if os.path.splitext(file)[1].lower()
                      in ACCEPTABLE_IMAGE_EXTENSIONS]

    rel_files = [os.path.relpath(file, dir_) for file in filtered_files]
    map_files = {os.path.splitext(file)[0]: os.path.normpath(os.path.join(dir_, file))
                 for file in rel_files}

    return map_files


parser = argparse.ArgumentParser(description='batch compare multiple images')
parser.add_argument('-l', '--loss', required=False, default=None,
                    action='append', choices=LOSS_FUNCTION_MAP.keys(),
                    help='loss functions need to be used')
parser.add_argument('-s', '--src_dir', required=True, help='source directory for comparision')
parser.add_argument('-d', '--dst_dir', required=True, help='destination directory for comparision')
parser.add_argument('-m', '--map', required=False, default=None,
                    help='filename map relationship to process different names')
arguments = parser.parse_args()

if __name__ == "__main__":
    print("Comparision start, command line:", sys.argv)
    losses = sorted(list(set(arguments.loss or DEFAULT_LOSS_FUNCTIONS)))
    loss_functions = {name: LOSS_FUNCTION_MAP[name] for name in losses}

    src_dir = os.path.abspath(arguments.src_dir)
    print("Configuring source directory {src}...".format(src=repr(src_dir)))
    src_images = load_images(src_dir)
    print("Finished, {cnt} source image(s) found.".format(cnt=len(src_images.keys())))

    dst_dir = os.path.abspath(arguments.dst_dir)
    print("Configuring destination directory {dst}...".format(dst=repr(dst_dir)))
    dst_images = load_images(dst_dir)
    print("Finished, {cnt} destination image(s) found.".format(cnt=len(dst_images.keys())))

    print("Configuring relationships...")
    if arguments.map is None:
        mapping = {}
    else:
        with codecs.open(arguments.map, 'r') as f:
            _value = json.load(f)
        mapping = {item['from']: item['to'] for item in _value}
    valid_src_names = [name for name in src_images.keys()
                       if mapping.get(name, name) in dst_images.keys()]
    print("Finished, {cnt} valid relationship(s) found.".format(cnt=len(valid_src_names)))

    diffs = {name: [] for name, _ in loss_functions.items()}
    for src_name in tqdm(valid_src_names):
        src_file = src_images[src_name]
        dst_file = dst_images[mapping.get(src_name, src_name)]
        print("Comparing from {src} to {dst}...".format(src=repr(src_file), dst=repr(dst_file)))

        for loss_name, loss_function in loss_functions.items():
            loss_value = loss_function(src_file, dst_file)
            print("Loss function %s's result: %.12f" % (loss_name, loss_value))
            diffs[loss_name].append(loss_value)

        print()

    print()
    print("Comparision completed!")
    for name, result in diffs.items():
        print("%s: %.12f" % (name, sum(result) / len(result)))
