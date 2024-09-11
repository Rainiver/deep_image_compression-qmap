import torch
import numpy as np
import pickle
from argparse import ArgumentParser

from pprint import pprint
import io


def trans(x):
    if isinstance(x, torch.Tensor):
        return x.size()
    elif isinstance(x, np.ndarray):
        return x.shape
    elif isinstance(x, dict):
        if all([isinstance(i, int) for i in x.keys()]):
            x = {
                k: trans(v) for i, (k, v) in enumerate(x.items()) if i == 0
            }
            x.update({'...': '...'})
            return x
        else:
            return {k: trans(v) for k, v in x.items()}
    elif isinstance(x, list):
        if all([isinstance(i, int) for i in x]):
            return [trans(x[0]), '...']
        else:
            return [trans(i) for i in x]
    else:
        return x


def pretty_print_pth(x):
    pprint(trans(x))


def print_pth(x: dict, pre=""):
    if isinstance(x, torch.Tensor):
        print(f"{x.size()}")
    elif isinstance(x, np.ndarray):
        print(f"{x.shape}")
    elif isinstance(x, dict):
        print()
        for k, v in x.items():
            print(f"{pre + '    '}{k}: ", end='')
            print_pth(v, pre + '    ' + '    ')
    elif isinstance(x, list):
        print()
        for item in x:
            print(f"{pre + '  - '}", end="")
            print_pth(item, pre + "    ")
            print("")
    else:
        print(f"{x}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--dir", type=str, required=True, help="dir of the ckpt")
    args = parser.parse_args()

    pretty_print_pth(torch.load(args.dir))

    # print_pth(torch.load(args.dir))
