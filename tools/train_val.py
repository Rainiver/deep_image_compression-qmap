"""train_rt.py is the old script for real time
   this one intends to be a configurable train script, maybe should work with an additional config file?
   namely, change exp settings should not change the train.py
"""
from argparse import ArgumentParser

from train_val_helper import main as train_val

from tools.config import load_yaml


def main(args, **kwargs):
    if 'verbose' in kwargs and kwargs['verbose']:
        print(args)

    train_val(args, **kwargs)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='the directory of experiment')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--test_quality_entropy', default=False, action='store_true')
    parser.add_argument('--test_real_bpp', default=False, action='store_true')
    parser.add_argument('--test_time', default=False, action='store_true')
    parser.add_argument('--test_lambda4_id', type=int, default=0)
    parser.add_argument('--test_b', type=float, default=0.)
    parser.add_argument('--to_caffe', default=False, action='store_true')

    _args = parser.parse_args()
    train_args = load_yaml(_args.root)

    kwargs = {k: getattr(_args, k) for k in [
        'test_quality_entropy',
        'test_real_bpp',
        'verbose',
        'test_lambda4_id',
        'test_b',
        'test_time',
        'to_caffe'
    ]}
    main(train_args, **kwargs)
