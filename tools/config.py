import argparse
import os
from collections.abc import Iterable

import yaml
from easydict import EasyDict
import warnings

required_args = [
    # (name, type),
    # (parent_name, [(child_name1, type1), (child_name2, type2)]
    ('log_dir', str),
    ('lr', float),
    ('lr_milestion', list),
    ('lr_scheduler_gamma', float),
    ('epoch', int),
    # ('show_interval', int),
    ('show_bpp_interval', int),
    # ('snapshot_interval', int),
    ('linklink', bool),
    ('load_epoch', int),
    # ('num_features_encode', int),
    # ('num_features_decode', int),
    # ('num_features_entropy_model', int),
    # ('y_encode', str),
    # ('y_decode', str),
    # ('y_quant', str),
    # ('z_encode', str),
    # ('z_decode', str),
    # ('z_quant', str),
    # ('entropy_model', str),
    # ('z_entropy_model', str),
    ('BIT', int),
    ('b_range', list),
    ('lambda1', float),
    ('lambda2', float),
    ('lambda4', float),
    ('lambda4s', list),
    ('path', list),
    ('entropy_AE', bool),
    ('compress_per_channel', bool),
    ('adapt', bool),
    ('ae_backend', str),
]


def _convert_config_args(exp_root, args: dict, required=None) -> EasyDict:
    if not required:
        required = required_args

    for name, dtype in required:
        # if name not in args:
        #     raise AssertionError('miss expected arg: ' + name)

        if isinstance(dtype, Iterable):
            # if dtype is a list-like object,
            # current argument has child nodes
            child = args[name]
            assert isinstance(child, dict)
            args[name] = _convert_config_args(exp_root, child, dtype)  # convert child args by recursion
            continue

        if name not in args:
            warnings.warn(f'arg named {name} is null')
            args[name] = dtype()

        if not isinstance(args[name], dtype):
            try:
                args[name] = dtype(eval(args[name]))
                assert isinstance(args[name], dtype)
            except Exception as e:
                raise AssertionError('{} should be an instance of{}'.format(name, dtype))

        if isinstance(args[name], str) and args[name].startswith('$EXP_ROOT'):
            args[name] = args[name].replace('$EXP_ROOT', exp_root)

    return EasyDict(args)


def load_yaml(root):
    path = os.path.join(root, 'config.yml')
    with open(path, 'r') as f:
        args = yaml.load(f.read())
    return _convert_config_args(root, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', default='./')
    args = parser.parse_args()
    res = load_yaml(args.root)
    print(res)
