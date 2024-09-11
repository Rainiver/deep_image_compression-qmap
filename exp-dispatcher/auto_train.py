import os
from argparse import ArgumentParser
import yaml
import sys
import copy
import importlib


def merge_config(ori, diff):
    r"""
    diff不是dict如list，val则直接覆写
    diff是dict且ori不是dict则覆写
    都是dict则改写或添加
    """
    ori = copy.deepcopy(ori)
    if isinstance(diff, dict):
        if isinstance(ori, dict):
            dst = ori
            for k, v in diff.items():
                dst[k] = merge_config(ori.get(k, {}), diff[k])
        else:
            dst = diff
    elif isinstance(diff, list):
        dst = diff
    else:
        dst = diff

    return dst


def exec_exp(exp_path, partition):
    os.system(f'cd {exp_path} && sh auto_train_helper.sh {partition} && cd -')
    print(f'cd {exp_path} && sh auto_train_helper.sh {partition} && cd -', flush=True)


def build_exp_config(config, merge, base_exp_path):
    exp_path_set = []
    for i, exp in enumerate(config["exps"]):
        exp_path = base_exp_path + f'_{i}'

        if os.path.exists(exp_path):
            continue

        flag_cp = os.system('cp -r {} {}'.format(base_exp_path, exp_path))

        for i_key, (exp_lambda, lambda_path) in enumerate(zip(exp, config["keys"])):
            yml_path = os.path.join(exp_path, lambda_path[0])
            with open(yml_path, 'r') as yaml_file_r:
                d = yaml.load(yaml_file_r.read())

            d_item = d
            for dir_in_yaml in lambda_path[1:-1]:
                d_item = d_item[dir_in_yaml]

            d_item[lambda_path[-1]] = exp_lambda

            with open(yml_path, 'w') as yaml_file_w:
                yaml.dump(d, yaml_file_w)

        for idx, merge_info in enumerate(merge):
            ori_path = merge_info[0]
            diff_path = os.path.join(exp_path, merge_info[1])
            dst_path = os.path.join(exp_path, merge_info[2])
            with open(ori_path, 'r') as ori_file_r:
                ori = yaml.load(ori_file_r.read())

            with open(diff_path, 'r') as diff_file_r:
                diff = yaml.load(diff_file_r.read())

            dst = merge_config(ori, diff)

            with open(dst_path, 'w') as dst_file_w:
                yaml.dump(dst, dst_file_w)

        exp_path_set.append(exp_path)

    return exp_path_set


def main(args):
    sys.path.insert(0, args.root)
    config = importlib.import_module('config').__dict__.get('config', {
        "exps": [
            [],
        ],
        "keys": [
        ],
    })
    merge = importlib.import_module('config').__dict__.get('merge', [])
    exp_path_set = build_exp_config(config, merge, os.path.join(args.root, args.base))
    for idx, path in enumerate(exp_path_set):
        exec_exp(path, args.partition)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--root', '-r', type=str, required=True,
                        help='the directory of experiments')
    parser.add_argument('--partition', '-p', type=str, default='VI_AIC_TITANXP')
    parser.add_argument('--base', '-b', type=str, default='base')
    args = parser.parse_args()
    main(args)
