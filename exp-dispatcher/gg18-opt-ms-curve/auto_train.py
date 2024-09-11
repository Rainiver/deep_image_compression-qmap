import os
import argparse
import yaml
import sys

DOC = 'auto_train'

partition = 'VI_AIC_TITANXP'
base_exp_path = 'base'

config = {
    # 40 80 120
    "exps": [
        [40.],
        [80.],
        [120.],
        [3.],
        [12.],
        [24.],
        # add
        # [180.],
        # [240.],
        # [480.],
        # [960.],
        # [1920.],
    ],
    "keys": [
        ["gg18.yaml", "config", "lambda1"],
    ],
}


def main():
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

        os.system(f'cd {exp_path} && sh auto_train_helper.sh {partition} && cd ..')


if __name__ == '__main__':
    main()
