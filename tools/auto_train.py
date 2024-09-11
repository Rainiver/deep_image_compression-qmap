import os
import argparse
import yaml
import sys

DOC = 'auto_train'

global_config = {
    'psnr': {
        'lambda': [[0., 0.01, 0.01], [0., 0.01, 0.2], [0., 0.01, 0.7],
                   [0., 0.01, 1.], [0., 0.01, 20.], [0., 0.01, 4.],
                   [0., 0.01, 8.], [0., 0.01, 0.1], [0., 0.01, 13.],
                   [0., 0.01, 2.]],
        'dir_path': 'auto_Train',
        'key': ['lambda1', 'lambda2', 'lambda4'],
    },
    'msssim': {
        'lambda': [[1000., 0., 0.01], [1000., 0., 50.], [1000., 0., 100.],
                   [1000., 0., 400.], [1000., 0., 1.], [1000., 0., 8.],
                   [1000., 0., 20.], [1000., 0., 30.], [1000., 0., 50.],
                   [1000., 0., 200.], [1000., 0., 0.2], [1000., 0., 4.],
                   [1000., 0., 13.]],
        'dir_path': 'auto_Train_ms',
        'key': ['lambda1', 'lambda2', 'lambda4'],
    },
    'hybrid': {
        'lambda': [[1000., 0.004, 0.01], [1000., 0.004, 50.], [1000., 0.004, 100.],
                   [1000., 0.004, 400.], [1000., 0.004, 1.], [1000., 0.004, 8.],
                   [1000., 0.004, 20.], [1000., 0.004, 30.], [1000., 0.004, 50.],
                   [1000., 0.004, 200.], [1000., 0.004, 0.2], [1000., 0.004, 4.],
                   [1000., 0.004, 13.]],
        'dir_path': 'auto_Train_hybrid',
        'key': ['lambda1', 'lambda2', 'lambda4'],
    },
    'grad': {
        'lambda': [
            # [1000., 0.004, 400.],
            # [1000., 0.004, 200.],
            # [1000., 0.004, 100.],
            # [1000., 0.004, 50.],
            # [1000., 0.004, 30.],
            [1000., 0.04, 20.],
            [1000., 0.04, 13.],
            [1000., 0.04, 8.],  # id5, key point
            [1000., 0.04, 4.],
            # [1000., 0.004, 1.],
            # [1000., 0.004, 0.2],
            # [1000., 0.004, 0.01],  # id1, biggest bpp
        ],
        'dir_path': 'auto_Train_grad',
        'key': ['lambda1', 'lambda_grad_simple_v2', 'lambda4'],
    },
    'grad_mse': {
        'lambda': [
            [1000., 0.04, 400.,0.4],
            [1000., 0.04, 200.,0.4],
            [1000., 0.04, 100.,0.4],
            [1000., 0.04, 50.,0.4],
            [1000., 0.04, 30.,0.4],
            [1000., 0.04, 20.,0.4],
            [1000., 0.04, 13.,0.4],
            [1000., 0.04, 8.,0.4],  # id5, key point
            [1000., 0.04, 4.,0.4],
            [1000., 0.04, 1.,0.4],
            [1000., 0.04, 0.2,0.4],
            [1000., 0.04, 0.01,0.4],  # id1, biggest bpp
        ],
        'dir_path': 'auto_Train_grad_mse',
        'key': ['lambda1', 'lambda_grad_simple_v2', 'lambda4' , 'lambda2'],
    },
    'grad_rmse': {
        'lambda': [
            [1000., 0.04, 400.,0.,0.4],
            [1000., 0.04, 200.,0.,0.4],
            [1000., 0.04, 100.,0.,0.4],
            [1000., 0.04, 50.,0.,0.4],
            [1000., 0.04, 30.,0.,0.4],
            [1000., 0.04, 20.,0.,0.4],
            [1000., 0.04, 13.,0.,0.4],
            [1000., 0.04, 8.,0.,0.4],  # id5, key point
            [1000., 0.04, 4.,0.,0.4],
            [1000., 0.04, 1.,0.,0.4],
            [1000., 0.04, 0.2,0.,0.4],
            [1000., 0.04, 0.01,0.,0.4],  # id1, biggest bpp
        ],
        'dir_path': 'auto_Train_grad_rmse',
        'key': ['lambda1', 'lambda_grad_simple_v2', 'lambda4' , 'lambda2','lambda2_1'],
    },
    '0.2.3': {
        'lambda': [
            [1000., 400.,0.4],
            [1000., 200.,0.4],
            [1000., 100.,0.4],
            [1000., 50.,0.4],
            [1000., 30.,0.4],
            [1000., 20.,0.4],
            [1000., 13.,0.4],
            [1000., 8.,0.4],  # id5, key point
            [1000., 4.,0.4],
            [1000., 1.,0.4],
            [1000., 0.2,0.4],
            [1000., 0.01,0.4],  # id1, biggest bpp
        ],
        'dir_path': 'auto_Train_0.2.3',
        'key': ['lambda1','lambda4' , 'lambda2'],
    },
    'cheng20mse': {
        'lambda': [
            [0., 1., 0.0008],  # 0
            [0., 1., 0.0016],  # 1
            [0., 1., 0.0032],  # 2
            [0., 1., 0.0075],  # 3
            [0., 1., 0.015],   # 4
            [0., 1., 0.03],    # 5
            [0., 1., 0.045],   # 6
            [0., 1., 0.08],    # 7
        ],
        'vars': [
            [128 * i for i in (1, 2, 4, 6, 9)],
            [128 * i for i in (1, 2, 4, 6, 9)],
            [128 * i for i in (1, 2, 4, 6, 9)],
            [128 * i for i in (1, 2, 4, 6, 9)],
            [192 * i for i in (1, 2, 4, 6, 9)],
            [192 * i for i in (1, 2, 4, 6, 9)],
            [192 * i for i in (1, 2, 4, 6, 9)],
            [192 * i for i in (1, 2, 4, 6, 9)],
        ],
        'dir_path': 'auto_Train_cheng20mse',
        'key': ['lambda1', 'lambda4', 'lambda2'],
        'key_vars': ['C', '2C', '4C', '6C', '9C']
    },
    'gg18mse': {
        'lambda': [
            [0., 1., 0.0008],  # 0
            [0., 1., 0.0016],  # 1
            [0., 1., 0.0032],  # 2
            [0., 1., 0.0075],  # 3

            [0., 1., 0.015],   # 4
            [0., 1., 0.03],    # 5
            [0., 1., 0.045],   # 6
            [0., 1., 0.08],    # 7
        ],
        'vars': [
            [128, 192],
            [128, 192],
            [128, 192],
            [128, 192],

            [192, 320],
            [192, 320],
            [192, 320],
            [192, 320],
        ],
        'dir_path': 'auto_Train_gg18mse',
        'key': ['lambda1', 'lambda4', 'lambda2'],
        'key_vars': ['N', 'M']
    },
    'all192mse': {
        'lambda': [
            [0., 1., 0.0008],  # 0
            [0., 1., 0.0016],  # 1
            [0., 1., 0.0032],  # 2
            [0., 1., 0.0075],  # 3

            [0., 1., 0.015],   # 4
            [0., 1., 0.03],    # 5
            [0., 1., 0.045],   # 6
            [0., 1., 0.08],    # 7
        ],
        'vars': [
            [192, 320],
            [192, 320],
            [192, 320],
            [192, 320],
            [192, 320],
            [192, 320],
            [192, 320],
            [192, 320],
        ],
        'dir_path': 'auto_Train_gg18mse',
        'key': ['lambda1', 'lambda4', 'lambda2'],
        'key_vars': ['N', 'M']
    }

}


def arg_parser():
    parser = argparse.ArgumentParser(description=DOC)
    parser.add_argument('-tp', '--loss_weight_parameter_type',
                        choices=global_config.keys(), default='psnr', type=str, required=True,
                        help='Enter the loss type')
    parser.add_argument('-pi', '--input_dir_name',
                        type=str, required=True,
                        help='auto_train dir_path_name')  # GG18_CENIC
    parser.add_argument('-dp', '--use_default_cfg_dir',
                        type=bool, default=True,
                        help='default cfg dir prefix ../experiments else none')
    parser.add_argument('-re', '--restore_dir',
                        type=str, default='../experiments',
                        help='default exp dir ../experiments else input')
    parser.add_argument('--range', type=str, default='*',
                        help='range of exp index, a python expression or "*"')
    parser.add_argument('--partition', '-P', type=str, default='spring_scheduler')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        return False
    else:
        return True


def main(args):
    base_config = os.path.join('../experiments' if args.use_default_cfg_dir else '', args.input_dir_name, 'config.yml')
    exp_config = global_config[args.loss_weight_parameter_type]

    exp_config_lambda = exp_config['lambda']
    n_exp = len(exp_config_lambda)  # number of experiments

    range = args.range
    if range == '*':
        range = range(n_exp)
    else:
        range = eval(range)

    if 'key_vars' in exp_config:
        exp_config_vars = exp_config['vars']
        assert n_exp == len(exp_config_vars)
    else:
        exp_config_vars = None

    dir_path = os.path.join(args.restore_dir, exp_config['dir_path'])
    _ = check_dir(dir_path)

    for exp_ind, lambda_values in enumerate(exp_config_lambda):
        if exp_ind not in range:
            continue

        sub_dir_name = args.input_dir_name + '_{}'.format(exp_ind)

        # ./experiments/auto_Train/exp/exp_0
        sub_dir_path = os.path.join(dir_path, args.input_dir_name, sub_dir_name)
        flag = check_dir(sub_dir_path)
        # 如果sub_dir_path这个目录已经存在了，略过
        if not flag:
            '''
            cp base config到每一个sub_dir_path内，并按照lambda_item修改其中的lambda1 2 4
            '''
            os.system('cp {} {}'.format(base_config, sub_dir_path))
            sub_dir_path_yml = os.path.join(sub_dir_path, 'config.yml')

            with open(sub_dir_path_yml, 'r') as f:
                d = yaml.load(f.read())

            # update lambda values
            lambda_keys = exp_config['key']
            assert len(lambda_keys) == len(lambda_values)
            for k, v in zip(lambda_keys, lambda_values):
                d[k] = v

            # update variable values
            if exp_config_vars is not None:
                vars_values = exp_config_vars[exp_ind]
                vars_keys = exp_config['key_vars']
                assert len(vars_keys) == len(vars_values)
                for k, v in zip(vars_keys, vars_values):
                    d['vars'][k] = v

            with open(sub_dir_path_yml, 'w') as f:
                yaml.dump(d, f)
            os.system(f'sh auto_train_helper.sh {args.partition} {sub_dir_path}')
        if resume_all:
            os.system(f'sh auto_train_helper.sh {args.partition} {sub_dir_path}')


if __name__ == '__main__':
    args = arg_parser()
    resume_all = False
    main(args)
    
    # python3 auto_train.py -tp psnr -pi GG18_CENIC_quant_mixq_dequant --range '[0,1,2]'
