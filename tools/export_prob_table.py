from argparse import ArgumentParser
import numpy as np
import torch

from codes.prob_table_utils import get_named_prob_table
from nets.entropy import *

from config import load_yaml
from nets.model_builder import model_builder
from pipelines.models_helper import get_codec
from train_val_helper import load


def export_prob_table(configs):
    models = model_builder(configs)
    codec = get_codec(configs.codec, models)
    load({'codec': codec}, None, None, configs.log_dir, -1, 0, 0)

    if configs.device == 'gpu' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    configs.device = device
    print('device:', device)
    print('entropy model names:', configs.entropy_models)

    # configs.entropy_models: [model_name:model_type]
    entropy_models = map(lambda x: x.split(':'), configs.entropy_models)
    entropy_models = {name: (models[name].to(device), type) for name, type in entropy_models}
    tables = {name: get_named_prob_table(*model, device=device) for name, model in entropy_models.items()}
    print(tables)

    with open(configs.output, 'w') as f:
        for name, (table, tails) in tables.items():
            print(name)
            f.write(name)
            f.write(' ')

            print(tails)
            f.write(str(tails[0]))
            f.write(' ')
            f.write(str(tails[1]))
            f.write(' ')

            print(table.shape)
            f.write(str(table.shape[0]))
            f.write(' ')
            f.write(str(table.shape[1]))
            f.write(' ')
            f.write(' '.join(map(str, table.flatten())))
            f.write(' ')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='the directory of experiment')
    parser.add_argument('--entropy-models', '-M', nargs='+',
                        help='name(s) of entropy model(s), e.g. entropy_pre:scale-only z_entropy_pre:factorized')
    parser.add_argument('--device', default='gpu', choices=['cpu', 'gpu'],
                        help='device where the entropy be calculated.')
    parser.add_argument('--tail-mass', type=float, default=2 ** -8)
    parser.add_argument('--output', '-O', type=str, default='prob_table.dip')
    args = parser.parse_args()
    configs = load_yaml(args.root)
    for aname, avalue in args.__dict__.items():
        configs[aname] = avalue
    export_prob_table(configs)
