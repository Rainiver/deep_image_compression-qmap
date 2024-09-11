import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('exp_dir', default='../experiments')
args = parser.parse_args()


def _detect_latest(prefix, suffix, log_dir):
    """
    detect the latest file in log_dir with format <prefix><epoch><suffix>
    :param prefix:
    :param suffix:
    :return: epoch, if here's no checkpoints, return a negative value
    """
    checkpoints = os.listdir(log_dir)
    checkpoints = [f for f in checkpoints if f.startswith(prefix) and f.endswith(suffix)]
    checkpoints = [int(f[len(prefix):-len(suffix)]) for f in checkpoints]
    checkpoints = sorted(checkpoints)
    max_epoch = checkpoints[-1] if len(checkpoints) > 0 else None
    min_epoch = checkpoints[0] if len(checkpoints) > 0 else None
    return max_epoch, min_epoch


prefixes = [
    'opt_epoch-',
    'sch_epoch-',
    'codec_epoch-'
]

for dirpath, dirnames, filenames in os.walk(args.exp_dir):
    if 'logs' in dirpath:
        print('inspect dir {}'.format(dirpath), flush=True)
        log_dir = dirpath
    else:
        continue
    for prefix in prefixes:
        max_epoch, min_epoch = _detect_latest(prefix, '.pth', log_dir)
        if max_epoch is None:
            print('{} is None'.format(prefix), flush=True)
            continue
        for e in range(min_epoch, max_epoch - 1):
            name = prefix + str(e) + '.pth'
            path = os.path.join(log_dir, name)
            commend = 'rm {}'.format(path)
            print(commend, flush=True)
            os.system(commend)
