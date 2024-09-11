import sys
import os
import logging
from utils.distributed_utils import get_rank

logs = set()


def rank_0_print(s):
    if get_rank() == 0:
        print(s)


def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return

    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(level)

    rank = 0
    if 'SLURM_PROCID' in os.environ:
        # only print log for rank 0
        rank = int(os.environ['SLURM_PROCID'])
        logger.addFilter(lambda record: rank == 0)

    format_str = f'%(asctime)s-rk{rank}-%(filename)s#%(lineno)d:%(message)s'
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
