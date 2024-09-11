import torch
import numpy as np
from tqdm import tqdm

precison = 15

def _get_tails(cmf, pmf_type, *args):
    return (-64, 63) if pmf_type == "factorized" else (-64, 63)


def get_table_factorized(pmf, dummy, base=2**precison):
    CHANNEL_SIZE = pmf.channel_size
    len_table = len(dummy)
    dummy = dummy.reshape(1, 1, 1, len_table).repeat(1, CHANNEL_SIZE, 1, 1)
    table = pmf(dummy).reshape(CHANNEL_SIZE, len_table)
    table = table.cpu()
    table = pmf_to_quantized_prob_table(table, base).numpy()

    return table


def get_table_mean_scale(pmf, dummy, base=2**precison):
    SCALES_MIN = 0.11
    SCALES_MAX = 256
    SCALES_LEVELS = 64

    scale_table = list(np.exp(np.linspace(
        np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS)))

    MEANS_MIN = 0.01
    MEANS_MAX = 64
    MEANS_LEVELS = 256

    mean_table = \
        list(-np.exp(np.linspace(
            np.log(MEANS_MIN), np.log(MEANS_MAX), MEANS_LEVELS)))[::-1] + \
        [0] + \
        list(np.exp(np.linspace(
            np.log(MEANS_MIN), np.log(MEANS_MAX), MEANS_LEVELS)))

    _min, _max = dummy.min().int().item(), dummy.max().int().item()
    scale_factor = 1  # deprecated
    len_range_table = len(dummy)
    len_scale_table = len(scale_table)
    len_mean_table = len(mean_table)
    len_params_table = len_scale_table * len_mean_table

    # each channel for each sigma
    y_range_map = np.array([list(range(_min, _max + 1))] * len_params_table) \
        .reshape([1, len_params_table, 1, len_range_table])
    y_range_rescaled_map = y_range_map / scale_factor

    scale_map = np.repeat(scale_table, len_range_table * len_mean_table) \
        .reshape([1, len_params_table, 1, len_range_table])
    mean_map = np.tile(np.repeat(mean_table, len_range_table), len_scale_table) \
        .reshape([1, len_params_table, 1, len_range_table])

    y_range_rescaled_map = torch.tensor(y_range_rescaled_map).float()
    scale_map = torch.tensor(scale_map).float()
    mean_map = torch.tensor(mean_map).float()

    y_range_rescaled_map = y_range_rescaled_map.to(dummy.device)
    scale_map = scale_map.to(dummy.device)
    mean_map = mean_map.cuda(dummy.device)

    # now we can get a [LEN_SCALE_TABLE X LEN_Y_SCALED] shaped prob table
    # and we will compress data according to this
    y_prob_table_per_param = pmf(y_range_rescaled_map, mean_map, scale_map)
    y_prob_table = y_prob_table_per_param.reshape([len_params_table, len_range_table])
    y_prob_table = y_prob_table.cpu()
    y_prob_table = pmf_to_quantized_prob_table(y_prob_table, base).numpy()
    return y_prob_table


def get_table_scale_only(pmf, dummy, base=2**precison):
    SCALES_MIN = 0.11
    SCALES_MAX = 256
    SCALES_LEVELS = 64

    scale_table = list(np.exp(np.linspace(
        np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS)))

    _min, _max = dummy.min().int().item(), dummy.max().int().item()
    scale_factor = 1  # deprecated
    len_range_table = len(dummy)
    len_scale_table = len(scale_table)

    # each channel for each sigma
    y_range_map = np.array([list(range(_min, _max + 1))] * len_scale_table) \
        .reshape([1, len_scale_table, 1, len_range_table])
    y_range_rescaled_map = y_range_map / scale_factor
    scale_map = np.repeat(scale_table, len_range_table) \
        .reshape([1, len_scale_table, 1, len_range_table])

    y_range_rescaled_map = torch.tensor(y_range_rescaled_map).float()
    scale_map = torch.tensor(scale_map).float()
    if torch.cuda.is_available():
        y_range_rescaled_map = y_range_rescaled_map.cuda()
        scale_map = scale_map.cuda()

    # now we can get a [LEN_SCALE_TABLE X LEN_Y_SCALED] shaped prob table
    # and we will compress data according to this
    y_prob_table_per_sigma = pmf(y_range_rescaled_map, 0., scale_map)
    y_prob_table = y_prob_table_per_sigma.reshape([len_scale_table, len_range_table])
    y_prob_table = y_prob_table.cpu()
    y_prob_table = pmf_to_quantized_prob_table(y_prob_table, base).numpy()
    return y_prob_table

def pmf_to_quantized_prob_table(pmf, base):
    '''
    :param pmf: probability mass function
    :param base: sum of frequency of quantized table(2**precison)
    :return : quantized prob table
    '''
    def normalize_table(table):
        '''
        normalize table to base
        reference from PmfToQuantizedCdf op of tensorflow compression
        '''
        sum = table.sum()
        len = table.shape[0]

        if sum > base:
            i = 0
            index = torch.argsort(table)
            while sum > base:
                value = table[index[i]]
                if value == 1:
                    i += 1
                    continue
                else:
                    table[index[i]] = value - 1
                    sum -= 1
                    j = i + 1
                    # find next greater j
                    while j < len and table[index[j]] <= table[index[i]]:
                        j += 1
                    if j == len: continue
                    # find succeed, rotate index[i:j+1] once to left
                    index[i:j], index[j] = index[i + 1:j + 1].clone(), index[i].clone()
        elif sum < base:
            i = 0
            index = torch.argsort(table, descending=True)
            while sum < base:
                value = table[index[i]]
                table[index[i]] = value + 1
                sum += 1
                j = i + 1
                # find next lesser j
                while j < len and table[index[j]] >= table[index[i]]:
                    j += 1

                if j == len: continue
                # find succeed, rotate index[i:j+1] once to left
                index[i:j], index[j] = index[i + 1:j + 1].clone(), index[i].clone()
        # verify table
        assert table.min() >= 1, "frequency table can not have zero value!"
        assert table.sum() == base, "sum of table must be equal to 2**precision!"

    prob_table = (pmf * base).int()
    prob_table[prob_table < 1] = 1
    len_scale_table = prob_table.shape[0]
    for i in tqdm(range(len_scale_table)):
        normalize_table(prob_table[i])

    return prob_table

def get_named_prob_table(pmf, pmf_type, get_tail_fn=_get_tails, get_tail_fn_args=None, device=None):
    if device is None:
        device = torch.cuda.current_device()
    if get_tail_fn_args is None:
        get_tail_fn_args = []
    tails = get_tail_fn(pmf, pmf_type, *get_tail_fn_args)
    dummy = torch.arange(tails[0], tails[1] + 1, device=device, dtype=torch.float)
    get_table_fn_map = {
        'scale-only': get_table_scale_only,
        'mean-scale': get_table_mean_scale,
        'factorized': get_table_factorized,
    }
    table = get_table_fn_map[pmf_type](pmf, dummy)
    return table, tails


def read_table(path):
    """
    read .dip format prob table file and yield (table_name, table, tails) tuples
    :param path: path of .dip file
    :return: generator of (table_name, table, tails) tuples
    """
    with open(path, 'r') as f:
        buffer = f.read()
    buffer = buffer.strip().split(' ')

    while len(buffer) > 0:
        table_name, buffer = buffer[0], buffer[1:]

        def _int(str_list):
            try:
                return list(map(int, str_list))
            except:
                raise ValueError(f'{table_name}: cannot convert given data to int list')

        print('reading table:', table_name)
        tails, buffer = _int(buffer[:2]), buffer[2:]
        shape, buffer = _int(buffer[:2]), buffer[2:]
        table_len = shape[0] * shape[1]
        table, buffer = _int(buffer[:table_len]), buffer[table_len:]
        print('tails:', tails, 'shape:', shape)

        table = np.array(table, dtype=np.int)
        table = np.reshape(table, shape)

        yield table_name, table, tails
