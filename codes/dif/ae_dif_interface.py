import ctypes
import numpy as np
import os

try:
    libc = ctypes.cdll.LoadLibrary(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'libcoder.so'))
except OSError:
    try:
        libc = ctypes.cdll.LoadLibrary(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'libcoder.dll'))
    except OSError:
        raise OSError('cannot open libcoder.so/libcoder.dll, please try to access it')


def dif_compress(data, data_type, save_path, prob_table, table_min, index, adapt, append, **kwargs):
    prob_table = np.array(prob_table)
    assert prob_table.ndim == 2
    num_table = prob_table.shape[0]
    table_len = prob_table.shape[1]
    table_max = table_min + table_len - 1
    assert data.ndim == 4
    assert data.shape[0] == 1
    data = list(np.array(data).flatten())
    index = list(np.array(index).flatten())
    assert len(data) == len(index)
    assert len(data_type) == 1

    save_path_c = (ctypes.c_char * (len(save_path) + 1))(*(save_path.encode() + '\0'.encode()))
    data_type_c = ctypes.c_char(data_type.encode())
    data_c = (ctypes.c_int * len(data))(*data)
    data_len_c = ctypes.c_int(len(data))
    index_c = (ctypes.c_int * len(index))(*index)
    data_min_c = ctypes.c_int(min(data))
    data_max_c = ctypes.c_int(max(data))

    table_c = (ctypes.c_int * (table_len * num_table))(*prob_table.flatten())
    num_table_c = ctypes.c_int(num_table)
    table_min_c = ctypes.c_int(table_min)
    table_max_c = ctypes.c_int(table_max)

    adapt_c = ctypes.c_bool(adapt)
    append_c = ctypes.c_bool(append)

    if kwargs == {}:
        r = libc.compress(save_path_c, data_type_c,
                          data_c, data_len_c, index_c,
                          table_c, num_table_c, table_min_c,
                          table_max_c, data_min_c, data_max_c, adapt_c, append_c)
    else:
        raise NotImplementedError('dif ae backend have not support GMM yet')

    assert r == 0