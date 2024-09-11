import ctypes
import numpy as np
import os

try:
    libc = ctypes.cdll.LoadLibrary(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'libae.so'))
except OSError:
    raise OSError('cannot open libae.so, please try to compile it using Makefile under codes/cc')


# int compress(char *file_dir,int **p_tabel,int table_sum,int table_len,char *a,int *index,int len,bool adaptive);
def cc_compress(data, save_path, prob_table, index, adapt, append, **kwargs):
    prob_table = np.array(prob_table)
    assert prob_table.ndim == 2
    num_table = prob_table.shape[0]
    table_len = prob_table.shape[1]
    data = list(np.array(data).flatten())
    index = list(np.array(index).flatten())
    assert min(data) == 0  # TODO: check here
    # assert len(data) == len(index)

    save_path_c = (ctypes.c_char * (len(save_path) + 1))(*(save_path.encode() + '\0'.encode()))
    data_c = (ctypes.c_int * len(data))(*data)
    index_c = (ctypes.c_int * len(index))(*index)
    data_len = ctypes.c_int(len(data))
    num_table_c = ctypes.c_int(num_table)
    table_len_c = ctypes.c_int(table_len)
    ptable_c = (ctypes.c_int * (table_len * num_table))(*prob_table.flatten())
    adapt_c = ctypes.c_bool(adapt)
    append_c = ctypes.c_bool(append)

    if kwargs == {}:
        r = libc.compress(save_path_c,
                          ptable_c, num_table_c, table_len_c,
                          data_c, index_c, data_len,
                          adapt_c, append_c)
    else:
        omega_table = list(np.array(kwargs['omega_t']).flatten())
        omega_table_len = len(omega_table)
        gmm_k = len(index) // len(data)
        assert len(index) % len(data) == 0
        omega_table_c = (ctypes.c_int * omega_table_len)(*omega_table)
        omega_table_len_c = ctypes.c_int(omega_table_len)
        gmm_k_c = ctypes.c_int(gmm_k)
        r = libc.compress_gmm(save_path_c,
                          ptable_c, num_table_c, table_len_c,
                          omega_table_c, omega_table_len_c,
                          data_c, data_len,
                          index_c, gmm_k_c,
                          adapt_c, append_c)

    assert r == 0


# int decompress(char *file_dir,int offset,int *p_tabel,int table_sum,int table_len,int *a,int *index,int len,bool adaptive)
def cc_decompress(save_path, prob_table, index, adapt, offset):
    prob_table = np.array(prob_table)
    assert prob_table.ndim == 2
    num_table = prob_table.shape[0]
    table_len = prob_table.shape[1]

    index = list(np.array(index).flatten())

    save_path_c = (ctypes.c_char * (len(save_path) + 1))(*(save_path.encode() + '\0'.encode()))
    index_c = (ctypes.c_int * len(index))(*index)
    data_len = len(index)
    data_len_c = ctypes.c_int(data_len)
    data_buf_c = ctypes.c_buffer(data_len * 4)
    num_table_c = ctypes.c_int(num_table)
    table_len_c = ctypes.c_int(table_len)
    ptable_c = (ctypes.c_int * (table_len * num_table))(*prob_table.flatten())
    adapt_c = ctypes.c_bool(adapt)
    offset_c = ctypes.c_int(offset)

    r = libc.decompress(save_path_c, offset_c, ptable_c, num_table_c, table_len_c,
                        data_buf_c, index_c, data_len_c,
                        adapt_c)
    assert r == 0

    data_res_c = ctypes.cast(data_buf_c, ctypes.POINTER(ctypes.c_int))
    return [data_res_c[i] for i in range(data_len)]


if __name__ == '__main__':
    from codes.AE.adaptive_arithmetic_compress import compress_with_index
    from codes.AE.adaptive_arithmetic_decompress import decompress_with_index

    import time


    def test(adapt):
        prob_table = [[125, 250, 500, 125, 1], [12, 25, 50, 12, 1], [31, 41, 51, 1, 1]]
        data = np.array([1, 2, 0, 2, 3, 1, 2, 2]).reshape([1, 2, 2, 2])
        index = np.array([0, 1] * 4).reshape([1, 2, 2, 2])

        compress_with_index(data, index, 'test_cc.bin', adapt, prob_table, backend='cc')
        dec = decompress_with_index('test_cc.bin', index, adapt, prob_table, backend='cc')
        assert (dec == data).all()

        compress_with_index(data, index, 'test_py.bin', adapt, prob_table, backend='py')
        size_cc = os.path.getsize('test_cc.bin')
        size_py = os.path.getsize('test_py.bin')
        assert size_cc > 0
        print('cc:', size_cc, 'py:', size_py)


    def test_large(adapt):
        prob_table = [[12500000, 25465476, 34, 12500000, 10000000],
                      [12500000, 25000000, 123, 12500000, 10000000],
                      [31, 41, 51, 1, 1]]
        data = np.array([1, 2, 0, 2, 3, 1, 2, 2] * 100).reshape([1, 200, 2, 2])
        index = np.array([0, 1] * 4 * 100).reshape([1, 200, 2, 2])

        t = time.time()
        compress_with_index(data, index, 'test_cc.bin', adapt, prob_table, backend='cc')
        cc_duration = time.time() - t

        t = time.time()
        dec = decompress_with_index('test_cc.bin', index, adapt, prob_table, backend='cc')
        cc_dec_duration = time.time() - t
        assert (dec == data).all()

        t = time.time()
        compress_with_index(data, index, 'test_py.bin', adapt, prob_table, backend='py')
        py_duration = time.time() - t

        t = time.time()
        print(prob_table)
        dec = decompress_with_index('test_py.bin', index, adapt, prob_table, backend='py')
        py_dec_duration = time.time() - t
        assert (dec == data).all()

        print('compressing time: cc: {}s\tpy : {}s'.format(cc_duration, py_duration))
        print('decompressing time: cc: {}s\tpy : {}s'.format(cc_dec_duration, py_dec_duration))

        size_cc = os.path.getsize('test_cc.bin')
        size_py = os.path.getsize('test_py.bin')
        assert size_cc > 0
        print('cc:', size_cc, 'py:', size_py)


    test(True)
    test(False)
    test_large(True)
    test_large(False)
