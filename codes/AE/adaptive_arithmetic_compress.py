# 
# Compression application using adaptive arithmetic coding
# 
# Usage: python adaptive-arithmetic-compress.py InputFile OutputFile
# Then use the corresponding adaptive-arithmetic-decompress.py application to recreate the original input file.
# Note that the application starts with a flat frequency table of 257 symbols (all set to a frequency of 1),
# and updates it after each byte encoded. The corresponding decompressor program also starts with a flat
# frequency table and updates it after each byte decoded. It is by design that the compressor and
# decompressor have synchronized states, so that the data can be decompressed properly.
# 
# Copyright (c) Project Nayuki
# 
# https://www.nayuki.io/page/reference-arithmetic-coding
# https://github.com/nayuki/Reference-arithmetic-coding
# 

import os
import contextlib, sys
import numpy as np
import struct
from codes.AE import arithmeticcoding

verbose = False
cc_compress = None
dif_compress = None

python3 = sys.version_info.major >= 3


# Command line main application function.
def main(args):
    # Handle command line arguments
    if len(args) != 2:
        sys.exit("Usage: python adaptive-arithmetic-compress.py InputFile OutputFile")
    inputfile, outputfile = args

    # Perform file compression
    with open(inputfile, "rb") as inp, \
            contextlib.closing(arithmeticcoding.BitOutputStream(open(outputfile, "wb"))) as bitout:
        compress(inp, bitout)


"""
def compress(inp, bitout):
    initfreqs = arithmeticcoding.FlatFrequencyTable(257)
    freqs = arithmeticcoding.SimpleFrequencyTable(initfreqs)
    enc = arithmeticcoding.ArithmeticEncoder(32, bitout)
    while True:
        # Read and encode one byte
        symbol = inp.read(1)
        if len(symbol) == 0:
            break
        symbol = symbol[0] if python3 else ord(symbol)
        enc.write(freqs, symbol)
        freqs.increment(symbol)
    enc.write(freqs, 256)  # EOF
    enc.finish()  # Flush remaining code bits
"""


def save_int(f, x, bytes):
    s = struct.pack('i', x)[0:bytes]
    f.write(s)


def save_shape(f, shape):
    shape_len = len(shape)
    save_int(f, shape_len, 1)
    for i in range(0, shape_len):
        save_int(f, shape[i], 2)


# 8bits: shape_len -> {16bits: shape[x]} -> 16bits: table len -> 16bits: Min
# x: numpy array(to be compressed)
# output_file : string(filename to be written)
# prob_table
def compress(x: np.ndarray, output_file, per_channel=False, adapt=True, prob_table=None, test=False):
    """
    compress the given encoded features

    :param x: numpy array(to be compressed). shape should be [N, C, W, H]
    :param output_file: string(filename to be written)
    :param per_channel:
    :param adapt:
    :param prob_table: shape should be [C, L] where L denotes table length
    :return:
    """
    x = np.array(x)
    if not test:
        assert x.ndim == 4
        assert x.shape[0] == 1  # currently can compress only one image per time

    channels = x.shape[1] if per_channel else 1

    Min = np.min(x)
    Max = np.max(x)
    x -= Min
    shape = x.shape
    shape_max = max(shape)
    table_len = Max - Min + 2
    # print('table_len {}'.format(table_len))
    f = open(output_file, 'wb')
    save_shape(f, shape)
    save_int(f, table_len, 2)
    save_int(f, Min, 2)
    if verbose:
        print("compressing!!! shape:{}".format(shape), table_len, Min)

    f.close()
    x_all = x

    with contextlib.closing(arithmeticcoding.BitOutputStream(open(output_file, "ab+"))) as bitout:
        enc = arithmeticcoding.ArithmeticEncoder(32, bitout)
        for c in range(channels):
            if per_channel:
                x = x_all[:, c:c + 1, ...]
            if prob_table is not None:
                initfreqs = arithmeticcoding.EntropyFrequencyTable(table_len, prob_table[c])
            else:
                initfreqs = arithmeticcoding.FlatFrequencyTable(table_len)
            freqs = arithmeticcoding.SimpleFrequencyTable(initfreqs)
            x = x.ravel('C')
            l = len(x)
            for i in range(0, l):
                enc.write(freqs, x[i])
                if adapt:
                    freqs.increment(x[i])
            enc.write(freqs, table_len - 1)  # note this is EOF
        enc.finish()


def compress_with_index(x: np.ndarray, index: np.ndarray, output_file,
                        adapt=True, prob_table=None, test=False, backend='py', shift_to_zero=True, **kwargs):
    x = np.array(x)
    index = np.array(index, dtype=int)
    if not test:
        assert x.ndim == 4
        assert x.shape[0] == 1  # currently can compress only one image per time
        # assert x.shape == index.shape

    _min = np.min(x)
    _max = np.max(x)
    if shift_to_zero:
        x -= _min
    shape = x.shape
    table_len = _max - _min + 2
    if prob_table is not None:
        prob_table = np.array(prob_table)
        assert prob_table.ndim == 2
        assert table_len == prob_table.shape[1] or backend == 'dif'

    if backend == 'py' or backend == 'cc':
        with open(output_file, 'wb') as f:
            save_shape(f, shape)
            save_int(f, table_len, 2)
            save_int(f, _min, 2)
    # print("compressing!!! shape:{}".format(shape), table_len, _min)

    if backend == 'py':
        _compress_py(x, index, output_file, prob_table, table_len, adapt)
    elif backend == 'cc':
        global cc_compress
        if cc_compress is None:
            from codes.cc.ae_cc_interface import cc_compress
        if 'omega_t' not in kwargs:
            cc_compress(x, output_file, prob_table, index, adapt, True)
        else:
            cc_compress(x, output_file, prob_table, index, adapt, True, omega_t=kwargs['omega_t'])
    elif backend == 'dif':
        # using difencoder format coder
        global dif_compress
        if dif_compress is None:
            from codes.dif.ae_dif_interface import dif_compress
        table_min = -1024  # TODO pass tails in

        data_type = 'y'
        if output_file.endswith('_side.bin'):  # TODO improve hardcode here
            data_type = 'z'
            f = open(output_file, 'w')
            f.close()  # touch an empth file, this is a hack for YZRealBPPEvaluation
            output_file = output_file.replace('_side.bin', '.bin')

            table_min = -128
            x_shape = kwargs['x_shape']
            #write frame size
            with open(output_file, 'wb') as f:
                _, _, height, width = x_shape
                align = 64
                head_len = struct.pack('i', 17)
                head_type = struct.pack('B', 127)
                head_body = struct.pack('iii', height, width, align)
                head = head_len + head_type + head_body
                f.write(head)

        dif_compress(x, data_type, output_file, prob_table, table_min, index, adapt, True)
    else:
        raise ValueError('unsupported compress backend {}'.format(backend))


def _compress_py(x, index, output_file, prob_table, table_len, adapt):
    with contextlib.closing(arithmeticcoding.BitOutputStream(open(output_file, "ab+"))) as bitout:
        enc = arithmeticcoding.ArithmeticEncoder(32, bitout)
        if prob_table is not None:
            initfreqs = [arithmeticcoding.EntropyFrequencyTable(table_len, pt) for pt in prob_table]
        else:
            initfreqs = [arithmeticcoding.FlatFrequencyTable(table_len)]
        freqs = [arithmeticcoding.SimpleFrequencyTable(f) for f in initfreqs]

        x = x.ravel('C')
        index = index.ravel('C')

        for sym, ind in zip(x, index):
            enc.write(freqs[ind], sym)
            if adapt:
                freqs[ind].increment(sym)
        enc.write(freqs[0], table_len - 1)  # note this is EOF
        enc.finish()


# Main launcher
if __name__ == "__main__":
    x = np.array([[-1, 1, 1], [2, 3, 3]])
    compress(x, 'test.bin', test=True)

    x = np.array([[0, 1, 2, 1, 0], [0, 1, 1, 1, 2]])
    compress(x, 'test2.bin', test=True)

    x = np.array([[0, 1, 5, 1, 0], [0, 1, 1, 1, 2]])
    prob = {'0': 0.3, '1': 0.5, '2': 0.1, '3': 0, '4': 0, '5': 0.1, '6': 0}
    for k in prob:
        prob[k] *= x.size
    prob_table = list(prob.values())
    prob_table = [round(f) for f in prob_table]
    prob_table = [[1 if f == 0 else f for f in prob_table]]
    print(prob_table)
    compress(x, 'test3.bin', prob_table=prob_table, test=True)
    compress(x, 'test4.bin', adapt=False, prob_table=prob_table, test=True)

    # use table
    x = np.array([[0, 1, 5, 1, 0], [0, 1, 1, 1, 2]]).reshape([1, 2, 1, -1])
    prob1 = {'0': 0.3, '1': 0.5, '2': 0.1, '3': 0, '4': 0, '5': 0.1, '6': 0}
    prob2 = {'0': 0.6, '1': 0.7, '2': 0.8, '3': 0.2, '4': 0.3, '5': 0.9, '6': 0}
    prob_table = []
    for prob in [prob1, prob2]:
        for k in prob:
            prob[k] *= x.size // 2
        _prob_table = list(prob.values())
        _prob_table = [round(f) for f in _prob_table]
        _prob_table = [1 if f == 0 else f for f in _prob_table]
        prob_table.append(_prob_table)
    print(prob_table)
    compress(x, 'test5.bin', True, adapt=False, prob_table=prob_table, test=True)
    compress(x, 'test6.bin', True, adapt=True, prob_table=prob_table, test=True)

    # test compress_with_index
    prob_table = [[1, 3, 2, 4, 2, 4, 1], [1, 56435, 343, 5, 2, 3, 5], [122, 21, 2, 2, 442, 2, 4]]
    index = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]).reshape([1, 3, 2, 2])
    # index = [0, 1, 2, 2, 1, 0, 0, 1, 2, 0, 1, 2]
    data = np.array([1, 2, 3, 4, 5, 6, 1, 3, 5, 2, 4, 6]).reshape([1, 3, 2, 2])
    compress(data, 'compress_test.bin', True, False, prob_table)
    print(os.path.getsize('compress_test.bin'))
    compress_with_index(data, index, 'compress_test_with_index.bin', False, prob_table)
    print(os.path.getsize('compress_test_with_index.bin'))
