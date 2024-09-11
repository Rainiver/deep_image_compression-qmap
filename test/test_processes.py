import unittest.result
from typing import Optional
from unittest import TestCase
import torch

from pipelines.processes import *


def _get_dummy():
    return torch.randn(3, 16, 16, 64)


def _get_single_model():
    def _black_box(*args):
        if len(args) > 5:
            return 2 * (sum(args) + 1)
        else:
            r = 0
            for i, x in enumerate(args):
                r += x * (i + 1)
            return r

    return _black_box


def _get_single_models(names):
    if not isinstance(names, list):
        names = [names]
    return {name: _get_single_model() for name in names}


def single_case(process_cls, pre_cond, post_cond, side_effect, models):
    """
    check whether the given process update specified data fields

    :param process_cls: the process class
    :param pre_cond: list or int. name(s) of required data
    :param post_cond: list or int. name(s) of created data
    :param side_effect: list or int. name(s) of updated/removed data
    :param models: dict. name - model (which is a callable) pairs
    :return: tuple, input data and output results, which are all dict
    """
    assert issubclass(process_cls, BaseProcess)

    def _wrap_list(x):
        if not x:
            return []
        if not isinstance(x, list):
            return [x]
        return x

    pre_cond = _wrap_list(pre_cond)
    post_cond = _wrap_list(post_cond)
    side_effect = _wrap_list(side_effect)
    if not models:
        models = {}

    data = {field: _get_dummy() for field in pre_cond}
    inputs = {k: v for k, v in data.items()}

    proc = process_cls(models, data)
    proc.run()

    for field in post_cond:
        assert field in data
    for field in side_effect:
        if field in data:
            assert (data[field] != inputs[field]).any()
    outputs = data
    return inputs, outputs


def test_y_encode_process():
    single_case(YEncodeProcess,
                'x',
                'y',
                None,
                _get_single_models('y_encoder'))


def test_y_quantize_process():
    single_case(YQuantizeProcess,
                'y',
                'y_tilde',
                None,
                _get_single_models('y_quant'))


def test_y_decode_process():
    single_case(YDecodeProcess,
                'y_tilde',
                'x_hat',
                None,
                _get_single_models('y_decoder'))


def test_z_encode_process():
    single_case(ZEncodeProcess,
                pre_cond='y',
                post_cond='z',
                side_effect=None,
                models=_get_single_models('z_encoder'))


def test_z_abs_encode_process():
    models = _get_single_models('z_encoder')
    inputs, actual = single_case(ZAbsEncodeProcess,
                                 pre_cond='y',
                                 post_cond='z',
                                 side_effect=None,
                                 models=models)
    y = inputs['y']
    expected = models['z_encoder'](y.abs())
    assert (actual['z'] == expected).all()


def test_z_quantize_process():
    single_case(ZQuantizeProcess,
                pre_cond='z',
                post_cond='z_tilde',
                side_effect=None,
                models=_get_single_models('z_quant'))


def test_z_decode_process():
    single_case(ZDecodeProcess,
                pre_cond='z_tilde',
                post_cond='prior',
                side_effect=None,
                models=_get_single_models('z_decoder'))


def test_z_factorized_entropy_process():
    single_case(ZFactorizedEntropyProcess,
                pre_cond='z_tilde',
                post_cond='z_likelihoods',
                side_effect=None,
                models=_get_single_models('z_entropy_pre'))


def _y_entropy_case(cls, extra_pre_cond, side_effect):
    return single_case(cls,
                       ['y_tilde'] + (extra_pre_cond if extra_pre_cond else []),
                       'y_likelihoods',
                       side_effect,
                       _get_single_models('entropy_pre'))


def test_y_factorized_entropy_process():
    return _y_entropy_case(YFactorizedEntropyProcess, None, None)


def test_gmm_entropy_process():
    return _y_entropy_case(GMMCompleteEntropyProcess, ['sigma', 'mu'], None)


def test_gsm_entropy_process():
    return _y_entropy_case(GSMEntropyProcess, ['sigma'], None)
