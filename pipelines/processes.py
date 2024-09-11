import itertools
from abc import ABC
import importlib
import warnings
import torch
from torch import nn
try:
    import spring.linklink as link
except:
    link = None

from codes.AE.adaptive_arithmetic_compress import compress_with_index
from codes.AE.adaptive_arithmetic_decompress import decompress_with_index
from codes.prob_table_utils import get_named_prob_table
from losses.SSIM_Loss import msssim
from losses.TV_Loss import Loss as TV
from losses.PSNR_Loss import Loss as PSNR
from losses.entropy_Loss import Loss as Entropy_loss
from losses.YCbCr_Loss import Loss as YUV_Loss
from losses.grad_Loss import Loss as grad_Loss
from losses.grad_Loss import Loss_Simple as grad_Loss_simple
from losses.grad_Loss import TVLoss2
from nets.flow import encode_patches
# from nets.layers import PixelwiseRateDistortionLoss
# from nets.layers import Metrics
import numpy as np

from tools.train_val_helper import calc_bpp_by_shape, to_prob_table
from tools.compare_image import diff_image_torch


class BaseProcess:
    def __init__(self, models, data_pool: dict):
        self._models = models
        self._data_pool = data_pool

    def run(self, training=True):
        raise NotImplementedError(self.__class__.__name__)

    def check(self):
        pass

    @classmethod
    def create(cls, created_process_name=None, **attrs):
        class CreatedClass(cls):
            pass

        for k, v in attrs.items():
            setattr(CreatedClass, k, v)

        if not created_process_name:
            created_process_name = cls.__name__ + '_Created'
        CreatedClass.__name__ = created_process_name

        return CreatedClass

    def __repr__(self):
        return f'<{self.__class__.__name__} object>'


class IOMixin:
    """
    a mixin describe processes with properties 'input_name' and 'output_name
    """
    input_name = None
    output_name = None

    @classmethod
    def create_process(cls, *args, **kwargs):
        msg = f'{cls.__name__}.create_process() is deprecated, use create() instead.'
        warnings.warn(msg)
        return cls.create_io_process(*args, **kwargs)

    @classmethod
    def create_io_process(cls, input=None, output=None, created_process_name=None):
        """
        re-generate a process inheriting current class with input_name=input and output_name=output
        :param input: input data name
        :param output: output data name
        :param created_process_name: name of new class. if None, use auto generated name
        :return: created class
        """

        class CreatedClass(cls):
            input_name = input if input is not None else cls.input_name
            output_name = output if output is not None else cls.output_name

        if not created_process_name:
            created_process_name = cls.__name__ + '_Created'
        CreatedClass.__name__ = created_process_name

        return CreatedClass


def _list(x):
    if isinstance(x, list) or x is None:
        return x
    return [x]


def _isinstance_or_none(x, *type):
    return isinstance(x, type) or x is None


class MultiIOMixin(BaseProcess):
    """
    a mixin describe processes with multiple input and output 'input_names', 'kwarg_names' and 'output_names

    This is extension of IOMixin
    """
    input_names = None
    kwargs_names = None
    lambda_name = None
    output_names = None

    @classmethod
    def create_multi_io(cls, inputs=None, outputs=None, foo_kwargs=None, foo=None, cls_name=None):
        """
        re-generate a process inheriting current class with input_name=input and output_name=output

        :param inputs: input data names. if None, use original input name
        :type inputs: list of str or str or None
        :param outputs: output data names. if None, use original output name
        :type outputs: list of str or str or None
        :param foo_kwargs: key-name dict of kwards. if None, use original kwarg name
        :type foo_kwargs: dict or None
        :param foo: callable function name. if None, use original foo name
        :type foo: str or None
        :param cls_name: name of new class. if None, use auto generated name
        :type cls_name: str or None
        :return: created class
        """

        assert _isinstance_or_none(inputs, list, str), inputs
        assert _isinstance_or_none(outputs, list, str), outputs
        assert _isinstance_or_none(foo_kwargs, dict), foo_kwargs

        attrs = {}
        for name, val in [('input_names', _list(inputs)),
                          ('output_names', _list(outputs)),
                          ('lambda_name', foo),
                          ('kwargs_names', foo_kwargs), ]:
            if val is not None:
                attrs[name] = val

        return cls.create(created_process_name=cls_name, **attrs)

    @classmethod
    def create_single_io(cls, input=None, output=None, foo=None, cls_name=None):
        return cls.create_multi_io(input, output, foo=foo, cls_name=cls_name)


class ModelInvokeProcess(MultiIOMixin):
    """
    Model Process interface V2
    """

    def run(self, training=True):
        data = self._data_pool
        model = self._models[self.lambda_name]
        inputs = [data[name] for name in self.input_names]
        kwargs_names = self.kwargs_names or {}
        kwargs = {k: data[v] for k, v in kwargs_names.items()}
        outputs = model(*inputs, **kwargs)
        output_names = self.output_names
        if len(output_names) > 1:
            assert len(output_names) == len(outputs)
            for name, out in zip(output_names, outputs):
                data[name] = out
        else:
            data[output_names[0]] = outputs


class ComposedProcess(BaseProcess):
    processes = None

    def run(self, training=True):
        for proc in self.processes:
            proc.run(training)


class BaseSingleModelProcess(BaseProcess, IOMixin):
    """
    basic process with a model io based behavior
    """
    model_name = None

    def run(self, training=True):
        data = self._data_pool

        inputs = data[self.input_name]
        inputs = self.pre_model(inputs)
        if self._models[self.model_name] is None:
            # for safety
            # all used models should not be None
            raise AssertionError('missed model: {}'.format(self.model_name))

        # print(self._models[self.model_name].__class__)
        outputs = self._models[self.model_name](inputs)
        outputs = self.post_model(outputs)
        data[self.output_name] = outputs

    def pre_model(self, inputs):
        """
        to perform a pre-processing, override this method

        :param inputs: data(input_name)
        :return: processed data.
        """
        return inputs

    def post_model(self, outputs):
        """
        to perform a post-processing, override this method

        :param outputs: calculated value: model(input)
        :return: processed data.
        """
        return outputs

    def check(self):
        if self.input_name is None or self.output_name is None:
            raise NotImplementedError(self.__class__.__name__)
        if self.model_name is None:
            raise NotImplementedError(self.__class__.__name__)


class YEncodeProcess(BaseSingleModelProcess):
    """
    y <- y_encoder(x_after_pre)
    """
    input_name = 'tmp/x_after_pre'
    output_name = 'y'
    model_name = 'y_encoder'


class YVideoEncodeProcess(BaseSingleModelProcess):
    """
    y <- y_encoder(xts1)
    """
    input_name = 'xts1'
    output_name = 'yts1'
    model_name = 'y_encoder'


class YQuantizeProcess(BaseSingleModelProcess):
    """
    y_tilde <- y_quant(y)
    """
    input_name = 'y'
    output_name = 'y_tilde'
    model_name = 'y_quant'


class YRoundQuantizeProcess(BaseSingleModelProcess):
    """
    y_round <- y_quant_round(y)
    """
    input_name = 'y'
    output_name = 'y_round'
    model_name = 'y_quant_round'


class YDecodeProcess(BaseSingleModelProcess):
    """
    x_hat <- y_decoder(y_tilde)
    """
    input_name = 'y_tilde'
    output_name = 'x_hat'
    model_name = 'y_decoder'


class   ZEncodeProcess(BaseSingleModelProcess):
    """
    z <- z_encoder(y)
    """
    input_name = 'y'
    output_name = 'z'
    model_name = 'z_encoder'


class ZVideoEncodeProcess(BaseSingleModelProcess):
    """
    z <- z_encoder(y, yts1)
    """
    output_name = 'z'
    model_name = 'z_encoder'

    def run(self, training=True):
        data_pool = self._data_pool
        y = data_pool['y']
        yts1 = data_pool['yts1']
        inputs = torch.cat((y, yts1), 1)
        model = self._models[self.model_name]
        data_pool[self.output_name] = model(inputs)


class ZAttentionProcess(BaseSingleModelProcess):
    input_name = 'z'
    output_name = 'z'
    model_name = 'z_attention'


class ZAbsEncodeProcess(ZEncodeProcess):
    """
    z <- z_encoder(y.abs())
    """

    def __init__(self, *args):
        super(ZAbsEncodeProcess, self).__init__(*args)

        # (hedailan 2021/03/19)
        # here we do not modify inputs directly
        # but hack the encoder models via adding extra abs op to em
        # in order to conveniently convert/export the abs-encoders
        model = self._models[self.model_name]
        model.register_forward_pre_hook(lambda m, inp: torch.abs(*inp))


class postProcess(BaseSingleModelProcess):
    input_name = 'x_hat'
    output_name = 'x_hat'
    model_name = 'post'

    def pre_model(self, inputs):
        if hasattr(self._models[self.model_name], 'rate'):
            self._data_pool['post_rate'] = self._models[self.model_name].rate
        return inputs


class preProcess(BaseSingleModelProcess):
    input_name = 'x'
    output_name = 'tmp/x_after_pre'
    model_name = 'pre'

    def run(self, training=True):
        data_pool = self._data_pool
        inputs = data_pool[self.input_name]
        if self.model_name in self._models:
            model = self._models[self.model_name]
            if data_pool.get('arg/pre', 'NONE') == "format":
                outputs = model(inputs, data_pool['arg/rank'])
            # TODO: delete
            elif data_pool.get('arg/pre', 'NONE') == "dct-coef":
                # y_coef, cb_coef, cr_coef = data_pool["y_coefficients"], \
                #                            data_pool["cb_coefficients"], \
                #                            data_pool["cr_coefficients"]
                x = data_pool["img"]
                # y_coef, cb_coef, cr_coef = model(y_coef, cb_coef, cr_coef)
                x = model(x)
                # data_pool["tmp/y_coef_after_pre"] = y_coef
                # data_pool["tmp/cb_coef_after_pre"] = cb_coef
                # data_pool["tmp/cr_coef_after_pre"] = cr_coef
                # data_pool["tmp/x_after_pre"] = y_coef
                data_pool["tmp/x_after_pre"] = x
                return
            else:
                outputs = model(inputs)
        else:
            # no pre-process provided
            outputs = inputs
        data_pool[self.output_name] = outputs


class preVideoProcess(BaseProcess):
    """
    x -> (xt, xts1)
    """

    def run(self, training=True):
        x = self._data_pool["x"]
        self._data_pool["tmp/x_after_pre"] = x[:, :3]
        self._data_pool["xt"] = x[:, :3]
        self._data_pool["xts1"] = x[:, 3:]


class ZQuantizeProcess(BaseSingleModelProcess):
    """
    z_tilde <- z_quant(z)
    """
    input_name = 'z'
    output_name = 'z_tilde'
    model_name = 'z_quant'


class ZRoundQuantizeProcess(BaseSingleModelProcess):
    """
    z_round <- z_quant_round(z)
    """
    input_name = 'z'
    output_name = 'z_round'
    model_name = 'z_quant_round'


class ZDecodeProcess(BaseSingleModelProcess):
    """
    prior <- z_decoder(z_tilde)
    """
    input_name = 'z_tilde'
    output_name = 'prior'
    model_name = 'z_decoder'


class ZVideoDecodeProcess(BaseSingleModelProcess):
    """
    prior <- z_decoder(z_tilde, yts1)
    """
    output_name = 'prior'
    model_name = 'z_decoder'

    def run(self, training=True):
        data_pool = self._data_pool
        zt = data_pool['z_tilde']
        yts1 = data_pool['yts1']
        model = self._models[self.model_name]
        data_pool[self.output_name] = model(yts1, zt)


class ZDecode2Process(BaseSingleModelProcess):
    """
    prior2 <- z_decoder2(z_tilde)
    """
    input_name = 'z_tilde'
    output_name = 'prior2'
    model_name = 'z_decoder2'


class BaseEntropyProcess(BaseProcess):
    scope = None
    model_name = None
    input_suffix = '_tilde'
    output_suffix = '_likelihoods'

    def get_prior(self, training):
        raise NotImplementedError(self.__class__.__name__)

    def run(self, training=True):
        scope, model_name = self.scope, self.model_name
        if scope is None or model_name is None:
            raise NotImplementedError(self.__class__.__name__)

        input_name = scope + self.input_suffix
        output_name = scope + self.output_suffix

        data = self._data_pool
        models = self._models

        entropy_model = models[model_name]

        inputs = data[input_name]
        prior = self.get_prior(training)
        outputs = entropy_model(inputs, *prior)
        data[output_name] = outputs


class BaseFactorizedEntropyProcess(BaseEntropyProcess):
    def get_prior(self, training):
        return []


class ZFactorizedEntropyProcess(BaseFactorizedEntropyProcess):
    """
    z_likelihoods <- z_entropy_pre(z_tilde)
    """

    scope = 'z'
    model_name = 'z_entropy_pre'


class YFactorizedEntropyProcess(BaseFactorizedEntropyProcess):
    """
    y_likelihoods <- entropy_pre(y_tilde)
    """

    scope = 'y'
    model_name = 'entropy_pre'


class BaseYEntropyProcess(BaseEntropyProcess, ABC):
    scope = 'y'
    model_name = 'entropy_pre'


class GMMEntropyProcess(BaseEntropyProcess):
    """
    Gaussian Mixture Model (Mock)

    y_likelihoods <- entropy_pre(y_tilde, mu, sigma)
    """

    scope = 'y'
    model_name = 'entropy_pre'
    sigma_name = 'sigma'
    mu_name = 'mu'

    def get_sigma(self, training):
        if training:
            return self._data_pool[self.sigma_name]
        else:
            # TODO: use sigma_aligned or not?
            return self._data_pool[self.sigma_name]

    def get_mu(self, training):
        return self._data_pool[self.mu_name]

    def get_prior(self, training):
        return [self.get_mu(training), self.get_sigma(training)]


class GMMCompleteEntropyProcess(GMMEntropyProcess):
    """
    Gaussian Mixture Model

    y_likelihoods <- entropy_pre(y_tilde, mu, sigma, omega)


    omega : weight of different Gaussian Model

    $$
    \sum_{i \in chancel\_index} \omega_i = 1
    $$

    """
    omega_name = 'omega'

    def get_omega(self, training):
        # assert self._data_pool.get('omega', None) is not None
        return self._data_pool.get(self.omega_name, None)

    def get_prior(self, training):
        return [self.get_mu(training), self.get_sigma(training), self.get_omega(training)]


class GMMCompleteEntropy1Process(GMMCompleteEntropyProcess):
    mu_name = 'mu1'
    sigma_name = 'sigma1'
    omega_name = 'omega1'
    output_suffix = '_likelihoods_1'


class GMMCompleteEntropy2Process(GMMCompleteEntropyProcess):
    mu_name = 'mu2'
    sigma_name = 'sigma2'
    omega_name = 'omega2'
    output_suffix = '_likelihoods_2'


class GSMEntropyProcess(GMMCompleteEntropyProcess):
    """
    Gaussian Scale Model

    y_likelihoods <- entropy_pre(y_tilde, 0, sigma)

    """

    def get_mu(self, training):
        return 0.


class GMMEntropy1Process(GMMEntropyProcess):
    """
    y_likelihoods_1 <- entropy_pre(y_tilde, mu1, sigma1)

    This is for two-path models
    """
    mu_name = 'mu1'
    sigma_name = 'sigma1'
    output_suffix = '_likelihoods_1'


class GMMEntropy2Process(GMMEntropyProcess):
    """
    y_likelihoods_2 <- entropy_pre(y_tilde, mu2, sigma2)

    This is for two-path models
    """
    mu_name = 'mu2'
    sigma_name = 'sigma2'
    output_suffix = '_likelihoods_2'


class GMMEntropy3Process(GMMEntropyProcess):
    """
    y_likelihoods_3 <- entropy_pre(y_tilde, mu3, sigma3)

    This is for five-step models
    """
    mu_name = 'mu3'
    sigma_name = 'sigma3'
    output_suffix = '_likelihoods_3'


class GMMEntropy4Process(GMMEntropyProcess):
    """
    y_likelihoods_4 <- entropy_pre(y_tilde, mu4, sigma4)

    This is for five-step models
    """
    mu_name = 'mu4'
    sigma_name = 'sigma4'
    output_suffix = '_likelihoods_4'


class GMMEntropy5Process(GMMEntropyProcess):
    """
    y_likelihoods_5 <- entropy_pre(y_tilde, mu5, sigma5)

    This is for five-step models
    """
    mu_name = 'mu5'
    sigma_name = 'sigma5'
    output_suffix = '_likelihoods_5'


class TwoPathEntropyLossProcess(BaseProcess):
    """
    y_entropy1 <- y_likelihoods_1
    (side-effect: eval/y_entropy1 <- y_likelihoods_1)
    y_entropy2 <- y_likelihoods_2
    (side-effect: eval/y_entropy2 <- y_likelihoods_2)

    (dep: x.shape)

    compute entropy losses from given likelihoods

    notice: y_entropy1 and y_entropy2 will not involve in backward
    without summing them up by processes like Entropy211WeightedSumProcess
    """

    def run(self, training=True):
        data = self._data_pool
        p1 = data['y_likelihoods_1']
        p2 = data['y_likelihoods_2']
        entropy_loss = Entropy_loss()
        x = data['x']
        num_pixels = x.shape[0] * x.shape[2] * x.shape[3]
        e1, e2 = map(entropy_loss, (p1, p2))
        e1, e2 = (x / num_pixels for x in (e1, e2))
        data['y_entropy1'] = e1
        data['y_entropy2'] = e2
        data['eval/y_entropy1'] = e1
        data['eval/y_entropy2'] = e2


class FiveStepsEntropyLossProcess(BaseProcess):
    """
    y_entropy1 <- y_likelihoods_1
    (side-effect: eval/y_entropy1 <- y_likelihoods_1)
    y_entropy2 <- y_likelihoods_2
    (side-effect: eval/y_entropy2 <- y_likelihoods_2)
    y_entropy3 <- y_likelihoods_3
    (side-effect: eval/y_entropy3 <- y_likelihoods_3)
    y_entropy4 <- y_likelihoods_4
    (side-effect: eval/y_entropy4 <- y_likelihoods_4)
    y_entropy5 <- y_likelihoods_5
    (side-effect: eval/y_entropy5 <- y_likelihoods_5)

    (dep: x.shape)

    compute entropy losses from given likelihoods

    notice: y_entropy1 ~ y_entropy5 will not involve in backward
    without summing them up by processes like Entropy211WeightedSumProcess
    """

    def run(self, training=True):
        data = self._data_pool
        p1, p2, p3, p4, p5 = (data['y_likelihoods_' + str(i)] for i in (1, 2, 3, 4, 5))
        entropy_loss = Entropy_loss()
        x = data['x']
        num_pixels = x.shape[0] * x.shape[2] * x.shape[3]
        e1, e2, e3, e4, e5 = map(entropy_loss, (p1, p2, p3, p4, p5))
        e1, e2, e3, e4, e5 = (x / num_pixels for x in (e1, e2, e3, e4, e5))
        data['y_entropy1'] = e1
        data['y_entropy2'] = e2
        data['y_entropy3'] = e3
        data['y_entropy4'] = e4
        data['y_entropy5'] = e5
        data['eval/y_entropy1'] = e1
        data['eval/y_entropy2'] = e2
        data['eval/y_entropy3'] = e3
        data['eval/y_entropy4'] = e4
        data['eval/y_entropy5'] = e5


class Entropy211WeightedSumProcess(BaseProcess):
    """
    weighted sum up given entropy

    loss/y_entropy_loss <- 0.5 * y_entropy1 + 0.25 * y_entropy2 + 0.25 * y_entropy3

    this is for multi-mask spatial context models
    """

    def run(self, training=True):
        data = self._data_pool
        e1, e2, e3 = (data['y_entropy' + str(i)] for i in (1, 2, 3))
        data['loss/y_entropy_loss'] = 0.5 * e1 + 0.25 * e2 + 0.25 * e3


class EntropyFiveWeightedSumProcess(BaseProcess):
    """
    weighted sum up given entropy

    loss/y_entropy_loss <- 0.2 * sum_i(y_entropyi)

    this is for multi-mask spatial context models
    """

    def run(self, training=True):
        data = self._data_pool
        e1, e2, e3, e4, e5 = (data['y_entropy' + str(i)] for i in (1, 2, 3, 4, 5))
        data['loss/y_entropy_loss'] = 0.2 * e1 + 0.2 * e2 + 0.2 * e3 + 0.2 * e4 + 0.2 * e5


class KnightMaskProcess(BaseProcess):
    """
    five masked y_tilde for further context processing

    y_masked1 <- y_mask1(y_tilde)
    y_masked2 <- y_mask2(y_tilde)
    y_masked3 <- y_mask3(y_tilde)
    y_masked4 <- y_mask4(y_tilde)
    y_masked5 <- y_mask5(y_tilde)

    this is for KnightsPosition Share-Weight Context model
    """

    def run(self, training=True):
        data_pool, models = self._data_pool, self._models
        y_tilde = data_pool['y_tilde']
        y_masked = ['y_masked1', 'y_masked2', 'y_masked3', 'y_masked4', 'y_masked5']
        y_mask = ['y_mask1', 'y_mask2', 'y_mask3', 'y_mask4', 'y_mask5']
        for i in range(5):
            data_pool[y_masked[i]] = models[y_mask[i]](y_tilde)


class ContextProcess(BaseSingleModelProcess):
    """
    y_context <- y_context(y_tilde)
    """
    input_name = 'y_tilde'
    output_name = 'y_context'
    model_name = 'y_context'


class Context2Process(BaseSingleModelProcess):
    """
    y_context2 <- y_context2(y_tilde)
    """
    input_name = 'y_tilde'
    output_name = 'y_context2'
    model_name = 'y_context2'


class Context3Process(BaseSingleModelProcess):
    """
    y_context3 <- y_context3(y_tilde)
    """
    input_name = 'y_tilde'
    output_name = 'y_context3'
    model_name = 'y_context3'


class Context4Process(BaseSingleModelProcess):
    """
    y_context4 <- y_context4(y_tilde)
    """
    input_name = 'y_tilde'
    output_name = 'y_context4'
    model_name = 'y_context4'


class Context5Process(BaseSingleModelProcess):
    """
    y_context5 <- y_context5(y_tilde)
    """
    input_name = 'y_tilde'
    output_name = 'y_context5'
    model_name = 'y_context5'


class KnightShareWeightContextProcess(BaseProcess):
    """
    get five y_context from five y_masked

    y_context  <- y_context(y_masked1)
    y_context2 <- y_context(y_masked2)
    y_context3 <- y_context(y_masked3)
    y_context4 <- y_context(y_masked4)
    y_context5 <- y_context(y_masked5)

    this is for KnightsPosition Share-Weight Context model
    """

    def run(self, training=True):
        data_pool, models = self._data_pool, self._models
        context_names = ['y_context', 'y_context2', 'y_context3', 'y_context4', 'y_context5']
        y_masked = ['y_masked1', 'y_masked2', 'y_masked3', 'y_masked4', 'y_masked5']
        context_model = models['y_context']
        for i in range(5):
            data_pool[context_names[i]] = context_model(data_pool[y_masked[i]])


class ChannelCatProcess(BaseProcess):
    input_names = None
    output_name = None

    def run(self, training=True):
        data = self._data_pool
        inputs = [data[input] for input in self.input_names]
        data[self.output_name] = torch.cat(inputs, 1)


class PriorAndContextParameterProcess(BaseProcess):
    prior_name = 'prior'
    context_name = 'y_context'
    param_model = 'y_parameter'
    sigma_name = 'sigma'
    mu_name = 'mu'

    def run(self, training=True):
        data_pool, models = self._data_pool, self._models

        prior = data_pool[self.prior_name]
        context = data_pool[self.context_name]
        inputs = torch.cat([prior, context], 1)

        param_model = models[self.param_model]
        params = param_model(inputs)
        n_channel = params.shape[1]
        mean = params[:, :n_channel // 2, ...]
        std = params[:, n_channel // 2:, ...]

        data_pool[self.sigma_name] = std
        data_pool[self.mu_name] = mean


class PriorParameterProcess(BaseProcess):
    prior_name = 'prior'
    sigma_name = 'sigma'
    mu_name = 'mu'

    def run(self, training=True):
        data_pool, models = self._data_pool, self._models

        prior = data_pool[self.prior_name]
        c_param = prior.shape[1]
        mean, std = torch.split(prior, c_param // 2, 1)

        data_pool[self.sigma_name] = std
        data_pool[self.mu_name] = mean



class ChannelSliceParameterProcess(BaseProcess):
    y_name = 'y_round'
    y_lrp_name = 'y_lrp'
    hyper_name = 'prior'
    param_model_name = 'y_parameter'
    param_name = 'tmp/param_group'

    def run(self, training=True):
        data = self._data_pool
        y = data[self.y_name]
        hyper = data[self.hyper_name]
        param_model = self._models[self.param_model_name]
        n_groups = param_model.n_groups  # TODO ugly here
        ys = torch.split(y, y.shape[1] // n_groups, 1)
        out = list(param_model(ys, hyper))
        param = [o[0] for o in out]
        y_lrp = torch.cat([o[1] for o in out], 1)
        data[self.param_name] = param
        data[self.y_lrp_name] = y_lrp

class ChannelSliceParameterProcessTwoPath(BaseProcess):     #get param_group of two context (cross_context and zero context)
    y_name = 'y_round'
    y_lrp_name = 'y_lrp'
    hyper_name = 'prior'
    param_model_name = 'y_parameter'
    param_name1 = 'tmp/param_group1'
    param_name2 = 'tmp/param_group2'

    def run(self, training=True):
        data = self._data_pool
        y = data[self.y_name]
        hyper = data[self.hyper_name]
        param_model = self._models[self.param_model_name]
        n_groups = param_model.n_groups  # TODO ugly here
        ys = torch.split(y, y.shape[1] // n_groups, 1)
        out = list(param_model(ys, hyper))
        param1 = [o[0] for o in out]
        param2 = [o[1] for o in out]
        y_lrp = torch.cat([o[2] for o in out], 1)
        data[self.param_name1] = param1
        data[self.param_name2] = param2
        data[self.y_lrp_name] = y_lrp

class ParameterGroupSplitProcess(BaseProcess):
    sigma_name = 'sigma'
    mu_name = 'mu'
    param_name = 'tmp/param_group'

    def run(self, training=True):
        data = self._data_pool
        sigma_list = []
        mu_list = []
        for param in data[self.param_name]:
            c_param = param.shape[1]
            mu, sigma = torch.split(param, c_param // 2, 1)
            sigma_list.append(sigma)
            mu_list.append(mu)
        sigma = torch.cat(sigma_list, 1)
        mu = torch.cat(mu_list, 1)
        data[self.sigma_name] = sigma
        data[self.mu_name] = mu

class ParameterGroupSplitProcessTwoPath(BaseProcess):
    sigma_name = 'sigma'
    mu_name = 'mu'
    param_name = 'tmp/param_group'
    # sigma_name1 = 'sigma1'
    # mu_name1 = 'mu1'
    # param_name1 = 'tmp/param_group1'
    # sigma_name2 = 'sigma2'
    # mu_name2 = 'mu2'
    # param_name2 = 'tmp/param_group2'
    def run(self, training=True):
        data = self._data_pool
        for i in range(1,3):
            sigma_list = []
            mu_list = []
            for param in data[self.param_name+str(i)]:
                c_param = param.shape[1]
                mu, sigma = torch.split(param, c_param // 2, 1)
                sigma_list.append(sigma)
                mu_list.append(mu)
            sigma = torch.cat(sigma_list, 1)
            mu = torch.cat(mu_list, 1)
            data[self.sigma_name+str(i)] = sigma
            data[self.mu_name+str(i)] = mu

class FakeSerialPriorAndContextParameterProcess(BaseProcess):
    prior_name = 'prior'
    context_name = 'y_context'
    param_model = 'y_parameter'
    sigma_name = 'sigma'
    mu_name = 'mu'
    y_tilde_name = 'y_tilde'
    context_model_name = 'y_context'

    def run(self, training=True):
        data_pool, models = self._data_pool, self._models

        y_tilde = data_pool[self.y_tilde_name]
        n, c, h, w = y_tilde.shape
        params = torch.zeros(
            (n, 2 * c, h, w), device=y_tilde.device, dtype=y_tilde.dtype)
        prior = data_pool['prior']
        ctx_model = models[self.context_model_name]
        param_model = models['y_parameter']
        for i in range(h):
            for j in range(w):
                ctx = ctx_model(y_tilde[..., i:i + 1, j:j + 1])
                hyper = prior[..., i:i + 1, j:j + 1]
                inp = torch.cat((hyper, ctx), 1)
                params[..., i:i + 1, j:j + 1] = param_model(inp)

        n_channel = params.shape[1]
        mean = params[:, :n_channel // 2, ...]
        std = params[:, n_channel // 2:, ...]

        data_pool[self.sigma_name] = std
        data_pool[self.mu_name] = mean


class PriorAndContextParameter1Process(PriorAndContextParameterProcess):
    """
    (mu1, sigma1) <- y_parameter(hyper, y_context)
    """
    context_name = 'y_context'
    param_model = 'y_parameter'
    sigma_name = 'sigma1'
    mu_name = 'mu1'


class PriorAndContextShareWeightParameterProcess(PriorAndContextParameterProcess):
    """
    (mu2, sigma2) <- y_parameter(hyper, y_context2)
    """
    prior_name = 'prior'
    context_name = 'y_context2'
    param_model = 'y_parameter'
    sigma_name = 'sigma2'
    mu_name = 'mu2'


class PriorAndContextShareWeightParameter2Process(PriorAndContextParameterProcess):
    """
    (mu2, sigma2) <- y_parameter(hyper, y_context2)

    notice: this is the same as PriorAndContextShareWeightParameterProcess for making appendix agree with attrs
    """
    prior_name = 'prior'
    context_name = 'y_context2'
    param_model = 'y_parameter'
    sigma_name = 'sigma2'
    mu_name = 'mu2'


class PriorAndContextShareWeightParameter3Process(PriorAndContextParameterProcess):
    """
    (mu3, sigma3) <- y_parameter(hyper, y_context3)
    """
    prior_name = 'prior'
    context_name = 'y_context3'
    param_model = 'y_parameter'
    sigma_name = 'sigma3'
    mu_name = 'mu3'


class PriorAndContextShareWeightParameter4Process(PriorAndContextParameterProcess):
    """
    (mu4, sigma4) <- y_parameter(hyper, y_context4)
    """
    prior_name = 'prior'
    context_name = 'y_context4'
    param_model = 'y_parameter'
    sigma_name = 'sigma4'
    mu_name = 'mu4'


class PriorAndContextShareWeightParameter5Process(PriorAndContextParameterProcess):
    """
    (mu5, sigma5) <- y_parameter(hyper, y_context5)
    """
    prior_name = 'prior'
    context_name = 'y_context5'
    param_model = 'y_parameter'
    sigma_name = 'sigma5'
    mu_name = 'mu5'


class ContextLRPProcess(BaseProcess, IOMixin):
    input_name = 'y_round'
    prior_name = 'prior'
    mu_name = 'mu1'
    lrp_name = 'lrp'
    output_name = 'y_lrp'

    def run(self, training=True):
        data = self._data_pool
        y = data[self.input_name]
        prior = data[self.prior_name]
        mu = data[self.mu_name]
        lrp_model = self._models[self.lrp_name]
        lrp = lrp_model(torch.cat((prior, mu), 1))
        res = y + lrp
        data[self.output_name] = res


class PriorAndContext2Parameter2Process(PriorAndContextParameterProcess):
    """
    (mu2, sigma2) <- y_parameter(hyper2, y_context2)

    this is for two-path models
    """
    prior_name = 'prior2'
    context_name = 'y_context2'
    param_model = 'y_parameter2'
    sigma_name = 'sigma2'
    mu_name = 'mu2'


class ParameterGateProcess(BaseProcess):
    """
    summarize entropy parameters using given gate model


    mu <- gate(mu1, mu2)

    sigma <- gate(sigma1, sigma2)
    """

    def run(self, training=True):
        gate = self._models['gate']
        data = self._data_pool
        sigma = [data['sigma' + str(i)] for i in [1, 2]]
        sigma = gate(*sigma, sigma[1])  # 0, 1, 1
        mu = [data['mu' + str(i)] for i in [1, 2]]
        mu = gate(*mu, mu[1])  # 0, 1, 1

        data['sigma'] = sigma
        data['mu'] = mu


class ParameterGate2Process(BaseProcess):
    """
    summarize entropy parameters using given gate model


    mu <- gate(mu1, mu2, mu3, mu4, mu5)

    sigma <- gate(sigma1, sigma2, sigma3, sigma4, sigma5)
    """

    def run(self, training=True):
        gate = self._models['gate']
        data = self._data_pool
        sigma = [data['sigma' + str(i)] for i in [1, 2, 3, 4, 5]]
        sigma = gate(*sigma)
        mu = [data['mu' + str(i)] for i in [1, 2, 3, 4, 5]]
        mu = gate(*mu)

        data['sigma'] = sigma
        data['mu'] = mu


class CompleteGMMParameterGateProcess(BaseProcess):
    """
    summarize entropy parameters using given gate model


    mu <- gate(mu1, mu2)

    sigma <- gate(sigma1, sigma2)

    omega <- gate(omega1, omega2)
    """

    def run(self, training=True):
        gate = self._models['gate']
        data = self._data_pool
        for param_name in ['mu', 'sigma', 'omega']:
            param = [data[param_name + str(i)] for i in [1, 2]]
            param = gate(param[0], param[1], param[1])  # 0, 1, 1
            data[param_name] = param


class PriorGMMParameterProcess(BaseProcess):
    def run(self, training=True):
        data_pool, models = self._data_pool, self._models

        inputs = data_pool['prior']

        # print(inputs.size())
        param_model = models['y_parameter']
        params = param_model(inputs)
        n_channel = params.shape[1]

        gmm_k = data_pool['arg/entropy_model_args'].get('K', 1)
        assert n_channel % (3 * gmm_k) == 0

        data_pool['sigma'] = params[:, :n_channel // 3, ...]
        data_pool['mu'] = params[:, n_channel // 3:n_channel // 3 * 2, ...]

        omega = params[:, n_channel // 3 * 2:, ...]
        size = omega.shape
        omega = omega.view(size[0], gmm_k, -1).softmax(dim=1)
        data_pool['omega'] = omega.view(size)


class ContextAndPriorGMMParameterProcess(BaseProcess):
    prior_name = 'prior'
    context_name = 'y_context'
    sigma_name = 'sigma'
    mu_name = 'mu'
    omega_name = 'omega'

    def run(self, training=True):
        data_pool, models = self._data_pool, self._models

        inputs = torch.cat([data_pool[self.prior_name],
                            data_pool[self.context_name]], 1)

        # print(inputs.size())
        param_model = models['y_parameter']
        params = param_model(inputs)
        n_channel = params.shape[1]

        gmm_k = data_pool['arg/entropy_model_args'].get('K', 1)
        assert n_channel % (3 * gmm_k) == 0

        data_pool[self.sigma_name] = params[:, :n_channel // 3, ...]
        data_pool[self.mu_name] = params[:,
                                         n_channel // 3:n_channel // 3 * 2, ...]

        omega = params[:, n_channel // 3 * 2:, ...]
        size = omega.shape
        omega = omega.view(size[0], gmm_k, -1).softmax(dim=1)
        data_pool[self.omega_name] = omega.view(size)


class FakeSerialContextAndPriorGMMParameterProcess(BaseProcess):
    prior_name = 'prior'
    y_tilde_name = 'y_tilde'
    sigma_name = 'sigma'
    mu_name = 'mu'
    omega_name = 'omega'
    context_model_name = 'y_context'

    def run(self, training=True):
        data_pool, models = self._data_pool, self._models

        y_tilde = data_pool[self.y_tilde_name]
        n, c, h, w = y_tilde.shape
        params = torch.zeros(
            (n, 9 * c, h, w), device=y_tilde.device, dtype=y_tilde.dtype)
        prior = data_pool['prior']
        ctx_model = models[self.context_model_name]
        param_model = models['y_parameter']
        for i in range(h):
            for j in range(w):
                ctx = ctx_model(y_tilde[..., i:i + 1, j:j + 1])
                hyper = prior[..., i:i + 1, j:j + 1]
                inp = torch.cat((hyper, ctx), 1)
                params[..., i:i + 1, j:j + 1] = param_model(inp)

        n_channel = params.shape[1]

        gmm_k = data_pool['arg/entropy_model_args'].get('K', 1)
        assert n_channel % (3 * gmm_k) == 0

        data_pool[self.sigma_name] = params[:, :n_channel // 3, ...]
        data_pool[self.mu_name] = params[:,
                                         n_channel // 3:n_channel // 3 * 2, ...]

        omega = params[:, n_channel // 3 * 2:, ...]
        size = omega.shape
        omega = omega.view(size[0], gmm_k, -1).softmax(dim=1)
        data_pool[self.omega_name] = omega.view(size)


class ContextAndPriorGMMParameter1Process(ContextAndPriorGMMParameterProcess):
    prior_name = 'prior'
    context_name = 'y_context'
    sigma_name = 'sigma1'
    mu_name = 'mu1'
    omega_name = 'omega1'


class ContextAndPriorGMMShareWeightParameter2(ContextAndPriorGMMParameterProcess):
    prior_name = 'prior'
    context_name = 'y_context2'
    sigma_name = 'sigma2'
    mu_name = 'mu2'
    omega_name = 'omega2'


class LossProcess(BaseProcess):
    def run(self, training=True):
        # TODO: refactor, add more options
        data_pool = self._data_pool

        tv = TV()
        entropy_loss = Entropy_loss()

        x_hat = data_pool['x_hat']

        gt_conf = data_pool.get('arg/train_gt', 'x')
        x = data_pool[gt_conf]

        # TODO: refactor this
        # distortion loss functions should obey common protocol
        # with input x(gt) and x_hat(reconstruction)
        # so that a fancy config is possible

        lambda1 = data_pool['arg/lambda1']
        lambda2 = data_pool['arg/lambda2']
        lambda2_1 = 0
        lambda2_2 = 0
        lambda2_3 = 0
        if 'arg/lambda2_1' in data_pool:
            lambda2_1 = data_pool['arg/lambda2_1']
        if 'arg/lambda2_2' in data_pool:
            lambda2_2 = data_pool['arg/lambda2_2']
        if 'arg/lambda2_3' in data_pool:
            lambda2_3 = data_pool['arg/lambda2_3']
        lambda4 = data_pool['arg/lambda4']
        lambda_grad = data_pool.get('arg/lambda_grad', 0.)
        lambda_grad_simple = data_pool.get('arg/lambda_grad_simple', 0.)
        lambda_grad_simple_v2 = data_pool.get('arg/lambda_grad_simple_v2', 0.)

        # fast_train = training and data_pool['arg/fast_train']
        fast_train = True
        # TODO: delete
        cur_epoch = data_pool.get("tmp/cur_epoch", None)
        # if cur_epoch and cur_epoch <= 500:
        #     lambda2 = 0.5
        # else:
        #     lambda2 = 0.

        if lambda1 == 0. and fast_train:
            loss1 = 0.
        else:
            loss1 = 1 - msssim(x_hat, x, normalize=True)
            data_pool['loss/msssim'] = loss1

        if lambda2 == 0. and fast_train:
            loss2 = 0.
        else:
            loss2 = nn.MSELoss(reduction='mean')(x_hat, x) * 65025  # 255 ** 2
            data_pool['loss/mse'] = loss2

        if lambda2_1 == 0. and fast_train:
            loss2_1 = 0.
        else:
            #loss2_1 = torch.sqrt(loss2)
            loss2_tmp = nn.MSELoss(reduction='mean')(
                x_hat, x) * 65025  # 255 ** 2
            loss2_1 = torch.sqrt(loss2_tmp)
            data_pool['loss/rmse'] = loss2_1

        if lambda2_2 == 0. and fast_train:
            loss2_2 = 0.
        else:
            loss2_2 = YUV_Loss()(x_hat * 255, x * 255)
            data_pool['loss/yuv'] = loss2_2

        if fast_train:
            loss2_2_1 = 0.
        else:
            loss2_2_1 = YUV_Loss('Y')(x_hat * 255, x * 255)
            data_pool['loss/Y'] = loss2_2_1

        if lambda2_3 == 0. and fast_train:
            loss2_3 = 0.
        else:
            loss2_3 = nn.SmoothL1Loss(reduction='mean')(x_hat * 255, x * 255)
            data_pool['loss/smoothL1'] = loss2_3

        if fast_train:
            loss3 = 0.
        else:
            loss3 = tv(x_hat)
            data_pool['loss/tv'] = loss3

        if lambda_grad == 0. and fast_train:
            loss_grad = 0.
        else:
            loss_grad = grad_Loss()(x_hat * 255, x * 255)
            data_pool['loss/grad'] = loss_grad

        if lambda_grad_simple == 0. and fast_train:
            loss_grad_simple = 0.
        else:
            loss_grad_simple = grad_Loss_simple()(x_hat * 255, x * 255)
            data_pool['loss/grad_simple'] = loss_grad_simple

        if lambda_grad_simple_v2 == 0. and fast_train:
            loss_grad_simple_v2 = 0.
        else:
            loss_grad_simple_v2 = TVLoss2()(x_hat * 255, x * 255)
            data_pool['loss/grad_simple_v2'] = loss_grad_simple_v2

        lambda_dssim = data_pool.get('arg/lambda_dssim', 0)
        if lambda_dssim != 0:
            loss_module = self._models['dssim_loss']
            dssim_loss = loss_module(x_hat * 255, x * 255)
            data_pool['loss/dssim'] = dssim_loss
        else:
            dssim_loss = 0.

        # N * W * H
        num_pixels = x.shape[0] * x.shape[2] * x.shape[3]

        loss4 = torch.zeros(1, device=x.device)
        entropies = []
        for name in data_pool:
            if name.endswith('_likelihoods'):
                entropy = entropy_loss(self._data_pool[name]) / num_pixels
                prefix = name[:-len('_likelihoods')]
                entropies.append(
                    ('loss/{}_entropy_loss'.format(prefix), entropy))

        for e_name, entropy in entropies:
            data_pool[e_name] = entropy

        for name in data_pool:
            if name.endswith('_entropy_loss'):
                loss4 += data_pool[name]

        data_pool['loss/entropy_total'] = loss4

        loss_total = loss1 * lambda1 \
            + loss2 * lambda2 + loss2_1 * lambda2_1 + loss2_2 * lambda2_2 + loss2_3 * lambda2_3 \
            + loss4 * lambda4 \
            + loss_grad * lambda_grad + loss_grad_simple * lambda_grad_simple + loss_grad_simple_v2 * lambda_grad_simple_v2 \
            + dssim_loss * lambda_dssim

        lfs = self._data_pool.get('arg/lfs', None)
        if lfs:
            # print('use lfs get loss succeed!')
            # print(id(lfs), "loss", link.get_rank())
            loss_total = loss4 * lambda4 + lfs.get_loss(x, x_hat)

        # force_bpp = self._data_pool.get('arg/force_bpp', {'force': False})
        # if force_bpp and force_bpp['force']:
        #     bpp_goal = force_bpp['bpp']
        #     if loss4 < bpp_goal:
        #         print('pre')
        #         loss_total = loss_total - loss4 * lambda4
        #     else:
        #         print('post')
        #         loss_total = loss4 * lambda4

        data_pool['loss/total'] = loss_total


class EvalProcess(BaseProcess):
    def run(self, training=True):
        # TODO: refactor, add more options

        data_pool = self._data_pool

        psnr = PSNR()

        x_hat = data_pool['x_hat']
        gt_conf = data_pool.get('arg/test_gt', 'x')
        x = data_pool[gt_conf]

        loss_evals = []
        for name, value in data_pool.items():
            if name.startswith('loss/'):
                loss_evals.append(
                    ['eval/' + name[len('loss/'):], value.item()])
        for name, value in loss_evals:
            data_pool[name] = value

        data_pool['eval/msssim'] = msssim(x_hat, x, normalize=False).item()
        data_pool['eval/ms(db)'] = -10 * \
            torch.log10(torch.tensor(1 - data_pool['eval/msssim']))
        data_pool['eval/psnr'] = psnr(x_hat, x).item()
        # and use YRealBPPEvalProcess or YZRealBPPEvalProcess to calculate bpp


class YRealBPPEvalProcess(BaseProcess):
    def run(self, training=True):
        data_pool = self._data_pool
        y_save_path = data_pool['arg/y_save_path']
        x_shape = data_pool['tmp/x_shape']  # shape of x (before padding)
        data_pool['eval/bpp'] = calc_bpp_by_shape(x_shape, y_save_path)


class YZRealBPPEvalProcess(BaseProcess):
    def run(self, training=True):
        data_pool = self._data_pool
        y_save_path = data_pool['arg/y_save_path']
        z_save_path = data_pool['arg/z_save_path']
        x_shape = data_pool['tmp/x_shape']  # shape of x (before padding)
        y_bpp = calc_bpp_by_shape(x_shape, y_save_path)
        z_bpp = calc_bpp_by_shape(x_shape, z_save_path)
        data_pool['eval/y_bpp'] = y_bpp
        data_pool['eval/z_bpp'] = z_bpp
        data_pool['eval/bpp'] = y_bpp + z_bpp


class SampleImageProcess(BaseProcess):
    def run(self, training=True):
        data_pool = self._data_pool
        data_pool['image/original'] = data_pool['x']
        data_pool['image/reconstruction'] = data_pool['x_hat']
        data_pool['image/diff'] = diff_image_torch(
            data_pool['x'], data_pool['x_hat'])
        # x_d1 = (data_pool['x'] - data_pool['x_hat']).clamp(min=0)
        # data_pool['image/delta_x_x_hat'] = x_d1
        # x_d2 = (data_pool['x_hat'] - data_pool['x']).clamp(min=0)
        # data_pool['image/delta_x_hat_x'] = x_d2
        # data_pool['image/delta_norm_x_x_hat'] = x_d1 / x_d1.max()
        # data_pool['image/delta_norm_x_hat_x'] = x_d2 / x_d2.max()


class VideoSampleImageProcess(BaseProcess):
    def run(self, training=True):
        data_pool = self._data_pool
        data_pool['image/x_t'] = data_pool['xt']
        data_pool['image/x_t-1'] = data_pool['xts1']
        data_pool['image/reconstruction'] = data_pool['x_hat']
        data_pool['image/diff'] = diff_image_torch(
            data_pool['xt'], data_pool['x_hat'])


class BaseCompressProcess(BaseProcess):
    scope = None
    code_suffix = None
    save_path_arg = None

    def compress(self, save_path):
        raise NotImplementedError(self.__class__.__name__)

    def run(self, training=True):
        assert not training
        arg_name = 'arg/test_real_bpp'
        if arg_name not in self._data_pool or not self._data_pool[arg_name]:
            return

        save_path = self._data_pool[self.save_path_arg]
        try:
            self.compress(save_path)
        except Exception as e:
            print('caught exception in ' + self.__class__.__name__)
            raise e


class Base2DProbTableCompressProcess(BaseCompressProcess):
    scope = None
    code_suffix = '_tilde'
    prob_table_name = None

    def get_index(self, symbols):
        raise NotImplementedError(self.__class__.__name__)

    def get_symbols(self, code):
        bit = self._data_pool['arg/BIT']
        scale_factor = (1 << bit) - 1
        if 'arg/scale_factor_from_delta' in self._data_pool.keys():
            scale_factor = self._data_pool['arg/scale_factor_from_delta']
        return torch.round(code.detach() * scale_factor).cpu().int().numpy()

    def get_prob_table(self, symbols):
        raise NotImplementedError(self.__class__.__name__)

    def compress(self, save_path):
        data, models = self._data_pool, self._models
        code_name = self.scope + self.code_suffix
        code = data[code_name]
        symbols = self.get_symbols(code)
        index = self.get_index(symbols)
        prob_table = self.get_prob_table(symbols)
        data[self.prob_table_name] = prob_table
        adapt = data['arg/adapt']
        backend = data['arg/ae_backend']
        shift_to_zero = backend != 'dif'  # for dif backend, should keep the original data
        x_shape = data['tmp/x_shape']
        compress_with_index(symbols, index, save_path, adapt, prob_table, False, backend, shift_to_zero,
                            x_shape=x_shape)


class Base2DProbTableMockDecompressProcess(BaseProcess):
    decompress_arg_name = 'arg/decompress'
    save_path_arg_name = None
    prob_table_name = None
    output_name = None

    def get_index(self):
        return NotImplementedError(self.__class__.__name__)

    def run(self, training=True):
        data = self._data_pool
        if self.decompress_arg_name not in data or not data[self.decompress_arg_name]:
            return
        if training:
            return  # or can cause error
        codes = decompress_with_index(data[self.save_path_arg_name],
                                      self.get_index(),
                                      data['arg/adapt'],
                                      data[self.prob_table_name],
                                      backend='cc')
        codes = torch.tensor(codes, device=data['arg/device']).float()
        data[self.output_name] = codes


class YGSMMockDecompress(Base2DProbTableMockDecompressProcess):
    save_path_arg_name = 'arg/y_save_path'
    output_name = 'y_tilde'  # replace quantized y_tilde
    prob_table_name = 'tmp/y_prob_table'

    def get_index(self):
        return self._data_pool['tmp/y_index']


class ZFactorizedMockDecompress(Base2DProbTableMockDecompressProcess):
    save_path_arg_name = 'arg/z_save_path'
    output_name = 'z_tilde'
    prob_table_name = 'tmp/z_prob_table'

    @staticmethod
    def per_channel_idx_fn(_min, _table_len, shape):
        assert len(shape) == 4
        assert shape[0] == 1
        c, w, h = shape[1:]
        index = np.arange(c).repeat(w * h).reshape(shape)
        return index

    def get_index(self):
        return self.per_channel_idx_fn


class BaseFactorizedModelCompressProcess(Base2DProbTableCompressProcess):
    model_name = None

    def get_index(self, symbols):
        # each channel has a prob table
        flatten = np.arange(symbols.shape[1]).repeat(
            symbols.shape[2] * symbols.shape[3])
        return flatten.reshape(symbols.shape)

    def get_prob_table(self, symbols):
        data, models = self._data_pool, self._models

        entropy_pre = models[self.model_name]

        if data['arg/ae_backend'] == 'dif':
            return get_named_prob_table(entropy_pre, 'factorized', device=data['arg/device'])[0]

        bit = data['arg/BIT']
        scale_factor = (1 << bit) - 1
        if 'arg/scale_factor_from_delta' in data.keys():
            scale_factor = data['arg/scale_factor_from_delta']

        # todo refactor the method
        return to_prob_table(symbols, entropy_pre, data['arg/device'], scale_factor, True)


class YFactorizedModelCompressProcess(BaseFactorizedModelCompressProcess):
    scope = 'y'
    model_name = 'entropy_pre'
    save_path_arg = 'arg/y_save_path'
    prob_table_name = 'tmp/y_prob_table'


class ZFactorizedModelCompressProcess(BaseFactorizedModelCompressProcess):
    scope = 'z'
    model_name = 'z_entropy_pre'
    save_path_arg = 'arg/z_save_path'
    prob_table_name = 'tmp/z_prob_table'

    def get_prob_table(self, symbols):
        if self._data_pool.get('arg/use_table_file'):
            return self._data_pool['arg/prob_table/z_entropy_pre']
        else:
            return super().get_prob_table(symbols)


class YAlignSigmaProcess(BaseProcess):
    # TODO: add options for scale table
    SCALES_MIN = 0.11
    SCALES_MAX = 256
    SCALES_LEVELS = 64

    scale_table = list(np.exp(np.linspace(
        np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS)))

    @staticmethod
    def get_sigma_aligned_and_index(sigma, scale_table):
        sigma_aligned_idx = torch.zeros_like(sigma, dtype=torch.int)
        sigma_aligned = torch.full_like(
            sigma, scale_table[0], dtype=torch.float32)
        for ind, scale in enumerate(scale_table[:-1]):
            cmp_mask = (sigma > scale).int()
            cmp_mask_fp = cmp_mask.float()
            sigma_aligned_idx += cmp_mask
            sigma_aligned = (1 - cmp_mask_fp) * sigma_aligned + \
                cmp_mask_fp * scale_table[ind + 1]
        sigma_aligned_idx = sigma_aligned_idx.cpu().numpy()
        return sigma_aligned, sigma_aligned_idx

    def run(self, training=True):
        sigma = self._data_pool['sigma']
        sigma_aligned, sigma_aligned_idx = self.get_sigma_aligned_and_index(
            sigma, self.scale_table)
        self._data_pool['tmp/sigma_aligned'] = sigma_aligned
        self._data_pool['tmp/y_index'] = sigma_aligned_idx


class YMapSigmaProcess(BaseProcess):
    def run(self, training=True):
        prior = self._data_pool['prior']
        sigma = self._data_pool['sigma']
        # note that sigma_aligne here is the sigma calculated by SigmaCalculateProcess
        # which may have very small and unimportant difference with the values in the scale_table of YGSMCompressProcess and YAlignSigmaProcess
        # this process is to ensure that the y_index is correct, if we align again, y_index may shift one due to the above methoned difference
        self._data_pool['tmp/sigma_aligned'] = sigma
        self._data_pool['tmp/y_index'] = prior.clone().cpu().numpy()


class YAlignMuProcess(YAlignSigmaProcess):
    # TODO refactor this
    MEANS_MIN = 0.01
    MEANS_MAX = 64
    MEANS_LEVELS = 512

    mean_table = \
        list(-np.exp(np.linspace(
            np.log(MEANS_MIN), np.log(MEANS_MAX), MEANS_LEVELS)))[::-1] + \
        [0] + \
        list(np.exp(np.linspace(
            np.log(MEANS_MIN), np.log(MEANS_MAX), MEANS_LEVELS)))

    def run(self, training=True):
        mu = self._data_pool['mu']
        mu_aligned, mu_idx = self.get_sigma_aligned_and_index(
            mu, self.mean_table)
        self._data_pool['tmp/mu_aligned'] = mu_aligned
        self._data_pool['tmp/y_index'] = self._data_pool['tmp/y_index'] * \
            len(self.mean_table) + mu_idx


class YAlignOmegaProcess(YAlignMuProcess):
    # TODO refactor this
    OMEGAS_MIN = 0.01
    OMEGAS_MAX = 0.99
    OMEGAS_LEVELS = 64

    omega_table = \
        list(np.linspace(OMEGAS_MIN, OMEGAS_MAX, OMEGAS_LEVELS))

    def run(self, training=True):
        omega = self._data_pool['omega']
        omega_aligned, omega_idx = self.get_sigma_aligned_and_index(
            omega, self.omega_table)
        self._data_pool['tmp/omega_aligned'] = omega_aligned
        self._data_pool['tmp/y_index'] = self._data_pool['tmp/y_index'] * \
            len(self.omega_table) + omega_idx
        tmp = (torch.Tensor(self.omega_table) * 100).int()
        tmp[tmp < 1] = 1
        self._data_pool['tmp/omega_table'] = tmp.numpy()
        #
        # # gmm : mu : 1 K*C H W
        # N = omega.shape[0]
        # K = self._data_pool['arg/entropy_model_args'].get('K', 1)
        # if K > 0:
        #     assert N == 1
        #     len_scale_table = len(self.scale_table)
        #     len_mean_table = len(self.mean_table)
        #     len_omega_table = len(self.omega_table)
        #     len_params_table = len_scale_table * len_mean_table * len_omega_table
        #     weight = torch.ones(omega.shape, dtype=torch.int).view(N, K, -1)
        #     for i in range(1, K):
        #         weight[:, i, ...] = weight[:, i - 1, ...] * len_params_table
        #
        #     index = self._data_pool['tmp/y_index']
        #     index = torch.tensor(index, dtype=torch.int).view(N, K, -1)
        #     index *= weight
        #     index = index.sum(dim=1)
        #     size = self._data_pool['y'].size()
        #     index = index.view(size)
        #     self._data_pool['tmp/y_index'] = index.numpy()


class YGSMCompressProcess(Base2DProbTableCompressProcess):
    scope = 'y'
    model_name = 'entropy_pre'
    save_path_arg = 'arg/y_save_path'
    prob_table_name = 'tmp/y_prob_table'

    # TODO: add options for scale table
    SCALES_MIN = 0.11
    SCALES_MAX = 256
    SCALES_LEVELS = 64

    scale_table = list(np.exp(np.linspace(
        np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS)))

    @staticmethod
    def get_prob_table_with_sigma(_min, len_range_table, shape, entropy_pre, scale_table, scale_factor, device):
        _max = _min - 2 + len_range_table
        len_scale_table = len(scale_table)
        # each channel for each sigma
        y_range_map = np.array([list(range(_min, _max + 2))] * len_scale_table) \
            .reshape([1, len_scale_table, 1, len_range_table])
        y_range_rescaled_map = y_range_map / scale_factor
        scale_map = np.repeat(scale_table, len_range_table) \
            .reshape([1, len_scale_table, 1, len_range_table])

        y_range_rescaled_map = torch.tensor(
            y_range_rescaled_map, device=device).float()
        scale_map = torch.tensor(scale_map, device=device).float()

        # now we can get a [LEN_SCALE_TABLE X LEN_Y_SCALED] shaped prob table
        # and we will compress data according to this
        y_prob_table_per_sigma = entropy_pre(
            y_range_rescaled_map, 0., scale_map)
        y_prob_table = y_prob_table_per_sigma.reshape(
            [len_scale_table, len_range_table])
        y_prob_table = y_prob_table.cpu()
        y_prob_table *= shape[1] * shape[2] * shape[3]
        y_prob_table[y_prob_table < 1] = 1
        y_prob_table = y_prob_table.int().numpy()
        return y_prob_table

    def get_index(self, symbols):
        # calculated in previous process (YAlignSigmaProcess)
        index_name = 'tmp/y_index'
        assert index_name in self._data_pool
        return self._data_pool[index_name]

    def get_prob_table(self, symbols):
        data, models = self._data_pool, self._models
        entropy_pre = models[self.model_name]

        if data.get('arg/use_table_file'):
            return data['arg/prob_table/entropy_pre']

        if data['arg/ae_backend'] == 'dif':
            return get_named_prob_table(entropy_pre, 'scale-only', device=data['arg/device'])[0]

        bit = data['arg/BIT']
        scale_factor = (1 << bit) - 1
        if 'arg/scale_factor_from_delta' in data.keys():
            scale_factor = data['arg/scale_factor_from_delta']

        _min = symbols.min()
        _max = symbols.max()
        len_range_table = _max - _min + 2

        return self.get_prob_table_with_sigma(_min, len_range_table, symbols.shape,
                                              entropy_pre, self.scale_table, scale_factor,
                                              data['arg/device'])


class YGMMCompressProcess(YGSMCompressProcess):
    code_suffix = ''  # use symbol = quant(y-mu), so code is y but not y_tilde

    def get_symbols(self, code):
        mu = self._data_pool['mu']
        return super().get_symbols(code - mu)


class YMapMuSigmaProcess(BaseProcess):
    MEANS_MIN = -64.
    MEANS_MAX = 64.
    MEANS_LEVELS = 255

    mean_table = \
        list(np.tan(np.linspace(
            np.arctan(MEANS_MIN), np.arctan(MEANS_MAX), MEANS_LEVELS)))

    def run(self, training=True):
        sigma_ind = self._data_pool['tmp/sigma_index']
        mu_ind = self._data_pool['tmp/mu_index']
        self._data_pool['tmp/sigma_aligned'] = self._data_pool['sigma']
        self._data_pool['tmp/mu_aligned'] = self._data_pool['mu']
        self._data_pool['tmp/y_index'] = (sigma_ind * len(
            self.mean_table) + mu_ind + 127).detach().cpu().numpy()


class YSubMuQuantProcess(BaseProcess):
    """
    y_tilde = Round(y - mu) + mu
    """
    y_name = 'y'
    mu_name = 'mu'
    output_name = 'y_tilde'

    def run(self, training=True):
        data = self._data_pool
        y = data[self.y_name]
        mu = data[self.mu_name]
        out = torch.round(y - mu) + mu
        data[self.output_name] = out


class YGMMWithMuTableCompressProcess(YGSMCompressProcess):
    # TODO: should not inherit from YGSMCompressProcess
    MEANS_MIN = 0.01
    MEANS_MAX = 64
    MEANS_LEVELS = 512

    mean_table = \
        list(-np.exp(np.linspace(
            np.log(MEANS_MIN), np.log(MEANS_MAX), MEANS_LEVELS)))[::-1] + \
        [0] + \
        list(np.exp(np.linspace(
            np.log(MEANS_MIN), np.log(MEANS_MAX), MEANS_LEVELS)))

    @staticmethod
    def get_prob_table_with_params(_min, len_range_table, shape, entropy_pre,
                                   scale_table, mean_table, scale_factor, device):
        _max = _min - 2 + len_range_table
        len_scale_table = len(scale_table)
        len_mean_table = len(mean_table)
        len_params_table = len_scale_table * len_mean_table
        # each channel for each sigma
        y_range_map = np.array([list(range(_min, _max + 2))] * len_params_table) \
            .reshape([1, len_params_table, 1, len_range_table])
        y_range_rescaled_map = y_range_map / scale_factor

        scale_map = np.repeat(scale_table, len_range_table * len_mean_table) \
            .reshape([1, len_params_table, 1, len_range_table])
        mean_map = np.tile(np.repeat(mean_table, len_range_table), len_scale_table) \
            .reshape([1, len_params_table, 1, len_range_table])

        y_range_rescaled_map = torch.tensor(
            y_range_rescaled_map, device=device).float()
        scale_map = torch.tensor(scale_map, device=device).float()
        mean_map = torch.tensor(mean_map, device=device).float()

        # now we can get a [LEN_SCALE_TABLE X LEN_Y_SCALED] shaped prob table
        # and we will compress data according to this
        y_prob_table_per_param = entropy_pre(
            y_range_rescaled_map, mean_map, scale_map)
        y_prob_table = y_prob_table_per_param.reshape(
            [len_params_table, len_range_table])
        y_prob_table = y_prob_table.cpu()
        y_prob_table *= shape[1] * shape[2] * shape[3]
        y_prob_table[y_prob_table < 1] = 1
        y_prob_table = y_prob_table.int().numpy()
        return y_prob_table

    def get_prob_table(self, symbols):
        data, models = self._data_pool, self._models
        bit = data['arg/BIT']
        scale_factor = (1 << bit) - 1
        if 'arg/scale_factor_from_delta' in data.keys():
            scale_factor = data['arg/scale_factor_from_delta']

        _min = symbols.min()
        _max = symbols.max()
        len_range_table = _max - _min + 2

        entropy_pre = models[self.model_name]

        return self.get_prob_table_with_params(_min, len_range_table, symbols.shape,
                                               entropy_pre,
                                               self.scale_table, self.mean_table,
                                               scale_factor,
                                               data['arg/device'])

    """
    use gmm_k to decide whether Gaussian or Gaussian Mixture
    """

    def compress(self, save_path):
        data, models = self._data_pool, self._models
        code_name = self.scope + self.code_suffix
        code = data[code_name]
        symbols = self.get_symbols(code)
        index = self.get_index(symbols)
        prob_table = self.get_prob_table(symbols)
        data[self.prob_table_name] = prob_table
        adapt = data['arg/adapt']
        gmm_k = data.get('arg/entropy_model_args', {}).get('K', None)
        if gmm_k is None:
            compress_with_index(symbols, index, save_path,
                                adapt, prob_table, False, 'cc')
        else:
            # print('use my compress')
            compress_with_index(symbols, index, save_path, adapt, prob_table, False, 'cc',
                                omega_t=data['tmp/omega_table'])


class YGMMMultiTableCompressProcess(YGSMCompressProcess, YAlignOmegaProcess):
    """
    a implement of gmm compress process but occupy too much memory
    see:  YGMMWithMuTableCompressProcess
    """

    @staticmethod
    def get_prob_table_with_params(_min, len_range_table, shape, entropy_pre,
                                   scale_table, mean_table, omega_table, scale_factor, K, device):
        _max = _min - 2 + len_range_table
        len_scale_table = len(scale_table)
        len_mean_table = len(mean_table)
        len_omega_table = len(omega_table)
        len_params_table = len_scale_table * len_mean_table * len_omega_table
        len_table = len_params_table ** K
        # each channel for each sigma
        y_range_map = np.array([list(range(_min, _max + 2))] * len_table) \
            .reshape([1, len_table, 1, len_range_table])
        y_range_rescaled_map = y_range_map / scale_factor

        y_range_rescaled_map = torch.tensor(y_range_rescaled_map).float()

        l = []
        for i in range(K):
            l.append(scale_table)
            l.append(mean_table)
            l.append(omega_table)

        # TODO : use np.nditer
        raw_map = torch.Tensor([*itertools.product(*l)])
        scale_map = raw_map[..., range(0, 3 * K, 3)]
        mean_map = raw_map[..., range(1, 3 * K, 3)]
        omega_map = raw_map[..., range(2, 3 * K, 3)]

        # print(raw_map.size())

        def warp(x):
            # print("QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ")
            # print(type(x))
            y = x.repeat_interleave(int(len_range_table), dim=1)
            # print(x.size())
            # print(y.size())
            y = y.reshape(len_table, K, len_range_table).permute(
                1, 0, 2).contiguous()
            return y.reshape(
                [1, K * len_table, 1, len_range_table]).float()

        scale_map = warp(scale_map)
        mean_map = warp(mean_map)
        omega_map = warp(omega_map)
        y_range_rescaled_map = y_range_rescaled_map.to(device)
        scale_map = scale_map.to(device)
        mean_map = mean_map.to(device)
        omega_map = omega_map.to(device)

        y_prob_table_per_param = entropy_pre(
            y_range_rescaled_map, mean_map, scale_map, omega_map)
        y_prob_table = y_prob_table_per_param.reshape(
            [len_table, len_range_table])
        y_prob_table = y_prob_table.cpu()
        y_prob_table *= shape[1] * shape[2] * shape[3]
        y_prob_table[y_prob_table < 1] = 1
        y_prob_table = y_prob_table.int().numpy()
        return y_prob_table

    def get_prob_table(self, symbols):
        data, models = self._data_pool, self._models
        bit = data['arg/BIT']
        scale_factor = (1 << bit) - 1
        if 'arg/scale_factor_from_delta' in data.keys():
            scale_factor = data['arg/scale_factor_from_delta']

        _min = symbols.min()
        _max = symbols.max()
        len_range_table = _max - _min + 2

        entropy_pre = models[self.model_name]

        K = self._data_pool['arg/entropy_model_args'].get('K', 1)

        return self.get_prob_table_with_params(_min, len_range_table, symbols.shape,
                                               entropy_pre,
                                               self.scale_table, self.mean_table, self.omega_table,
                                               scale_factor, K, data['arg/device'])


class BaseMapProcess(BaseProcess):
    input_name = None
    output_name = None

    def run(self, training=True):
        if self.input_name is None or self.output_name is None:
            raise NotImplementedError(self.__class__.__name__)
        self._data_pool[self.output_name] = self._data_pool[self.input_name]

    @classmethod
    def get_map_cls(cls, input_name, output_name):
        _in, _out = input_name, output_name

        class WrappedMapProcess(BaseMapProcess):
            input_name = _in
            output_name = _out

        return WrappedMapProcess


class SigmaCalculateProcess(BaseProcess):
    input_name = 'sigma'
    output_name = 'sigma'

    # TODO: add options for scale table
    SCALES_MIN = 0.11
    SCALES_MAX = 256.
    SCALES_LEVELS = 64.0
    LOG_MIN = np.log(SCALES_MIN)
    LOG_MAX = np.log(SCALES_MAX)
    step = (LOG_MAX - LOG_MIN) / (SCALES_LEVELS - 1)

    def calculate_sigma_from_prior(self, prior):
        # GG19I EQ15, only used in training
        return torch.exp(self.LOG_MIN + self.step * prior)

    def run(self, training=True):
        if self.input_name is None or self.output_name is None:
            raise NotImplementedError(self.__class__.__name__)
        self._data_pool[self.output_name] = \
            self.calculate_sigma_from_prior(self._data_pool[self.input_name])

    @classmethod
    def get_cls(cls, input_name, output_name):
        _in, _out = input_name, output_name

        class WrappedSigmaCalculateProcess(SigmaCalculateProcess):
            input_name = _in
            output_name = _out

        return WrappedSigmaCalculateProcess


class MuCalculateProcess(BaseProcess):
    input_name = 'mu'
    output_name = 'mu'

    # TODO: add options for scale table
    MEANS_MIN = 0.01
    MEANS_MAX = 64.
    MEANS_LEVELS = 512.
    LOG_MIN = np.log(MEANS_MIN)
    LOG_MAX = np.log(MEANS_MAX)
    step = (LOG_MAX - LOG_MIN) / (MEANS_LEVELS - 1)

    def run(self, training=True):
        if self.input_name is None or self.output_name is None:
            raise NotImplementedError(self.__class__.__name__)

        prior = self._data_pool[self.input_name]  # prior: [0, 1024]
        prior -= self.MEANS_LEVELS  # [0, 1024] -> [-512, 512]
        sign = torch.sign(prior)  # {0, 1, -1}
        prior *= sign  # [0, 512]
        prior -= 1.  # [-1, 511], -1 for zero
        abs_val = torch.exp(self.LOG_MIN + self.step * prior)
        mu = sign * abs_val
        self._data_pool[self.output_name] = mu

    @classmethod
    def get_cls(cls, input_name, output_name):
        _in, _out = input_name, output_name

        class WrappedProcess(cls):
            input_name = _in
            output_name = _out

        WrappedProcess.__name__ = 'Wrapped' + cls.__name__

        return WrappedProcess


class MuCalculateTanProcess(BaseProcess):
    # use tan function to discretize mu
    input_name = 'mu'
    output_name = 'mu'

    # TODO: add options for scale table
    MEANS_MAX = 64.
    MEANS_LEVELS = 128.
    ATAN_MAX = np.arctan(MEANS_MAX)
    step = ATAN_MAX / (MEANS_LEVELS - 1)

    def run(self, training=True):
        if self.input_name is None or self.output_name is None:
            raise NotImplementedError(self.__class__.__name__)

        prior = self._data_pool[self.input_name]  # prior: [0, 1024]

        # try to use linear function , but gradient is small
        # mu = 6. * prior / 127.
        mu = torch.tan(self.step * prior)
        self._data_pool[self.output_name] = mu

    @classmethod
    def get_cls(cls, input_name, output_name):
        _in, _out = input_name, output_name

        class WrappedProcess(cls):
            input_name = _in
            output_name = _out

        WrappedProcess.__name__ = 'Wrapped' + cls.__name__

        return WrappedProcess


class BaseLambdaProcess(BaseProcess):
    input_name: None
    output_name: None
    lambda_str: None

    def __init__(self, models, data_pool: dict, *args, **kwargs):
        super().__init__(models, data_pool)
        lambda_fn = eval(self.lambda_str)
        assert callable(lambda_fn)
        self.fn = lambda_fn
        try:
            # after python36
            self.n_args = lambda_fn.__code__.co_argcount
        except:
            # before python36
            self.n_args = lambda_fn.func_code.co_argcount

    def run(self, training=True):
        n_args = self.n_args
        if n_args == 0:
            args = []
        elif n_args == 1:
            args = [self._data_pool[self.input_name]]
        elif n_args == 2:
            args = [self._data_pool[self.input_name], self._data_pool]
        else:
            raise NotImplementedError(
                'cannot handle lambda/function with n_arg={}'.format(n_args))

        self._data_pool[self.output_name] = self.fn(*args)


class AlignAndPadInputProcess(BaseProcess):
    def run(self, training=True):
        data_pool = self._data_pool
        x = data_pool['x']
        _, _, x_h, x_w = _, _, x_h_ori, x_w_ori = data_pool['tmp/x_shape'] = x.shape
        align = 64
        x_h = (x_h // align + int(x_h % align > 0)) * align
        x_w = (x_w // align + int(x_w % align > 0)) * align
        if (x_h, x_w) != (x_h_ori, x_w_ori):
            if data_pool.get('arg/zero_pad', False):
                x = nn.ZeroPad2d((0, x_w - x_w_ori, 0, x_h - x_h_ori))(x)
            else:
                x = nn.ReflectionPad2d((0, x_w - x_w_ori, 0, x_h - x_h_ori))(x)
            data_pool['x'] = x


class VideoAlignAndPadInputProcess(BaseProcess):
    def run(self, training=True):
        data_pool = self._data_pool
        x = data_pool['x']
        _, _, x_h, x_w = x_n, x_c, x_h_ori, x_w_ori = x.shape
        data_pool['tmp/x_shape'] = (x_n, x_c // 2, x_h, x_w)
        align = 64
        x_h = (x_h // align + int(x_h % align > 0)) * align
        x_w = (x_w // align + int(x_w % align > 0)) * align
        if (x_h, x_w) != (x_h_ori, x_w_ori):
            if data_pool.get('arg/zero_pad', False):
                x = nn.ZeroPad2d((0, x_w - x_w_ori, 0, x_h - x_h_ori))(x)
            else:
                x = nn.ReflectionPad2d((0, x_w - x_w_ori, 0, x_h - x_h_ori))(x)
            data_pool['x'] = x


class RemovePadProcess(BaseProcess):
    def run(self, training=True):
        data_pool = self._data_pool
        x = data_pool['x']
        x_hat = data_pool['x_hat']
        x_pre = data_pool['tmp/x_after_pre']
        assert x.shape == x_hat.shape
        assert x.shape == x_pre.shape
        n, c, h, w = shape = data_pool['tmp/x_shape']
        if x.shape != shape:
            assert x.shape[0] == n
            assert x.shape[1] == c
            x = x[..., :h, :w]
            x_pre = x_pre[..., :h, :w]
            x_hat = x_hat[..., :h, :w]
        data_pool['x'] = x
        data_pool['x_hat'] = x_hat
        data_pool['tmp/x_after_pre'] = x_pre


class VideoRemovePadProcess(BaseProcess):
    def run(self, training=True):
        data_pool = self._data_pool
        x = data_pool['x']
        x_hat = data_pool['x_hat']
        x_pre = data_pool['tmp/x_after_pre']
        n, c, h, w = shape = data_pool['tmp/x_shape']
        if x.shape != shape:
            x = x[..., :h, :w]
            x_pre = x_pre[..., :h, :w]
            x_hat = x_hat[..., :h, :w]
        data_pool['x'] = x
        data_pool['x_hat'] = x_hat
        data_pool['xt'] = x[:, :3]
        data_pool['xts1'] = x[:, 3:]
        data_pool['tmp/x_after_pre'] = x_pre


class SplitProcess(BaseProcess):
    def run(self, training=True):
        if training:
            pass
        else:
            data_pool = self._data_pool
            x = data_pool['x']
            assert x.shape[0] == 1
            x = x.reshape([-1, x.shape[1], 64, x.shape[3]])
            data_pool['x'] = x


class MergeProcess(BaseProcess):
    def merge(self, x):
        x.reshape([1, 3, -1, x.shape[3]])

    def run(self, training=True):
        if training:
            pass
        else:
            data_pool = self._data_pool
            data_pool['x'] = self.merge(data_pool['x'])
            data_pool['x_hat'] = self.merge(data_pool['x_hat'])


class DataAugmentProcess(BaseProcess):
    def run(self, training=True):
        if training:
            inputs = self._data_pool['x']
            outputs = self._models['data_augment'](inputs)
            if isinstance(outputs, dict):
                for k, v in outputs.items():
                    self._data_pool[k] = v
            elif isinstance(outputs, torch.Tensor):
                self._data_pool['x'] = outputs


class DataCollectProcess(BaseProcess):
    def run(self, training=True):
        if training:
            x = self._data_pool['x']
            x_hat = self._data_pool['x_hat']
            flag = self._data_pool.get('tmp/x_collect_flag', None)
            self._models['data_collect'](x, x_hat, flag=flag)


class YecProcess(BaseSingleModelProcess):
    input_name = 'img'
    output_name = 'y'
    kwargs = {'reverse': False}
    model_name = 'y_encoder'

    def run(self, training=True):
        data = self._data_pool
        assert self.input_name == 'img', f"input name is {self.input_name}"
        inputs = data[self.input_name]
        inputs = self.pre_model(inputs)
        qmap = data['qmap'].to(inputs.device)
        if self._models[self.model_name] is None:
            # for safety
            # all used models should not be None
            raise AssertionError('missed model: {}'.format(self.model_name))

        output = self._models[self.model_name](inputs, qmap)
        data[self.output_name] = output

class ZecProcess(BaseSingleModelProcess):
    input_name = 'y'
    output_name = 'z'
    kwargs = {'reverse': False}
    model_name = 'z_encoder'

    def run(self, training=True):
        data = self._data_pool
        inputs = data[self.input_name]
        inputs = self.pre_model(inputs)
        qmap = data['qmap'].to(inputs.device)
        if self._models[self.model_name] is None:
            # for safety
            # all used models should not be None
            raise AssertionError('missed model: {}'.format(self.model_name))

        output = self._models[self.model_name](inputs, qmap)
        data[self.output_name] = output


class ZdcProcess(BaseSingleModelProcess):
    input_name = 'z_tilde'
    output_name = 'w'
    model_name = 'z_condition'

class YdcProcess(BaseSingleModelProcess):
    input_name = 'y_tilde'
    output_name = 'x_hat'
    kwargs = {'reverse': False}
    model_name = 'y_decoder'

    def run(self, training=True):
        data = self._data_pool
        inputs = data[self.input_name]
        w = data['w'].to(inputs.device)
        if self._models[self.model_name] is None:
            # for safety
            # all used models should not be None
            raise AssertionError('missed model: {}'.format(self.model_name))

        output = self._models[self.model_name](inputs, w)
        data[self.output_name] = output

class QmapLossProcess(BaseProcess):
    #
    def quality2lambda(self, qmap):
        return 1e-2 * torch.exp(4.382 * qmap)

    def run(self, qmap, training=True):
        data_pool = self._data_pool

        tv = TV()
        entropy_loss = Entropy_loss()
        #
        lmbdamap = quality2lambda(qmap)
        # criterion = PixelwiseRateDistortionLoss()
        # metric = Metrics()

        x_hat = data_pool['x_hat']

        gt_conf = data_pool.get('arg/train_gt', 'x')
        x = data_pool[gt_conf]


        lambda1 = data_pool['arg/lambda1']
        lambda2 = data_pool['arg/lambda2']
        lambda2_1 = 0
        lambda2_2 = 0
        lambda2_3 = 0
        if 'arg/lambda2_1' in data_pool:
            lambda2_1 = data_pool['arg/lambda2_1']
        if 'arg/lambda2_2' in data_pool:
            lambda2_2 = data_pool['arg/lambda2_2']
        if 'arg/lambda2_3' in data_pool:
            lambda2_3 = data_pool['arg/lambda2_3']
        lambda4 = data_pool['arg/lambda4']
        lambda_grad = data_pool.get('arg/lambda_grad', 0.)
        lambda_grad_simple = data_pool.get('arg/lambda_grad_simple', 0.)
        lambda_grad_simple_v2 = data_pool.get('arg/lambda_grad_simple_v2', 0.)


        fast_train = True
        cur_epoch = data_pool.get("tmp/cur_epoch", None)

        if lambda1 == 0. and fast_train:
            loss1 = 0.
        else:
            loss1 = 1 - msssim(x_hat, x, normalize=True)
            data_pool['loss/msssim'] = loss1

        if lambda2 == 0. and fast_train:
            loss2 = 0.
        else:
            loss2 = nn.MSELoss(reduction='mean')(x_hat, x) * 65025 * lmbdamap# 255 ** 2
            data_pool['loss/mse'] = loss2

        if lambda2_1 == 0. and fast_train:
            loss2_1 = 0.
        else:
            #loss2_1 = torch.sqrt(loss2)
            loss2_tmp = nn.MSELoss(reduction='mean')(
                x_hat, x) * 65025  # 255 ** 2
            loss2_1 = torch.sqrt(loss2_tmp)
            data_pool['loss/rmse'] = loss2_1

        if lambda2_2 == 0. and fast_train:
            loss2_2 = 0.
        else:
            loss2_2 = YUV_Loss()(x_hat * 255, x * 255)
            data_pool['loss/yuv'] = loss2_2

        if fast_train:
            loss2_2_1 = 0.
        else:
            loss2_2_1 = YUV_Loss('Y')(x_hat * 255, x * 255)
            data_pool['loss/Y'] = loss2_2_1

        if lambda2_3 == 0. and fast_train:
            loss2_3 = 0.
        else:
            loss2_3 = nn.SmoothL1Loss(reduction='mean')(x_hat * 255, x * 255)
            data_pool['loss/smoothL1'] = loss2_3

        if fast_train:
            loss3 = 0.
        else:
            loss3 = tv(x_hat)
            data_pool['loss/tv'] = loss3

        if lambda_grad == 0. and fast_train:
            loss_grad = 0.
        else:
            loss_grad = grad_Loss()(x_hat * 255, x * 255)
            data_pool['loss/grad'] = loss_grad

        if lambda_grad_simple == 0. and fast_train:
            loss_grad_simple = 0.
        else:
            loss_grad_simple = grad_Loss_simple()(x_hat * 255, x * 255)
            data_pool['loss/grad_simple'] = loss_grad_simple

        if lambda_grad_simple_v2 == 0. and fast_train:
            loss_grad_simple_v2 = 0.
        else:
            loss_grad_simple_v2 = TVLoss2()(x_hat * 255, x * 255)
            data_pool['loss/grad_simple_v2'] = loss_grad_simple_v2

        lambda_dssim = data_pool.get('arg/lambda_dssim', 0)
        if lambda_dssim != 0:
            loss_module = self._models['dssim_loss']
            dssim_loss = loss_module(x_hat * 255, x * 255)
            data_pool['loss/dssim'] = dssim_loss
        else:
            dssim_loss = 0.

        # N * W * H
        num_pixels = x.shape[0] * x.shape[2] * x.shape[3]

        loss4 = torch.zeros(1, device=x.device)
        entropies = []
        for name in data_pool:
            if name.endswith('_likelihoods'):
                entropy = entropy_loss(self._data_pool[name]) / num_pixels
                prefix = name[:-len('_likelihoods')]
                entropies.append(
                    ('loss/{}_entropy_loss'.format(prefix), entropy))

        for e_name, entropy in entropies:
            data_pool[e_name] = entropy

        for name in data_pool:
            if name.endswith('_entropy_loss'):
                loss4 += data_pool[name]

        data_pool['loss/entropy_total'] = loss4

        loss_total = loss1 * lambda1 \
            + loss2 * lambda2 + loss2_1 * lambda2_1 + loss2_2 * lambda2_2 + loss2_3 * lambda2_3 \
            + loss4 * lambda4 \
            + loss_grad * lambda_grad + loss_grad_simple * lambda_grad_simple + loss_grad_simple_v2 * lambda_grad_simple_v2 \
            + dssim_loss * lambda_dssim


        data_pool['loss/total'] = loss_total





class IDFEncodeProcess(BaseSingleModelProcess):
    input_name = ['tmp/x_after_pre']
    output_name = ['z_tilde', 'tmp/pys', 'tmp/ys', 'tmp/zs']
    kwargs = {'reverse': False}
    model_name = 'flow'

    def run(self, training=True):
        data = self._data_pool

        inputs = []
        for key in self.input_name:
            inputs.append(data[key])

        if self._models[self.model_name] is None:
            # for safety
            # all used models should not be None
            raise AssertionError('missed model: {}'.format(self.model_name))

        outputs = self._models[self.model_name](*inputs, **self.kwargs)
        for key, value in zip(self.output_name, outputs):
            data[key] = value


class IDFDecodeProcess(IDFEncodeProcess):
    input_name = ['z_tilde', 'tmp/pys', 'tmp/ys']
    output_name = ['x_hat']
    kwargs = {'reverse': True}


class IDFCodingProcess(BaseProcess):
    input_name = 'tmp/x_after_pre'
    output_name = ['eval/real_bpd', 'eval/recon_error']
    model_name = 'flow'

    def run(self, training=True):
        data = self._data_pool
        with torch.no_grad():
            outputs = encode_patches(data[self.input_name], self._models[self.model_name])
        if outputs[1] != 0:
            print(f'\treal_bpd: {outputs[0]},recon_error: {outputs[1]}')
        for key, value in zip(self.output_name, outputs):
            data[key] = value


class SplitYsProcess(BaseProcess):
    def run(self, training=True):
        data = self._data_pool
        for i, y in enumerate(data['tmp/ys']):
            data[f'y{i + 1}_tilde'] = y


class SplitPysProcess(BaseProcess):
    def run(self, training=True):
        data = self._data_pool
        for i, py in enumerate(data['tmp/pys']):
            data[f'y{i + 1}_mu'] = py[0]
            data[f'y{i + 1}_sigma'] = py[1]


class IDFYsEntropyProcess(BaseProcess):
    """
        y_likelihoods <- entropy_pre(pys, ys)
    """
    model_name = 'entropy_pre'
    output_name = 'tmp/y_likelihoods'

    def run(self, training=True):
        # Add likelihoods of intermediate representations.
        data = self._data_pool
        pys = data['tmp/pys']
        ys = data['tmp/ys']
        likelihoods = []
        entropy = self._models[self.model_name]
        for px, x in zip(pys, ys):
            mean = px[0]
            scale = torch.exp(px[1])
            likelihood = entropy(x, mean, scale)
            likelihoods.append(likelihood)

        data[self.output_name] = likelihoods


class IDFZLMMParameterProcess(BaseProcess):
    """
        Logistic Mixture Model

        get parameters of logistic mixture model of z_tilde

        mu,sigma,omega <- prior(z_tilde)
    """
    input_name = 'z_tilde'
    output_name = ['mu', 'sigma', 'omega']
    model_name = 'prior'

    def run(self, training=True):
        data = self._data_pool

        if self._models[self.model_name] is None:
            # for safety
            # all used models should not be None
            raise AssertionError('missed model: {}'.format(self.model_name))

        pz, _, _ = self._models[self.model_name](data[self.input_name], None)
        for key, value in zip(self.output_name, pz):
            data[key] = value


class IDFZLMMCompleteEntropyProcess(GMMCompleteEntropyProcess):
    """
        Logistic Mixture Model

        z_likelihoods <- entropy_pre(z_tilde, mu, sigma, omega)
    """
    scope = 'z'


class IDFLossProcess(BaseProcess):
    """
        loss <- y_likelihoods,z_likelihoods
    """

    def run(self, training=True):
        data = self._data_pool
        shape = data['x'].shape
        y_likelihoods = data['tmp/y_likelihoods']
        z_likelihoods = data['z_likelihoods']

        log_pz = torch.sum(torch.log(z_likelihoods), dim=[1, 2, 3])
        data['eval/loss_z'] = -log_pz.mean()
        log_p = log_pz
        for i, py in enumerate(y_likelihoods):
            log_py = torch.sum(torch.log(py), dim=[1, 2, 3])
            data[f'eval/loss_y{i}'] = -log_py.mean()
            log_p += log_py

        loss = -log_p
        bpd = loss.detach() / (np.prod(shape[1:]) * np.log(2.))

        data['loss/total'] = loss.mean()
        data['eval/bpd'] = bpd.mean()


class LosslessCheckProcess(BaseProcess):
    """
        Check whether x_after_pre and x_hat are the same
    """

    def run(self, training=True):
        data = self._data_pool
        x_in = data['tmp/x_after_pre'].detach().cpu().numpy()
        x_out = data['x_hat'].detach().cpu().numpy()
        distance = np.max(np.abs(x_in - x_out))
        if distance > 1/255:
            print('distance between x_in and x_out:', distance)


class JPEGEncodeProcess(ModelInvokeProcess):
    '''
    jpeg split 8x8 block, dct, table quant + ste.
    '''
    input_names = ['tmp/x_after_pre']
    output_names = ['y_tilde', 'u_tilde', 'v_tilde']
    lambda_name = 'jpeg_encoder'

class JPEGDecodeProcess(ModelInvokeProcess):
    '''
    jpeg table dequant, idct, combine 8x8 block.
    '''
    input_names = ['y_tilde', 'u_tilde', 'v_tilde']
    output_names = ['x_hat']
    lambda_name = 'jpeg_decoder'

class DCTFlattenProcess(ModelInvokeProcess):
    '''
    dct coefficent flatten for entropy estimate
    '''
    input_names = ['y_tilde', 'u_tilde', 'v_tilde']
    output_names = ['y_flat_tilde', 'u_flat_tilde', 'v_flat_tilde']
    lambda_name = 'dct_flatten'


class YDCTFactorizedEntropyProcess(BaseFactorizedEntropyProcess):
    """
    estimate entropy of dct coefficent
    """
    scope = 'y_flat'
    model_name = 'y_entropy_pre'
class UDCTFactorizedEntropyProcess(BaseFactorizedEntropyProcess):
    """
    estimate entropy of dct coefficent
    """
    scope = 'u_flat'
    model_name = 'u_entropy_pre'
class VDCTFactorizedEntropyProcess(BaseFactorizedEntropyProcess):
    """
    estimate entropy of dct coefficent
    """
    scope = 'v_flat'
    model_name = 'v_entropy_pre'

class ResidualProcess(ModelInvokeProcess):
    """Residual between original frame and reconstructed frame.
    """
    input_names = ["tmp/x_after_pre", "x_hat"]
    kwargs_names = {}
    lambda_name = "residual"
    output_names = ["x_residual"]


class ResYEncodeProcess(BaseSingleModelProcess):
    """Residual Y encoder
    """
    input_name = 'x_residual'
    output_name = 'ry'
    model_name = 'ry_encoder'


class ResYQuantizeProcess(BaseSingleModelProcess):
    """Residual Y quantize
    """
    input_name = 'ry'
    output_name = 'ry_tilde'
    model_name = 'ry_quant'


class ResYDecodeProcess(BaseSingleModelProcess):
    """Residual Y decoder
    """
    input_name = 'ry_tilde'
    output_name = 'residual_hat'
    model_name = 'ry_decoder'


class ResZAbsEncodeProcess(ZEncodeProcess):
    """Residual Z Abs encoder
    """
    input_name = 'ry'
    output_name = 'rz'
    model_name = 'rz_encoder'

    def pre_model(self, inputs):
        return inputs.abs()


class ResZQuantizeProcess(BaseSingleModelProcess):
    """Residual Z quantize
    """
    input_name = 'rz'
    output_name = 'rz_tilde'
    model_name = 'rz_quant'


class ResZDecodeProcess(BaseSingleModelProcess):
    """Residual Z decoder
    """
    input_name = 'rz_tilde'
    output_name = 'r_prior'
    model_name = 'rz_decoder'


class ResContextProcess(BaseSingleModelProcess):
    """Residual context model
    """
    input_name = 'ry_tilde'
    output_name = 'ry_context'
    model_name = 'ry_context'


class ResPriorAndContextParameterProcess(PriorAndContextParameterProcess):
    """Residual Prior and Context model
    """
    prior_name = 'r_prior'
    context_name = 'ry_context'
    param_model = 'ry_parameter'
    sigma_name = 'r_sigma'
    mu_name = 'r_mu'


class ResZFactorizedEntropyProcess(BaseFactorizedEntropyProcess):
    """
    z_likelihoods <- z_entropy_pre(z_tilde)
    """
    scope = 'rz'
    model_name = 'rz_entropy_pre'


class ResGMMEntropyProcess(GMMEntropyProcess):
    """
    Gaussian Mixture Model (Mock)

    y_likelihoods <- entropy_pre(y_tilde, mu, sigma)
    """

    scope = 'ry'
    model_name = 'r_entropy_pre'
    sigma_name = 'r_sigma'
    mu_name = 'r_mu'


class ResYAlignSigmaProcess(YAlignSigmaProcess):

    def run(self, training=True):
        sigma = self._data_pool['r_sigma']
        sigma_aligned, sigma_aligned_idx = self.get_sigma_aligned_and_index(sigma, self.scale_table)
        self._data_pool['tmp/r_sigma_aligned'] = sigma_aligned
        self._data_pool['tmp/ry_index'] = sigma_aligned_idx


class ResYAlignMuProcess(YAlignMuProcess):

    def run(self, training=True):
        mu = self._data_pool['r_mu']
        mu_aligned, mu_idx = self.get_sigma_aligned_and_index(mu, self.mean_table)
        self._data_pool['tmp/r_mu_aligned'] = mu_aligned
        self._data_pool['tmp/ry_index'] = self._data_pool['tmp/ry_index'] * len(self.mean_table) + mu_idx


class ResYGMMWithMuTableCompressProcess(YGMMWithMuTableCompressProcess):
    scope = 'ry'
    model_name = 'r_entropy_pre'
    save_path_arg = 'arg/ry_save_path'
    prob_table_name = 'tmp/ry_prob_table'

    def get_index(self, symbols):
        # calculated in previous process (YAlignSigmaProcess)
        index_name = 'tmp/ry_index'
        assert index_name in self._data_pool
        return self._data_pool[index_name]


class ResZFactorizedModelCompressProcess(BaseFactorizedModelCompressProcess):
    scope = 'rz'
    model_name = 'rz_entropy_pre'
    save_path_arg = 'arg/rz_save_path'
    prob_table_name = 'tmp/rz_prob_table'

    def get_prob_table(self, symbols):
        if self._data_pool.get('arg/use_table_file'):
            return self._data_pool['arg/prob_table/rz_entropy_pre']
        else:
            return super().get_prob_table(symbols)


class ResYZRealBPPEvalProcess(BaseProcess):
    def run(self, training=True):
        data_pool = self._data_pool
        y_save_path = data_pool['arg/y_save_path']
        ry_save_path = data_pool["arg/ry_save_path"]
        rz_save_path = data_pool['arg/rz_save_path']
        x_shape = data_pool['tmp/x_shape']  # shape of x (before padding)
        y_bpp = calc_bpp_by_shape(x_shape, y_save_path)
        ry_bpp = calc_bpp_by_shape(x_shape, ry_save_path)
        rz_bpp = calc_bpp_by_shape(x_shape, rz_save_path)
        data_pool['eval/y_bpp'] = y_bpp
        data_pool["eval/ry_bpp"] = ry_bpp
        data_pool['eval/rz_bpp'] = rz_bpp
        data_pool['eval/bpp'] = y_bpp + ry_bpp + rz_bpp


def get_process(name: str, attrs=None):
    name_spl = name.split('.')
    try:
        module, name = '.'.join(name_spl[:-1]), name_spl[-1]
        process_cls = importlib.import_module(module).__dict__[name]

        assert attrs is None or isinstance(attrs, dict)

        class WrappedProcess(process_cls):
            def __init__(self, *args, **kwargs):
                if attrs:
                    for a_name, attr in attrs.items():
                        self.__setattr__(a_name, attr)
                self.__class__.__name__ = 'Wrapped' + name
                print('inited dynamic class: ' + self.__class__.__name__)
                super().__init__(*args, **kwargs)

        return WrappedProcess
    except Exception:
        raise ValueError('Unsupported process name: {}'.format(name))


if __name__ == '__main__':
    from nets.entropy import entropy_models, SymmetricConditional

    e = entropy_models('gaussian')
    sigma_t = list(range(10, 13))
    mu_t = list(range(20, 23))
    omega_t = list(range(30, 33))
    prob_table: np.ndarray = YGMMMultiTableCompressProcess.get_prob_table_with_params(
        1, 5, [1, 3, 5, 5], e, sigma_t, mu_t, omega_t, 1., 2, torch.device('cpu:0'))
    # raw_table 729 * s1 m1 o1 s2 m2 o2
    # 3 * 3 * 3 = 27, 27 ** 2 = 729
    print(prob_table.shape)

class Compressor(BaseSingleModelProcess):
    """
    y_tilde <- y_quant(y)
    """
    input_name = 'x'
    output_name = 'bits'
    model_name = 'Compressor'

    def run(self, training=True):
        data = self._data_pool

        inputs = data['x']
        inputs = torch.round(inputs * 255)
        inputs = self.pre_model(inputs)
        if self._models[self.model_name] is None:
            # for safety
            # all used models should not be None
            raise AssertionError('missed model: {}'.format(self.model_name))

        outputs = self._models[self.model_name](inputs)
        outputs = self.post_model(outputs) #Bits
        data['loss/total'] = outputs.get_total_bits() / inputs.shape[0]
        data['eval/bpd'] = data['loss/total'].detach() / (np.prod(inputs.shape[1:]))
