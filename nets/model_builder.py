import collections

from nets.augment.data_augment import data_augment
from nets.augment.data_collect import data_collect
from nets.context import context_models
from nets.decoder import decoders
from nets.encoder import encoders
from nets.entropy import entropy_models
from nets.pre.pre_builder import pre
from nets.post.post_builder import post
from nets.attention import attention
from nets.param import param_models
from nets.flow import flows
from quant.quantizator import quantizators
import importlib

DYNAMIC_MODEL_ARCH_FIELD_NAME = 'model_arch'
DYNAMIC_VARIABLE_FIELD_NAME = 'vars'
DYNAMIC_DEFAULT_REQUIRE_GRAD = 'default_require_grad'


def model_builder(configs) -> dict:
    """
    get networks according to configs
    :param configs: config dict
    :return: dict consists of name-model pairs
    """
    if DYNAMIC_MODEL_ARCH_FIELD_NAME in configs:
        # dynamic building mode
        return _dynamic_model_builder(configs)

    y_encoders_kwargs = configs.get('y_encode_args', {})
    z_encoders_kwargs = configs.get('z_encode_args', {})
    entropy_model_kwargs = configs.get('entropy_model_args', {})
    y_decoders_kwargs = configs.get('y_decode_args', {})
    z_decoders_kwargs = configs.get('z_decode_args', {})
    y_parameter_kwargs = configs.get('y_parameter_args', {})
    y_context_kwargs = configs.get('y_context_args', {})
    z_attention_kwargs = configs.get('z_attention_args', {})
    y_post_kwargs = configs.get('y_post_args', {})
    y_pre_kwargs = configs.get('y_pre_args', {})
    y_quant_kwargs = configs.get('y_quant_args', {})
    z_quant_kwargs = configs.get('z_quant_args', {})

    que = collections.deque(maxlen=256)

    models = {
        'y_encoder': encoders(configs.y_encode, out_channels=configs.num_features_encode,
                              **y_encoders_kwargs),
        'y_decoder': decoders(configs.y_decode, out_channels=configs.num_features_decode,
                              **y_decoders_kwargs),
        'entropy_pre': entropy_models(configs.entropy_model, configs.num_features_entropy_model,
                                      **entropy_model_kwargs),
        'z_encoder': encoders(configs.z_encode,
                              in_channels=configs.num_features_encode,
                              out_channels=configs.num_features_encode,
                              **z_encoders_kwargs),
        'z_attention': attention(configs.get('z_attention', 'NONE'),
                                 in_channels=configs.num_features_encode,
                                 out_channels=configs.num_features_encode,
                                 **z_attention_kwargs),
        'z_decoder': decoders(configs.z_decode,
                              in_channels=configs.num_features_encode,
                              out_channels=configs.num_features_decode,
                              **z_decoders_kwargs),
        'z_entropy_pre': entropy_models(configs.z_entropy_model, configs.num_features_entropy_model),
        'y_parameter': param_models(configs.get('y_parameter', 'NONE'),
                                    in_channels=y_parameter_kwargs.get('in_c',
                                                                       configs.num_features_encode * 4),
                                    out_channels=y_parameter_kwargs.get('out_c',
                                                                        configs.num_features_encode * 2),
                                    **y_parameter_kwargs),
        'y_context': context_models(configs.get('y_context', 'NONE'),
                                    in_channels=configs.num_features_encode,
                                    out_channels=configs.num_features_encode * 2,
                                    **y_context_kwargs),
        'y_quant': quantizators(configs.y_quant, train=True, **y_quant_kwargs),
        'z_quant': quantizators(configs.z_quant, train=True, **z_quant_kwargs),
        'y_quant_round': quantizators(configs.get('y_quant_round', 'round'), train=True),
        'z_quant_round': quantizators(configs.get('z_quant_round', 'round'), train=True),
        'post': post(configs.get('post', 'NONE'), **y_post_kwargs),
        'pre': pre(configs.get('pre', 'NONE'), **y_pre_kwargs),
        'data_augment': data_augment(configs.get('data_augment', 'NONE'), que=que),
        'data_collect': data_collect(configs.get('data_collect', 'NONE'), que=que),
        'flow': flows(configs.get('flow', 'NONE')),
        'z_condition': decoders(configs.get('z_con', 'NONE'),
                              in_channels=configs.num_features_encode,
                              out_channels=configs.num_features_decode,
                              **z_decoders_kwargs),
    }
    return models


def _dynamic_model_builder(configs):
    """
    dynamically build models according to given configs

    :param configs: experiment configuration
    :return: models dict
    """
    arch = configs[DYNAMIC_MODEL_ARCH_FIELD_NAME]
    variables = configs[DYNAMIC_VARIABLE_FIELD_NAME]
    default_require_grad = configs.get(DYNAMIC_DEFAULT_REQUIRE_GRAD, True)
    models = {}
    for model_name, model_config in arch.items():
        import_path = model_config['import']
        module_spl = import_path.split('.')
        module, cls_name = '.'.join(module_spl[:-1]), module_spl[-1]
        try:
            callable_cls_or_func = importlib.import_module(module).__dict__[cls_name]
        except Exception:
            raise ValueError('unexpected network: ' + import_path)

        def var_replacer(val):
            """
            replace variable placeholder string $VAL_NAME to its corresponding value

            :param val: value to be checked
            :return: if `val` is a var string, return its corresponding value, otherwise return `val` itself
            """
            if isinstance(val, str) and val.startswith('$'):
                var_name = val[1:]
                coef = 1
                if var_name[0] == '(':
                    for i in range(1, len(var_name)):
                        if var_name[i] == ')':
                            coef_str = var_name[1:i]
                            try:
                                coef = eval(coef_str)
                                var_name = var_name[i + 1:]
                            except ValueError:
                                coef = 1
                            break
                if var_name not in variables:
                    raise ValueError('Unexpected config variable:{} (parsed to {})'.format(val, var_name))
                value = variables[var_name]
                if coef == 1:
                    return value
                else:
                    t = type(value)
                    try:
                        return t(coef * value)  # may raise an exception
                    except Exception:
                        raise ValueError(f'failed to parse {coef} * {value} to {t} ')
            return val

        def var_replacer_recurrent(val):
            val = var_replacer(val)
            if isinstance(val, list):
                val = list(map(var_replacer_recurrent, val))
            elif isinstance(val, dict):
                val = {k: var_replacer_recurrent(v) for k, v in val.items()}

            return val

        args = var_replacer_recurrent(model_config.get('args', []))
        kwargs = var_replacer_recurrent(model_config.get('kwargs', {}))
        requires_grad = model_config.get('requires_grad', default_require_grad)
        model = callable_cls_or_func(*args, **kwargs)

        # NOTICE:
        # we do not use an anonymous wrapper class
        # because the var `callable_cls_or_func` may be a function
        # so hack the forward method with a decorator
        if not requires_grad:
            import types

            for param in model.parameters():
                param.requires_grad = False

            extra_repr = model.extra_repr

            def decorated_extra_repr(self):
                return '[NoGrad]\n' + extra_repr()

            model.extra_repr = types.MethodType(decorated_extra_repr, model)

        models[model_name] = model
    return models
