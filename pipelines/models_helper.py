import importlib
import pprint

from pipelines.models import *
# print("check namespace!")
# assert "QmapCompressionCodec" in dir()

from pipelines.processes import get_process

try:
    from nas.codec_oneshot import CodecOneShot
except ImportError:
    CodecOneShot = None


def build_codec(args):
    """
    build a codec with given args

    fields in the args:

    - `processes`: list, each element in it defines a process.

      - If an element is a string, will get a process using `get_process` with the given name.
      - Otherwise, will build a process using `get_process` with `process` and `attrs` fields in the element

    :param args: dict-like arguments defining the processes in a codec pipeline
    :return: a class inheriting `BaseCodec`
    """
    try:
        processes = args['processes']

        def _load_process(proc):
            assert isinstance(proc, str) or isinstance(proc, dict)
            if isinstance(proc, str):
                return get_process(proc)
            proc['process'] = get_process(proc['process'], proc.get('attrs', None))
            print(proc)
            return proc

        class WrappedCodec(BaseCodec):
            process_cls_list = list(map(_load_process, processes))

        return WrappedCodec
    except Exception as e:
        print('Bad args:')
        pprint.pprint(args)
        raise e


def get_codec(args, models: dict, nas=False, codec_cfg={}):
    """
    get(or build) a `BaseCodec` instance by given args/name

    :param args: str or dict-like args.
      If is a str, return builtin codec whose name is the given name string (gg17, gg18 or gg18c).
      Otherwise will build the codec using `build_codec`.
    :param models: the sub-models
    :return: a `BaseCodec` instance
    """
    if not isinstance(args, str):
        return build_codec(args)(models)
    name: str = args

    if name == 'gg17':
        codec_cls = GG17Codec
    elif name == 'gg18':
        codec_cls = GG18Codec
    elif name == 'gg18-mix-quant':
        codec_cls = MixQuantGG18Codec
    elif name == 'gg18q':
        codec_cls = GG18QCodec
    elif name == 'sigma-q':
        codec_cls = SigmaQuantCodec
    elif name == 'sigma-q-context':
        codec_cls = ContextualSigmaQuantCodec
    elif name == 'gg18q-mix-quant':
        codec_cls = MixQuantGG18QCodec
    elif name == 'gg18c':
        codec_cls = ContextualCodec
    elif name == 'gg18c-fixed':
        codec_cls = ContextualFixedCodec
    elif name == 'gg18c-fake-serial':
        codec_cls = FakeSerialContextualCodec
    elif name == 'gg18cq':
        codec_cls = ContextualQuantCodec
    elif name in ['gg20c', 'gg18c-mix-quant']:
        codec_cls = MixQuantContextualCodec
    elif name == 'gg20c-iter':
        codec_cls = MixQuantGG20IterCodec
    elif name == 'gg20c-iter-context':
        codec_cls = MixQuantGG20IterContextualCodec
    elif name == 'edic':
        codec_cls = EDICCodec
    elif name == 'edic-context':
        codec_cls = EDICContextualCodec
    elif name == 'edic-context-fake-serial':
        codec_cls = FakeSerialEDICContextualCodec
    elif name == 'edic-share-weight':
        codec_cls = EDICTwoPathShareWeightCodec
    elif name == 'edic-share-weight-gate':
        codec_cls = EDICTwoPathShareWeightGateCodec
    elif name == 'gg18da':
        codec_cls = DataAugmentCodec
    elif name == 'checkerboard':
        codec_cls = CheckerboardTwoPathContextualCodec
    elif name == 'checkerboard-share-weight':
        codec_cls = CheckerboardShareWeightContextualCodec
    elif name == 'checkerboard-share-weight-fixed':
        codec_cls = CheckerboardShareWeightContextualFixedCodec
    elif name == 'checkerboard-share-weight-fixed-mq':
        codec_cls = MixQuantCheckerboardShareWeightContextualFixedCodec
    elif name == 'checkerboard-md':
        codec_cls = MultiDimensionContextualCodec
    elif name == 'cevideo':
        codec_cls = CEVideoCodec
    elif name == 'idf':
        codec_cls = IDFCodec
    elif name == 'idf_bottleneck':
        codec_cls = IDF_Bottleneck
    # DCT related codec
    elif name == "dct-idf":
        codec_cls = DCTIDFCodec
    elif name == "dct-gg18c":
        codec_cls = DCTGG18CTX
    elif name == "dct-rc":
        codec_cls = DCTRC
    elif name == "gg20c-iter-cc":
        codec_cls = MixQuantGG20IterCodecCC
    elif name == "gg20c-iter-cc-two-path":
        codec_cls = MixQuantGG20IterCodecCCTwoPath
    elif name == 'knights-position':
        codec_cls = KnightContextualCodec
    elif name == 'knights-position-share-weight':
        codec_cls = KnightContextualShareWeightCodec
    elif name == 'multiscale':
        codec_cls = Multiscale

    elif name == 'qualitymap':
        codec_cls = QmapCompressionCodec
    else:
        # dynamically import
        try:
            name_spl = name.split('.')
            module, name = '.'.join(name_spl[:-1]), name_spl[-1]
            codec_cls = importlib.import_module(module).__dict__[name]
        except Exception:
            raise NotImplementedError('unsupported codec: ' + name)

    if nas:
        codec = codec_cls(models)
        return CodecOneShot(codec, codec_cfg=codec_cfg)

    return codec_cls(models)


if __name__ == '__main__':
    from nets.context import context_models
    from nets.decoder import decoders
    from nets.encoder import encoders
    from nets.entropy import entropy_models
    from nets.param import param_models
    from nets.post.enhance import enhance
    from nets.attention import attention
    from quant.quantizator import quantizators
    import pprint

    models = {
        'y_encoder': encoders(),
        'y_decoder': decoders(),
        'y_enhance': enhance('NONE'),
        'entropy_pre': entropy_models("NONE"),
        'z_encoder': encoders(),
        'z_attention': attention('NONE'),
        'z_decoder': decoders(),
        'z_entropy_pre': entropy_models("NONE"),
        'y_parameter': param_models(),
        'y_context': context_models(),
        'y_quant': quantizators(),
        'z_quant': quantizators(),
    }

    # get_codec('gg18', models, nas=True)
    test_gg17 = get_codec('gg17', models)
    pprint.pprint(test_gg17.processes)
    pprint.pprint(test_gg17.data_pool)

    for stage in CodecStageEnum:
        test_gg17.print_processes(stage)

    test_gg18 = get_codec('gg18', models)
    pprint.pprint(test_gg17.processes)

    for stage in CodecStageEnum:
        test_gg18.print_processes(stage)

    print(test_gg18._modules['sub_models']['y_decoder'])
    print(test_gg18._modules['sub_models'].y_decoder)
