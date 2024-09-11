from nets.base import PreparableMixin
from pipelines.processes import *
from enum import Enum
from utils.test_time import mark
import inspect


class CodecStageEnum(Enum):
    TRAIN = 1
    VALID = 2
    TEST = 3
    COMPRESS = 4
    DECOMPRESS = 5
    SEARCH = 6


class BaseCodec(nn.Module, PreparableMixin):
    process_cls_list = None
    common_pre_process_list = [
        AlignAndPadInputProcess,
        preProcess,
    ]
    common_post_process_list = [
        postProcess,
        RemovePadProcess,
        {
            'process': LossProcess,
            'exclude': {CodecStageEnum.COMPRESS, CodecStageEnum.DECOMPRESS}
        },
        {
            'process': SampleImageProcess,
            'exclude': {CodecStageEnum.COMPRESS, CodecStageEnum.DECOMPRESS}
        },
        {
            'process': EvalProcess,
            'only': {CodecStageEnum.VALID, CodecStageEnum.TEST, CodecStageEnum.SEARCH},
        },
    ]

    def __init__(self, models):
        super().__init__()
        self.sub_models = nn.ModuleDict(models)

        self._data_pool = {}
        process_cls_list = self.get_process_cls_list()
        if process_cls_list is None:
            raise NotImplementedError

        def _proc_wrapper(proc):
            if inspect.isclass(proc):
                proc = {'process': proc, 'only': set(CodecStageEnum)}
            elif not isinstance(proc, dict):
                raise TypeError('should be a Process class or a dict: {}'.format(proc))

            if 'only' not in proc.keys():
                proc['only'] = set(CodecStageEnum)
            if 'exclude' not in proc.keys():
                proc['exclude'] = {}

            for key in ['only', 'exclude']:
                if isinstance(proc[key], CodecStageEnum):
                    proc[key] = {proc[key]}
                elif isinstance(proc[key], str):
                    proc[key] = {CodecStageEnum[proc[key]]}
                proc[key] = {CodecStageEnum[p] if isinstance(p, str) else p
                             for p in proc[key]}
            proc['only'] = set(proc['only']) - set(proc['exclude'])
            return proc

        process_cls_list = map(_proc_wrapper, process_cls_list)

        self.processes = [
            {
                'process': p['process'](self.sub_models, self._data_pool),
                'only': p['only']
            } for p in process_cls_list]

    def print_processes(self, stage):
        for process in self.processes:
            if stage not in process['only']:
                continue

            print('{} stage include {}'.format(stage, process['process']), flush=True)

    @property
    def data_pool(self):
        return self._data_pool

    def get_process_cls_list(self):
        return self.common_pre_process_list + self.process_cls_list + self.common_post_process_list

    def forward(self, x, stage=None, extra_inputs=None, args=None, clear_pool=True):
        """

        the data in _data_pool will be updated INPLACE during the forward

        :param x: the tensor of input images
        :param stage: the stage of codec. If none, set to CodecStageEnum.TRAIN
        :param extra_inputs: other tensors that will be used during forward
        :param args: args that will be used during forward
        :param clear_pool: whether clear data pool before forward
        :return:
        """

        if stage is None:
            if self.training:
                stage = CodecStageEnum.TRAIN
            else:
                raise ValueError('Cannot inference codec stage when codec.training is False')

        assert self.training == (stage == CodecStageEnum.TRAIN)

        test_prec_time = args.get('test_time', False)
        if test_prec_time:
            print('testing precious running speed')

        if clear_pool:
            self._data_pool.clear()
        # TODO: delete
        if isinstance(x, dict):
            for k, v in x.items():
                if torch.is_tensor(v):
                    self._data_pool[k] = v
            self._data_pool["x"] = x.get("img", None)
            self._data_pool["tmp/x_shape"] = self._data_pool["x"].shape
            self._data_pool["tmp/cur_epoch"] = x["cur_epoch"]
        else:
            self._data_pool['x'] = x

        if extra_inputs:
            assert isinstance(extra_inputs, dict)
            for k, v in extra_inputs.items():
                self._data_pool[k] = v

        if args:
            assert isinstance(args, dict)
            for k, v in args.items():
                self._data_pool['arg/' + k] = v

        for process in self.processes:
            if stage not in process['only']:
                continue
            try:
                process['process'].run(self.training)  # 调用执行process，返回值放入到data_pool
            except Exception as e:
                print('Exception in process {}'.format(process['process']))
                raise e
            if test_prec_time:
                torch.cuda.synchronize(args.device)
            mark(name=process['process'].__class__.__name__)

        return self._data_pool


class GG17Codec(BaseCodec):
    process_cls_list = [
        # Y auto-encoder
        YEncodeProcess,
        YQuantizeProcess,
        YDecodeProcess,

        # Entropy
        YFactorizedEntropyProcess,

        # Compress
        {
            'process': YFactorizedModelCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },

        # Eval BPP
        {
            'process': YRealBPPEvalProcess,
            'only': {CodecStageEnum.VALID, CodecStageEnum.TEST},
        },

        # TODO Decompress
    ]


class GG18Codec(BaseCodec):
    process_cls_list = [
        # Y & Z encoding
        YEncodeProcess,
        YQuantizeProcess,
        ZAbsEncodeProcess,
        ZQuantizeProcess,

        # Z entropy
        ZFactorizedEntropyProcess,

        # Z compressing
        {
            'process': ZFactorizedModelCompressProcess,
            'exclude': {CodecStageEnum.TRAIN, CodecStageEnum.SEARCH},
        },

        # Z mock decompressing
        {
            'process': ZFactorizedMockDecompress,
            'exclude': {CodecStageEnum.TRAIN, CodecStageEnum.SEARCH},
        },

        # Z decoding
        ZDecodeProcess,

        # Y entropy
        BaseMapProcess.get_map_cls('prior', 'sigma'),
        GSMEntropyProcess,

        # Y compressing
        {
            'process': YAlignSigmaProcess,
            'exclude': {CodecStageEnum.TRAIN, CodecStageEnum.SEARCH},
        },
        {
            'process': YGSMCompressProcess,
            'exclude': {CodecStageEnum.TRAIN, CodecStageEnum.SEARCH},
        },

        # Y mock decompressing
        {
            'process': YGSMMockDecompress,
            'exclude': {CodecStageEnum.TRAIN, CodecStageEnum.SEARCH},
        },

        # Y decoding
        YDecodeProcess,

        # Eval BPP
        {
            'process': YZRealBPPEvalProcess,
            'only': {CodecStageEnum.VALID, CodecStageEnum.TEST},
        },
    ]


class MixQuantGG18Codec(BaseCodec):
    """
    this is GG18Codec with mix-quantizator trick introduced by gg20c

    see: CHANNEL-WISE AUTOREGRESSIVE ENTROPY MODELS FOR LEARNED IMAGE COMPRESSION
    """
    process_cls_list = [
        # Y & Z encoding
        YEncodeProcess,
        YQuantizeProcess,  # noise quant, -> y_tilde
        YRoundQuantizeProcess,  # ste quant    -> y_round
        ZAbsEncodeProcess,
        ZQuantizeProcess,  # noise quant, -> z_tilde
        ZRoundQuantizeProcess,  # ste quant    -> z_round

        # Z entropy
        ZFactorizedEntropyProcess,

        # Z compressing
        {
            'process': ZFactorizedModelCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },

        # Z mock decompressing
        {
            'process': ZFactorizedMockDecompress,
            'exclude': CodecStageEnum.TRAIN
        },

        # Z decoding
        ZDecodeProcess.create_io_process(input='z_round'),

        # Y entropy
        BaseMapProcess.get_map_cls('prior', 'sigma'),
        GSMEntropyProcess,

        # Y compressing
        {
            'process': YAlignSigmaProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YGSMCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },

        # Y mock decompressing
        {
            'process': YGSMMockDecompress,
            'exclude': CodecStageEnum.TRAIN
        },

        # Y decoding
        YDecodeProcess.create_io_process(input='y_round'),

        # Eval BPP
        {
            'process': YZRealBPPEvalProcess,
            'only': {CodecStageEnum.VALID, CodecStageEnum.TEST},
        },
    ]


class MixQuantGG20IterCodec(BaseCodec):
    """
    this is GG18Codec with mix-quantizator trick introduced by gg20c

    see: CHANNEL-WISE AUTOREGRESSIVE ENTROPY MODELS FOR LEARNED IMAGE COMPRESSION
    """
    process_cls_list = [
        # Y auto-encoder
        YEncodeProcess,
        YQuantizeProcess,
        YRoundQuantizeProcess,

        # Z auto-encoder
        ZEncodeProcess,
        ZQuantizeProcess,
        ZRoundQuantizeProcess,
        ZDecodeProcess.create_io_process(input='z_round'),

        # Context
        ChannelSliceParameterProcess,
        ParameterGroupSplitProcess,

        # Y Decode
        YDecodeProcess.create_io_process(input='y_lrp'),

        # Entropy
        ZFactorizedEntropyProcess,
        GMMEntropyProcess,

        # Compress
        {
            'process': YAlignSigmaProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YGMMCompressProcess,  # NOTICE: q(y-mu) trick
            'exclude': CodecStageEnum.TRAIN,
        },
        {
            'process': ZFactorizedModelCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },

        # Eval BPP
        {
            'process': YZRealBPPEvalProcess,
            'only': {CodecStageEnum.VALID, CodecStageEnum.TEST},
        },
    ]


class MixQuantGG20IterContextualCodec(BaseCodec):
    """
    this is GG18Codec with mix-quantizator trick introduced by gg20c

    see: CHANNEL-WISE AUTOREGRESSIVE ENTROPY MODELS FOR LEARNED IMAGE COMPRESSION
    """
    process_cls_list = [
        # Y auto-encoder
        YEncodeProcess,
        YQuantizeProcess,
        YRoundQuantizeProcess,

        # Z auto-encoder
        ZEncodeProcess,
        ZQuantizeProcess,
        ZRoundQuantizeProcess,
        ZDecodeProcess.create_io_process(input='z_round'),

        # Context
        ChannelSliceParameterProcess,
        ParameterGroupSplitProcess,
        BaseMapProcess.get_map_cls('mu', 'mu1'),
        BaseMapProcess.get_map_cls('sigma', 'sigma1'),

        BaseMapProcess.get_map_cls('prior', 'prior2'),
        Context2Process,
        PriorAndContext2Parameter2Process,

        # Y Decode
        YDecodeProcess.create_io_process(input='y_lrp'),

        ################
        # Entropy
        {
            'process': ParameterGateProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': GMMEntropyProcess,
            'exclude': CodecStageEnum.TRAIN
        },

        ZFactorizedEntropyProcess,
        GMMEntropy1Process,
        GMMEntropy2Process,
        TwoPathEntropyLossProcess,

        BaseMapProcess.get_map_cls('y_entropy2', 'y_entropy3'),  # hack, for use GammaEntropyWeightedSumUp

        {
            'process': Entropy211WeightedSumProcess,
            'only': CodecStageEnum.TRAIN
        },
        ########################

        # Compress
        {
            'process': YAlignSigmaProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YGMMCompressProcess,  # NOTICE: q(y-mu) trick
            'exclude': CodecStageEnum.TRAIN,
        },
        {
            'process': ZFactorizedModelCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },

        # Eval BPP
        {
            'process': YZRealBPPEvalProcess,
            'only': {CodecStageEnum.VALID, CodecStageEnum.TEST},
        },
    ]


class MixQuantContextualCodec(BaseCodec):
    """
    see: CHANNEL-WISE AUTOREGRESSIVE ENTROPY MODELS FOR LEARNED IMAGE COMPRESSION

    this is ContextualCodec with mix-quantizator trick introduced by gg20c
    """

    process_cls_list = [
        # Y auto-encoder
        YEncodeProcess,
        YQuantizeProcess,
        YRoundQuantizeProcess,
        YDecodeProcess.create_io_process(input='y_round'),

        # Z auto-encoder
        ZAbsEncodeProcess,
        ZQuantizeProcess,
        ZRoundQuantizeProcess,
        ZDecodeProcess.create_io_process(input='z_round'),

        # Context
        # TODO decompress
        ContextProcess.create_io_process(input='y_round'),

        # Entropy
        PriorAndContextParameterProcess,
        ZFactorizedEntropyProcess,
        GMMEntropyProcess,

        # Compress
        {
            'process': YAlignSigmaProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        # {
        #     'process': YAlignMuProcess,
        #     'exclude': CodecStageEnum.TRAIN
        # },
        # {
        #     'process': YGMMWithMuTableCompressProcess,
        #     'exclude': CodecStageEnum.TRAIN,
        # },
        {
            'process': YGMMCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },
        {
            'process': ZFactorizedModelCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },

        # Eval BPP
        {
            'process': YZRealBPPEvalProcess,
            'only': {CodecStageEnum.VALID, CodecStageEnum.TEST},
        },

        # TODO Decompress
    ]


class GG18QCodec(BaseCodec):
    process_cls_list = [
        # Y auto-encoder
        YEncodeProcess,
        YQuantizeProcess,
        YDecodeProcess,

        # Z auto-encoder
        ZAbsEncodeProcess,
        ZQuantizeProcess,
        ZDecodeProcess,

        # Entropy
        SigmaCalculateProcess.get_cls('prior', 'sigma'),
        ZFactorizedEntropyProcess,
        GSMEntropyProcess,

        # Compress
        {
            'process': YMapSigmaProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': ZFactorizedModelCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },
        {
            'process': YGSMCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },

        # Eval BPP
        {
            'process': YZRealBPPEvalProcess,
            'only': {CodecStageEnum.VALID, CodecStageEnum.TEST},
        },

        # TODO Decompress
    ]

class QmapCompressionCodec(BaseCodec):
    process_cls_list = [
        # Y & Z encoding
        YecProcess,
        YQuantizeProcess,
        ZecProcess,
        ZQuantizeProcess,

        # Z entropy
        ZFactorizedEntropyProcess,

        # Z compressing
        {
            'process': ZFactorizedModelCompressProcess,
            'exclude': {CodecStageEnum.TRAIN, CodecStageEnum.SEARCH},
        },

        # Z mock decompressing
        {
            'process': ZFactorizedMockDecompress,
            'exclude': {CodecStageEnum.TRAIN, CodecStageEnum.SEARCH},
        },

        # Z decoding
        ZDecodeProcess,

        # mu,sigma split
        PriorParameterProcess,

        # Y entropy
        GSMEntropyProcess,

        # Y compressing
        {
            'process': YAlignSigmaProcess,
            'exclude': {CodecStageEnum.TRAIN, CodecStageEnum.SEARCH},
        },
        {
            'process': YGSMCompressProcess,
            'exclude': {CodecStageEnum.TRAIN, CodecStageEnum.SEARCH},
        },

        # Y mock decompressing
        {
            'process': YGSMMockDecompress,
            'exclude': {CodecStageEnum.TRAIN, CodecStageEnum.SEARCH},
        },

        # Y decoding
        ZdcProcess,
        YdcProcess,

        # Eval BPP
        {
            'process': YZRealBPPEvalProcess,
            'only': {CodecStageEnum.VALID, CodecStageEnum.TEST},
        },
    ]
    common_post_process_list = [
        postProcess,
        RemovePadProcess,
        {
            'process': QmapLossProcess,
            'exclude': {CodecStageEnum.COMPRESS, CodecStageEnum.DECOMPRESS}
        },
        {
            'process': SampleImageProcess,
            'exclude': {CodecStageEnum.COMPRESS, CodecStageEnum.DECOMPRESS}
        },
        {
            'process': EvalProcess,
            'only': {CodecStageEnum.VALID, CodecStageEnum.TEST, CodecStageEnum.SEARCH},
        },
    ]

class MixQuantGG18QCodec(BaseCodec):
    """
    GG18QCodec with mix-quant trick

    refer to: MixQuantContextualCodec
    """
    process_cls_list = [
        # Y auto-encoder
        YEncodeProcess,
        YQuantizeProcess,
        YRoundQuantizeProcess,
        YDecodeProcess.create_io_process('y_round'),

        # Z auto-encoder
        ZAbsEncodeProcess,
        ZQuantizeProcess,
        ZRoundQuantizeProcess,
        ZDecodeProcess.create_io_process('z_round'),

        # Entropy
        SigmaCalculateProcess.get_cls('prior', 'sigma'),
        ZFactorizedEntropyProcess,
        GSMEntropyProcess,

        # Compress
        {
            'process': YMapSigmaProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YGSMCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },
        {
            'process': ZFactorizedModelCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },

        # Eval BPP
        {
            'process': YZRealBPPEvalProcess,
            'only': {CodecStageEnum.VALID, CodecStageEnum.TEST},
        },

        # TODO Decompress
    ]


class ContextualCodec(BaseCodec):
    """
    > Joint Autoregressive and Hierarchical Priors for Learned Image Compression

    see: https://arxiv.org/abs/1809.02736
    """
    process_cls_list = [
        # Y auto-encoder
        YEncodeProcess,
        YQuantizeProcess,
        YDecodeProcess,

        # Z auto-encoder
        ZAbsEncodeProcess,
        ZQuantizeProcess,
        ZDecodeProcess,

        # Context
        # TODO decompress
        ContextProcess,

        # Entropy
        PriorAndContextParameterProcess,
        ZFactorizedEntropyProcess,
        GMMEntropyProcess,

        # Compress
        {
            'process': YAlignSigmaProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YAlignMuProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YGMMWithMuTableCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },
        {
            'process': ZFactorizedModelCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },

        # Eval BPP
        {
            'process': YZRealBPPEvalProcess,
            'only': {CodecStageEnum.VALID, CodecStageEnum.TEST},
        },

        # TODO Decompress
    ]


class ContextualFixedCodec(BaseCodec):
    """
    > Joint Autoregressive and Hierarchical Priors for Learned Image Compression

    see: https://arxiv.org/abs/1809.02736

    Fixed: AbsZDecode -> ZDecode
    """
    process_cls_list = [
        # Y auto-encoder
        YEncodeProcess,
        YQuantizeProcess,
        YDecodeProcess,

        # Z auto-encoder
        ZEncodeProcess,
        ZQuantizeProcess,
        ZDecodeProcess,

        # Context
        # TODO decompress
        ContextProcess,

        # Entropy
        PriorAndContextParameterProcess,
        ZFactorizedEntropyProcess,
        GMMEntropyProcess,

        # Compress
        {
            'process': YAlignSigmaProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YAlignMuProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YGMMWithMuTableCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },
        {
            'process': ZFactorizedModelCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },

        # Eval BPP
        {
            'process': YZRealBPPEvalProcess,
            'only': {CodecStageEnum.VALID, CodecStageEnum.TEST},
        },

        # TODO Decompress
    ]


class FakeSerialContextualCodec(BaseCodec):
    """
    Fake serial codec for running speed test
    """
    process_cls_list = [
        # Y auto-encoder
        YEncodeProcess,
        YQuantizeProcess,
        YDecodeProcess,

        # Z auto-encoder
        ZAbsEncodeProcess,
        ZQuantizeProcess,
        ZDecodeProcess,

        # Context
        # TODO decompress
        FakeSerialPriorAndContextParameterProcess,

        # Entropy
        ZFactorizedEntropyProcess,
        GMMEntropyProcess,

        # Compress
        {
            'process': YAlignSigmaProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YAlignMuProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YGMMWithMuTableCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },
        {
            'process': ZFactorizedModelCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },

        # Eval BPP
        {
            'process': YZRealBPPEvalProcess,
            'only': {CodecStageEnum.VALID, CodecStageEnum.TEST},
        },

        # TODO Decompress
    ]


class ContextualQuantCodec(BaseCodec):
    process_cls_list = [
        # Y auto-encoder
        YEncodeProcess,
        YQuantizeProcess,
        YDecodeProcess,

        # Z auto-encoder
        ZAbsEncodeProcess,
        ZQuantizeProcess,
        ZDecodeProcess,

        # Context
        # TODO decompress
        ContextProcess,

        # Entropy
        PriorAndContextParameterProcess,
        ZFactorizedEntropyProcess,

        BaseMapProcess.get_map_cls('sigma', 'tmp/sigma_index'),
        BaseMapProcess.get_map_cls('mu', 'tmp/mu_index'),
        SigmaCalculateProcess.get_cls('sigma', 'sigma'),
        MuCalculateTanProcess.get_cls('mu', 'mu'),

        GMMEntropyProcess,
        # Compress
        {
            'process': YMapMuSigmaProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YGMMWithMuTableCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },
        {
            'process': ZFactorizedModelCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },

        # Eval BPP
        {
            'process': YZRealBPPEvalProcess,
            'only': {CodecStageEnum.VALID, CodecStageEnum.TEST},
        },

        # TODO Decompress
    ]


class SigmaQuantCodec(BaseCodec):
    """
    gg18c without context model
    using discretized sigma only and en/de-code s = Round(y-mu)

    now x_hat = y_decoder(s+mu)
    """
    process_cls_list = [
        # Y auto-encoder
        YEncodeProcess,
        YQuantizeProcess,

        # Z auto-encoder
        ZEncodeProcess,
        ZQuantizeProcess,
        ZDecodeProcess,  # sigma, should be int net
        ZDecode2Process,  # mu

        # ContextProcess,
        # ChannelCatProcess.create(
        #     'ContextAndPriorChannelCatProcess',
        #     input_names=['y_context', 'prior2'],
        #     output_name='context_prior2',
        # ),
        BaseMapProcess.get_map_cls('prior2', 'context_prior2'),

        ModelInvokeProcess.create_single_io(
            input='prior',
            output='sigma',
            foo='sigma_parameter',
        ),
        ModelInvokeProcess.create_single_io(
            input='context_prior2',
            output='mu',
            foo='mu_parameter',
        ),

        {
            'process': YSubMuQuantProcess,  # y_tilde = Round(y-mu)+mu
            'exclude': CodecStageEnum.TRAIN
        },
        YDecodeProcess,

        # Entropy
        ZFactorizedEntropyProcess,

        # Discrete sigma
        BaseMapProcess.get_map_cls('sigma', 'tmp/sigma_index'),
        SigmaCalculateProcess.get_cls('sigma', 'sigma'),

        GMMEntropyProcess,
        # Compress
        {
            'process': YMapSigmaProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YGMMCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },
        {
            'process': ZFactorizedModelCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },

        # Eval BPP
        {
            'process': YZRealBPPEvalProcess,
            'only': {CodecStageEnum.VALID, CodecStageEnum.TEST},
        },

        # TODO Decompress
    ]


class ContextualSigmaQuantCodec(BaseCodec):
    """
    gg18c using discretized sigma only and en/de-code s = Round(y-mu)

    now x_hat = y_decoder(s+mu)
    """
    process_cls_list = [
        # Y auto-encoder
        YEncodeProcess,
        YQuantizeProcess,

        # Z auto-encoder
        ZEncodeProcess,
        ZQuantizeProcess,
        ZDecodeProcess,  # sigma, should be int net
        ZDecode2Process,  # mu

        ContextProcess,  # non-anchor
        Context2Process,  # anchor, zero-context
        ChannelCatProcess.create(
            'ContextAndPriorChannelCatProcess',
            input_names=['y_context', 'prior2'],
            output_name='context_prior2',
        ),
        ChannelCatProcess.create(
            'ContextAndPriorChannelCatProcess',
            input_names=['y_context2', 'prior2'],
            output_name='context2_prior2',
        ),

        ModelInvokeProcess.create_single_io(
            input='prior',
            output='sigma',
            foo='sigma_parameter',
        ),
        ModelInvokeProcess.create_single_io(
            input='context_prior2',
            output='mu1',
            foo='mu_parameter',
        ),
        ModelInvokeProcess.create_single_io(
            input='context2_prior2',
            output='mu2',
            foo='mu_parameter',
        ),
        ModelInvokeProcess.create_multi_io(
            inputs=['mu1', 'mu2', 'mu2'],
            outputs='mu',
            foo='gate'
        ),

        {
            'process': YSubMuQuantProcess,  # y_tilde = Round(y-mu)+mu
            'exclude': CodecStageEnum.TRAIN
        },
        YDecodeProcess,

        # Entropy
        ZFactorizedEntropyProcess,

        # Discrete sigma
        BaseMapProcess.get_map_cls('sigma', 'tmp/sigma_index'),
        SigmaCalculateProcess.get_cls('sigma', 'sigma'),

        GMMEntropyProcess,
        # Compress
        {
            'process': YMapSigmaProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YGMMCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },
        {
            'process': ZFactorizedModelCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },

        # Eval BPP
        {
            'process': YZRealBPPEvalProcess,
            'only': {CodecStageEnum.VALID, CodecStageEnum.TEST},
        },

        # TODO Decompress
    ]


class EDICCodec(BaseCodec):
    """
    > A Unified End-to-End Framework for Efficient Deep Image Compression
    see: https://arxiv.org/abs/1809.02736
    """
    process_cls_list = [
        # Y auto-encoder
        YEncodeProcess,
        YQuantizeProcess,
        YDecodeProcess,

        # Z auto-encoder
        ZAbsEncodeProcess,
        ZAttentionProcess,
        ZQuantizeProcess,
        ZDecodeProcess,

        # Entropy
        PriorGMMParameterProcess,
        ZFactorizedEntropyProcess,
        GMMCompleteEntropyProcess,

        # Compress
        {
            'process': YAlignSigmaProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YAlignMuProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YAlignOmegaProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YGMMWithMuTableCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },
        {
            'process': ZFactorizedModelCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },

        # Eval BPP
        {
            'process': YZRealBPPEvalProcess,
            'only': {CodecStageEnum.VALID, CodecStageEnum.TEST},
        },

        # TODO Decompress
    ]


class EDICContextualCodec(BaseCodec):
    """
    > A Unified End-to-End Framework for Efficient Deep Image Compression
    see: https://arxiv.org/abs/1809.02736
    """
    process_cls_list = [
        # Y auto-encoder
        YEncodeProcess,
        YQuantizeProcess,
        YDecodeProcess,

        # Z auto-encoder
        ZAbsEncodeProcess,
        ZAttentionProcess,
        ZQuantizeProcess,
        ZDecodeProcess,

        # Entropy
        ContextProcess,
        ContextAndPriorGMMParameterProcess,
        ZFactorizedEntropyProcess,
        GMMCompleteEntropyProcess,

        # Compress
        {
            'process': YAlignSigmaProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YAlignMuProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YAlignOmegaProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YGMMWithMuTableCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },
        {
            'process': ZFactorizedModelCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },

        # Eval BPP
        {
            'process': YZRealBPPEvalProcess,
            'only': {CodecStageEnum.VALID, CodecStageEnum.TEST},
        },

        # TODO Decompress
    ]


class CEVideoCodec(BaseCodec):
    """
    > Conditional Entropy Coding for Efficient Video Compression
    """
    common_pre_process_list = [
        VideoAlignAndPadInputProcess,
        preVideoProcess,
    ]
    common_post_process_list = [
        postProcess,
        VideoRemovePadProcess,
        {
            'process': LossProcess,
            'exclude': {CodecStageEnum.COMPRESS, CodecStageEnum.DECOMPRESS}
        },
        {
            'process': VideoSampleImageProcess,
            'exclude': {CodecStageEnum.COMPRESS, CodecStageEnum.DECOMPRESS}
        },
        {
            'process': EvalProcess,
            'only': {CodecStageEnum.VALID, CodecStageEnum.TEST},
        },
    ]
    process_cls_list = [
        # Y auto-encoder
        YEncodeProcess,
        YVideoEncodeProcess,
        YQuantizeProcess,
        YDecodeProcess,

        # Z auto-encoder
        ZVideoEncodeProcess,
        ZQuantizeProcess,
        ZVideoDecodeProcess,

        # Entropy
        PriorGMMParameterProcess,
        ZFactorizedEntropyProcess,
        GMMCompleteEntropyProcess,

        # Compress
        {
            'process': YAlignSigmaProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YAlignMuProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YAlignOmegaProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YGMMWithMuTableCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },
        {
            'process': ZFactorizedModelCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },

        # Eval BPP
        {
            'process': YZRealBPPEvalProcess,
            'only': {CodecStageEnum.VALID, CodecStageEnum.TEST},
        },

        # TODO Decompress
    ]


class FakeSerialEDICContextualCodec(BaseCodec):
    """
    Fake serial codec for running speed test
    """
    process_cls_list = [
        # Y auto-encoder
        YEncodeProcess,
        YQuantizeProcess,
        YDecodeProcess,

        # Z auto-encoder
        ZAbsEncodeProcess,
        ZAttentionProcess,
        ZQuantizeProcess,
        ZDecodeProcess,

        # Entropy
        # ContextProcess,
        # ContextAndPriorGMMParameterProcess,
        FakeSerialContextAndPriorGMMParameterProcess,
        ZFactorizedEntropyProcess,
        GMMCompleteEntropyProcess,

        # Compress
        {
            'process': YAlignSigmaProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YAlignMuProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YAlignOmegaProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YGMMWithMuTableCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },
        {
            'process': ZFactorizedModelCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },

        # Eval BPP
        {
            'process': YZRealBPPEvalProcess,
            'only': {CodecStageEnum.VALID, CodecStageEnum.TEST},
        },

        # TODO Decompress
    ]


class EDICTwoPathShareWeightCodec(BaseCodec):
    process_cls_list = [
        # Y auto-encoder
        YEncodeProcess,
        YQuantizeProcess,
        YDecodeProcess,

        # Z auto-encoder
        ZAbsEncodeProcess,
        ZAttentionProcess,
        ZQuantizeProcess,
        ZDecodeProcess,

        # Entropy
        ContextProcess,
        Context2Process,

        ContextAndPriorGMMParameter1Process,
        ContextAndPriorGMMShareWeightParameter2,
        ZFactorizedEntropyProcess,

        {
            'process': CompleteGMMParameterGateProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': GMMCompleteEntropyProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        # GMMCompleteEntropyProcess,

        GMMCompleteEntropy1Process,
        GMMCompleteEntropy2Process,
        TwoPathEntropyLossProcess,

        BaseMapProcess.get_map_cls('y_entropy2', 'y_entropy3'),  # hack, for use GammaEntropyWeightedSumUp

        {
            'process': Entropy211WeightedSumProcess,
            'only': CodecStageEnum.TRAIN
        },

        # Compress
        {
            'process': YAlignSigmaProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YAlignMuProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YAlignOmegaProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YGMMWithMuTableCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },
        {
            'process': ZFactorizedModelCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },

        # Eval BPP
        {
            'process': YZRealBPPEvalProcess,
            'only': {CodecStageEnum.VALID, CodecStageEnum.TEST},
        },

        # TODO Decompress
    ]


class EDICTwoPathShareWeightGateCodec(BaseCodec):
    process_cls_list = [
        # Y auto-encoder
        YEncodeProcess,
        YQuantizeProcess,
        YDecodeProcess,

        # Z auto-encoder
        ZAbsEncodeProcess,
        ZAttentionProcess,
        ZQuantizeProcess,
        ZDecodeProcess,

        # Entropy
        ContextProcess,
        Context2Process,

        ContextAndPriorGMMParameter1Process,
        ContextAndPriorGMMShareWeightParameter2,
        ZFactorizedEntropyProcess,

        {
            'process': CompleteGMMParameterGateProcess,
        },
        {
            'process': GMMCompleteEntropyProcess,
        },

        # Compress
        {
            'process': YAlignSigmaProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YAlignMuProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YAlignOmegaProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YGMMWithMuTableCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },
        {
            'process': ZFactorizedModelCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },

        # Eval BPP
        {
            'process': YZRealBPPEvalProcess,
            'only': {CodecStageEnum.VALID, CodecStageEnum.TEST},
        },

        # TODO Decompress
    ]


class DataAugmentCodec(GG18Codec):
    """
    this codec support complex data augment
    base data augment can also be find in dataset
    """
    common_pre_process_list = [
        {
            'process': DataAugmentProcess,
            'only': CodecStageEnum.TRAIN,
        },
        AlignAndPadInputProcess,
        preProcess,
    ]
    common_post_process_list = [
        postProcess,
        RemovePadProcess,
        {
            'process': DataCollectProcess,
            'only': CodecStageEnum.TRAIN,
        },
        {
            'process': LossProcess,
            'exclude': {CodecStageEnum.COMPRESS, CodecStageEnum.DECOMPRESS}
        },
        {
            'process': SampleImageProcess,
            'exclude': {CodecStageEnum.COMPRESS, CodecStageEnum.DECOMPRESS}
        },
        {
            'process': EvalProcess,
            'only': {CodecStageEnum.VALID, CodecStageEnum.TEST},
        },
    ]


class CheckerboardTwoPathContextualCodec(BaseCodec):
    """
    2-path structure designed for a checkerboard shaped context model
    """
    process_cls_list = [
        # Y auto-encoder
        YEncodeProcess,
        YQuantizeProcess,
        YDecodeProcess,

        # Z auto-encoder
        ZAbsEncodeProcess,
        ZQuantizeProcess,
        ZDecodeProcess,
        ZDecode2Process,

        # Context
        # TODO decompress
        ContextProcess,
        Context2Process,

        PriorAndContextParameter1Process,
        PriorAndContext2Parameter2Process,

        # Entropy
        {
            'process': ParameterGateProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': GMMEntropyProcess,
            'exclude': CodecStageEnum.TRAIN
        },

        ZFactorizedEntropyProcess,
        GMMEntropy1Process,
        GMMEntropy2Process,
        TwoPathEntropyLossProcess,

        BaseMapProcess.get_map_cls('y_entropy2', 'y_entropy3'),  # hack, for use GammaEntropyWeightedSumUp

        {
            'process': Entropy211WeightedSumProcess,
            'only': CodecStageEnum.TRAIN
        },

        # Compress
        {
            'process': YAlignSigmaProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        # {
        #     'process': YAlignMuProcess,
        #     'exclude': CodecStageEnum.TRAIN
        # },
        # {
        #     'process': YGMMWithMuTableCompressProcess,
        #     'exclude': CodecStageEnum.TRAIN,
        # },
        {
            'process': YGMMCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },
        {
            'process': ZFactorizedModelCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },

        # Eval BPP
        {
            'process': YZRealBPPEvalProcess,
            'only': {CodecStageEnum.VALID, CodecStageEnum.TEST},
        },

        # TODO Decompress
    ]


class CheckerboardShareWeightContextualCodec(BaseCodec):
    """
    2-path structure designed for a checkerboard shaped context model
    """
    process_cls_list = [
        # Y auto-encoder
        YEncodeProcess,
        YQuantizeProcess,
        YDecodeProcess,

        # Z auto-encoder
        ZAbsEncodeProcess,
        ZQuantizeProcess,
        ZDecodeProcess,

        # Context
        # TODO decompress
        ContextProcess,
        Context2Process,

        PriorAndContextParameter1Process,
        PriorAndContextShareWeightParameterProcess,

        # Entropy
        {
            'process': ParameterGateProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': GMMEntropyProcess,
            'exclude': CodecStageEnum.TRAIN
        },

        ZFactorizedEntropyProcess,
        GMMEntropy1Process,
        GMMEntropy2Process,
        TwoPathEntropyLossProcess,

        BaseMapProcess.get_map_cls('y_entropy2', 'y_entropy3'),  # hack, for use GammaEntropyWeightedSumUp

        {
            'process': Entropy211WeightedSumProcess,
            'only': CodecStageEnum.TRAIN
        },

        # Compress
        {
            'process': YAlignSigmaProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YAlignMuProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YGMMWithMuTableCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },
        {
            'process': ZFactorizedModelCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },

        # Eval BPP
        {
            'process': YZRealBPPEvalProcess,
            'only': {CodecStageEnum.VALID, CodecStageEnum.TEST},
        },

        # TODO Decompress
    ]


class CheckerboardShareWeightContextualFixedCodec(BaseCodec):
    """
    2-path structure designed for a checkerboard shaped context model
    """
    process_cls_list = [
        # Y auto-encoder
        YEncodeProcess,
        YQuantizeProcess,
        YDecodeProcess,

        # Z auto-encoder
        ZEncodeProcess,
        ZQuantizeProcess,
        ZDecodeProcess,

        # Context
        # TODO decompress
        ContextProcess,
        Context2Process,

        PriorAndContextParameter1Process,
        PriorAndContextShareWeightParameterProcess,

        # Entropy
        {
            'process': ParameterGateProcess,    #得到gatemodel过滤之后的sigma与mu   其实就是整合到一起
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': GMMEntropyProcess,       #得到y_likelihoods 和 y_tilde
            'exclude': CodecStageEnum.TRAIN
        },

        ZFactorizedEntropyProcess,
        GMMEntropy1Process,        #得到y_likelihoods_1
        GMMEntropy2Process,        #得到y_likelihoods_2
        TwoPathEntropyLossProcess,    #得到y_entropy1  y_entropy2

        BaseMapProcess.get_map_cls('y_entropy2', 'y_entropy3'),  # hack, for use GammaEntropyWeightedSumUp

        {
            'process': Entropy211WeightedSumProcess,
            'only': CodecStageEnum.TRAIN
        },

        # Compress
        {
            'process': YAlignSigmaProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YAlignMuProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YGMMWithMuTableCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },
        {
            'process': ZFactorizedModelCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },

        # Eval BPP
        {
            'process': YZRealBPPEvalProcess,
            'only': {CodecStageEnum.VALID, CodecStageEnum.TEST},
        },

        # TODO Decompress
    ]


class MixQuantCheckerboardShareWeightContextualFixedCodec(BaseCodec):
    """
    2-path structure designed for a checkerboard shaped context model
    """
    process_cls_list = [
        # Y auto-encoder
        YEncodeProcess,
        YQuantizeProcess,
        YRoundQuantizeProcess,
        YDecodeProcess.create_io_process(input='y_round'),

        # Z auto-encoder
        ZEncodeProcess,
        ZQuantizeProcess,
        ZRoundQuantizeProcess,
        ZDecodeProcess.create_io_process(input='z_round'),

        # Context
        # TODO decompress
        ContextProcess.create_io_process(input='y_round'),
        Context2Process.create_io_process(input='y_round'),

        PriorAndContextParameter1Process,
        PriorAndContextShareWeightParameterProcess,

        # Entropy
        {
            'process': ParameterGateProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': GMMEntropyProcess,
            'exclude': CodecStageEnum.TRAIN
        },

        ZFactorizedEntropyProcess,
        GMMEntropy1Process,
        GMMEntropy2Process,
        TwoPathEntropyLossProcess,

        BaseMapProcess.get_map_cls('y_entropy2', 'y_entropy3'),  # hack, for use GammaEntropyWeightedSumUp

        {
            'process': Entropy211WeightedSumProcess,
            'only': CodecStageEnum.TRAIN
        },

        # Compress
        {
            'process': YAlignSigmaProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YAlignMuProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YGMMWithMuTableCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },
        {
            'process': ZFactorizedModelCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },

        # Eval BPP
        {
            'process': YZRealBPPEvalProcess,
            'only': {CodecStageEnum.VALID, CodecStageEnum.TEST},
        },

        # TODO Decompress
    ]


class MultiDimensionContextualCodec(BaseCodec):
    """
    TwoPathContextualCodec with mix-quant trick like MixQuantContextualCodec
    """
    process_cls_list = [
        # Y auto-encoder
        YEncodeProcess,
        YQuantizeProcess,
        YRoundQuantizeProcess,
        YDecodeProcess.create_io_process(input='y_round'),

        # Z auto-encoder
        ZAbsEncodeProcess,
        ZQuantizeProcess,
        ZRoundQuantizeProcess,
        ZDecodeProcess.create_io_process(input='z_round'),
        ZDecode2Process.create_io_process(input='z_round'),

        # Context
        # TODO decompress
        ContextProcess.create_io_process(input='y_round'),
        Context2Process.create_io_process(input='y_round'),

        PriorAndContextParameter1Process,
        PriorAndContext2Parameter2Process,

        # Entropy
        {
            'process': ParameterGateProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': GMMEntropyProcess,
            'exclude': CodecStageEnum.TRAIN
        },

        ZFactorizedEntropyProcess,
        GMMEntropy1Process,
        GMMEntropy2Process,
        TwoPathEntropyLossProcess,

        BaseMapProcess.get_map_cls('y_entropy2', 'y_entropy3'),

        {
            'process': Entropy211WeightedSumProcess,
            'only': CodecStageEnum.TRAIN
        },

        # Compress
        {
            'process': YAlignSigmaProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YAlignMuProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YGMMWithMuTableCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },
        {
            'process': ZFactorizedModelCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },

        # Eval BPP
        {
            'process': YZRealBPPEvalProcess,
            'only': {CodecStageEnum.VALID, CodecStageEnum.TEST},
        },

        # TODO Decompress
    ]


class IDFCodec(BaseCodec):
    '''
    > Integer Discrete Flows and Lossless Compression
    see: https://arxiv.org/abs/1905.07376
    '''
    process_cls_list = [
        # encoder
        IDFEncodeProcess,

        # decoder
        # {
        #     'process': IDFDecodeProcess,
        #     'only': {CodecStageEnum.VALID, CodecStageEnum.TEST}
        # },
        # {
        #     'process': LosslessCheckProcess,
        #     'only': {CodecStageEnum.VALID, CodecStageEnum.TEST}
        # },

        # entropy
        IDFYsEntropyProcess,
        IDFZLMMParameterProcess,
        IDFZLMMCompleteEntropyProcess,
    ]

    common_post_process_list = [
        IDFLossProcess,
    ]


class IDF_Bottleneck(BaseCodec):
    '''
    > Integer Discrete Flows and Lossless Compression
    see: https://arxiv.org/abs/1905.07376
    '''
    process_cls_list = [
        # encoder
        IDFEncodeProcess,

        # decoder
        # {
        #     'process': IDFDecodeProcess,
        #     'only': {CodecStageEnum.VALID, CodecStageEnum.TEST}
        # },
        # {
        #     'process': LosslessCheckProcess,
        #     'only': {CodecStageEnum.VALID, CodecStageEnum.TEST}
        # },

        # entropy
        IDFYsEntropyProcess,
        ZFactorizedEntropyProcess
    ]

    common_post_process_list = [
        IDFLossProcess,
    ]



class DCTIDFCodec(BaseCodec):
    """DCT coefficients lossless compression model using IDF.
    """
    common_pre_process_list = [preProcess]

    process_cls_list = [
        # encoder
        IDFEncodeProcess,

        # decoder
        # {
        #     'process': IDFDecodeProcess,
        #     'only': {CodecStageEnum.VALID, CodecStageEnum.TEST}
        # },
        # {
        #     'process': LosslessCheckProcess,
        #     'only': {CodecStageEnum.VALID, CodecStageEnum.TEST}
        # },

        # entropy
        IDFYsEntropyProcess,
        IDFZLMMParameterProcess,
        IDFZLMMCompleteEntropyProcess,
    ]

    common_post_process_list = [
        IDFLossProcess,
    ]


class DCTGG18CTX(BaseCodec):
    """Compress DCT coef losslessly with GG18 context model.

    Lossless compression is implemented by entropy modeling DCT coef directly with context model.
    """
    common_pre_process_list = [preProcess]

    process_cls_list = [
        # Y auto-encoder
        YEncodeProcess,
        YQuantizeProcess,
        YDecodeProcess,

        # Z auto-encoder
        ZAbsEncodeProcess,
        ZQuantizeProcess,
        ZDecodeProcess,

        # Context
        # TODO decompress
        ContextProcess,

        # Entropy
        PriorAndContextParameterProcess,
        ZFactorizedEntropyProcess,
        GMMEntropyProcess,

        # Compress
        {
            'process': YAlignSigmaProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YAlignMuProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YGMMWithMuTableCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },
        {
            'process': ZFactorizedModelCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },

        # Eval BPP
        {
            'process': YZRealBPPEvalProcess,
            'only': {CodecStageEnum.VALID, CodecStageEnum.TEST},
        },

        # TODO Decompress
    ]

    common_post_process_list = [
        LossProcess,
        SampleImageProcess,
        {
            'process': EvalProcess,
            'only': {CodecStageEnum.VALID, CodecStageEnum.TEST},
        }
    ]


class DCTRC(BaseCodec):
    """Compress DCT coef losslessly with residual compression.
    """
    common_pre_process_list = [preProcess]

    process_cls_list = [
        # lossy
        # Y auto-encoder
        YEncodeProcess,
        YQuantizeProcess,
        YDecodeProcess,

        # residual
        ResidualProcess,

        # lossless
        # Residual Y auto-encoder
        ResYEncodeProcess,
        ResYQuantizeProcess,
        ResYDecodeProcess,

        # Z auto-encoder
        ResZAbsEncodeProcess,
        ResZQuantizeProcess,
        ResZDecodeProcess,

        # Context
        # TODO decompress
        ResContextProcess,

        # Entropy
        YFactorizedEntropyProcess,
        ResPriorAndContextParameterProcess,
        ResZFactorizedEntropyProcess,
        ResGMMEntropyProcess,

        # Compress
        {
            'process': YFactorizedModelCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },
        {
            'process': ResYAlignSigmaProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': ResYAlignMuProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': ResYGMMWithMuTableCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },
        {
            'process': ResZFactorizedModelCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },

        # Eval BPP
        {
            'process': ResYZRealBPPEvalProcess,
            'only': {CodecStageEnum.VALID, CodecStageEnum.TEST},
        },

        # TODO Decompress
    ]

    common_post_process_list = [
        LossProcess,
        SampleImageProcess,
        {
            'process': EvalProcess,
            'only': {CodecStageEnum.VALID, CodecStageEnum.TEST},
        }
    ]


class MixQuantGG20IterCodecCC(MixQuantGG20IterCodec):
    """
    this is GG18Codec with mix-quantizator trick introduced by gg20c

    see: CHANNEL-WISE AUTOREGRESSIVE ENTROPY MODELS FOR LEARNED IMAGE COMPRESSION
    """

    common_pre_process_list = [preProcess]

    process_cls_list = [
        # Y auto-encoder
        YEncodeProcess,
        YQuantizeProcess,
        YRoundQuantizeProcess,

        # Z auto-encoder
        ZEncodeProcess,
        ZQuantizeProcess,
        ZRoundQuantizeProcess,
        ZDecodeProcess.create_io_process(input='z_round'),

        # Context
        ChannelSliceParameterProcess,
        ParameterGroupSplitProcess,

        # Y Decode
        YDecodeProcess.create_io_process(input='y_lrp'),

        # Entropy
        ZFactorizedEntropyProcess,
        GMMEntropyProcess,

        # Compress
        {
            'process': YAlignSigmaProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YGMMCompressProcess,  # NOTICE: q(y-mu) trick
            'exclude': CodecStageEnum.TRAIN,
        },
        {
            'process': ZFactorizedModelCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },

        # Eval BPP
        {
            'process': YZRealBPPEvalProcess,
            'only': {CodecStageEnum.VALID, CodecStageEnum.TEST},
        },
    ]

class MixQuantGG20IterCodecCCTwoPath(MixQuantGG20IterCodec):
    """
    this is GG18Codec with mix-quantizator trick introduced by gg20c

    see: CHANNEL-WISE AUTOREGRESSIVE ENTROPY MODELS FOR LEARNED IMAGE COMPRESSION
    """

    common_pre_process_list = [preProcess]

    process_cls_list = [
        # Y auto-encoder
        YEncodeProcess,
        YQuantizeProcess,
        YRoundQuantizeProcess,

        # Z auto-encoder
        ZEncodeProcess,
        ZQuantizeProcess,
        ZRoundQuantizeProcess,
        ZDecodeProcess.create_io_process(input='z_round'),

        # Context
        ChannelSliceParameterProcessTwoPath,
        ParameterGroupSplitProcessTwoPath,

        # Y Decode
        YDecodeProcess.create_io_process(input='y_lrp'),

        # Entropy
        # ZFactorizedEntropyProcess,
        # GMMEntropyProcess,
        # Entropy
        {
            'process': ParameterGateProcess,  # 得到gatemodel过滤之后的sigma与mu   其实就是整合到一起
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': GMMEntropyProcess,  # 得到y_likelihoods 和 y_tilde
            'exclude': CodecStageEnum.TRAIN
        },

        ZFactorizedEntropyProcess,
        GMMEntropy1Process,  # 得到y_likelihoods_1
        GMMEntropy2Process,  # 得到y_likelihoods_2
        TwoPathEntropyLossProcess,  # 得到y_entropy1  y_entropy2

        BaseMapProcess.get_map_cls('y_entropy2', 'y_entropy3'),  # hack, for use GammaEntropyWeightedSumUp

        {
            'process': Entropy211WeightedSumProcess,
            'only': CodecStageEnum.TRAIN
        },

        # Compress
        {
            'process': YAlignSigmaProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YGMMCompressProcess,  # NOTICE: q(y-mu) trick
            'exclude': CodecStageEnum.TRAIN,
        },
        {
            'process': ZFactorizedModelCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },

        # Eval BPP
        {
            'process': YZRealBPPEvalProcess,
            'only': {CodecStageEnum.VALID, CodecStageEnum.TEST},
        },
    ]

class JEPGCodec(BaseCodec):
    '''
    > differentiable jpeg
    '''
    process_cls_list = [
        # encoder
        JPEGEncodeProcess,

        # decoder
        JPEGDecodeProcess,

        # flatten
        DCTFlattenProcess,

        # entropy
        YDCTFactorizedEntropyProcess,
        UDCTFactorizedEntropyProcess,
        VDCTFactorizedEntropyProcess,
        ]


    common_post_process_list = [
        RemovePadProcess,
        {
            'process': LossProcess,
            'exclude': {CodecStageEnum.COMPRESS, CodecStageEnum.DECOMPRESS}
        },
        {
            'process': SampleImageProcess,
            'exclude': {CodecStageEnum.COMPRESS, CodecStageEnum.DECOMPRESS}
        },
        {
            'process': EvalProcess,
            'only': {CodecStageEnum.VALID, CodecStageEnum.TEST},
        },
    ]


class KnightContextualCodec(BaseCodec):
    """
    a knight's position shaped context model
    """
    process_cls_list = [
        # Y auto-encoder
        YEncodeProcess,
        YQuantizeProcess,
        YDecodeProcess,

        # Z auto-encoder
        ZEncodeProcess,
        ZQuantizeProcess,
        ZDecodeProcess,

        # Context
        # TODO decompress
        ContextProcess,
        Context2Process,
        Context3Process,
        Context4Process,
        Context5Process,

        PriorAndContextParameter1Process,
        PriorAndContextShareWeightParameter2Process,
        PriorAndContextShareWeightParameter3Process,
        PriorAndContextShareWeightParameter4Process,
        PriorAndContextShareWeightParameter5Process,

        # Entropy
        {
            'process': ParameterGate2Process,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': GMMEntropyProcess,
            'exclude': CodecStageEnum.TRAIN
        },

        ZFactorizedEntropyProcess,
        GMMEntropy1Process,
        GMMEntropy2Process,
        GMMEntropy3Process,
        GMMEntropy4Process,
        GMMEntropy5Process,
        FiveStepsEntropyLossProcess,

        {
            'process': EntropyFiveWeightedSumProcess,
            'only': CodecStageEnum.TRAIN
        },

        # Compress
        {
            'process': YAlignSigmaProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YAlignMuProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YGMMWithMuTableCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },
        {
            'process': ZFactorizedModelCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },

        # Eval BPP
        {
            'process': YZRealBPPEvalProcess,
            'only': {CodecStageEnum.VALID, CodecStageEnum.TEST},
        },

        # TODO Decompress
    ]


class KnightContextualShareWeightCodec(BaseCodec):
    """
    a share-weight knight's position shaped context model
    """
    process_cls_list = [
        # Y auto-encoder
        YEncodeProcess,
        YQuantizeProcess,
        YDecodeProcess,

        # Z auto-encoder
        ZEncodeProcess,
        ZQuantizeProcess,
        ZDecodeProcess,

        # Context after Mask
        # TODO decompress
        KnightMaskProcess,
        KnightShareWeightContextProcess,

        PriorAndContextParameter1Process,
        PriorAndContextShareWeightParameter2Process,
        PriorAndContextShareWeightParameter3Process,
        PriorAndContextShareWeightParameter4Process,
        PriorAndContextShareWeightParameter5Process,

        # Entropy
        {
            'process': ParameterGate2Process,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': GMMEntropyProcess,
            'exclude': CodecStageEnum.TRAIN
        },

        ZFactorizedEntropyProcess,
        GMMEntropy1Process,
        GMMEntropy2Process,
        GMMEntropy3Process,
        GMMEntropy4Process,
        GMMEntropy5Process,
        FiveStepsEntropyLossProcess,

        {
            'process': EntropyFiveWeightedSumProcess,
            'only': CodecStageEnum.TRAIN
        },

        # Compress
        {
            'process': YAlignSigmaProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YAlignMuProcess,
            'exclude': CodecStageEnum.TRAIN
        },
        {
            'process': YGMMWithMuTableCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },
        {
            'process': ZFactorizedModelCompressProcess,
            'exclude': CodecStageEnum.TRAIN,
        },

        # Eval BPP
        {
            'process': YZRealBPPEvalProcess,
            'only': {CodecStageEnum.VALID, CodecStageEnum.TEST},
        },

        # TODO Decompress
    ]

class Multiscale(BaseCodec):
    '''
    《Lossless Image Compression Using a Multi-Scale Progressive Statistical Model》
    '''
    process_cls_list = [
        Compressor,



    ]
    common_post_process_list=[]