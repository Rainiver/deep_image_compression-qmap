import torch
from nets.encoder import encoders
from nets.decoder import decoders

try:
    from springnas_lite import SearchSpace
    from springnas_lite.base_mixop import MixedOp
except ImportError as e:
    SearchSpace = object
    MixedOp = object


class InitOneShot(SearchSpace):
    def __init__(self):
        # eg. YGG18, conv=mixop, normalization=gdn, in_channels=3, out_channels=256
        super(InitOneShot, self).__init__()

    def _build(self):
        self.path_width = 0

        for m_idx, module in enumerate(self.layer._modules['_layers']):
            if isinstance(module, MixedOp):
                self.path_width += 1

        self.path_choosens = [-1] * self.path_width

    def get_path_op_nums(self):
        path_op_nums = []
        for m_idx, module in enumerate(self.layer._modules['_layers']):
            if isinstance(module, MixedOp):
                path_op_nums.append(module.num_ops)
        return path_op_nums


    def forward(self, x):
        mixop_id = 0
        for m_idx, module in enumerate(self.layer._modules['_layers']):
            if mixop_id >= len(self.path_choosens):
                raise OverflowError('number of module is larger than path_choosens can provide')
                # path_choosen = -1
            else:
                path_choosen = self.path_choosens[mixop_id]

            if isinstance(module, MixedOp):
                x = module(x, path_choosen)
                mixop_id += 1
            else:
                x = module(x)

        return x


class EncoderInitOneShot(InitOneShot):
    def __init__(self, *args, **kwargs):
        super(EncoderInitOneShot, self).__init__()
        self.layer = encoders(*args, **kwargs)
        self._build()


class DecoderInitOneShot(InitOneShot):
    def __init__(self, *args, **kwargs):
        super(DecoderInitOneShot, self).__init__()
        self.layer = decoders(*args, **kwargs)
        self._build()
