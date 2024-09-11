from pipelines.models import BaseCodec
from collections import OrderedDict

try:
    from springnas_lite import SearchSpace
    import spring.linklink as link
except ImportError as e:
    raise e


class CodecOneShot(SearchSpace):
    def __init__(self, codec: BaseCodec, codec_cfg={}):
        super(CodecOneShot, self).__init__()
        search_info = OrderedDict()
        for name, model in codec.sub_models.items():
            if isinstance(model, SearchSpace):
                search_info[name] = model.path_width

        assert isinstance(codec_cfg, dict) and codec_cfg != {}
        assert set(codec_cfg.search_order) == set(search_info.keys())

        split_path_info = OrderedDict()
        for search_layer in codec_cfg.search_order:
            split_path_info[search_layer] = search_info[search_layer]

        self.codec = codec
        self.split_path_info = split_path_info
        self.path_choosens = [-1] * sum(split_path_info.values())
        self.path_width = len(self.path_choosens)
        if link.get_rank() == 0:
            print('path information:\n', split_path_info)
            print('path op nums:\n', self.get_path_op_nums())

    def get_path_helper(self, name):
        start_id = 0
        for model_name, model_width in self.split_path_info.items():
            if name != model_name:
                start_id += model_width
            else:
                path = self.path_choosens[start_id: start_id + model_width]
                return path
        raise ValueError(f'model name {name} not find')

    def get_path_op_nums(self):
        path_op_nums = []
        for model_name in self.split_path_info.keys():
            model = self.codec.sub_models[model_name]
            path_op_nums += model.get_path_op_nums()
        return path_op_nums

    def forward(self, *args, **kwargs):
        codec = self.codec
        for name, model in codec.sub_models.items():
            if isinstance(model, SearchSpace):
                model.path_choosens = self.get_path_helper(name)
        return self.codec(*args, **kwargs)

    def __getattr__(self, item):
        try:
            return super(CodecOneShot, self).__getattr__(item)
        except AttributeError as err:
            if 'codec' in self.__dict__['_modules'].keys():
                return getattr(self.__dict__['_modules']['codec'], item)
            else:
                raise AttributeError(item)
