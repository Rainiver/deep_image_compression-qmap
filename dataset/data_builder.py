import copy
from .base_dataset import build_baseloader
from .roi_dataset import build_roiloader
from .video_dataset import build_videoloader
from .dct_dataset import build_dctloader
from .qualitymap_dataset import build_qmaploader


def build_dataloader(cfg, training, **kwargs):
    cfg = copy.deepcopy(cfg)
    if training == 'train':
        cfg.update(cfg.get('train', {}))
    elif training == 'test':
        cfg.update(cfg.get('test', {}))
    elif training == 'fast_test':
        cfg.update(cfg.get('fast_test', {}))
    else:
        raise TypeError(f'unsupported type {training}')

    dataset = cfg['type']
    if dataset == 'base':
        data_loader = build_baseloader(cfg, training)
    elif dataset == 'roi':
        data_loader = build_roiloader(cfg, training)
    elif dataset == 'video':
        data_loader = build_videoloader(cfg, training == 'train', **kwargs)
    elif dataset == 'qmap':
        data_loader = build_qmaploader(cfg, training)
    elif dataset == "dct":
        data_loader = build_dctloader(cfg, training)
    else:
        raise NotImplementedError(f"{dataset} is not supported")
    return data_loader
