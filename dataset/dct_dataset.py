import os
import cv2
import logging
import json
import time
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset as dataset
from .dataloader import DataLoader
from torchvision import transforms
from utils.distributed_utils import get_rank, get_world_size
from utils.log_helper import rank_0_print
from .data_utils import remove_duplicate
from .transform import transforms_zoo
from .base_dataset import parse_base_meta, BaseTransform
from .sampler import RangeSample, MySubsetRandomSampler
from torchvision.transforms.functional import to_tensor
try:
    import torchjpeg.codec as torchjpeg_codec
except ImportError:
    torchjpeg_codec = None


ZIGZAG_ORDER = [
    0,  1,  8, 16,  9,  2,  3, 10, 17, 24, 32, 25, 18, 11,  4,  5, 12,
    19, 26, 33, 40, 48, 41, 34, 27, 20, 13,  6,  7, 14, 21, 28, 35, 42,
    49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52,
    45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63
]


INVERSE_ZIGZAG_ORDER = [
    0, 1,  5,  6, 14, 15, 27, 28,  2,  4,  7, 13, 16, 26, 29, 42,  3,
    8, 12, 17, 25, 30, 41, 43,  9, 11, 18, 24, 31, 40, 44, 53, 10, 19,
    23, 32, 39, 45, 52, 54, 20, 22, 33, 38, 46, 51, 55, 60, 21, 34, 37,
    47, 50, 56, 59, 61, 35, 36, 48, 49, 57, 58, 62, 63
]


def build_dctloader(cfg, training):
    phase = training
    if training != "train":
        cfg["batch_size"] = 1
    transform_fn = BaseTransform(cfg['data_augment'][phase], True if training == 'train' else False)
    rank_0_print('building dataset from: {}'.format(cfg['meta_file_list']))
    dataset = DCTDataset(
        cfg['meta_file_list'],
        quality=cfg.get("quality", None),
        inverse_zigzag=cfg.get("inverse_zigzag", None),
        flatten=cfg.get("flatten", False),
        transform_fn=transform_fn,
        color_image=cfg.get('color_image', True),
        for_training=True if training == 'train' else False,
        kept=cfg.get('kept', None),
        memcached=True,
    )
    dataset_size = len(dataset)
    index = list(range(dataset_size))
    if training == 'train':
        # np.random.seed(int(time.time()) % 100)
        # np.random.shuffle(index)
        pass

    rank = get_rank()
    world_size = get_world_size()

    num_groups = cfg.get("num_groups", None)

    def get_l_r(_dataset_size, my_rank, my_world_size):
        l_bound = [0]
        for i in range(my_world_size):
            l_bound.append(
                l_bound[-1] + _dataset_size // my_world_size + (i < _dataset_size % my_world_size)
            )
        return l_bound[my_rank], l_bound[my_rank + 1]
        # l = my_rank * ((_dataset_size + my_world_size - 1) // my_world_size)
        # r = min(l + (_dataset_size + my_world_size - 1) // my_world_size, _dataset_size)
        # return l, r

    if num_groups is not None:
        group_size = world_size // num_groups
        local_rank = rank % group_size

        data_l, data_r = get_l_r(dataset_size, local_rank, group_size)
        index = index[data_l:data_r]
        print(f'rank: {rank}, data_sum: {len(index)}')
    else:

        data_l, data_r = get_l_r(dataset_size, rank, world_size)
        index = index[data_l:data_r]
        print(f'rank: {rank}, data_sum: {len(index)}')

    dist_sampler = MySubsetRandomSampler(index)
    if training == 'test':
        dist_sampler = None
    elif training == 'fast_test':
        dist_sampler = RangeSample(index)
    batch_size = cfg["batch_size"]
    if training == 'train':
        if world_size > 1:
            if num_groups is not None:
                group_size = world_size // num_groups
                assert batch_size % group_size == 0
                batch_size //= group_size
            else:
                assert batch_size % world_size == 0
                batch_size //= world_size

    elif training == 'test' or training == 'fast_test':
        batch_size = 1
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=cfg['workers'],
                        pin_memory=True,
                        persistent_workers=True,
                        sampler=dist_sampler)
    return loader


class DCTDataset(dataset):
    def __init__(self,
                 meta_file_list,
                 quality=None,
                 inverse_zigzag=None,
                 flatten=False,
                 transform_fn=None,
                 color_image=True,
                 for_training=True,
                 kept=None,
                 memcached=False):
        super(DCTDataset, self).__init__()

        assert torchjpeg_codec is not None, \
            ImportError('cannot load torchjpeg. see: https://gitlab.bj.sensetime.com/facedet/codec/torchjpeg')

        self.memcached = memcached
        if memcached:
            self.initialized = False

        self.for_training = for_training
        self.color_image = color_image
        self.transform_fn = transform_fn
        self.metas = []
        self.meta_file_list = meta_file_list
        for i, par_file_path in enumerate(meta_file_list):
            try:
                meta = parse_base_meta(i, par_file_path)
                self.metas.extend(meta)
                rank_0_print('Partition {} {} loaded, nums {}.'.format(i, par_file_path, len(meta)))
            except:
                rank_0_print('Partition {} {} missing.'.format(i, par_file_path))
        if not self.for_training:
            self.metas = remove_duplicate(self.metas)
        if kept is not None:
            self.metas = self.metas[0:kept]
        self.num = len(self.metas)
        self.to_tensor = transforms.ToTensor()
        # self.quality = quality
        self.quality = 75
        self.inverse_zigzag = inverse_zigzag
        self.flatten = flatten

        # zigzag order
        self.zigzag_order = torch.Tensor(ZIGZAG_ORDER).type(torch.LongTensor)
        # inverse zigzag order
        self.inverse_zigzag_order = torch.Tensor(INVERSE_ZIGZAG_ORDER).type(torch.LongTensor)

    def __len__(self):
        return self.num

    def get_jpeg_quality(self):
        if self.quality:
            return self.quality
        quality = np.random.randint(10, 101)
        return quality

    def read_dct(self, img_path):
        if self.memcached:
            img = self.read_img_by_mc(img_path)
        else:
            img = self.read_img_by_OpenCV(img_path)
        if self.color_image:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform_fn(img)
        img = to_tensor(img)     #normalized to [0, 1] and with shape (C,H,W)
        # # TODO: fixed quality=75?
        _quality = self.get_jpeg_quality()
        dimensions, quantization, y_coefficients, cbcr_coefficients = torchjpeg_codec.quantize_at_quality(img, _quality)
        cb_coefficients, cr_coefficients = cbcr_coefficients[0][0:1], cbcr_coefficients[0][1:2]
        return dimensions, quantization, y_coefficients, cb_coefficients, cr_coefficients

    def read_img_by_OpenCV(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            rank_0_print(img_path)
        return img

    def _init_memcached(self):
        if not self.initialized:
            import mc
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def read_img_by_mc(self, img_path):
        import mc
        self._init_memcached()
        value = mc.pyvector()
        assert len(img_path) < 250, 'memcached requires length of path < 250'
        self.mclient.Get(img_path, value)
        value_buf = mc.ConvertBuffer(value)
        img_array = np.frombuffer(value_buf, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img

    def get_zigzag_data(self, x):
        # hwc
        indices = self.zigzag_order.expand_as(x)
        return torch.gather(x, 2, indices)

    def get_inverse_zigzag_data(self, x):
        # hwc
        indices = self.inverse_zigzag_order.expand_as(x)
        return torch.gather(x, 2, indices)

    def __getitem__(self, idx):
        meta = self.metas[idx].to_dict()
        img_path = meta['path']
        dimensions, quantization, y_coefficients, cb_coefficients, cr_coefficients = self.read_dct(img_path)

        if self.flatten:
            # In this mode, y is transformed to (4, h/2, w/2), final img has 6 channels
            # (1, h/8, w/8, 8, 8) -> (1, h, w)
            _, h, w, _, _ = y_coefficients.shape
            y_coefficients = y_coefficients.view(h, w, 4, 2, 4, 2)
            y_coefficients = y_coefficients.permute((3, 5, 0, 2, 1, 4)).contiguous()
            y_coefficients = y_coefficients.view(4, h * 4, w * 4)
            _, h, w, _, _ = cb_coefficients.shape
            cb_coefficients = cb_coefficients.permute((0, 1, 3, 2, 4)).contiguous()
            cb_coefficients = cb_coefficients.view(1, h * 8, w * 8)
            cr_coefficients = cr_coefficients.permute((0, 1, 3, 2, 4)).contiguous()
            cr_coefficients = cr_coefficients.view(1, h * 8, w * 8)
            img = torch.cat((y_coefficients, cb_coefficients, cr_coefficients), dim=0)
            _data = {
                "img": img,
                "dimensions": dimensions,
                "quantization": quantization,
                "path": img_path
            }
        else:
            # (1, h, w, 8, 8) -> (h, w, 64)
            # raster scan
            _, h, w, _, _ = y_coefficients.shape
            y_coefficients = y_coefficients[0].view(h, w, 64)
            _, h, w, _, _ = cb_coefficients.shape
            cb_coefficients, cr_coefficients = (x[0].view(h, w, 64) for x in (cb_coefficients, cr_coefficients))
            # zigzag order
            if self.inverse_zigzag:
                y_coefficients = self.get_zigzag_data(y_coefficients)
                cb_coefficients = self.get_zigzag_data(cb_coefficients)
                cr_coefficients = self.get_zigzag_data(cr_coefficients)
                y_coefficients = torch.flip(y_coefficients, [2])
                cb_coefficients = torch.flip(cb_coefficients, [2])
                cr_coefficients = torch.flip(cr_coefficients, [2])
            else:
                y_coefficients = self.get_zigzag_data(y_coefficients)
                cb_coefficients = self.get_zigzag_data(cb_coefficients)
                cr_coefficients = self.get_zigzag_data(cr_coefficients)
            # HWC -> CHW
            y_coefficients, cb_coefficients, cr_coefficients = \
                (x.permute(2, 0, 1) for x in (y_coefficients, cb_coefficients, cr_coefficients))
            img = y_coefficients
            _data = {
                "img": img,
                "dimensions": dimensions,
                "quantization": quantization,
                "y_coefficients": y_coefficients,
                "cb_coefficients": cb_coefficients,
                "cr_coefficients": cr_coefficients,
                "path": img_path
            }
        return _data
