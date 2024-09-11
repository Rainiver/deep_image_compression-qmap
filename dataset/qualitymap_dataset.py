import cv2
import logging
import json
import time
import torch
import numpy as np
from torch.utils.data import Dataset as dataset
# from torch.utils.data import DataLoader
from .dataloader import DataLoader
from torchvision import transforms
from utils.distributed_utils import get_rank, get_world_size
from utils.log_helper import rank_0_print
from .data_utils import remove_duplicate
from .transform import transforms_zoo
from .sampler import RangeSample, MySubsetRandomSampler
import warnings
import pandas as pd
import random
from PIL import Image, ImageFile, ImageFilter
ImageFile.LOAD_TRUNCATED_IMAGES = True
#from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import hflip, to_tensor
from torch.distributions.multivariate_normal import MultivariateNormal


def build_qmaploader(cfg, training):
    phase = training
    if training != "train":
        cfg["batch_size"] = 1
    transform_fn = BaseTransform(cfg['data_augment'][phase], True if training == 'train' else False)
    rank_0_print('building dataset from: {}'.format(cfg['meta_file_list']))
    if cfg.get("video"):
        dataset = VideoDataset(
            cfg['meta_file_list'],
            transform_fn=transform_fn,
            color_image=cfg.get('color_image', True),
            for_training=True if training == 'train' else False,
            kept=cfg.get('kept', None),
        )
    else:
        # dataset = BaseDataset(
        dataset = QualityMapDataset(
            cfg['meta_file_list'],
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

class BaseDataset(dataset):
    def __init__(self,
                 meta_file_list,
                 transform_fn=None,
                 color_image=True,
                 for_training=True,
                 kept=None,
                 memcached=False):
        super(BaseDataset, self).__init__()
        try:
            import mc
        except ImportError:
            memcached = False
            warnings.warn('failed to import memcached. reading images from disk directly.')
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

    def __len__(self):
        return self.num

    def read_img_by_OpenCV(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            rank_0_print(img_path)
        # if self.color_image:
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # else:
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _init_memcached(self):
        if not self.initialized:
            import mc
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def read_img_by_mc(self, filename):
        import mc
        self._init_memcached()
        value = mc.pyvector()
        assert len(filename) < 250, 'memcached requires length of path < 250'
        self.mclient.Get(filename, value)
        value_buf = mc.ConvertBuffer(value)
        img_array = np.frombuffer(value_buf, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img

    def __getitem__(self, idx):
        meta = self.metas[idx].to_dict()
        img_path = meta['path']
        if self.memcached:
            img = self.read_img_by_mc(img_path)
        else:
            img = self.read_img_by_OpenCV(img_path)
        img = self.transform_fn(img)
        if len(img.shape) == 2:
            img = self.to_tensor(np.expand_dims(img.astype(np.uint8), axis=2))
        else:
            img = self.to_tensor(img)

        # img.img_path = img_path
        return {'img': img, 'path': img_path}

class QualityMapDataset(BaseDataset):
    def __init__(self,
                 meta_file_list,
                 transform_fn=None,
                 color_image=True,
                 for_training=True,
                 kept=None,
                 memcached=False,
                 cropsize=256,
                 mode='train',
                 level_range=(0, 100),
                 level=0, p=0.2):
        BaseDataset.__init__(self, meta_file_list, transform_fn,
                             color_image, for_training, kept, memcached)
        self.paths = [x.to_dict()['path'] for x in self.metas]


        self.cropsize = cropsize
        self.mode = mode
        self.level_range = level_range
        self.level = level
        self.p = p
        self.grid = self._get_grid((self.cropsize, cropsize))

        """
        NOTE: map_paths should be consistent with paths, which means map_paths[i] refers to
        the map derived from the image refered by paths[i]
        """
        self.map_paths = None
        assert self.map_paths is None or len(self.paths) == len(self.map_paths)
        assert level_range[0] == 0 and level_range[1] == 100
        if self.mode == 'train':
            print(f'[{mode}set] {len(self.paths)} images')
        elif self.mode == 'test':
            print(f'[{mode}set] {len(self.paths)} images for quality {level/100}')
            self.paths.sort()


    def _get_crop_params(self, img):
        w, h = img.size
        # if w == self.cropsize and h == self.cropsize:
        #     return 0, 0, h, w

        if self.mode == 'train':
            top = random.randint(0, h - self.cropsize)
            left = random.randint(0, w - self.cropsize)
        else:
            # center
            top = int(round((h - self.cropsize) / 2.))
            left = int(round((w - self.cropsize) / 2.))
        return top, left

    def _get_grid(self, size):
        x1 = torch.tensor(range(size[0]))
        x2 = torch.tensor(range(size[1]))
        grid_x1, grid_x2 = torch.meshgrid(x1, x2)

        grid1 = grid_x1.view(size[0], size[1], 1)
        grid2 = grid_x2.view(size[0], size[1], 1)
        grid = torch.cat([grid1, grid2], dim=-1)
        return grid

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        if self.memcached:
            img = self.read_img_by_mc(img_path)
        else:
            img = self.read_img_by_OpenCV(img_path)
        img = self.transform_fn(img)
        if len(img.shape) == 2:
            img = self.to_tensor(np.expand_dims(img.astype(np.uint8), axis=2))
        else:
            img = self.to_tensor(img)

        ####################### get map #################################
        img = transforms.ToPILImage()(img).convert('RGB')
        segmap = Image.open(self.map_paths[idx]) if self.map_paths else Image.fromarray(
            # np.zeros(img.size[::-1], dtype=np.uint8))
            np.ones(img.size, dtype=np.uint8))

        # crop if training
        if self.mode == 'train':
            top, left = self._get_crop_params(img)
            # print(f'2, image shape is {img.size}')
            region = (left, top, left + self.cropsize, top + self.cropsize)
            img = img.crop(region)
            segmap = segmap.crop(region)

        # horizontal flip
        if random.random() < 0.5 and self.mode == 'train':
            img = hflip(img)
            segmap = hflip(segmap)

        # filter segmap to remove some artifacts
        segmap = segmap.filter(ImageFilter.MedianFilter(7))

        # random rate for each class
        segmap = np.array(segmap)
        qmap = np.zeros_like(segmap, dtype=float)
        uniques = np.unique(segmap)
        if self.mode == 'train':
            sample = random.random()
            if sample < self.p:
                # uniform
                if random.random() < 0.01:
                    qmap[:] = 0
                else:
                    qmap[:] = (self.level_range[1] + 1) * random.random()
            elif sample < 2 * self.p:
                # uniform for each class
                for v in uniques:
                    level = (self.level_range[1] + 1) * random.random()
                    qmap[segmap == v] = level
            elif sample < 3 * self.p:
                # gradation between two levels
                v1 = random.random() * self.level_range[1]
                v2 = random.random() * self.level_range[1]
                qmap = np.tile(np.linspace(v1, v2, self.cropsize), (self.cropsize, 1)).astype(float)
                if random.random() < 0.5:
                    qmap = qmap.T
            else:
                # gaussian kernel
                gaussian_num = int(1 + random.random() * 20)
                for i in range(gaussian_num):
                    mu_x = self.cropsize * random.random()
                    mu_y = self.cropsize * random.random()
                    var_x = 2000 * random.random() + 1000
                    var_y = 2000 * random.random() + 1000

                    m = MultivariateNormal(torch.tensor([mu_x, mu_y]), torch.tensor([[var_x, 0], [0, var_y]]))
                    p = m.log_prob(self.grid)
                    kernel = torch.exp(p).numpy()
                    qmap += kernel
                qmap *= 100 / qmap.max() * (0.5 * random.random() + 0.5)
        else:
            uniques.sort()
            if self.level == -100:
                w, h = img.size
                # gradation
                if idx % 3 == 0:
                    v1 = idx/len(self.paths) * self.level_range[1]
                    v2 = (1-idx/len(self.paths)) * self.level_range[1]
                    qmap = np.tile(np.linspace(v1, v2, w), (h, 1)).astype(float)
                # gaussian kernel
                else:
                    gaussian_num = 1
                    for i in range(gaussian_num):
                        mu_x = h / 4 + (h/2)*idx/len(self.paths)
                        mu_y = w / 4 + (w/2)*(1-idx/len(self.paths))
                        var_x = 20000 * (1-idx/len(self.paths)) + 5000
                        var_y = 20000 * idx/len(self.paths) + 5000

                        m = MultivariateNormal(torch.tensor([mu_x, mu_y]), torch.tensor([[var_x, 0], [0, var_y]]))
                        grid = self._get_grid((h, w))
                        p = m.log_prob(grid)
                        kernel = torch.exp(p).numpy()
                        qmap += kernel
                    qmap *= 100 / qmap.max() * (0.4 * idx/len(self.paths) + 0.6)
            else:
                # uniform level
                qmap[:] = self.level

        # to tensor
        qmap = torch.FloatTensor(qmap).unsqueeze(dim=0)
        qmap *= 1 / self.level_range[1]  # 0~100 -> 0~1

        ### convert PIL.image class to tensor of shape (C,H,W)
        # maybe need normalization??
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        img = transform(img) / 255.0
        return {'path': img_path, 'img': img,  'qmap': qmap}


def parse_base_meta(partition, meta_file_path):
    metas = []
    with open(meta_file_path, encoding="utf-8") as f:
        lines = f.readlines()
    for i in range(len(lines)):
        img_info = lines[i].strip().split()
        path = img_info[0]
        img_height = float(img_info[1])
        img_width = float(img_info[2])
        bpp = float(img_info[3])
        metas.append(ImageData(partition, path, img_height, img_width, bpp))
    return metas

class ImageData():
    __slots__ = ['partition', 'path', 'height', 'width', 'bpp']

    def __init__(self, partition, path, height, width, bpp):
        self.partition = partition
        self.path = path
        self.height = height
        self.width = width
        self.bpp = bpp

    def to_json(self):
        res = {}
        res['partition'] = self.partition
        res['path'] = self.path
        res['height'] = self.height
        res['width'] = self.width
        res['bpp'] = self.bpp
        return json.dumps(res)

    def to_dict(self):
        res = {}
        res['partition'] = self.partition
        res['path'] = self.path
        res['height'] = self.height
        res['width'] = self.width
        res['bpp'] = self.bpp
        return res



class BaseTransform(object):
    def __init__(self, cfg, training):
        self.cfg = cfg
        transforms = list()
        if cfg is not None:
            for trans in cfg:
                for k, v in trans.items():
                    if k in transforms_zoo:
                        if not isinstance(v, dict):
                            v = {}
                        v.update({'training': training})
                        transform = transforms_zoo[k](**v)
                        transforms.append(transform)
        self.final_transforms = transforms_zoo['compose'](transforms)

    def __call__(self, img):
        img = self.final_transforms(img)
        return img
