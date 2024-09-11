import time
import torch
import os
import cv2
import logging
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import SubsetRandomSampler, Sampler
from prefetch_generator import BackgroundGenerator

from video.utils.misc_helper import clock
from utils.distributed_utils import get_rank, get_world_size
from dataset.base_dataset import BaseTransform


logger = logging.getLogger('global')


def build_videoloader(cfg, training, meta_file):
    phase = "train"
    if not training:
        # In testing procedure, to avoid input image with different size, batch_size is set to 1
        cfg["batch_size"] = 1
        phase = "test"

    transform_fn = BaseTransform(cfg['data_augment'][phase], training)
    logger.info('building dataset from: {}'.format(meta_file))

    dataset = GopDataset(
        meta_file=meta_file,
        transform_fn=transform_fn,
        training=training,
        task=cfg.get("task", None),
    )

    index = dataset.get_index()
    dataset_size = len(index)
    if training:
        np.random.seed(int(time.time()) % 100)
        np.random.shuffle(index)
    rank = get_rank()
    world_size = get_world_size()
    data_l = rank * dataset_size // world_size
    data_r = data_l + dataset_size // world_size
    index = index[data_l:data_r]
    print(f'rank: {rank}, data_sum: {len(index)}')

    batch_size = cfg["batch_size"]

    if training:
        dist_sampler = SubsetRandomSampler(index)
        if world_size > 1:
            assert batch_size % world_size == 0
            batch_size //= world_size
    else:
        dist_sampler = SubsetSequentialSampler(index)
        batch_size = 1

    loader = DataLoaderX(dataset,
                        batch_size=batch_size,
                        num_workers=cfg['workers'],
                        pin_memory=True,
                        sampler=dist_sampler)
    loader.gop_size = dataset.gop_size

    return loader

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__(), max_prefetch=2)


def parse_seq_meta(meta_file):
    """Read meta file."""
    with open(meta_file, "r") as f:
        metas = [l.strip().split(" ") for l in f.readlines()]
    assert len(metas) > 0, "empty file: {}".format(meta_file)
    return metas


# TODO: merge with GopDataset._to_tensor
def _image_to_tensor(pic, normalize=True):
    """
    Convert a ``PIL Image`` to tensor.
    Copied from torchvision.transforms.functional.to_tensor, adapted
    to only support PIL inputs and normalize flag
    :param pic PIL Image
    :param normalize If False, return uint8, otherwise return float32 in range [0,1]
    """
    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    elif pic.mode == 'F':
        img = torch.from_numpy(np.array(pic, np.float32, copy=False))
    elif pic.mode == '1':
        img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor) and normalize:
        return img.float().div(255)
    elif pic.mode == "I" and normalize:
        return img.float().div(65535)
    else:
        return img


def _upsample_nearest_neighbor(t, output_shape):
    """ Upsample tensor `t` by `factor`. """
    return F.interpolate(t.unsqueeze(0), size=output_shape, mode='nearest').squeeze(0)


def yuv_420_to_444(y, u, v):
    """ Convert Y, U, V, given in 420, to RGB 444. Expects CHW dataformat """
    # u, v = map(_upsample_nearest_neighbor, (u, v))  # upsample U, V
    u = _upsample_nearest_neighbor(u, y.shape[1:])
    v = _upsample_nearest_neighbor(v, y.shape[1:])
    return torch.cat((y, u, v), dim=0)  # merge


class GopDataset(Dataset):
    def __init__(self, meta_file, transform_fn, training, task):

        assert task in ["normal", "bg_model", "clic2020"], \
            "unsupported task type: {}".format(task)

        self.task = task
        self.training = training
        self.transform_fn = transform_fn

        self.metas = []
        self._to_tensor = transforms.ToTensor()

        assert os.path.isfile(meta_file), \
            "Meta file does not exists: {}".format(meta_file)
        self.metas = parse_seq_meta(meta_file)
        self.gop_size = len(self.metas[0])

    def __len__(self):
        return len(self.metas)

    #@clock
    def __getitem__(self, idx):
        gop_data = self.metas[idx]
        return self.load_frames(gop_data)

    def load_frames(self, gop_data):
        if self.task == "normal":
            return self._load_frames_normal(gop_data)
        elif self.task == "bg_model":
            return self._load_frames_bg(gop_data)
        elif self.task == "clic2020":
            return self._load_frames_clic2020(gop_data)
        else:
            raise NotImplementedError

    def _load_frame(self, frame_path):
        assert os.path.isfile(frame_path), "invalid frame path: {}".format(frame_path)
        img = cv2.imread(frame_path)
        img = self._to_tensor(img)
        return img

    def _load_frames_normal(self, gop_data):
        paths, frames = [], []
        for frame_path in gop_data:
            assert os.path.isfile(frame_path), "invalid frame path: {}".format(frame_path)
            img = cv2.imread(frame_path)
            img = self._to_tensor(img)
            paths.append(frame_path)
            frames.append(img)
        frames = self.transform_fn(frames)
        frames = torch.stack(frames)
        outputs = {
            "paths": paths,
            "xts": frames
        }
        return outputs

    def _load_frames_bg(self, gop_data):
        paths, frames, fgs, bgs = [], [], [], []
        for data in gop_data:
            frame_path, fg_path, bg_path = data.split(',')
            fg_img = self._load_frame(fg_path)
            bg_img = self._load_frame(bg_path)
            img = self._load_frame(frame_path)

            paths.append(frame_path)
            frames.append(img)
            fgs.append(fg_img)
            bgs.append(bg_img)

        frames = self.transform_fn(frames)
        frames = torch.stack(frames)
        fgs = self.transform_fn(fgs)
        fgs = torch.stack(fgs)
        bgs = self.transform_fn(bgs)
        bgs = torch.stack(bgs)

        outputs = {
            "paths": paths,
            "xts": frames,
            "fts": fgs,
            "bts": bgs
        }
        return outputs

    def _load_frames_clic2020(self, gop_data):
        paths, frames = [], []
        for frame_path in gop_data:
            y_path = frame_path
            u_path = frame_path.replace("_y.png", "_u.png")
            v_path = frame_path.replace("_y.png", "_v.png")

            y_frame, u_frame, v_frame = (Image.open(p) for p in (y_path, u_path, v_path))
            y_frame, u_frame, v_frame = (_image_to_tensor(f, True) for f in (y_frame, u_frame, v_frame))
            # merge channel
            yuv = yuv_420_to_444(y_frame, u_frame, v_frame)

            paths.append(frame_path)
            frames.append(yuv)
        frames = self.transform_fn(frames)
        frames = torch.stack(frames)
        outputs = {
            "paths": paths,
            "xts": frames,
        }
        return outputs

    def get_index(self):
        """
        Get Valid index of data.
        :return: valid index list
        """
        idx_list = list(range(0, len(self.metas)))
        return idx_list


class SubsetSequentialSampler(Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        super(SubsetSequentialSampler, self).__init__(indices)
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


if __name__ == "__main__":
    data_list = "/mnt/lustre/share/wangyuan/datasets/data_list/testset_bpg_22_10.txt"
    transform_fn = BaseTransform(None, False)
    dataset = GopDataset([data_list], transform_fn, False, 10)
    for data in dataset:
        print(data)
        break
