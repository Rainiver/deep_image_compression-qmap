import numpy as np
import random
import cv2
import torch
from torch import Tensor
from typing import List, Tuple


class BaseTransform(object):
    def __init__(self):
        pass

    def __call__(self, image):
        return self.process(image)

    def process(self, image):
        NotImplemented


class ImgRandomCrop(BaseTransform):
    def __init__(self, crop_size, training=True):
        self.crop_size = crop_size
        self.training = training
        super(ImgRandomCrop).__init__()

    def process(self, img):
        while img.shape[0] < self.crop_size:
            img = np.concatenate([img, img], 0)
        while img.shape[1] < self.crop_size:
            img = np.concatenate([img, img], 1)
        H, W = img.shape[0] - self.crop_size, img.shape[1] - self.crop_size
        if self.training:
            h, w = random.randint(0, H), random.randint(0, W)
        else:
            h, w = 0, 0
        img = img[h:h + self.crop_size, w:w + self.crop_size, :]
        return img


class TransformCompose(BaseTransform):
    def __init__(self, transforms):
        self.transforms = transforms
        super(TransformCompose).__init__()

    def process(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class Resize(BaseTransform):
    def __init__(self, size, training=True):
        super().__init__()
        self.size = size

    def process(self, image):
        size = self.size
        return cv2.resize(image, (size, size))


class VideoRandomCrop(BaseTransform):
    """Video random crop transform"""

    def __init__(self, crop_size: int, training: bool):
        super(VideoRandomCrop, self).__init__()
        self.crop_size = crop_size
        self.training = training

    def process(self, vdo: List[Tensor]) -> List[Tensor]:
        """
        Video random crop

        Args:
            vdo (List[Tensor]): List of video frames to be randomly cropped.
                The frame shape should be [C, H, W].
        Returns:
            vdo_cropped (List[Tensor])
        """
        if not self.training:
            return vdo

        height, width = vdo[0].shape[1:]
        assert self.crop_size <= height and self.crop_size <= width, \
            "crop size should not more than framse size, but got {}x{} and {}x{}"\
                .format(self.crop_size, self.crop_size, height, width)

        h_range, w_range = height - self.crop_size, width - self.crop_size
        h, w = random.randint(0, h_range), random.randint(0, w_range)
        vdo_cropped = []
        for frame in vdo:
            if isinstance(frame, Tuple):
                raise NotImplementedError
            vdo_cropped.append(frame[:, h:h + self.crop_size, w:w + self.crop_size])
        return vdo_cropped


class VideoRandomFlip(BaseTransform):
    """Video random flip transform"""

    def __init__(self, training: bool):
        super(VideoRandomFlip, self).__init__()
        self.training = training

    def process(self, vdo: List[Tensor]) -> List[Tensor]:
        """
        Video random flip

        Args:
            vdo (List[Tensor]): List of video frames to be randomly flipped.
                The frame shape should be [C, H, W].
        Returns:
            vdo_flipped (List[Tensor])
        """
        if not self.training:
            return vdo

        # vertical flip
        vdo_flipped_v = []
        if random.randint(0, 1) == 1:
            for frame in vdo:
                if isinstance(frame, Tuple):
                    raise NotImplementedError
                vdo_flipped_v.append(torch.flip(frame, [1]))
        else:
            vdo_flipped_v = vdo
        # horizontal flip
        vdo_flipped_h = []
        if random.randint(0, 1) == 1:
            for frame in vdo_flipped_v:
                if isinstance(frame, Tuple):
                    raise NotImplementedError
                vdo_flipped_h.append(torch.flip(frame, [2]))
        else:
            vdo_flipped_h = vdo_flipped_v
        return vdo_flipped_h


transforms_zoo = {
    'img_random_crop': ImgRandomCrop,
    'compose': TransformCompose,
    'img_resize': Resize,
    "video_random_crop": VideoRandomCrop,
    "video_random_flip": VideoRandomFlip,
}
