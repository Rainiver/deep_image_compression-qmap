from typing import Callable, Union

import cv2
import numpy as np
from PIL import Image
from torch import Tensor
from torchvision import transforms

from losses.PSNR_Loss import Loss as psnr_origin_class
from losses.SSIM_Loss import msssim as msssim_origin_func
from losses.SSIM_Loss import ssim as ssim_origin_func

_INPUT_TYPE = Union[str, np.ndarray, Image.Image]
_PROCESSABLE_TYPE = Union[np.ndarray, Image.Image]


def _to_tensor(pic: _INPUT_TYPE) -> Tensor:
    def _to_image(pic_: _INPUT_TYPE) -> _PROCESSABLE_TYPE:
        if isinstance(pic_, str):
            # return cv2.imread(pic_)
            return Image.open(pic_)
        else:
            return pic_

    return transforms.ToTensor()(_to_image(pic))


def _make_loss_function(func: Callable[[Tensor, Tensor], Tensor]) -> Callable[[_INPUT_TYPE, _INPUT_TYPE], float]:
    def _func(x: _INPUT_TYPE, y: _INPUT_TYPE) -> float:
        return float(func(
            _to_tensor(x).detach().unsqueeze(0),
            _to_tensor(y).detach().unsqueeze(0),
        ))

    return _func


psnr = _make_loss_function(psnr_origin_class())
msssim = _make_loss_function(msssim_origin_func)
ssim = _make_loss_function(ssim_origin_func)
