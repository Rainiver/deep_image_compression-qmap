from argparse import ArgumentParser
from PIL import Image
import numpy as np
import cv2
import torch


def diff_image(x, y):
    base = 20
    x_np = np.array(x)
    y_np = np.array(y)
    z_np = pow(x_np - y_np, 2)
    z_np = np.sum(z_np, 2)
    z_np = z_np.astype(np.float32)
    filter = np.ones([base, base])
    filter = filter / base / base
    z_np = cv2.filter2D(z_np, -1, filter)

    Max = np.max(z_np)
    Min = np.min(z_np)
    z_np = z_np / Max * 255.
    z_np = z_np.astype(np.uint8)
    z = cv2.applyColorMap(z_np, cv2.COLORMAP_JET)
    return z


def diff_image_torch(x: torch.Tensor, y):
    # NCHW -> HWC
    if x.shape[0] != 1:
        return x

    def prepare(value):
        img = torch.round(value.float() * 255).int().clamp(0, 255)
        img = img.detach().cpu().numpy()
        img = np.array(img, dtype=np.uint8).squeeze(0)
        img = np.transpose(img, [1, 2, 0])
        return img

    # x = torch.round(x * 255).int().clamp(0, 255)
    # y = torch.round(y * 255).int().clamp(0, 255)
    # x = x.squeeze().permute([1, 2, 0]).detach().cpu().numpy()
    # y = y.squeeze().permute([1, 2, 0]).detach().cpu().numpy()
    # z = diff_image(x, y)
    z = diff_image(prepare(x), prepare(y))
    # z = torch.from_numpy(z).cuda().float().permute([2, 0, 1]).unsqueeze(0) / 255.
    return z


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('image1', type=str, help='image file 1')
    parser.add_argument('image2', type=str, help='image file 2')
    args = parser.parse_args()

    x = Image.open(args.image1, mode="r")
    y = Image.open(args.image2, mode="r")
    z = diff_image(x, y)
    cv2.imwrite("diff.png", z)
