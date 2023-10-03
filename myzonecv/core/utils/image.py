import numpy as np
import skimage
import skimage.io as io
import cv2
import torch
from torchvision.transforms import functional as F

from ..consts import IMAGENET_RGB_MEAN, IMAGENET_RGB_STD


def img_as_float(img, min=None, max=None):
    if img.dtype == np.uint8:
        if min is not None:
            img = np.clip(img, min, max)
        return skimage.img_as_float(img)  # 0~255 => 0~1
    elif img.dtype in (np.float32, np.float64):
        assert img.min() >= 0 and img.max() <= 1
        if min is not None:
            img = np.clip(img, min, max)
        return img
    else:
        raise TypeError(f"Invalid img.dtype {img.dtype}")


def img_as_ubyte(img):
    if img.dtype in (np.float32, np.float64):
        return skimage.img_as_ubyte(img)
    elif img.dtype == np.unit8:
        return img
    else:
        raise TypeError(f"Invalid img.dtype {img.dtype}")


def transpose_img(img):
    if img.ndim == 4:
        return np.transpose(img, (0, 3, 2, 1))  # NHWC <=> NCHW
    else:
        return np.transpose(img)  # HWC <=> CHW


def to_tensor(img):
    assert isinstance(img, np.ndarray)
    _img = img
    if img.ndim == 4:  # n,h,w,c
        img_h, img_w, img_c = img.shape[1:]
        img = np.transpose(img, (1, 2, 0, 3)).reshape(img_h, img_w, -1)  # h,w,(n*c)
    elif img.ndim == 3:
        img_h, img_w, img_c = img.shape
    elif img.ndim == 2:
        img_h, img_w = img.shape
        img_c = 1
    else:
        raise ValueError(f"img.ndim can only be one of 2,3,4")

    # shape:
    #   1) h,w,c => c,h,w
    #   2) h,w => h,w
    # dtype:
    #   1) np.uint8 => torch.float32
    #   2) same dtype
    # value:
    #   1) if np.uint8, value.div(255), 0 ~ 1
    #   2) same value
    img = F.to_tensor(img)

    if _img.ndim == 4:
        img = img.reshape(-1, img_c, img_h, img_w)  # (n*c),h,w => n,c,h,w
    return img


def to_img_np(img):
    assert isinstance(img, torch.Tensor)
    if img.is_floating_point():
        img = img.mul(255).byte()
    img = img.cpu().numpy()
    img = np.clip(img, 0, 255)

    if img.ndim == 4:  # b,c,h,w => b,h,w,c
        img = np.transpose(img, (0, 2, 3, 1))
    elif img.ndim == 3:  # c,h,w => h,w,c
        img = np.transpose(img, (1, 2, 0))
    return img


def normalize(tensor, mean=IMAGENET_RGB_MEAN, std=IMAGENET_RGB_STD):
    assert isinstance(tensor, torch.Tensor)
    _tensor = tensor
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)  # h,w => 1,h,w

    tensor = F.normalize(tensor, mean, std)

    if _tensor.ndim == 2:
        tensor = tensor.squeeze(0)
    return tensor


def inv_normalize(tensor, mean=IMAGENET_RGB_MEAN, std=IMAGENET_RGB_STD):
    assert isinstance(tensor, torch.Tensor)
    _tensor = tensor
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)  # h,w => 1,h,w

    tensor = tensor.clone()
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError(f"std evaluated to zero after conversion to {tensor.dtype}, leading to division by zero.")
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)

    tensor.mul_(std).add_(mean)

    if _tensor.ndim == 2:
        tensor = tensor.squeeze(0)
    return tensor


def stack_imgs(imgs, squared=False, size=None):
    def resize(img, height, width):
        h, w = img.shape[1:]
        ratio = min(height / h, width / w)
        h, w = int(h * ratio), int(w * ratio)
        img = F.resize(img, size=(h, w))
        new_img = torch.zeros(3, height, width)
        t = int((height - h) / 2)
        l = int((width - w) / 2)
        new_img[:, t:t + h, l:l + w] = img
        return new_img

    if isinstance(imgs[0], np.ndarray):
        imgs = [to_tensor(img) for img in imgs]
    if size is None:
        height = max([img.shape[1] for img in imgs])
        width = max([img.shape[2] for img in imgs])
    else:
        height, width = size if isinstance(size, (list, tuple)) else (size, size)

    if squared:
        size = max(height, width)
        imgs = [resize(img, size, size) for img in imgs]
    else:
        imgs = [resize(img, height, width) for img in imgs]
    imgs = torch.stack(imgs, 0)
    return imgs


def save_img(img, path, use_cv2=False, convert=False):
    if use_cv2:
        try:
            if convert:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, img)
        except Exception:
            print(f"Error occurred when processing image (shape: {img.shape}) to path '{path}'")
            pass
    else:
        img = skimage.img_as_ubyte(img)
        io.imsave(path, img)
