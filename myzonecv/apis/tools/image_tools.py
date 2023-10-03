import numpy as np
import matplotlib.pyplot as plt
import pylab
import skimage
import skimage.io as io
import skimage.transform as transform


def show_image(img_or_path, figsize=None, as_gray=False, title=None, fontweight='bold', fontsize=16):
    if figsize:
        pylab.rcParams['figure.figsize'] = figsize

    if not isinstance(img_or_path, (list, tuple)):
        img_or_path = (img_or_path,)
    for im in img_or_path:
        if isinstance(im, str):
            im = io.imread(im)
            if im.ndim == 2:
                as_gray = True

        plt.axis('off')
        if title:
            plt.title(title, fontweight=fontweight, fontsize=fontsize)
        if as_gray:
            plt.imshow(im, cmap='gray')
        else:
            plt.imshow(im)
        plt.show()


def resize_image(img, height, width):
    if img.ndim == 2:
        img = img[..., None]
    h, w = img.shape[:2]
    ratio = min(height / h, width / w)
    h, w = int(h * ratio), int(w * ratio)
    pad_h, pad_w = int((height - h) / 2), int((width - w) / 2)
    new_img = np.zeros_like(img, shape=(height, width, 3))
    img = transform.resize(img, (h, w), anti_aliasing=True)
    if img.dtype != np.uint8:
        img = skimage.img_as_ubyte(img)
    new_img[pad_h:pad_h + h, pad_w:pad_w + w] = img
    return new_img


def make_image_grid(imgs, size, n_cols, inner_padding=0):  # size: (h, w)
    imgs = [io.imread(img) if isinstance(img, str) else img for img in imgs]
    height, width = size if isinstance(size, (list, tuple)) else (size, size)
    n_imgs = len(imgs)
    n_rows = (n_imgs + n_cols - 1) // n_cols
    total_h = height * n_rows + inner_padding * (n_rows - 1)
    total_w = width * n_cols + inner_padding * (n_cols - 1)
    out_img = np.zeros_like(imgs[0], shape=(total_h, total_w, 3))
    for i, img in enumerate(imgs):
        r = i // n_cols
        c = i % n_cols
        offset_h = r * (height + inner_padding)
        offset_w = c * (width + inner_padding)
        new_img = resize_image(img, height, width)
        out_img[offset_h:offset_h + height, offset_w:offset_w + width] = new_img
    return out_img
