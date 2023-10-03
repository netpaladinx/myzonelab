import math

import cv2
import numpy as np
import matplotlib.pyplot as plt

from .bboxseg import cxywh2xyxy, xywh2xyxy
from .dtype import to_numpy


def normalize_per_channel(imgs, v_min=0., v_max=1.):
    bs, _, _, nc = imgs.shape
    v_min = np.min(np.reshape(imgs, (bs, -1, nc)), axis=1)[:, None, None, :]
    v_max = np.max(np.reshape(imgs, (bs, -1, nc)), axis=1)[:, None, None, :]
    imgs = (imgs - v_min) / np.maximum(np.spacing(1), (v_max - v_min))
    return (imgs * 255).astype(np.uint8)


def plot_images(imgs,
                save_path,
                data_format='NHWC',
                img_format='RGB',
                normalize=False,
                point_data=None,
                point_size=4,
                point_color=(255, 0, 0),
                box_data=None,
                box_format='cxywh',
                box_thickness=1,
                box_color=(255, 0, 0),
                box_text_data=None,
                font_scale=0.5,
                text_thickness=1,
                text_color=(255, 0, 0),
                max_subplots=16,
                max_size=1920):
    """ data_format: 'NHWC', 'NCHW'
        img_type: 'RGB', 'GRAYSCALE'
    """
    imgs = to_numpy(imgs).copy()

    if data_format == 'NCHW':
        imgs = np.transpose(imgs, (0, 2, 3, 1))

    if normalize:
        imgs = normalize_per_channel(imgs)

    if np.max(imgs[0]) <= 1:
        imgs = (imgs * 255).astype(np.unit8)

    bs, h, w, c = imgs.shape
    n = min(bs, max_subplots)
    m = math.ceil(bs ** 0.5)

    mosaic = np.full((m * h, m * w, c), 255, dtype=np.uint8)
    for i in range(n):
        x0, y0 = w * (i % m), h * (i // m)
        mosaic[y0:y0 + h, x0:x0 + w, :] = imgs[i]

    scale = min(max_size / (m * max(h, w)), 1)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, (m * w, m * h))

    if box_text_data is not None:
        text_size, text_baseline = cv2.getTextSize('0', cv2.FONT_HERSHEY_DUPLEX, font_scale, text_thickness)

    for i in range(n):
        x0, y0 = w * (i % m), h * (i // m)

        if point_data is not None:
            points = point_data[i]
            for p in points:
                px, py = int(x0 + p[0] * scale), int(y0 + p[1] * scale)
                cv2.circle(mosaic, (px, py), point_size, point_color)

        if box_data is not None:
            boxes = box_data[i]
            texts = box_text_data[i] if box_text_data is not None else None
            if box_format == 'cxywh':
                boxes = cxywh2xyxy(boxes)
            elif box_format == 'xywh':
                boxes = xywh2xyxy(boxes)
            for i, box in enumerate(boxes):
                px0, py0, px1, py1 = box
                px0, py0 = int(x0 + px0 * scale), int(y0 + py0 * scale)
                px1, py1 = int(x0 + px1 * scale), int(y0 + py1 * scale)
                cv2.rectangle(mosaic, (px0, py0), (px1, py1), box_color, thickness=box_thickness)
                if texts is not None:
                    cv2.putText(mosaic, str(texts[i]), (px0 + 3, py0 + text_size[1] + 3),
                                cv2.FONT_HERSHEY_DUPLEX, font_scale, text_color, text_thickness)

    if img_format == 'RGB':
        mosaic = cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR)
    elif img_format == 'GRAYSCALE':
        mosaic = mosaic[..., 0]

    cv2.imwrite(save_path, mosaic)


def plot_hist(x, bins, save_path, title='Histogram', **kwargs):
    n, bins, patches = plt.hist(x, bins, facecolor='g', alpha=0.75, **kwargs)
    plt.title(title)
    plt.grid(True)
    if 'range' in kwargs:
        plt.xlim(*kwargs['range'])
    plt.savefig(save_path)
    plt.close()
