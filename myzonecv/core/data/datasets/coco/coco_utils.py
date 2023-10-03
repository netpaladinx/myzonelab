import os.path as osp
import math

import numpy as np
import skimage
import skimage.io as io
import skimage.color as color
import skimage.transform as transform
import cv2

from .coco_data import COCOData, ID
from .coco_consts import BBOX_SCALE_UNIT, BORDER_COLOR_VALUE
from . import mask as mask_utils

# img: np.ndarray, hxwx3, np.uint8 (0~255)
# mask: np.ndarray, hxw, np.float64 (0~1, default)
# seg: polygon (list(float)) or uncompressed RLE (dict(size=[], count=[]))
# rle: compressed RLE data


def npf(a, dtype=float):
    return np.array(a, dtype=dtype)


def npi(a, dtype=int):
    return np.array(a, dtype=dtype)


def size_tuple(sz):
    if isinstance(sz, int):
        return (sz, sz)
    elif isinstance(sz, (list, tuple)) and len(sz) == 2:
        return sz
    else:
        raise ValueError(f"Invalid size {sz}")


def resize_image(img, dst_size, border_value=BORDER_COLOR_VALUE, interpolation=None, position='center'):
    img_h, img_w = img.shape[:2]
    dst_h, dst_w = (dst_size, dst_size) if isinstance(dst_size, int) else dst_size
    h_ratio = dst_h / img_h
    w_ratio = dst_w / img_w
    ratio = min(h_ratio, w_ratio)
    if ratio < 1:
        img = cv2.resize(img, (int(img_w * ratio), int(img_h * ratio)), interpolation=(interpolation or cv2.INTER_AREA))
    elif ratio > 1:
        img = cv2.resize(img, (int(img_w * ratio), int(img_h * ratio)), interpolation=(interpolation or cv2.INTER_LINEAR))
    new_h, new_w = img.shape[:2]
    if new_h != dst_h or new_w != dst_w:
        if position == 'top-left':
            border_top = 0
            border_bottom = (dst_h - new_h) - border_top
            border_left = 0
            border_right = (dst_w - new_w) - border_left
        else:
            border_top = (dst_h - new_h) // 2
            border_bottom = (dst_h - new_h) - border_top
            border_left = (dst_w - new_w) // 2
            border_right = (dst_w - new_w) - border_left
        img = cv2.copyMakeBorder(img, border_top, border_bottom, border_left, border_right, cv2.BORDER_CONSTANT, value=border_value)
    return img, ratio, (border_top, border_bottom, border_left, border_right)


def int_bbox(bbox):
    x0, y0, w, h = bbox[:4]
    x1, y1 = x0 + w, y0 + h
    # coord: (0, w/h) => pixel index: 0 ~ w/h-1 (x1,y1 can take w/h)
    x0, y0, x1, y1 = math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1)
    w, h = x1 - x0, y1 - y0
    return x0, y0, w, h


def safe_bbox(bbox, img_h, img_w, include_x1y1=True):
    if img_h and img_w:
        x0, y0, w, h = bbox
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(img_w - 1, x0 + w) if include_x1y1 else min(img_w, x0 + w)
        y1 = min(img_h - 1, y0 + h) if include_x1y1 else min(img_h, y0 + h)
        return x0, y0, x1 - x0, y1 - y0
    else:
        return bbox


def bbox_area(bbox):
    x0, y0, w, h = bbox[:4]
    return w * h


def bbox_center(bbox):
    x0, y0, w, h = bbox[:4]
    center = x0 + w * 0.5, y0 + h * 0.5
    return center


def bbox_scale(bbox, aspect_ratio=None, padding_ratio=None):
    x0, y0, w, h = bbox[:4]
    if aspect_ratio:
        if w > h * aspect_ratio:
            h = w / aspect_ratio
        elif w < h * aspect_ratio:
            w = h * aspect_ratio
    scale = (w / BBOX_SCALE_UNIT, h / BBOX_SCALE_UNIT)
    if padding_ratio:
        scale = (scale[0] * padding_ratio, scale[1] * padding_ratio)
    return scale


def bbox2cs(bbox, aspect_ratio=None, padding_ratio=None):
    center = bbox_center(bbox)
    scale = bbox_scale(bbox, aspect_ratio, padding_ratio)
    return center, scale


def bbox2xyxy(bbox):
    x0, y0, w, h = bbox[:4]
    x1 = x0 + w
    y1 = y0 + h
    return x0, y0, x1, y1


def cs2bbox(center, scale, padding_ratio=None, aspect_ratio=None):
    cx, cy = center
    w, h = scale
    if padding_ratio:
        w, h = w / padding_ratio, h / padding_ratio
    w, h = w * BBOX_SCALE_UNIT, h * BBOX_SCALE_UNIT
    if aspect_ratio is not None:
        if w / h > aspect_ratio:
            w = h * aspect_ratio
        elif w / h < aspect_ratio:
            h = w / aspect_ratio
    x0, y0 = cx - w * 0.5, cy - h * 0.5
    return x0, y0, w, h


def xyxy2bbox(xyxy):
    x0, y0, x1, y1 = xyxy
    w = x1 - x0
    h = y1 - y0
    return x0, y0, w, h


def rle2bbox(rle, to_int=False):
    bbox = mask_utils.toBbox(rle).tolist()
    if to_int:
        bbox = int_bbox(bbox)
    return bbox


def seg2bbox(seg, width, height, to_int=False):
    rle = seg2rle(seg, width, height)
    return rle2bbox(rle, to_int=to_int)


def bbox_img(img, bbox):
    img_h, img_w = img.shape[:2]
    x0, y0, w, h = int_bbox(safe_bbox(bbox, img_h, img_w, include_x1y1=False))
    return img[y0:y0 + h, x0:x0 + w]


def img2img(img, size=None, rotate=None):  # img: 0 ~ 255, np.uint8; size: (w, h)
    if size:
        w, h = size
        img = transform.resize(img, (h, w))

    if rotate:
        img = transform.rotate(img, rotate)

    if img.dtype != np.uint8:
        img = skimage.img_as_ubyte(img)

    return img


def mask2mask(mask, size=None, rotate=None):  # size: (w, h)
    if mask.dtype == np.uint8:
        if np.max(mask) <= 1:
            mask = skimage.img_as_float(mask * 255)
        else:
            mask = skimage.img_as_float(mask)

    if size:
        if isinstance(size, int):
            size = (size, size)
        elif isinstance(size, (tuple, list)):
            size = tuple(size)
        else:
            raise TypeError

        mask_h, mask_w = mask.shape[:2]
        if mask_h != size[1] or mask_w != size[0]:
            mask = cv2.resize(mask, size, interpolation=cv2.INTER_AREA)

    if rotate:
        mask = transform.rotate(mask, rotate)

    return mask  # dtype: np.float64


def bboxmask2imgmask(bbox_mask, bbox, width, height, offset=None):  # offset: (w, h) for subimages
    x0, y0, w, h = int_bbox(bbox)
    if offset:
        x0 += offset[0]
        y0 += offset[1]
    mask = mask2mask(bbox_mask, size=(w, h))
    x1, y1 = x0 + w, y0 + h
    img_mask = np.zeros((height, width))
    assert x0 < width and y0 < height
    x1 = min(x1, width)
    y1 = min(y1, height)
    img_mask[y0:y1, x0:x1] = mask[:y1 - y0, :x1 - x0]
    return img_mask


def rle2seg(rle):
    return mask_utils.toUncompressedRLE(rle)


def rle2mask(rle, clip_bbox=None, size=None):
    mask = mask_utils.decode(rle)
    mask = np.ascontiguousarray(mask)  # 'F' order => 'C' order
    if clip_bbox:
        x0, y0, w, h = int_bbox(clip_bbox)
        mask = mask[y0:y0 + h, x0:x0 + w]
    if size:
        if isinstance(size, int):
            size = (size, size)
        elif isinstance(size, (tuple, list)):
            size = tuple(size)
        else:
            raise TypeError
    mask = mask2mask(mask, size)
    return mask  # dtype: np.float64


def seg2rle(seg, width, height):
    if isinstance(seg, list) or (isinstance(seg, dict) and isinstance(seg['counts'], list)):  # polygon or uncompressed RLE
        rle = mask_utils.frPyObjects(seg, height, width)
        if isinstance(rle, list):
            if len(rle) > 1:
                rle = mask_utils.merge(rle)
            else:
                rle = rle[0]
    else:
        rle = seg
    return rle


def seg2mask(seg, width, height, clip_bbox=None, size=None):  # size: (w, h)
    rle = seg2rle(seg, width, height)
    mask = rle2mask(rle, clip_bbox, size)
    return mask


def mask2rle(mask, clip_bbox=None, size=None, accept_thres=0):
    if clip_bbox:
        x0, y0, w, h = int_bbox(clip_bbox)
        mask = mask[y0:y0 + h, x0:x0 + w]
    if size:
        if isinstance(size, int):
            size = (size, size)
        elif isinstance(size, (tuple, list)):
            size = tuple(size)
        else:
            raise TypeError
    mask = mask2mask(mask, size)
    mask_b = (mask > accept_thres).astype(np.uint8)
    mask_b = np.asfortranarray(mask_b)
    rle = mask_utils.encode(mask_b)
    return rle


def mask2seg(mask, clip_bbox=None, size=None, accept_thres=0):  # size: (w, h)
    rle = mask2rle(mask, clip_bbox, size, accept_thres)
    seg = rle2seg(rle)
    return seg


def mask_area(mask, weighed=False):
    if weighed:
        return mask.clip(0, 1).sum()
    else:
        return (mask > 0).sum()


def get_rle(ann, img=None, coco_data=None, width=None, height=None):
    """ ann => one RLE """
    if width is None or height is None:
        width = ann.get('width')
        height = ann.get('height')

    if width is None or height is None:
        if not (isinstance(img, dict) or isinstance(img, np.ndarray)):
            img = coco_data.imgs[ann['image_id']]

        if isinstance(img, dict):
            width = img.get('width')
            height = img.get('height')
        elif isinstance(img, np.ndarray):
            height, width = img.shape[:2]

    assert width is not None and height is not None

    seg = ann['segmentation']
    rle = seg2rle(seg, width, height)
    return rle


def get_mask(ann, img=None, coco_data=None, width=None, height=None, resolution=None):
    """ ann => one mask """
    rle = get_rle(ann, img, coco_data, width, height)
    if resolution:
        height, width = rle['size']
        bbox = rle2bbox(rle, to_int=True)
        mask = rle2mask(rle, clip_bbox=bbox, size=resolution)
        mask = mask2mask(mask, size=bbox[2:4])
        mask = bboxmask2imgmask(mask, bbox, width, height)
    else:
        mask = rle2mask(rle)
    return mask


def get_img(ann=None, img=None, coco_data=None):
    if isinstance(img, dict) and 'file_path' in img:
        img_path = img['file_path']
        assert osp.isfile(img_path)
        img = io.imread(img_path)
    elif isinstance(coco_data, COCOData):
        if isinstance(img, (ID, tuple, list)):
            img_id = img
        elif isinstance(img, int):
            img_id = (img,)
        elif isinstance(img, dict) and 'id' in img:
            img_id = img['id']
        elif isinstance(ann, dict) and 'image_id' in img:
            img_id = ann['image_id']
        else:
            raise ValueError("img must have key 'id' or ann must have key 'image_id'")
        img = coco_data.read_img(img_id)

    assert isinstance(img, np.ndarray)

    if img.ndim == 2:
        img = color.gray2rgb(img)
    assert img.ndim == 3

    return img  # np.ndarray (hxwx3) (dtype=np.uint8)


def get_size(img, coco_data=None):
    if isinstance(img, dict):
        width = img.get('width')
        height = img.get('height')
        if width is None or height is None:
            img = get_img(img=img, coco_data=coco_data)
        else:
            return (width, height)
    assert isinstance(img, np.ndarray)
    height, width = img.shape[:2]
    return (width, height)


def update_offset(offset, new_offset):
    if offset is not None:
        return (offset[0] + new_offset[0], offset[1] + new_offset[1])
    else:
        return new_offset


def update_scale(scale, new_scale):
    if scale is not None:
        return (scale[0] * new_scale[0], scale[1] * new_scale[1])
    else:
        return new_scale


def update_rotate(rotate, new_rotate):
    if rotate is not None:
        return rotate + new_rotate
    else:
        return new_rotate


def update_flip(flip, new_flip):
    if flip is not None:
        return not flip if new_flip else flip
    else:
        return new_flip


def update_bbox(bbox, offset=(0, 0), scale=(1, 1), flip=False, width=None, height=None):
    if bbox is None:
        return bbox
    x, y, w, h = bbox
    x = max(0, (x - offset[0]) * scale[0])
    y = max(0, (y - offset[1]) * scale[1])
    w = w * scale[0]
    h = h * scale[1]
    if flip:
        assert width and height
        x = width - (x + w)
    if width:
        x = min(x, width)
        w = min(x + w, width) - x
    if height > 0:
        y = min(y, height)
        h = min(y + h, height) - y
    return [x, y, w, h]


def update_kpts(kpts, offset=(0, 0), scale=(1, 1), rotate=0, flip=False, width=None, height=None, trim=False):
    if kpts is None:
        return kpts, 0
    n_kpts = len(kpts) // 3
    n_out_kpts = 0
    out_kpts = []
    assert ((rotate != 0 or flip) and width and height) or not (rotate != 0 or flip)
    for i in range(n_kpts):
        x, y, v = kpts[3 * i: 3 * i + 3]
        x += 0.5  # pixel index => coord
        y += 0.5
        if v > 0:
            x = (x - offset[0]) * scale[0]
            y = (y - offset[1]) * scale[1]
            if width and height:
                cx, cy = width * 0.5, height * 0.5
                if rotate != 0:
                    dx, dy = x - cx, y - cy
                    rot = -rotate * math.pi / 180
                    dx2 = dx * math.cos(rot) - dy * math.sin(rot)
                    dy2 = dx * math.sin(rot) + dy * math.cos(rot)
                    x, y = cx + dx2, cy + dy2
                if flip:
                    x = 2 * cx - x
            x, y = int(x), int(y)  # coord => pixel index
        if trim and ((width and x > width) or (height > 0 and y > height) or x < 0 or y < 0):
            x, y, v = 0, 0, 0
        if v > 0:
            n_out_kpts += 1
        else:
            x, y, v = 0, 0, 0
        out_kpts += [x, y, v]
    if flip:
        for left, right in KEYPOINT_FLIP_PAIRS:
            tup = out_kpts[left * 3:left * 3 + 3]
            out_kpts[left * 3:left * 3 + 3] = out_kpts[right * 3:right * 3 + 3]
            out_kpts[right * 3:right * 3 + 3] = tup
    return out_kpts, n_out_kpts


def seg_img_with_mask(ann, img=None, coco_data=None):
    img = get_img(ann, img, coco_data)
    mask = get_mask(ann, img)
    mask_b = (mask > 0).astype(np.uint8)
    img = img * mask_b[:, :, None]
    return img, mask


def bbox_img_with_ann(ann, img=None, coco_data=None, return_seg_img_with_mask=False):
    img = get_img(ann, img, coco_data)
    x0, y0, w, h = int_bbox(ann['bbox'])
    bbox_img = img[y0:y0 + h, x0:x0 + w]

    seg_img, mask = seg_img_with_mask(ann, img)
    mask = mask[y0: y0 + h, x0:x0 + w]
    seg_img = seg_img[y0: y0 + h, x0:x0 + w]
    seg = mask2seg(mask)

    ann = copy.deepcopy(ann)
    ann['segmentation'] = seg
    ann['keypoints'], ann['num_keypoints'] = update_kpts(ann['keypoints'], offset=(x0, y0))
    ann['bbox'] = update_bbox(ann['bbox'], offset=(x0, y0), width=w, height=h)
    ann['offset'] = update_offset(ann.get('offset'), (x0, y0))
    ann['width'] = w
    ann['height'] = h

    if return_seg_img_with_mask:
        return bbox_img, ann, seg_img, mask
    else:
        return img, ann


def transform_img_with_ann(ann, img=None, coco_data=None, scale=(1, 1), rotate=0, flip=False):
    def _size_for_rotate(rot, w, h):
        rot = rotate * math.pi / 180
        new_w = max(w, abs(w * math.cos(rot)) + abs(h * math.sin(rot)))
        new_h = max(h, abs(w * math.sin(rot)) + abs(h * math.cos(rot)))
        return int(new_w + 0.5), int(new_h + 0.5)

    ann = copy.deepcopy(ann)
    img = get_img(ann, img, coco_data)
    img_h, img_w = img.shape[:2]
    mask = get_mask(ann, img, coco_data)
    seg = ann.get('segmentation')
    bbox = ann.get('bbox')
    area = bbox_area(bbox)
    kpts = ann.get('keypoints')
    n_kpts = ann.get('num_keypoints')
    prev_offset = ann.get('offset')
    prev_scale = ann.get('scale')
    prev_rotate = ann.get('rotate')
    prev_flip = ann.get('flip')

    if scale[0] != 1 or scale[1] != 1:
        img_w, img_h = int(img_w * scale[0] + 0.5), int(img_h * scale[1] + 0.5)
        img = img2img(img, size=(img_w, img_h))
        mask = mask2mask(mask, size=(img_w, img_h))
        rle = mask2rle(mask, accept_thres=0.5)
        seg = rle2seg(rle)
        bbox = update_bbox(bbox, scale=scale, width=img_w, height=img_h)
        area = bbox_area(bbox)
        kpts, n_kpts = update_kpts(kpts, scale=scale, width=img_w, height=img_h)
        prev_scale = update_scale(prev_scale, scale)

    if rotate != 0:
        rot_w, rot_h = _size_for_rotate(rotate, img_w, img_h)
        rot_img = np.zeros((rot_h, rot_w, 3), dtype=np.uint8)
        rot_mask = np.zeros((rot_h, rot_w))
        rot_offset = (int(rot_w / 2 - img_w / 2 + 0.5), int(rot_h / 2 - img_h / 2 + 0.5))
        rot_ox, rot_oy = rot_offset
        rot_img[rot_oy:rot_oy + img_h, rot_ox:rot_ox + img_w] = img
        rot_mask[rot_oy:rot_oy + img_h, rot_ox:rot_ox + img_w] = mask
        img = img2img(rot_img, rotate=rotate)
        mask = mask2mask(rot_mask, rotate=rotate)
        rle = mask2rle(mask, accept_thres=0.5)
        seg = rle2seg(rle)
        img_h, img_w = img.shape[:2]
        bbox = rle2bbox(rle)
        area = bbox_area(bbox)
        kpts, n_kpts = update_kpts(kpts, offset=(-rot_ox, -rot_oy), rotate=rotate, width=img_w, height=img_h)
        prev_offset = update_offset(prev_offset, (-rot_ox, -rot_oy))
        prev_rotate = update_rotate(prev_rotate, rotate)

    if flip:
        img = np.flip(img, 1)
        mask = np.flip(mask, 1)
        rle = mask2rle(mask, accept_thres=0.5)
        seg = rle2seg(rle)
        img_h, img_w = img.shape[:2]
        bbox = update_bbox(bbox, flip=flip, width=img_w, height=img_h)
        area = bbox_area(bbox)
        kpts, n_kpts = update_kpts(kpts, flip=flip, width=img_w, height=img_h)
        prev_flip = update_flip(prev_flip, flip)

    ann['segmentation'] = seg
    ann['bbox'] = bbox
    ann['area'] = area
    ann['keypoints'] = kpts
    ann['num_keypoints'] = n_kpts
    ann['offset'] = prev_offset
    ann['scale'] = prev_scale
    ann['rotate'] = prev_rotate
    ann['flip'] = prev_flip
    return img, ann


def overlay(bg_img, seg_img, seg_ann, offset=(0, 0), translate=(0, 0)):  # offset if for bg_img
    seg_mask = get_mask(seg_ann, seg_img)
    seg_h, seg_w = seg_img.shape[:2]
    bg_h, bg_w = bg_img.shape[:2]
    bg_seg_mask = np.zeros((bg_h, bg_w))
    bg_seg_img = np.zeros_like(bg_img)
    seg_offset = seg_ann.get('offset', (0, 0))
    seg_offset = (seg_offset[0] + translate[0], seg_offset[1] + translate[1])
    dx, dy = int(seg_offset[0] - offset[0]), int(seg_offset[1] - offset[1])
    bg_x0, bg_y0 = max(dx, 0), max(dy, 0)
    seg_x0, seg_y0 = max(-dx, 0), max(-dy, 0)
    w = min(bg_w - bg_x0, seg_w - seg_x0)
    h = min(bg_h - bg_y0, seg_h - seg_y0)
    img = np.copy(bg_img)
    ann = None
    if w > 0 and h > 0:
        bg_seg_mask[bg_y0:bg_y0 + h, bg_x0:bg_x0 + w] = seg_mask[seg_y0:seg_y0 + h, seg_x0:seg_x0 + w]
        bg_seg_img[bg_y0:bg_y0 + h, bg_x0:bg_x0 + w] = seg_img[seg_y0:seg_y0 + h, seg_x0:seg_x0 + w]
        mask = bg_seg_mask[:, :, None].astype(np.uint8)
        img = img * (1 - mask) + bg_seg_img * mask
        ann = copy.deepcopy(seg_ann)
        rle = mask2rle(bg_seg_mask, accept_thres=0.5)
        seg = rle2seg(rle)
        img_h, img_w = img.shape[:2]
        bbox = update_bbox(ann['bbox'], offset=(-dx, -dy), width=img_w, height=img_h)
        area = bbox_area(bbox)
        kpts, n_kpts = update_kpts(ann['keypoints'], offset=(-dx, -dy), width=img_w, height=img_h, trim=True)
        ann['segmentation'] = seg
        ann['bbox'] = bbox
        ann['area'] = area
        ann['keypoints'] = kpts
        ann['num_keypoints'] = n_kpts
        ann['offset'] = offset
    return img, ann


def update_visibility(anns, img):
    def _update_kpts_vis(kpts, mask):
        n = len(kpts) // 3
        new_kpts = []
        for i in range(n):
            x, y, v = kpts[3 * i:3 * i + 3]
            if v > 0:
                v = 2 if np.any(mask[max(y - 1, 0):y + 1, max(x - 1, 0):x + 1]) else 1
            new_kpts += [x, y, v]
        return new_kpts

    n = len(anns)
    new_anns = []
    for i in range(n):
        rles = []
        for j in range(i, n):
            rles.append(get_rle(anns[j], img))
        rle = mask_utils.diff(rles)
        mask = rle2mask(rle)
        seg = rle2seg(rle)
        ann = copy.deepcopy(anns[i])
        ann['segmentation'] = seg
        ann['keypoints'] = _update_kpts_vis(ann['keypoints'], mask)
        new_anns.append(ann)
    return new_anns
