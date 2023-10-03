import math

import numpy as np
import cv2

from ..consts import BBOX_SCALE_UNIT
from .dtype import np32f, npf


def get_inv(mat):
    assert mat.shape == (2, 3)
    mat = np.concatenate((mat, npf([[0., 0., 1.]])), 0)  # 3 x 3
    mat_inv = np.linalg.inv(mat)[:2]  # 2 x 3
    return mat_inv


def get_warp_matrix(center, scale, dst_size, rotate=0., scale_unit=BBOX_SCALE_UNIT, return_inv=False):
    """ Note that XoY of image is upside-down along Y-axis!
    """
    rot = np.deg2rad(-rotate)
    cos, sin = math.cos(rot), math.sin(rot)
    dx, dy = center
    src_w, src_h = scale[0] * scale_unit, scale[1] * scale_unit
    dst_w, dst_h = dst_size[0] - 1, dst_size[1] - 1

    s_x, s_y = dst_w / src_w, dst_h / src_h
    offset_x = -dx * cos + dy * sin + 0.5 * src_w  # anti-clockwise in the top-left-as-origin coordinate system
    offset_y = -dx * sin - dy * cos + 0.5 * src_h

    # center to (0,0) -> rotate around (0,0) -> left-top to (0,0) -> scale
    # [[s_x, 0,  0],  [[1, 0, 0.5*src_w],  [[cos, -sin, 0],  [[1, 0, -dx],
    #  [ 0, s_y, 0]]   [0, 1, 0.5*src_h],   [sin,  cos, 0],   [0, 1, -dy],
    #                  [0, 0 ,    1    ]]   [0,     0,  1]]   [0, 0,  1 ]]
    mat = npf([[cos * s_x, -sin * s_x, offset_x * s_x],
               [sin * s_y, cos * s_y, offset_y * s_y]])
    if return_inv:
        mat_inv = get_inv(mat)
        return mat, mat_inv

    return mat


def get_affine_matrix(center, scale, dst_size, rotate=0., shift=(0., 0.), scale_unit=BBOX_SCALE_UNIT, maintain_aspect_ratio=True, return_inv=False):
    """ order: center (translate) -> scale -> rotate 
    """
    rot = np.deg2rad(rotate)
    cos, sin = math.cos(rot), math.sin(rot)
    dx, dy = center
    src_w, src_h = scale[0] * scale_unit, scale[1] * scale_unit
    dst_w, dst_h = dst_size[0], dst_size[1]
    shift_x, shift_y = src_w * shift[0], src_h * shift[1]

    src_pnt0 = [dx + shift_x, dy + shift_y]
    src_dir1 = [0., src_w * -0.5] if maintain_aspect_ratio else [0., src_h * -0.5]
    src_dir1 = [src_dir1[0] * cos - src_dir1[1] * sin, src_dir1[0] * sin + src_dir1[1] * cos]
    src_pnt1 = [src_pnt0[0] + src_dir1[0], src_pnt0[1] + src_dir1[1]]
    src_dir2 = [src_dir1[1], -src_dir1[0]] if maintain_aspect_ratio else [src_dir1[1] * src_w / src_h, -src_dir1[0] * src_w / src_h]
    src_pnt2 = [src_pnt1[0] + src_dir2[0], src_pnt1[1] + src_dir2[1]]

    dst_pnt0 = [dst_w * 0.5, dst_h * 0.5]
    dst_dir1 = [0., dst_w * -0.5] if maintain_aspect_ratio else [0, dst_h * -0.5]
    dst_pnt1 = [dst_pnt0[0] + dst_dir1[0], dst_pnt0[1] + dst_dir1[1]]
    dst_dir2 = [dst_dir1[1], -dst_dir1[0]] if maintain_aspect_ratio else [dst_dir1[1] * dst_w / dst_h, -dst_dir1[0] * dst_w / dst_h]
    dst_pnt2 = [dst_pnt1[0] + dst_dir2[0], dst_pnt1[1] + dst_dir2[1]]

    mat = cv2.getAffineTransform(np32f([src_pnt0, src_pnt1, src_pnt2]), np32f([dst_pnt0, dst_pnt1, dst_pnt2]))

    if return_inv:
        mat_inv = cv2.getAffineTransform(np32f([dst_pnt0, dst_pnt1, dst_pnt2]), np32f([src_pnt0, src_pnt1, src_pnt2]))
        return mat, mat_inv

    return mat


def apply_warp_to_coord(coord, matrix):
    """ coord (np.ndarray): n x 2
    """
    ones = np.ones((len(coord), 1))
    coord = np.concatenate((coord, ones), axis=1)
    coord = coord @ matrix.T
    return coord


def apply_warp_to_map2d(map2d, matrix, dst_size, flags=cv2.INTER_LINEAR, border_value=0):
    """ map2d (np.ndarray): h x w x c or h x w  
    """
    dst_w, dst_h = int(dst_size[0]), int(dst_size[1])
    map2d = cv2.warpAffine(map2d, matrix, (dst_w, dst_h), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)
    return map2d


def revert_coord(coord, src_size, dst_size, dst_center):
    new_coord = coord.copy()
    new_coord[:, 0] = coord[:, 0] * dst_size[0] / src_size[0] + dst_center[0] - dst_size[0] * 0.5
    new_coord[:, 1] = coord[:, 1] * dst_size[1] / src_size[1] + dst_center[1] - dst_size[1] * 0.5
    return new_coord


def fliplr_coord(coord, img_width, flip_pairs=None):
    """ coord (np.ndarray): n x 2 
    """
    coord[:, 0] = img_width - 1 - coord[:, 0]
    if flip_pairs is not None:
        for left, right in flip_pairs:
            temp = coord[left].copy()
            coord[left] = coord[right]
            coord[right] = temp
    return coord
