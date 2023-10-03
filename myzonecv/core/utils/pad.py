import numpy as np


def get_safe_affine_padding(sz, translate=(0, 0), rotate=0, scale=1, order_mode='TRS'):
    w, h = sz
    coords = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=float).T  # 3 x 4
    T = np.eye(3)
    T[:2, 2] = translate
    R = np.eye(3)
    r = np.deg2rad(rotate)
    R[0, :2] = [np.cos(r), -np.sin(r)]
    R[1, :2] = [np.sin(r), np.cos(r)]
    S = np.eye(3)
    S[0, 0] = S[1, 1] = scale
    if order_mode == 'TRS':
        coords = S @ R @ T @ coords
    elif order_mode == 'SRT':
        coords = T @ R @ S @ coords
    else:
        raise ValueError(f'Invalid order_mode: {order_mode}')
    x_min, x_max = np.floor(coords[0].min()), np.ceil(coords[0].max())
    y_min, y_max = np.floor(coords[1].min()), np.ceil(coords[1].max())
    safe_padding = [max(0, -x_min), max(0, x_max - w), max(0, -y_min), max(0, y_max - h)]
    return safe_padding
