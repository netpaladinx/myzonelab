import numbers

import torch
import numpy as np


def to_numpy(data, copy=True, dtype=None):
    if isinstance(data, np.ndarray):
        return data.copy() if copy else data
    elif isinstance(data, (list, tuple)):
        return np.array(data) if dtype is None else np.array(data, dtype=dtype)
    elif isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
        return data.copy() if copy else data
    else:
        raise TypeError('Cannot convert to numpy array')


def npf(data, dtype=float):
    if isinstance(data, torch.Tensor):
        return to_numpy(data).astype(dtype)
    elif isinstance(data, np.ndarray):
        return data.astype(dtype)
    else:
        return np.array(data, dtype=dtype)


def np32f(data):
    if isinstance(data, torch.Tensor):
        return to_numpy(data).astype(np.float32)
    elif isinstance(data, np.ndarray):
        return data.astype(np.float32)
    else:
        return np.array(data, dtype=np.float32)


def pyscalar(data):
    if isinstance(data, (torch.Tensor, np.ndarray, np.generic)):
        return data.mean().item()  # mean
    elif isinstance(data, (int, float)):
        return data
    elif isinstance(data, (list, tuple)):
        return sum(pyscalar(v) for v in data)  # sum
    else:
        raise TypeError(f"Cannot merge {data} to a scalar")


def pydata(data):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    if isinstance(data, np.ndarray):
        data = data.tolist()
    while isinstance(data, (list, tuple)) and len(data) == 1:
        data = data[0]
    if isinstance(data, (list, tuple)):
        data = type(data)(pydata(v) for v in data)
    return data


def isscalar(data):
    if isinstance(data, numbers.Number):
        return True
    elif isinstance(data, np.ndarray) and data.ndim == 0:
        return True
    elif isinstance(data, torch.Tensor) and len(data) == 1:
        return True
    else:
        return False
