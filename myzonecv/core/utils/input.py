import torch


def to(data, device=None, dtype=None):
    if data is None:
        return None

    if isinstance(data, dict):
        return {key: to(value, device, dtype) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(to(elem, device, dtype) for elem in data)
    elif isinstance(data, torch.Tensor):
        if data.dtype in (torch.float16, torch.float32, torch.float64):
            return data.to(device=device, dtype=dtype)
        else:
            return data.to(device=device)
    else:
        return data
