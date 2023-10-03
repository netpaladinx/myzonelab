import torch


def get_gpu_memory(device):
    dev = torch.cuda.get_device_properties(device=device)
    mem = torch.cuda.max_memory_allocated(device=device)
    total_mb = int(dev.total_memory / 1024**2)
    mem_mb = int(mem / 1024**2)
    return mem_mb, total_mb
