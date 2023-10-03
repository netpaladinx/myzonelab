import sys
import os.path as osp
from collections import defaultdict
import subprocess

import cv2
import torch
import torchvision


def get_cuda_home():
    from torch.utils.cpp_extension import CUDA_HOME
    return CUDA_HOME

    
def collect_env(pretty_print=False):
    try:
        gcc = subprocess.check_output('gcc --version | head -n1', shell=True)
        gcc = gcc.decode('utf-8').strip()
    except subprocess.CalledProcessError:
        gcc = 'n/a'
    
    cuda_available = torch.cuda.is_available()
    gpu_info = {}
    if cuda_available:
        devices = defaultdict(list)
        for i in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(i)].append(str(i))
        for name, device_ids in devices.items():
            gpu_info['GPU' + ','.join(device_ids)] = name
        
        CUDA_HOME = get_cuda_home()
        gpu_info['CUDA_HOME'] = CUDA_HOME
        if CUDA_HOME and osp.isdir(CUDA_HOME):
            try:
                nvcc = osp.join(CUDA_HOME, 'bin/nvcc')
                nvcc = subprocess.check_output(f'"{nvcc}" -V | tail -n1', shell=True)
                nvcc = nvcc.decode('utf-8').strip()
            except subprocess.SubprocessError:
                nvcc = 'n/a'
            gpu_info['NVCC'] = nvcc

    env_info = {
        'sys.platform': sys.platform,
        'Python': sys.version.replace('\n', ''),
        'GCC': gcc,
        'PyTorch': torch.__version__,
        'PyTorch compiling details': torch.__config__.show(),
        'TorchVision': torchvision.__version__,
        'OpenCV': cv2.__version__,
        'CUDA available': cuda_available,
        **gpu_info
    }
    
    if pretty_print:
        env_info = '\n'.join([f'{k}: {v}' for k, v in env_info.items()])
    return env_info