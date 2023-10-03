from setuptools import setup, Extension
import numpy as np


setup(
    name='myzonelab',
    version='1.0',
    packages=['myzonecv'],
    package_dir={'myzonecv': 'myzonecv'},
    package_data={'myzonecv': ['configs/*/*.yaml', 'core/data/transform/coloradjust.json']},
    ext_modules=[
        Extension('myzonecv.core.data.datasets.coco._mask',
                  ['myzonecv/core/data/datasets/coco/c/mask_apis.c', 'myzonecv/core/data/datasets/coco/_mask.pyx'],
                  include_dirs=[np.get_include(), 'myzonecv/core/data/datasets/coco/c'],
                  extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c99'])
    ],
    scripts=[
        'bin/myzone-train',
        'bin/myzone-train-dev',
        'bin/myzone-torch2onnx',
        'bin/myzone-infer',
        'bin/myzone-test'
    ],
    install_requires=[
        'setuptools>=18.0',
        'cython>=0.27.3',
        'matplotlib>=2.1.0',
        'torch>=1.6.0'
    ]
)
