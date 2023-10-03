import random

import numpy as np

from ...registry import DATA_TRANSFORMS
from ...utils import tolist


@DATA_TRANSFORMS.register_class('compose')
class Compose:
    def __init__(self, transforms, dataset):
        self.dataset = dataset
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = DATA_TRANSFORMS.create(transform)
            assert callable(transform), f"transform must be callable or dict, but got {type(transform)}"
            self.transforms.append(transform)

    def __call__(self, input_dict, begin=0, end=0):  # end (not included)
        for step, trans in enumerate(self.transforms):
            if step < begin:
                continue

            input_dict = trans(input_dict, self.dataset, step)
            if input_dict is None:
                return None

            if end == step + 1:
                break
        return input_dict

    def __repr__(self):
        s = self.__class__.__name__ + '('
        for trans in self.transforms:
            s += f"\n   {trans}"
        s += '\n)'
        return s


@DATA_TRANSFORMS.register_class('collect')
class Collect:
    def __init__(self,
                 shared=None,
                 inputs=None,
                 targets=None,
                 arrays=None,
                 input_batching='stack',
                 target_batching='stack',
                 array_batching='stack'):
        # batching: 'stack' | 'concat'
        self.shared = tolist(shared)
        self.inputs = tolist(inputs)
        self.targets = tolist(targets)
        self.meta = {'input_batching': input_batching,
                     'target_batching': target_batching,
                     'array_batching': array_batching,
                     'array_keys': tolist(arrays)}

    def __call__(self, input_dict, dataset, step):
        if isinstance(input_dict, list):
            input_dict = [self._collect(in_dict) for in_dict in input_dict]
        else:
            input_dict = self._collect(input_dict)

        return input_dict

    def _collect(self, input_dict):
        shared = {}
        inputs = {}
        targets = {}
        for key in list(input_dict.keys()):
            if key in self.inputs:
                inputs[key] = input_dict.pop(key, None)
            elif key in self.targets:
                targets[key] = input_dict.pop(key, None)
            elif key in self.shared:
                shared[key] = input_dict.pop(key, None)

        input_dict['_shared'] = shared
        input_dict['_inputs'] = inputs
        input_dict['_targets'] = targets
        input_dict['_meta'] = self.meta
        return input_dict


@DATA_TRANSFORMS.register_class('repeat')
class Repeat:
    def __init__(self, apply_prob=0.25, begin=0, end=0):
        self.apply_prob = apply_prob
        self.begin = begin
        self.end = end

    def __call__(self, input_dict, dataset, step):
        if np.random.rand() < self.apply_prob:
            if self.end == 0 or self.end > step:
                self.end = step

            idx = random.randint(0, len(dataset) - 1)
            input_dict2 = dataset.get_unprocessed_item(idx)
            input_dict2 = dataset.data_transforms(input_dict2, self.begin, self.end)
            if 'input_dict2' in input_dict:
                input_dict['input_dict2'].append(input_dict2)
            else:
                input_dict['input_dict2'] = [input_dict2]
        return input_dict
