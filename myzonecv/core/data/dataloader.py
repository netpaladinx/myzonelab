import os
import re
import collections

import numpy as np
import torch
from torch.utils.data import SequentialSampler, RandomSampler
from torch.utils.data import DataLoader

from ..errors import DataModeError
from .datasets import BaseIterableDataset, BaseJITDataset, IterableBatchDataset

np_str_obj_array_pattern = re.compile(r'[SaUO]')


def default_collate(batch, stack_elem=True):
    elem = batch[0]
    elem_type = type(elem)

    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # In a background process, concatenate directly into a shared memory tensor to avoid extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            if stack_elem:
                out = elem.new(storage).view(-1, *list(elem.size()))  # avoid noisy warning
            else:
                out = elem.new(storage).view(-1, *list(elem.size()[1:]))  # avoid noisy warning

        if stack_elem:
            return torch.stack(batch, 0, out=out)
        else:
            return torch.cat(batch, 0, out=out)

    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError("default_collate: numpy array should not contain {elem.dtype}")

            return default_collate([torch.as_tensor(b) for b in batch], stack_elem=stack_elem)

        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)

    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, collections.abc.Mapping):  # dict
        return type(elem)({key: default_collate([d[key] for d in batch], stack_elem=stack_elem) for key in elem})
    elif isinstance(elem, collections.abc.Sequence):  # list
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return type(elem)([default_collate(samples, stack_elem=stack_elem) for samples in transposed])

    raise TypeError(f"default_collate: batch must contain tensors, numpy arrays, but found {elem_type}")


def collate(raw_batch):
    raw_batch = [elem for elem in raw_batch if elem]
    input_dict = raw_batch[0]
    meta = input_dict.pop('_meta')
    array_keys = meta.get('array_keys', [])
    array_batching = meta.get('array_batching', 'stack')

    batch = {'batch_size': len(raw_batch)}
    inputs = {}
    targets = {}
    for key, value in input_dict.items():
        if key == '_shared':
            for k, v in value.items():
                batch[k] = v
        elif key == '_inputs':
            for k, v in value.items():
                if v is not None:
                    inputs[k] = default_collate([elem[key][k] for elem in raw_batch],
                                                stack_elem=(meta['input_batching'] == 'stack'))
        elif key == '_targets':
            for k, v in value.items():
                if v is not None:
                    targets[k] = default_collate([elem[key][k] for elem in raw_batch],
                                                 stack_elem=(meta['target_batching'] == 'stack'))
        else:
            if array_batching == 'concat' and (not array_keys or key in array_keys):
                if isinstance(value, collections.abc.Sequence):  # concat list
                    batch[key] = [e for elem in raw_batch for e in elem[key]]
                    continue

                elif isinstance(value, np.ndarray):
                    batch[key] = np.concatenate([elem[key] for elem in raw_batch], axis=0)
                    continue

            batch[key] = [elem[key] for elem in raw_batch if elem[key] is not None]

    batch['inputs'] = inputs
    batch['targets'] = targets
    return batch


def get_dataloader(dataset, loader_cfg):
    if isinstance(dataset, BaseIterableDataset):
        return get_iterable_dataloader(dataset, loader_cfg)

    if isinstance(dataset, BaseJITDataset):
        return None

    batch_size = loader_cfg.get('batch_size', 1)
    num_workers = loader_cfg.get('num_workers', 0)
    pin_memory = loader_cfg.get('pin_memory', True)
    shuffle = loader_cfg.get('shuffle', False)
    drop_last = loader_cfg.get('drop_last', False)

    batch_size = min(batch_size, len(dataset))
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, num_workers])
    worker_init_fn = dataset.worker_init_fn if num_workers > 0 and hasattr(dataset, 'worker_init_fn') else None

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            worker_init_fn=worker_init_fn,
                            collate_fn=collate,
                            pin_memory=pin_memory,
                            shuffle=shuffle,
                            drop_last=drop_last)
    return dataloader


def get_iterable_dataloader(dataset, loader_cfg):
    batch_size = None if isinstance(dataset, IterableBatchDataset) else loader_cfg.get('batch_size', 1)
    num_workers = loader_cfg.get('num_workers', dataset.num_workers if hasattr(dataset, 'num_workers') else 0)
    pin_memory = loader_cfg.get('pin_memory', True)

    num_workers = min([os.cpu_count(), num_workers])
    if batch_size is not None:
        num_workers = min([batch_size if batch_size > 1 else 0, num_workers])
    worker_init_fn = dataset.worker_init_fn if num_workers > 0 and hasattr(dataset, 'worker_init_fn') else None

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            worker_init_fn=worker_init_fn,
                            collate_fn=collate,
                            pin_memory=pin_memory)
    return dataloader


class InfiniteSequentialSampler(SequentialSampler):
    def __iter__(self):
        while True:
            yield from super().__iter__()


class InfiniteRandomSampler(RandomSampler):
    def __iter__(self):
        while True:
            yield from super().__iter__()


def get_infinite_dataloader(dataset, loader_cfg):  # only for map-style datasets
    batch_size = loader_cfg.get('batch_size', 1)
    num_workers = loader_cfg.get('num_workers', 0)
    pin_memory = loader_cfg.get('pin_memory', True)
    shuffle = loader_cfg.get('shuffle', False)
    drop_last = loader_cfg.get('drop_last', False)

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, num_workers])
    worker_init_fn = dataset.worker_init_fn if num_workers > 0 and hasattr(dataset, 'worker_init_fn') else None

    if shuffle:
        sampler = InfiniteRandomSampler(dataset)
    else:
        sampler = InfiniteSequentialSampler(dataset)

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            worker_init_fn=worker_init_fn,
                            collate_fn=collate,
                            pin_memory=pin_memory,
                            drop_last=drop_last,
                            sampler=sampler)
    return dataloader


class DataIterator:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._iter = iter(dataloader)
        self._next = next(self._iter)
        self._epoch_begin = True
        self._epoch_end = False

    def next(self):
        ret = self._next
        try:
            self._next = next(self._iter)
            self._epoch_begin = False
            self._epoch_end = False
        except StopIteration:
            self._iter = iter(self.dataloader)
            self._next = next(self._iter)
            self._epoch_begin = True
            self._epoch_end = True
        return ret

    @property
    def epoch_begin(self):  # check before call next()
        return self._epoch_begin

    @property
    def epoch_end(self):  # check after call next()
        return self._epoch_end

    @property
    def epoch_steps(self):
        try:
            return len(self.dataloader)
        except DataModeError:
            return None


def get_data_iter(dataloader):
    return DataIterator(dataloader)
