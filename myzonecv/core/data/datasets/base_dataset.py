from abc import ABCMeta, abstractmethod

import random
import numpy as np
import torch


def worker_init_reseed(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        random.seed(worker_info.seed)
        np.random.seed(worker_info.seed % 2**32)
        torch.manual_seed(worker_info.seed)


class BaseDataset(torch.utils.data.Dataset, metaclass=ABCMeta):
    def __init__(self, data_source=None, data_params=None, data_transforms=None, data_eval=None, name='data'):
        self.data_source = data_source
        self.data_params = data_params or {}
        self.data_eval = data_eval
        self.name = name
        self.data_mode = self.data_params.get('mode')
        self.shuffle = self.data_params.get('shuffle', self.data_mode == 'train')
        self.input_data = None
        self.input_indices = None

        from myzonecv.core.data.transforms import Compose
        self.data_transforms = Compose(data_transforms, self)

    @abstractmethod
    def load_input_data(self):
        pass

    def __len__(self):
        return len(self.input_data)

    @property
    def size(self):
        return len(self)

    @abstractmethod
    def get_unprocessed_item(self, idx):
        pass

    def __getitem__(self, idx):
        input_dict = self.get_unprocessed_item(idx)
        input_dict = self.data_transforms(input_dict)
        return input_dict

    def pipe_item(self, idx):
        input_dict = self.get_unprocessed_item(idx)
        yield (input_dict, 'get_unprocessed_item')

        for i, transform in enumerate(self.data_transforms.transforms):
            input_dict = transform(input_dict, self, i)
            yield (input_dict, transform.__class__.__name__)

            if input_dict is None:
                break

    @property
    def worker_init_fn(self):
        return worker_init_reseed

    def cache_step(self, results, batch, **kwargs):
        pass

    def evaluate_begin(self, **kwargs):
        pass

    def evaluate_step(self, results, batch, **kwargs):
        pass

    def evaluate_all(self, all_results, **kwargs):
        return None

    def summarize(self, **kwargs):
        pass

    def dump_step(self, results, batch, **kwargs):
        pass

    def dump_all(self, all_results, **kwargs):
        pass


class BaseIterableDataset(torch.utils.data.IterableDataset, metaclass=ABCMeta):
    def __init__(self, data_source=None, data_params=None, data_transforms=None, data_eval=None, name='data'):
        self.data_source = data_source
        self.data_params = data_params
        self.data_eval = data_eval
        self.name = name
        self.data_mode = self.data_params.get('mode')

        from myzonecv.core.data.transforms import Compose
        self.data_transforms = Compose(data_transforms, self)

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def get_unprocessed_item(self):
        pass

    @abstractmethod
    def deinitialize(self):
        pass

    def __iter__(self):
        self.initialize()

        while True:
            quit, input_dict = self.get_unprocessed_item()
            if quit:
                break
            input_dict = self.data_transforms(input_dict)
            yield input_dict

        self.deinitialize()

    def pipe_item(self):
        quit, input_dict = self.get_unprocessed_item()
        if not quit:
            yield (input_dict, 'get_unprocessed_item')

            for i, transform in enumerate(self.data_transforms.transforms):
                input_dict = transform(input_dict, self, i)
                yield (input_dict, transform.__class__.__name__)

                if input_dict is None:
                    break

    @property
    def size(self):
        return -1

    @property
    def worker_init_fn(self):
        return worker_init_reseed

    def cache_step(self, results, batch, **kwargs):
        pass

    def evaluate_begin(self, **kwargs):
        pass

    def evaluate_step(self, results, batch, **kwargs):
        pass

    def evaluate_all(self, all_results, **kwargs):
        return None

    def summarize(self, **kwargs):
        pass

    def dump_step(self, results, batch, **kwargs):
        pass

    def dump_all(self, all_results, **kwargs):
        pass


class BaseJITDataset(metaclass=ABCMeta):
    def __init__(self, data_source=None, data_params=None, data_transforms=None, data_eval=None, name=None):
        self.data_source = data_source
        self.data_params = data_params
        self.data_eval = data_eval
        self.name = name
        self.data_mode = self.data_params.get('mode')

        from myzonecv.core.data.transforms import Compose
        self.data_transforms = Compose(data_transforms, self)

    @abstractmethod
    def get_input_batch(self, *args, **kwargs):
        pass
