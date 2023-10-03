from abc import ABCMeta, abstractmethod

from .base_dataset import BaseIterableDataset


class IterableBatchDataset(BaseIterableDataset, metaclass=ABCMeta):
    def __init__(self, data_source=None, data_params=None, data_transforms=None, data_eval=None, name=None):
        super().__init__(data_source, data_params, data_transforms, data_eval, name)
        self.batch_size = self.data_params.get('batch_size', 1)

    @abstractmethod
    def load_input_data(self):
        pass

    @abstractmethod
    def get_unprocessed_batch(self):
        pass

    def initialize(self):
        pass

    def get_unprocessed_item(self):
        return self.get_unprocessed_batch()

    def deinitialize(self):
        pass
