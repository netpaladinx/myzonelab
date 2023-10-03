import torch.nn as nn

from ..base_module import BaseModule


class BaseProcess(BaseModule):
    def __init__(self, default=None):
        super().__init__()
        self.default = default

    def __call__(self, *args, **kwargs):
        if self.default and hasattr(self, self.default):
            method = getattr(self, self.default)
            assert callable(method)
            return method(*args, **kwargs)
        return None
