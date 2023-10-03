from typing import Any

import torch


class no_grad(torch.no_grad):
    def __init__(self, enable=True):
        super().__init__()
        self.enable = enable

    def __enter__(self) -> None:
        if self.enable:
            super().__enter__()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self.enable:
            super().__exit__(exc_type, exc_value, traceback)
