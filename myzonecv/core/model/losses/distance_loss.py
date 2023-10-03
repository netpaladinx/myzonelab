import torch

from ...registry import LOSSES
from ..base_module import BaseModule


def reduce(x, reduction='mean'):
    if reduction == "sum":
        x = x.sum()
    elif reduction == "mean":
        x = x.mean()
    elif reduction == "none":
        x = x
    else:
        raise ValueError(f"Invalid reduction: {reduction}")
    return x


@LOSSES.register_class('l1_loss')
class L1Loss(BaseModule):
    def __init__(self, reduction='mean', dim=-1):
        super().__init__()
        assert reduction in ('mean', 'sum', 'none')
        self.reduction = reduction
        self.dim = dim

    def forward(self, x, y, mask=None):
        loss = torch.abs(x - y)
        if self.dim is not None:
            loss = loss.sum(self.dim)
        if mask is not None:
            loss = loss * mask
        loss = reduce(loss, reduction=self.reduction)
        return loss


@LOSSES.register_class('l2_loss')
class L2Loss(BaseModule):
    def __init__(self, use_sqrt=True, reduction='mean', dim=-1, eps=1e-8):
        super().__init__()
        self.use_sqrt = use_sqrt
        assert reduction in ('mean', 'sum', 'none')
        self.reduction = reduction
        self.dim = dim
        self.eps = eps

    def forward(self, x, y, mask=None):
        loss = torch.square(x - y)
        if self.dim is not None:
            loss = loss.sum(self.dim)
        if self.use_sqrt:
            loss = torch.sqrt(loss + self.eps)
        if mask is not None:
            loss = loss * mask
        loss = reduce(loss, reduction=self.reduction)
        return loss
