import torch
import torch.nn.functional as F


def reduce_loss(loss, reduction, mean_factor=None):
    # apply custom mean reduction if reduction is 'mean' and mean_factor is given

    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:  # reduction = 'none'
        pass
    elif reduction_enum == 1:  # reduction = 'mean'
        if mean_factor is not None:
            loss = loss.sum() / (mean_factor + torch.finfo(torch.float32).eps)
        else:
            loss = loss.mean()
    elif reduction_enum == 2:  # reduction = 'sum'
        loss = loss.sum()
    return loss


def weight_reduce_loss(loss, weight=None, reduction='mean', mean_factor=None):
    """ loss: element-wise loss
        weight: element-wise weight
    """
    if weight is not None:
        assert weight.dim() == loss.dim()
        loss = loss * weight

    loss = reduce_loss(loss, reduction, mean_factor)
    return loss
