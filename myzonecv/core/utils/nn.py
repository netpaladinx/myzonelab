import torch.nn as nn


def replace_syncbn(module):
    new_module = module

    if isinstance(module, nn.SyncBatchNorm):
        new_module = nn.BatchNorm3d(module.num_features,
                                    eps=module.eps,
                                    momentum=module.momentum,
                                    affine=module.affine,
                                    track_running_stats=module.track_running_stats)
        if module.affine:
            new_module.weight.data = module.weight.data.clone().detach()
            new_module.bias.data = module.bias.data.clone().detach()
            new_module.weight.requires_grad = module.weight.requires_grad
            new_module.bias.requires_grad = module.bias.requires_grad
        new_module.running_mean = module.running_mean
        new_module.running_var = module.running_var
        new_module.num_batches_tracked = module.num_batches_tracked

    for name, child in module.named_children():
        new_module.add_module(name, replace_syncbn(child))
    del module
    return new_module
