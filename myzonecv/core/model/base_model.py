import copy
import math
from abc import ABCMeta

import numpy as np
import torch

from ..utils import profile
from .base_module import BaseModule


class BaseModel(BaseModule, metaclass=ABCMeta):
    def __init__(self,
                 fp16_enabled=False,
                 ema_enabled=False,
                 train_cfg=None,
                 eval_cfg=None,
                 infer_cfg=None,
                 init_cfg=None,
                 ema_cfg=None):
        super().__init__(init_cfg)
        self.fp16_enabled = fp16_enabled
        self.ema_enabled = ema_enabled  # Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
        self.train_cfg = train_cfg
        self.eval_cfg = eval_cfg
        self.infer_cfg = infer_cfg
        self.ema_cfg = ema_cfg
        self._ema = None

    def train_step(self, *args, **kwargs):
        kwargs['training'] = True
        kwargs.update(self.train_cfg or {})
        return self(*args, **kwargs)

    def eval_step(self, *args, **kwargs):
        kwargs['training'] = False
        kwargs.update(self.eval_cfg or {})
        return self(*args, **kwargs)

    def infer_step(self, *args, **kwargs):
        kwargs['training'] = False
        kwargs.update(self.infer_cfg or {})
        return self(*args, **kwargs)

    @staticmethod
    def match_loss(cur_loss, target_loss):
        if target_loss is None:
            return True
        if isinstance(target_loss, str):
            return cur_loss == target_loss
        if isinstance(target_loss, (list, tuple)):
            return cur_loss in target_loss
        return False

    @staticmethod
    def merge_losses(results):
        loss_names = []
        for key, val in results.items():
            if 'loss' in key:
                loss_names.append(key)
                assert isinstance(val, torch.Tensor)
                results[key] = val.mean()  # for batch of loss

        if 'loss' not in loss_names:  # the total loss has not been defined
            results['loss'] = sum(results[name] for name in loss_names)
        return results

    @staticmethod
    def collect_summary(results):
        return results.get('summary_keys', [key for key in results])

    @staticmethod
    def merge_results(results, more_res):
        summary_keys = results.pop('summary_keys', []) + more_res.pop('summary_keys', [])
        results.update(more_res)
        results['summary_keys'] = summary_keys
        return results

    def info(self, input_size=None, in_channels=3, pretty_print=False):
        n_params = sum(p.numel() for p in self.parameters())
        n_grads = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_info = {'n_params': n_params, 'n_grads': n_grads}

        if input_size:
            self.estimate_batch_size(input_size, in_channels)

        if pretty_print:
            model_info = '\n'.join([f'{k}: {v}' for k, v in model_info.items()])
        return model_info

    def estimate_batch_size(self, input_size, in_channels=3, memory_fraction=0.9):
        device = next(self.parameters()).device
        assert device.type != 'cpu', "Convert model to GPU first"
        gb_bytes = 1024**3
        total_mem = torch.cuda.get_device_properties(device).total_memory / gb_bytes
        reserved_mem = torch.cuda.memory_reserved(device) / gb_bytes
        allocated_mem = torch.cuda.memory_allocated(device) / gb_bytes
        free_mem = total_mem - reserved_mem - allocated_mem
        print(f"{total_mem:.3g}G total, {reserved_mem:.3g}G reserved, {allocated_mem:.3g}G allocated, {free_mem:.3g}G free")

        batch_sizes = [2, 4, 8, 16]
        try:
            img = [torch.zeros(bs, in_channels, input_size[0], input_size[1]) for bs in batch_sizes]
            results = profile(img, self, device, n=3)
        except Exception as e:
            print(f"estimate_batch_size: {e}")

        mem_results = [res[2] for res in results if res]
        batch_sizes = batch_sizes[:len(mem_results)]
        p = np.polyfit(batch_sizes, mem_results, deg=1)  # first degree polynomial fit
        bs = int((free_mem * memory_fraction - p[1]) / p[0])
        print(f"Optimal batch size {bs} (at least {free_mem * (1 - memory_fraction):.2f}G left)")

    @property
    def ema(self):
        if self.ema_enabled and self._ema is None:
            self._ema = copy.deepcopy(self).eval()
            self.ema_updates = 0
            # decay exponential ramp (to help early epochs)
            self.ema_decay = lambda x: (self.ema_cfg or {}).get('decay', 0.9999) * (1 - math.exp(-x / 2000))
            for p in self.ema.parameters():
                p.requires_grad_(False)
        return self._ema

    def update_ema(self):
        # update EMA parameters
        with torch.no_grad():
            self.ema_updates += 1
            decay = self.ema_decay(self.ema_updates)
            state_dict = self.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= decay
                    v += (1. - decay) * state_dict[k].detach()

    def update_ema_attr(self, include=(), exclude=()):
        for k, v in self.__dict__.items():
            if (len(include) > 0 and k not in include) or k.startswith('_') or k in exclude:
                continue
            else:
                setattr(self.ema, k, v)

    def revise_state_dict(self, state_dict):
        return state_dict

    def call_before_train(self, ctx):
        pass

    def call_after_train(self, ctx):
        pass

    def call_before_eval(self, ctx):
        pass

    def call_after_eval(self, ctx):
        pass

    def call_before_infer(self, ctx):
        pass

    def call_after_infer(self, ctx):
        pass
