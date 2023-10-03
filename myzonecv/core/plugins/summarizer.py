import os.path as osp
from collections import deque, OrderedDict
import json
import datetime
from abc import ABCMeta, abstractmethod

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from ..registry import Registry
from ..context import ContextError
from ..utils import pyscalar, np32f, round_float, Timer, get_gpu_memory, mkdir, list2str
from ..optim.utils import get_momentum
from .base_plugin import BasePlugin

RECORDS = Registry('record')


class BaseRecord(metaclass=ABCMeta):
    @abstractmethod
    def update(self, val, cnt=1, tag=''):
        pass

    @abstractmethod
    def average(self, **kwargs):
        pass


@RECORDS.register_class('moving_record')
class MovingRecord(BaseRecord):
    def __init__(self, name, alpha=0, K=1):
        assert alpha < 1 and K >= 1
        self.name = name
        K_ = 1 / (1 - alpha)
        alpha_ = (K - 1) / K
        self.alpha = max(alpha, alpha_)
        self.K = max(K, K_)
        self.val = 0
        self.cnt = 0
        self.tag = 0
        self.k = 0

    def update(self, val, cnt=1, tag=''):
        if self.k >= self.K:
            self.cnt = self.cnt * (self.K - 1) / self.K

        alpha = self.cnt / (self.cnt + cnt)
        self.val = self.val * alpha + val * (1 - alpha)
        self.cnt += cnt
        self.tag = tag
        self.k += 1

    def average(self, **kwargs):
        return self.val


@RECORDS.register_class('latest_record')
class LatestRecord(BaseRecord):
    def __init__(self, name, K=1):
        assert K > 0
        self.name = name
        self.K = K
        self.vals = deque()
        self.cnts = deque()
        self.tags = deque()

    def update(self, val, cnt=1, tag=''):
        self.vals.append(val)
        self.cnts.append(cnt)
        self.tags.append(tag)
        while len(self.vals) > self.K:
            self.vals.popleft()
            self.cnts.popleft()
            self.tags.popleft()

    def average(self, latest=0):
        val = np32f(self.vals)[-latest:]
        cnt = np32f(self.cnts)[-latest:]
        avg = np.sum(val * cnt) / np.sum(cnt)
        return float(avg)


@RECORDS.register_class('all_record')
class AllRecord(BaseRecord):
    def __init__(self, name):
        self.name = name
        self.vals = []
        self.cnts = []
        self.tags = []

    def update(self, val, cnt=1, tag=''):
        self.vals.append(val)
        self.cnts.append(cnt)
        self.tags.append(tag)

    def average(self, latest=0):
        val = np32f(self.vals[-latest:])
        cnt = np32f(self.cnts[-latest:])
        avg = np.sum(val * cnt) / np.sum(cnt)
        return float(avg)


class SummaryBuffer:
    tb_summary_writer = None

    def __init__(self, record_cfg):
        self.record_cfg = record_cfg
        self.buffer = OrderedDict()

    def _create(self, key, val):
        if isinstance(val, (list, tuple)):
            return [self._create(f'{key}.{i}', v) for i, v in enumerate(val)]
        elif isinstance(val, dict):
            return {k: self._create(f'{key}.{k}', v) for k, v in val.items()}
        else:
            return RECORDS.create(self.record_cfg, args=(key,))

    def _update(self, buf, key, val, cnt=1, tag='', gs=None):
        if isinstance(val, (list, tuple)):
            for i, v in enumerate(val):
                self._update(buf[key], i, v, cnt, tag, gs)
        elif isinstance(val, dict):
            for k, v in val.items():
                self._update(buf[key], k, v, cnt, tag, gs)
        else:
            assert isinstance(buf[key], BaseRecord)
            if val is not None:
                buf[key].update(pyscalar(val), cnt=cnt, tag=tag)
                if self.tb_summary_writer is not None and gs is not None:
                    self.tb_summary_writer.add_scalar(buf[key].name, val, gs)

    def update(self, results, cnt=1, tag='', include=None, exclude=None, global_step=None):
        for key, val in results.items():
            if include and key not in include:
                continue
            if exclude and key in exclude:
                continue
            if key not in self.buffer:
                self.buffer[key] = self._create(key, val)
            self._update(self.buffer, key, val, cnt=cnt, tag=tag, gs=global_step)

    def _average(self, val, latest=0):
        if isinstance(val, (list, tuple)):
            return [self._average(v, latest) for i, v in enumerate(val)]
        elif isinstance(val, dict):
            return {k: self._average(v, latest) for k, v in val.items()}
        else:
            assert isinstance(val, BaseRecord)
            return val.average(latest=latest)

    def average(self, latest=0):
        results = OrderedDict()
        for key, val in self.buffer.items():
            results[key] = self._average(val, latest)
        return results

    def clear(self):
        if self.record_cfg['type'] == 'all_record':
            self.buffer.clear()


class Summarizer(BasePlugin):
    """ Summarize: 1) train losses 2) train metrics 3) val losses 4) val metrics 5) hyperparameters
    """

    def __init__(self,
                 interval=10,  # steps
                 ignore_last=False,
                 record_cfg={'type': 'latest_record', 'K': 100},
                 use_tensorboard=True):
        self.interval = interval
        self.ignore_last = ignore_last
        self.record_cfg = record_cfg
        self.use_tensorboard = use_tensorboard

        self.n = None
        self.N = None
        self.log_head = None
        self.train_buffer = None
        self.val_buffer = None
        self.summary_path = None
        self.timer = Timer()

    def call_before_run(self, ctx):
        self.train_buffer = SummaryBuffer(self.record_cfg)
        self.val_buffer = SummaryBuffer({'type': 'latest_record', 'K': 1})
        self.summary_path = osp.join(ctx.work_dir, f'summary_{ctx.timestamp}.jsonl')

        if self.use_tensorboard and SummaryBuffer.tb_summary_writer is None:
            tb_dir = osp.join(ctx.work_dir, 'tensorboard')
            mkdir(tb_dir, exist_ok=True)
            SummaryBuffer.tb_summary_writer = SummaryWriter(tb_dir)
            ctx.logger.info(f"TensorBoard: 'tensorboard --logdir {tb_dir}' http://localhost:6006/")

        self.timer.start(mark='data')

    def call_before_epoch(self, ctx):
        assert ctx.mode == 'train', f"Summarizer's `call_before_epoch` should be called in the train mode"
        self.train_buffer.clear()
        self.timer.start(mark='data')

    def call_before_step(self, ctx):
        assert ctx.mode == 'train', f"Summarizer's `call_before_step` should be called in the train mode"
        batch_size = ctx.train_batch_size
        data_time = self.timer.since_mark('data', mark='model')
        self.train_buffer.update({'data_time': data_time}, cnt=batch_size)

    def call_after_step(self, ctx, prefix=''):
        assert ctx.mode == 'train', f"Summarizer's `call_after_step` should be called in the train mode"
        batch_size = ctx.train_batch_size
        results = ctx.train_results
        summary_keys = results.get('summary_keys', ())
        self.train_buffer.update(results, cnt=batch_size, include=summary_keys, global_step=ctx.train_step)

        step_time, model_time = self.timer.since_mark(('data', 'model'), mark='data')
        self.train_buffer.update({'model_time': model_time, 'step_time': step_time}, cnt=batch_size)

        self._update_before_log(ctx)
        if (self.interval > 0 and (self.n + 1) % self.interval == 0) or (
                not self.ignore_last and (self.n + 1) == self.N):
            self._log_summary(ctx, prefix)

    def call_to_log(self, ctx, results, global_step=None, prefix=''):
        if ctx.mode == 'train':
            self.train_buffer.update(results, global_step=global_step)
        elif ctx.mode == 'val':
            self.val_buffer.update(results, global_step=global_step)
        else:
            raise ContextError(f"Context contains an invalid mode: {ctx.mode}")

        self._update_before_log(ctx)
        self._log_summary(ctx, prefix)

    def _update_before_log(self, ctx):
        if ctx.mode == 'train':
            if ctx.has('train_inner_step'):
                self.n, self.N = ctx.train_inner_step, ctx.steps_per_epoch
                progress = f'{self.n + 1}/{self.N}'
                self.log_head = f'Train[{progress}]@Epoch[{ctx.epoch + 1}]: '
            else:
                self.n, self.N = ctx.train_step, ctx.max_steps
                progress = f'{self.n + 1}/{self.N}'
                self.log_head = f'Train[{progress}]: '
        elif ctx.mode == 'val':
            self.log_head = 'Val@'
            if ctx.has('epoch'):
                self.log_head += f'Epoch[{ctx.epoch + 1}]'
            if ctx.has('train_step'):
                self.log_head += f'Step[{ctx.train_step + 1}]'
            self.log_head = self.log_head.rstrip('@') + ' '
        else:
            raise ContextError(f"Context contains an invalid mode: {ctx.mode}")

    @staticmethod
    def _log_kv(k, v):
        if k.endswith('loss'):
            return f'{k}: {round_float(v, ndigits=7):.7f}'
        elif k.endswith('dist'):
            return f'{k}: {round_float(v, ndigits=3):.3f}'
        else:
            return f'{k}: {round_float(v, ndigits=5):.5f}'

    def _log_summary(self, ctx, prefix=''):
        summary = OrderedDict()
        summary['head'] = self.log_head
        log_str = prefix + self.log_head

        if ctx.mode == 'train':
            lr = [param_group['lr'] for param_group in ctx.optimizer.param_groups]
            log_str += 'lr: ' + list2str(lr, reduce=True, tmpl='{:.3e}') + ', '
            summary['lr'] = lr

            momentum = [get_momentum(param_group) for param_group in ctx.optimizer.param_groups]
            log_str += 'momentum: ' + list2str(momentum, reduce=True, tmpl='{:.3e}') + ', '
            summary['momentum'] = momentum

        if ctx.mode == 'train':
            results = self.train_buffer.average(latest=self.interval)
        elif ctx.mode == 'val':
            results = self.val_buffer.average(latest=self.interval)
        else:
            raise ContextError(f"Context contains an invalid mode: {ctx.mode}")

        if 'step_time' in results:
            data_time = results.pop('data_time')
            model_time = results.pop('model_time')
            step_time = results.pop('step_time')
            eta = (self.N - self.n) * step_time
            eta_str = str(datetime.timedelta(seconds=int(eta)))
            alloc_mem, total_mem = get_gpu_memory(ctx.device)
            log_str += f'time: {step_time:.3f}({data_time:.3f},{model_time:.3f}), eta: {eta_str}, memory: {alloc_mem}/{total_mem}, '

        log_str += ', '.join(self._log_kv(k, v) for k, v in results.items())
        ctx.logger.info(log_str)

        summary.update(results)
        with open(self.summary_path, 'a+') as fout:
            json.dump(round_float(summary), fout)
            fout.write('\n')

    @property
    def train_summary(self):
        return self.train_buffer.average(latest=self.interval)

    @property
    def val_summary(self):
        return self.val_buffer.average(latest=self.interval)
