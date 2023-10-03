import os.path as osp
from collections import OrderedDict
import datetime
import json

from torch.utils.tensorboard import SummaryWriter

from ..utils import Timer, mkdir, round_float, list2str, get_gpu_memory, update_progressbar
from ..optim.utils import get_momentum
from .base_plugin import BasePlugin
from .summarizer import SummaryBuffer


class ThreadSummarizer(BasePlugin):
    def __init__(self,
                 thread_name,
                 interval=10,
                 ignore_last=False,
                 record_cfg=None,
                 use_tensorboard=True,
                 progressbar=None,
                 ndigits3_endswith=('dist',),
                 ndigits7_endswith=('loss',),
                 multiline_logging=False,
                 multiline_keywidth=0,
                 multiline_valwidth=0,
                 multiline_ncols=1):
        self.thread_name = thread_name
        self.interval = interval
        self.ignore_last = ignore_last
        self.record_cfg = record_cfg
        self.use_tensorboard = use_tensorboard
        self.progressbar = progressbar
        self.ndigits3_endswith = ndigits3_endswith
        self.ndigits7_endswith = ndigits7_endswith
        self.multiline_logging = multiline_logging
        self.multiline_keywidth = multiline_keywidth
        self.multiline_valwidth = multiline_valwidth
        self.multiline_ncols = multiline_ncols

        self.n = None
        self.N = None
        self.n2 = None
        self.N2 = None
        self.log_head = None
        self.buffer = None
        self.summary_path = None
        self.timer = Timer()

    def call_before_run(self, thread_ctx, ctx):
        if self.record_cfg is None:
            if thread_ctx.mode == 'train':
                self.record_cfg = {'type': 'latest_record', 'K': 100}
            else:
                self.record_cfg = {'type': 'latest_record', 'K': 1}

        self.buffer = SummaryBuffer(self.record_cfg)
        self.summary_path = osp.join(ctx.work_dir, f'{self.thread_name}_summary_{ctx.timestamp}.jsonl')

        if self.use_tensorboard and SummaryBuffer.tb_summary_writer is None:
            tb_dir = osp.join(ctx.work_dir, f'tensorboard_{ctx.timestamp}')
            mkdir(tb_dir, exist_ok=True)
            SummaryBuffer.tb_summary_writer = SummaryWriter(tb_dir)
            ctx.logger.info(f"TensorBoard: 'tensorboard --logdir {tb_dir}' http://localhost:6006/")

        self.timer.start(mark='data')

    def call_before_epoch(self, thread_ctx, ctx):
        self.buffer.clear()
        self.timer.start(mark='data')

        if self.progressbar:
            self.progressbar.start()

    def call_before_step(self, thread_ctx, ctx):
        if not self.progressbar:
            batch_size = thread_ctx.batch_size
            data_time = self.timer.since_mark('data', mark='model')
            self.buffer.update({'data_time': data_time}, cnt=batch_size)

    def call_after_step(self, thread_ctx, ctx, prefix=''):
        if not self.progressbar:
            batch_size = thread_ctx.batch_size
            results = thread_ctx.results
            global_step = thread_ctx.step
            summary_keys = results.get('summary_keys', ())
            self.buffer.update(results, cnt=batch_size, include=summary_keys, global_step=global_step)

            step_time, model_time = self.timer.since_mark(('data', 'model'), mark='data')
            self.buffer.update({'model_time': model_time, 'step_time': step_time}, cnt=batch_size)

            self._update_before_log(thread_ctx, ctx)
            if (self.interval > 0 and (self.n + 1) % self.interval == 0) or (
                    not self.ignore_last and (self.n + 1) == self.N):
                self._log_summary(thread_ctx, ctx, prefix)
        else:
            update_progressbar(self.progressbar, thread_ctx)

    def call_after_epoch(self, thread_ctx, ctx):
        if self.progressbar:
            self.progressbar.end()

    def call_to_log(self, thread_ctx, ctx, results, global_step=None, prefix=''):
        self.buffer.update(results, global_step=global_step)
        self._update_before_log(thread_ctx, ctx)
        self._log_summary(thread_ctx, ctx, prefix)

    def _update_before_log(self, thread_ctx, ctx):
        mode = thread_ctx.mode if thread_ctx.has('mode') else ''
        if mode == 'val':
            log_head = ''
            if ctx.has('epoch'):
                log_head += f'epoch[{ctx.epoch}]'  # no "+ 1" for global epoch
            if ctx.has('step'):
                log_head += f'step[{ctx.step}]'  # no "+ 1" for global step
            self.log_head = f'val@{log_head}: ' if log_head else 'val: '
        else:
            if thread_ctx.has('steps_per_epoch') and thread_ctx.steps_per_epoch:
                self.n, self.N = thread_ctx.step_in_epoch, thread_ctx.steps_per_epoch
                if thread_ctx.has('max_epochs') and thread_ctx.max_epochs:
                    self.n2, self.N2 = thread_ctx.epoch + 1, thread_ctx.max_epochs
                progress = f'{self.n + 1}/{self.N}'
                self.log_head = f'{mode}@epoch[{thread_ctx.epoch + 1}][{progress}]: '
            else:
                if thread_ctx.has('max_steps') and thread_ctx.max_steps:
                    self.n, self.N = thread_ctx.step, thread_ctx.max_steps
                    progress = f'{self.n + 1}/{self.N}'
                    self.log_head = f'{mode}@step[{progress}]: '
                else:
                    self.n = thread_ctx.step
                    self.log_head = f'{mode}@step[{thread_ctx.step + 1}]: '

    def _log_kv(self, k, v, keywidth=0, valwidth=0):
        if any(k.endswith(s) for s in self.ndigits3_endswith):
            return f"{k:<{keywidth}}: {round_float(v, ndigits=3, tmpl='{:.3f}'):<{valwidth}}"
        elif any(k.endswith(s) for s in self.ndigits7_endswith):
            return f"{k:<{keywidth}}: {round_float(v, ndigits=7, tmpl='{:.7f}'):<{valwidth}}"
        else:
            return f"{k:<{keywidth}}: {round_float(v, ndigits=5, tmpl='{:.5f}'):<{valwidth}}"

    def _log_summary(self, thread_ctx, ctx, prefix=''):
        summary = OrderedDict()
        summary['head'] = self.log_head
        log_head = prefix + self.log_head
        log_str = log_head

        if thread_ctx.has('optimizer'):
            lr = [param_group['lr'] for param_group in thread_ctx.optimizer.param_groups]
            log_str += 'lr: ' + list2str(lr, reduce=True, tmpl='{:.3e}') + ', '
            summary['lr'] = lr

            momentum = [get_momentum(param_group) for param_group in thread_ctx.optimizer.param_groups]
            log_str += 'momentum: ' + list2str(momentum, reduce=True, tmpl='{:.3e}') + ', '
            summary['momentum'] = momentum

        results = self.buffer.average(latest=self.interval)

        if 'step_time' in results:
            data_time = results.pop('data_time')
            model_time = results.pop('model_time')
            step_time = results.pop('step_time')
            log_str += f'time: {step_time:.3f}({data_time:.3f},{model_time:.3f}), '
            if self.N is not None:
                eta = (self.N - self.n) * step_time
                if self.N2 is not None:
                    eta += self.N * step_time * (self.N2 - self.n2)
                eta_str = str(datetime.timedelta(seconds=int(eta)))
                log_str += f'eta: {eta_str}, '

        alloc_mem, total_mem = get_gpu_memory(ctx.device)
        log_str += f'memory: {alloc_mem}/{total_mem}, '

        if not self.multiline_logging:
            log_str += ', '.join(self._log_kv(k, v) for k, v in results.items())
            ctx.logger.info(f"[Thread {self.thread_name}] {log_str}")
        else:
            ctx.logger.info(f"[Thread {self.thread_name}] {log_str}")
            log_kvs = []
            for i, (k, v) in enumerate(results.items()):
                log_kvs.append(self._log_kv(k, v, keywidth=self.multiline_keywidth, valwidth=self.multiline_valwidth))
                if (i + 1) % self.multiline_ncols == 0:
                    ctx.logger.info(f"[Thread {self.thread_name}] {log_head}  {', '.join(log_kvs)}")
                    log_kvs = []
            if len(log_kvs) > 0:
                ctx.logger.info(f"[Thread {self.thread_name}] {log_head}  {', '.join(log_kvs)}")

        summary.update(results)
        with open(self.summary_path, 'a+') as fout:
            json.dump(round_float(summary), fout)
            fout.write('\n')

    @property
    def summary(self):
        return self.buffer.average(latest=self.interval)
