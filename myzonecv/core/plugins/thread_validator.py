import os.path as osp

from ..utils import save_checkpoint, dump_json
from .base_plugin import BasePlugin


class ThreadValidator(BasePlugin):
    metrics_greater_better = ['acc', 'AP']
    metrics_less_better = ['loss']

    def __init__(self,
                 thread_name,
                 save_best_by=None,
                 save_prefix='',
                 metric_greater_better=None,
                 evaluate_begin_kwargs=None,
                 evaluate_step_kwargs=None,
                 evaluate_all_kwargs=None):
        self.thread_name = thread_name
        self.save_best_by = save_best_by
        self.save_prefix = save_prefix
        self.evaluate_begin_kwargs = evaluate_begin_kwargs or {}
        self.evaluate_step_kwargs = evaluate_step_kwargs or {}
        self.evaluate_all_kwargs = evaluate_all_kwargs or {}

        self.best_score = None
        self.all_results = None

        if save_best_by:
            if metric_greater_better is True:
                self.metrics_greater_better.append(save_best_by)
            elif metric_greater_better is False:
                self.metrics_less_better.append(save_best_by)

    def call_before_run(self, thread_ctx, ctx):
        pass

    def call_before_epoch(self, thread_ctx, ctx):
        thread_ctx.dataset.evaluate_begin(ctx=ctx, **self.evaluate_begin_kwargs)
        self.all_results = []

    def call_after_step(self, thread_ctx, ctx):
        thread_ctx.dataset.evaluate_step(thread_ctx.results, thread_ctx.batch, ctx=ctx, **self.evaluate_step_kwargs)
        intermediate_keys = thread_ctx.results.get('intermediate_keys', ())
        for key in intermediate_keys:
            thread_ctx.results.pop(key)
        self.all_results.append(thread_ctx.results)

    def call_after_epoch(self, thread_ctx, ctx):
        eval_results = thread_ctx.dataset.evaluate_all(self.all_results, ctx=ctx, **self.evaluate_all_kwargs)
        thread_ctx.summarizer.call_to_log(thread_ctx, ctx, eval_results, global_step=self.get_global_step(ctx)[0])
        if self.save_best_by:
            self._save_best(thread_ctx, ctx)

    @staticmethod
    def get_global_step(ctx):  # no "+ 1" for global epoch and step
        if ctx.has('epoch', 'step'):
            return ctx.epoch, f'epoch {ctx.epoch} step {ctx.step}'
        elif ctx.has('epoch'):
            return ctx.epoch, f'epoch {ctx.epoch}'
        elif ctx.has('step'):
            return ctx.step, f'step {ctx.step}'
        return None, ''

    def _save_best(self, thread_ctx, ctx):
        if self.save_best_by:
            score = thread_ctx.summarizer.summary.get(self.save_best_by)
            if score is not None:
                if self.best_score is None or (
                        self.save_best_by in self.metrics_greater_better and score > self.best_score) or (
                        self.save_best_by in self.metrics_less_better and score < self.best_score):
                    self.best_score = score

                    best_ckpt_path = osp.join(ctx.work_dir, 'save', f'{self.save_prefix}best.pth')
                    best_log_path = osp.join(ctx.work_dir, 'save', f'{self.save_prefix}best.json')
                    model = ctx.model
                    meta = {'score': self.best_score, 'metric': self.save_best_by}
                    if ctx.has('epoch'):
                        meta['epoch'] = ctx.epoch
                    if ctx.has('step'):
                        meta['step'] = ctx.step

                    save_checkpoint(model, best_ckpt_path, meta=meta)
                    dump_json(meta, best_log_path)

                    global_step_str = self.get_global_step(ctx)[1]
                    if global_step_str:
                        global_step_str = f'at {global_step_str}'
                    ctx.logger.info(f"[Thread {self.thread_name}] Best checkpoint is saved with "
                                    f"{self.save_best_by} {self.best_score:.4f} {global_step_str}")
