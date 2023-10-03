import os.path as osp

from ..registry import RUNNERS
from ..utils import save_checkpoint, dump_json, mkdir
from .base_plugin import BasePlugin


class Validator(BasePlugin):
    metrics_greater_better = ['acc', 'AP']
    metrics_less_better = ['loss']

    def __init__(self,
                 interval=1,
                 by_epoch=True,
                 save_best_by='AP',
                 val_at_start=False,
                 val_at_last=True,
                 save_prefix=''):
        self.interval = interval
        self.by_epoch = by_epoch
        self.save_best_by = save_best_by
        self.val_at_start = val_at_start
        self.val_at_last = val_at_last
        self.best_score = None
        self.save_prefix = save_prefix

    def call_before_run(self, ctx):
        if self.val_at_start:
            ctx.logger.info(f"Validate at start")
            self._validate(ctx, is_start=True)

    def call_after_step(self, ctx):
        assert ctx.mode == 'train', f"Validator's `call_after_step` should be called in the train mode"
        if self.by_epoch:
            return

        train_step = ctx.train_step
        if (self.interval > 0 and (train_step + 1) % self.interval == 0) or (
                self.val_at_last and (train_step + 1) == ctx.max_steps):
            ctx.logger.info(f"Validate after {train_step + 1} steps")

            if (train_step + 1) == ctx.max_steps:
                ctx.logger.info(f"This is the last validation")
                ctx.plot_dir = osp.join(ctx.work_dir, 'plot')
                mkdir(ctx.plot_dir, exist_ok=True)

            self._validate(ctx, train_step=train_step + 1)

    def call_after_epoch(self, ctx):
        assert ctx.mode == 'train', f"Validator's `call_after_epoch` should be called in the train mode"
        if not self.by_epoch:
            return

        epoch = ctx.epoch
        if (self.interval > 0 and (epoch + 1) % self.interval == 0) or (
                self.val_at_last and (epoch + 1) == ctx.max_epochs):
            ctx.logger.info(f"Validate after {epoch + 1} epochs")

            if (epoch + 1) == ctx.max_epochs:
                ctx.logger.info(f"This is the last validation")
                ctx.plot_dir = osp.join(ctx.work_dir, 'plot')
                mkdir(ctx.plot_dir, exist_ok=True)

            self._validate(ctx, epoch=epoch + 1)

    def _validate(self, ctx, is_start=False, **kwargs):
        runner = RUNNERS.create({'type': 'val'})
        runner.run(ctx)

        if self.save_best_by and ctx.has('summarizer'):
            val_score = ctx.summarizer.val_summary.get(self.save_best_by)
            if val_score is not None:
                if self.best_score is None or (
                        self.save_best_by in self.metrics_greater_better and val_score > self.best_score) or (
                        self.save_best_by in self.metrics_less_better and val_score < self.best_score):
                    self.best_score = val_score

                    best_ckpt_path = osp.join(ctx.work_dir, 'save', f'{self.save_prefix}best.pth')
                    best_log_path = osp.join(ctx.work_dir, 'save', f'{self.save_prefix}best.json')
                    model = ctx.model
                    meta = {'score': self.best_score, 'metric': self.save_best_by, **kwargs}

                    save_checkpoint(model, best_ckpt_path, meta=meta)
                    dump_json(meta, best_log_path)

                    if is_start:
                        ctx.logger.info(f'Best checkpoint is saved with {self.save_best_by} {self.best_score:.4f} at start')
                    else:
                        ctx.logger.info(f'Best checkpoint is saved with {self.save_best_by} {self.best_score:.4f} '
                                        f'at epoch {ctx.epoch+1} step {ctx.train_step+1}')
