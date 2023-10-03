import os
import os.path as osp
import shutil

from ..utils import save_checkpoint
from .base_plugin import BasePlugin


class ThreadSaver(BasePlugin):
    def __init__(self,
                 thread_name,
                 interval=-1,
                 by_epoch=True,
                 save_optimizer=False,
                 max_keep_ckpts=-1,
                 save_at_last=True):
        self.thread_name = thread_name
        self.interval = interval
        self.by_epoch = by_epoch
        self.save_optimizer = save_optimizer
        self.max_keep_ckpts = max_keep_ckpts
        self.save_at_last = save_at_last

    @staticmethod
    def get_epoch(thread_ctx, ctx):
        mode = thread_ctx.mode if thread_ctx.has('mode') else ''
        if mode == 'val':
            return ctx.epoch if ctx.has('epoch') else thread_ctx.epoch
        else:
            return thread_ctx.epoch

    @staticmethod
    def get_step(thread_ctx, ctx):
        mode = thread_ctx.mode if thread_ctx.has('mode') else ''
        if mode == 'val':
            return ctx.step if ctx.has('step') else thread_ctx.step
        else:
            return thread_ctx.step

    def call_before_run(self, thread_ctx, ctx):
        pass

    def call_after_step(self, thread_ctx, ctx):
        if self.by_epoch:
            return

        step = self.get_step(thread_ctx, ctx)
        if (self.interval > 0 and (step + 1) % self.interval == 0) or (
                self.save_at_last and thread_ctx.eq('max_steps', step + 1)):
            ctx.logger.info(f"[Thread {self.thread_name}] Save checkpoint after {step + 1} steps")
            self._save_checkpoint(step, thread_ctx, ctx)

    def call_after_epoch(self, thread_ctx, ctx):
        if not self.by_epoch:
            return

        epoch = self.get_epoch(thread_ctx, ctx)
        if (self.interval > 0 and (epoch + 1) % self.interval == 0) or (
                self.save_at_last and thread_ctx.eq('max_epochs', epoch + 1)):
            ctx.logger.info(f"[Thread {self.thread_name}] Save checkpoint after {epoch + 1} epochs")
            self._save_checkpoint(epoch, thread_ctx, ctx)

    def _save_checkpoint(self, step, thread_ctx, ctx):
        save_dir = osp.join(ctx.work_dir, 'save')
        ckpt_file = f'epoch_{step + 1}.pth' if self.by_epoch else f'step_{step + 1}.pth'
        ckpt_path = osp.join(save_dir, ckpt_file)
        latest_path = osp.join(save_dir, 'latest.pth')

        model = ctx.model
        optimizer = thread_ctx.optimizer if self.save_optimizer else None
        step_key = 'epoch' if self.by_epoch else 'step'
        meta = {step_key: step + 1}

        save_checkpoint(model, ckpt_path, optimizer=optimizer, meta=meta)
        shutil.copy(ckpt_path, latest_path)

        if self.max_keep_ckpts > 0:
            saved = range(0, step, self.interval)
            if len(saved) > self.max_keep_ckpts:
                for t in saved[:-self.max_keep_ckpts]:
                    path = osp.join(save_dir, f'epoch_{t}.pth' if self.by_epoch else f'step_{t}.pth')
                    if osp.exists(path):
                        os.remove(path)
