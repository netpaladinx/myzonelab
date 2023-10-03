import os
import os.path as osp
import shutil

from ..utils import save_checkpoint
from .base_plugin import BasePlugin


class Saver(BasePlugin):
    def __init__(self,
                 interval=-1,
                 by_epoch=True,
                 save_optimizer=False,
                 max_keep_ckpts=-1,
                 save_at_last=True,
                 save_at_val=False):
        self.interval = interval
        self.by_epoch = by_epoch
        self.save_optimizer = save_optimizer
        self.max_keep_ckpts = max_keep_ckpts
        self.save_at_last = save_at_last
        self.save_at_val = save_at_val

    def call_before_run(self, ctx):
        if self.save_at_val and ctx.has('validator'):
            self.interval = ctx.validator.interval
            self.by_epoch = ctx.validator.by_epoch
            self.save_at_last = ctx.validator.save_at_last

    def call_after_step(self, ctx):
        assert ctx.mode == 'train', f"Saver's `call_after_step` should be called in the train mode"
        if self.by_epoch:
            return

        train_step = ctx.train_step
        if (self.interval > 0 and (train_step + 1) % self.interval == 0) or (
                self.save_at_last and (train_step + 1) == ctx.max_steps):
            ctx.logger.info(f"Save checkpoint after {train_step + 1} steps")
            self._save_checkpoint(train_step, ctx)

    def call_after_epoch(self, ctx):
        assert ctx.mode == 'train', f"Saver's `call_after_epoch` should be called in the train mode"
        if not self.by_epoch:
            return

        epoch = ctx.epoch
        if (self.interval > 0 and (epoch + 1) % self.interval == 0) or (
                self.save_at_last and (epoch + 1) == ctx.max_epochs):
            ctx.logger.info(f"Save checkpoint after {epoch + 1} epochs")
            self._save_checkpoint(epoch, ctx)

    def _save_checkpoint(self, t, ctx):
        save_dir = osp.join(ctx.work_dir, 'save')
        ckpt_file = f'epoch_{t + 1}.pth' if self.by_epoch else f'step_{t + 1}.pth'
        ckpt_path = osp.join(save_dir, ckpt_file)
        latest_path = osp.join(save_dir, 'latest.pth')

        model = ctx.model
        optimizer = ctx.optimizer if self.save_optimizer else None
        meta = {'epoch' if self.by_epoch else 'train_step': t + 1}

        save_checkpoint(model, ckpt_path, optimizer=optimizer, meta=meta)
        shutil.copy(ckpt_path, latest_path)

        if self.max_keep_ckpts > 0:
            saved = range(0, t, self.interval)
            if len(saved) > self.max_keep_ckpts:
                for i in saved[:-self.max_keep_ckpts]:
                    path = osp.join(save_dir, f'epoch_{i + 1}.pth' if self.by_epoch else f'step_{i + 1}.pth')
                    if osp.exists(path):
                        os.remove(path)
