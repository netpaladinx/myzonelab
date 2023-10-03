from torch.nn.utils import clip_grad
from torch.cuda.amp import GradScaler  # PyTorch >= 1.6.0

from ..registry import OPTIMIZE_SCHEDULERS
from .utils import get_train_step


@OPTIMIZE_SCHEDULERS.register_class('optim')
class OptimScheduler:
    def __init__(self, grad_clip=None):
        self.grad_clip = grad_clip

    @staticmethod
    def clip_grads(optimizer, grad_clip):
        param_groups = optimizer.param_groups
        params = []
        for param_group in param_groups:
            for param in param_group['params']:
                if param.requires_grad and param.grad is not None:
                    params.append(param)
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **grad_clip)

    def call_before_run(self, ctx):
        pass

    @staticmethod
    def _stagename(callstage):
        if callstage[0] + 1 == callstage[1]:
            return 'end'
        elif callstage[0] == 0:
            return 'begin'
        elif callstage[0] > 0 and callstage[0] + 1 < callstage[1]:
            return 'middle'
        else:
            raise ValueError(f"Invalid callstage: {callstage}")

    def _call_at_step_beg(self, ctx):
        ctx.optimizer.zero_grad()
        ctx.loss.backward()

    def _call_at_step_mid(self, ctx):
        ctx.loss.backward()

    def _call_at_step_end(self, ctx):
        ctx.loss.backward()
        grad_norm = self.clip_grads(ctx.optimizer, self.grad_clip) if self.grad_clip else None
        ctx.optimizer.step()
        if grad_norm is not None:
            ctx.optim_info.update({'grad_norm': grad_norm})

    def call_at_step(self, ctx, callstage=None):
        if callstage is None:
            ctx.optimizer.zero_grad()
            ctx.loss.backward()
            grad_norm = self.clip_grads(ctx.optimizer, self.grad_clip) if self.grad_clip else None
            ctx.optimizer.step()
            if grad_norm is not None:
                ctx.optim_info.update({'grad_norm': grad_norm})
        elif self._stagename(callstage) == 'begin':
            self._call_at_step_beg(ctx)
        elif self._stagename(callstage) == 'middle':
            self._call_at_step_mid(ctx)
        elif self._stagename(callstage) == 'end':
            self._call_at_step_end(ctx)
        else:
            raise ValueError(f"Invalid callstage: {callstage}")


@OPTIMIZE_SCHEDULERS.register_class('grad_cumulative_optim')
class GradCumulativeOptimScheduler(OptimScheduler):
    def __init__(self, cumulative_steps=1, **kwargs):
        self.cumulative_steps = cumulative_steps
        super().__init__(**kwargs)

        self.max_steps = 0
        self.divisible_steps = 0
        self.remainder_steps = 0

    def call_before_run(self, ctx):
        self.max_steps = ctx.max_steps
        self.divisible_steps = self.max_steps // self.cumulative_steps * self.cumulative_steps
        self.remainder_steps = self.max_steps - self.divisible_steps

    def _call_at_step_beg(self, ctx):
        train_step = get_train_step(ctx)
        if train_step == 0:
            ctx.optimizer.zero_grad()

        if train_step < self.divisible_steps:
            ctx.loss = ctx.loss / self.cumulative_steps
        else:
            ctx.loss = ctx.loss / self.remainder_steps
        ctx.loss.backward()

    def _call_at_step_mid(self, ctx):
        train_step = get_train_step(ctx)
        if train_step < self.divisible_steps:
            ctx.loss = ctx.loss / self.cumulative_steps
        else:
            ctx.loss = ctx.loss / self.remainder_steps
        ctx.loss.backward()

    def _call_at_step_end(self, ctx):
        train_step = get_train_step(ctx)
        if train_step < self.divisible_steps:
            ctx.loss = ctx.loss / self.cumulative_steps
        else:
            ctx.loss = ctx.loss / self.remainder_steps
        ctx.loss.backward()

        if (train_step + 1) % self.cumulative_steps == 0 or (train_step + 1) == self.max_steps:
            grad_norm = self.clip_grads(ctx.optimizer, self.grad_clip) if self.grad_clip else None
            ctx.optimizer.step()
            ctx.optimizer.zero_grad()
            if grad_norm is not None:
                ctx.optim_info.update({'grad_norm': grad_norm})

    def call_at_step(self, ctx, callstage=None):
        if callstage is None:
            train_step = get_train_step(ctx)
            if train_step == 0:
                ctx.optimizer.zero_grad()

            if train_step < self.divisible_steps:
                ctx.loss = ctx.loss / self.cumulative_steps
            else:
                ctx.loss = ctx.loss / self.remainder_steps
            ctx.loss.backward()

            if (train_step + 1) % self.cumulative_steps == 0 or (train_step + 1) == self.max_steps:
                grad_norm = self.clip_grads(ctx.optimizer, self.grad_clip) if self.grad_clip else None
                ctx.optimizer.step()
                ctx.optimizer.zero_grad()
                if grad_norm is not None:
                    ctx.optim_info.update({'grad_norm': grad_norm})
        elif self._stagename(callstage) == 'begin':
            self._call_at_step_beg(ctx)
        elif self._stagename(callstage) == 'middle':
            self._call_at_step_mid(ctx)
        elif self._stagename(callstage) == 'end':
            self._call_at_step_end(ctx)
        else:
            raise ValueError(f"Invalid callstage: {callstage}")


@OPTIMIZE_SCHEDULERS.register_class('fp16_optim')
class Fp16OptimScheduler(OptimScheduler):
    def __init__(self,
                 coalesce=True,
                 bucket_size_mb=-1,
                 loss_scale=512.,  # 'dynamic', float, or dict
                 **kwargs):
        self.bucket_size_mb = bucket_size_mb
        self.coalesce = coalesce

        assert loss_scale == 'dynamic' or isinstance(loss_scale, (float, dict))
        if loss_scale == 'dynamic':
            self.loss_scale = None
            self.loss_scaler = GradScaler()
        elif isinstance(loss_scale, float):
            self.loss_scale = loss_scale
            self.loss_scaler = GradScaler(init_scale=loss_scale)
        elif isinstance(loss_scale, dict):
            self.loss_scale = None
            self.loss_scaler = GradScaler(**loss_scale)

        super().__init__(**kwargs)

    def call_before_run(self, ctx):
        if hasattr(ctx.model, 'fp16_enabled'):
            ctx.model.fp16_enabled = True

    def _call_at_step_beg(self, ctx):
        ctx.model.zero_grad()
        ctx.optimizer.zero_grad()
        self.loss_scale.scale(ctx.loss).backward()

    def _call_at_step_mid(self, ctx):
        self.loss_scale.scale(ctx.loss).backward()

    def _call_at_step_end(self, ctx):
        self.loss_scale.scale(ctx.loss).backward()
        self.loss_scale.unscale_(ctx.optimizer)
        grad_norm = self.clip_grads(ctx.optimizer, self.grad_clip) if self.grad_clip else None
        self.loss_scale.step(ctx.optimizer)
        self.loss_scale.update(self.loss_scale)
        if grad_norm is not None:
            ctx.optim_info.update({'grad_norm': grad_norm})

    def call_at_step(self, ctx, callstage=None):
        if callstage is None:
            ctx.model.zero_grad()
            ctx.optimizer.zero_grad()
            self.loss_scale.scale(ctx.loss).backward()
            self.loss_scale.unscale_(ctx.optimizer)
            grad_norm = self.clip_grads(ctx.optimizer, self.grad_clip) if self.grad_clip else None
            self.loss_scale.step(ctx.optimizer)
            self.loss_scale.update(self.loss_scale)
            if grad_norm is not None:
                ctx.optim_info.update({'grad_norm': grad_norm})
        elif self._stagename(callstage) == 'begin':
            self._call_at_step_beg(ctx)
        elif self._stagename(callstage) == 'middle':
            self._call_at_step_mid(ctx)
        elif self._stagename(callstage) == 'end':
            self._call_at_step_end(ctx)
        else:
            raise ValueError(f"Invalid callstage: {callstage}")

    @staticmethod
    def copy_grads_to_fp32(fp16_net, fp32_weights):
        for fp16_param, fp32_param in zip(fp16_net.parameters(), fp32_weights):
            if fp16_param.grad is not None:
                if fp32_param.grad is None:
                    fp32_param.grad = fp32_param.data.new(fp32_param.size())
                fp32_param.grad.copy_(fp16_param.grad)

    @staticmethod
    def copy_params_to_fp16(fp32_weights, fp16_net):
        for fp32_param, fp16_param in zip(fp32_weights, fp16_net.parameters()):
            fp16_param.data.copy_(fp32_param.data)


@OPTIMIZE_SCHEDULERS.register_class('grad_cumulative_fp16_optim')
class GradCumulativeFp16OptimScheduler(Fp16OptimScheduler):
    def __init__(self, cumulative_steps=1, **kwargs):
        self.cumulative_steps = cumulative_steps
        super().__init__(**kwargs)

        self.max_steps = 0
        self.divisible_steps = 0
        self.remainder_steps = 0

    def call_before_run(self, ctx):
        if hasattr(ctx.model, 'fp16_enabled'):
            ctx.model.fp16_enabled = True

        self.max_steps = ctx.max_steps
        self.divisible_steps = self.max_steps // self.cumulative_steps * self.cumulative_steps
        self.remainder_steps = self.max_steps - self.divisible_steps

    def _call_at_step_beg(self, ctx):
        train_step = get_train_step(ctx)
        if train_step == 0:
            ctx.model.zero_grad()
            ctx.optimizer.zero_grad()

        if train_step < self.divisible_steps:
            ctx.loss = ctx.loss / self.cumulative_steps
        else:
            ctx.loss = ctx.loss / self.remainder_steps
        self.loss_scale.scale(ctx.loss).backward()

    def _call_at_step_mid(self, ctx):
        train_step = get_train_step(ctx)
        if train_step < self.divisible_steps:
            ctx.loss = ctx.loss / self.cumulative_steps
        else:
            ctx.loss = ctx.loss / self.remainder_steps
        self.loss_scale.scale(ctx.loss).backward()

    def _call_at_step_end(self, ctx):
        train_step = get_train_step(ctx)
        if train_step < self.divisible_steps:
            ctx.loss = ctx.loss / self.cumulative_steps
        else:
            ctx.loss = ctx.loss / self.remainder_steps
        self.loss_scale.scale(ctx.loss).backward()

        if (train_step + 1) % self.cumulative_steps == 0 or (train_step + 1) == self.max_steps:
            self.loss_scale.unscale_(ctx.optimizer)
            grad_norm = self.clip_grads(ctx.optimizer, self.grad_clip) if self.grap_clip else None
            self.loss_scale.step(ctx.optimizer)
            self.loss_scale.update(self.loss_scale)
            ctx.model.zero_grad()
            ctx.optimizer.zero_grad()
            if grad_norm is not None:
                ctx.optim_info.update({'grad_norm': grad_norm})

    def call_at_step(self, ctx, callstage=None):
        if callstage is None:
            train_step = get_train_step(ctx)

            if train_step == 0:
                ctx.model.zero_grad()
                ctx.optimizer.zero_grad()

            if train_step < self.divisible_steps:
                ctx.loss = ctx.loss / self.cumulative_steps
            else:
                ctx.loss = ctx.loss / self.remainder_steps

            self.loss_scale.scale(ctx.loss).backward()

            if (train_step + 1) % self.cumulative_steps == 0 or (train_step + 1) == self.max_steps:
                self.loss_scale.unscale_(ctx.optimizer)
                grad_norm = self.clip_grads(ctx.optimizer, self.grad_clip) if self.grap_clip else None
                self.loss_scale.step(ctx.optimizer)
                self.loss_scale.update(self.loss_scale)
                ctx.model.zero_grad()
                ctx.optimizer.zero_grad()
                if grad_norm is not None:
                    ctx.optim_info.update({'grad_norm': grad_norm})
        elif self._stagename(callstage) == 'begin':
            self._call_at_step_beg(ctx)
        elif self._stagename(callstage) == 'middle':
            self._call_at_step_mid(ctx)
        elif self._stagename(callstage) == 'end':
            self._call_at_step_end(ctx)
        else:
            raise ValueError(f"Invalid callstage: {callstage}")
