from abc import ABCMeta, abstractmethod

from ..registry import LR_SCHEDULERS
from .utils import cosine_anneal, linear_anneal, get_max_epochs_and_max_steps, get_train_step


class BaseLRScheduler(metaclass=ABCMeta):
    def __init__(self,
                 by_epoch=True,
                 warmup=None,
                 warmup_ratio=0.1,
                 warmup_steps=0,
                 warmup_epochs=0,
                 min_warmup_steps=1000):
        if warmup:
            assert warmup in ('constant', 'linear', 'exp')
            assert warmup_steps > 0 or warmup_epochs > 0
            if isinstance(warmup_ratio, (list, tuple)):
                assert all([wr >= 0 for wr in warmup_ratio])
            else:
                assert warmup_ratio >= 0

        self.by_epoch = by_epoch
        self.warmup = warmup
        self.warmup_ratio = warmup_ratio
        self.warmup_steps = warmup_steps
        self.warmup_epochs = warmup_epochs
        self.min_warmup_steps = min_warmup_steps

        self.initial_lr = []  # for different param groups
        self.regular_lr = []
        self.warmup_lr = []

    @abstractmethod
    def compute_lr(self, lr0, epoch=None, train_step=None):
        pass

    def _update_regular_lr(self, epoch=None, train_step=None):
        self.regular_lr = [self.compute_lr(lr0, epoch, train_step) for lr0 in self.initial_lr]

    def _update_warmup_lr(self, train_step):
        warmup_ratio = self.warmup_ratio
        if isinstance(warmup_ratio, (list, tuple)):
            assert len(warmup_ratio) == len(self.regular_lr)
        else:
            warmup_ratio = [warmup_ratio] * len(self.regular_lr)

        if self.warmup == 'constant':
            self.warmup_lr = [lr1 * wr for wr, lr1 in zip(warmup_ratio, self.regular_lr)]
        elif self.warmup == 'linear':
            self.warmup_lr = [lr1 * (1 - (1 - train_step / self.warmup_steps) * (1 - wr))
                              for wr, lr1 in zip(warmup_ratio, self.regular_lr)]
        elif self.warmup == 'exp':
            self.warmup_lr = [lr1 * wr**(1 - train_step / self.warmup_steps)
                              for wr, lr1 in zip(warmup_ratio, self.regular_lr)]

    @staticmethod
    def update_lr(optimizer, lrs):
        param_groups = optimizer.param_groups
        for param_group, lr in zip(param_groups, lrs):
            param_group['lr'] = lr

    def call_before_run(self, ctx):
        param_groups = ctx.optimizer.param_groups
        for param_group in param_groups:
            param_group.setdefault('initial_lr', param_group['lr'])
        self.initial_lr = [param_group['initial_lr'] for param_group in param_groups]
        if self.warmup_steps == 0:
            self.warmup_steps = max(self.warmup_epochs * ctx.steps_per_epoch, self.min_warmup_steps)

    def call_before_epoch(self, ctx):
        if self.by_epoch:
            self._update_regular_lr(epoch=ctx.epoch)
            self.update_lr(ctx.optimizer, self.regular_lr)

    def call_before_step(self, ctx):
        train_step = get_train_step(ctx)
        if self.by_epoch:
            if self.warmup:
                if train_step < self.warmup_steps:
                    self._update_warmup_lr(train_step)
                    self.update_lr(ctx.optimizer, self.warmup_lr)
                elif train_step == self.warmup_steps:
                    self.update_lr(ctx.optimizer, self.regular_lr)
        else:
            self._update_regular_lr(train_step=train_step)
            if self.warmup and train_step < self.warmup_steps:
                self._update_warmup_lr(train_step)
                self.update_lr(ctx.optimizer, self.warmup_lr)
            else:
                self.update_lr(ctx.optimizer, self.regular_lr)


@LR_SCHEDULERS.register_class('fixed_lr')
class FixedLRScheduler(BaseLRScheduler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_lr(self, lr0, epoch=None, train_step=None):
        return lr0


@LR_SCHEDULERS.register_class('step_lr')
class StepLRScheduler(BaseLRScheduler):
    def __init__(self, step, gamma=0.1, min_lr=None, **kwargs):
        assert isinstance(step, int) or (
            isinstance(step, (list, tuple)) and all([isinstance(s, int) for s in step]))
        self.step = step
        self.gamma = gamma
        self.min_lr = min_lr
        super().__init__(**kwargs)

    def compute_lr(self, lr0, epoch=None, train_step=None):
        cur_step = epoch if self.by_epoch else train_step
        if isinstance(self.step, int):
            exp = cur_step // self.step
        else:
            exp = len(self.step)
            for i, s in enumerate(self.step):
                if cur_step < s:
                    exp = i
                    break
        lr = lr0 * (self.gamma**exp)
        if self.min_lr is not None:
            lr = max(lr, self.min_lr)
        return lr


@LR_SCHEDULERS.register_class('exp_lr')
class ExpLRScheduler(BaseLRScheduler):
    def __init__(self, gamma, **kwargs):
        self.gamma = gamma
        super().__init__(**kwargs)

    def compute_lr(self, lr0, epoch=None, train_step=None):
        cur_step = epoch if self.by_epoch else train_step
        return lr0 * (self.gamma**cur_step)


@LR_SCHEDULERS.register_class('poly_lr')
class PolyLRScheduler(BaseLRScheduler):
    def __init__(self, power=1., min_lr=0., **kwargs):
        self.power = power
        self.min_lr = min_lr
        super().__init__(**kwargs)

    def compute_lr(self, lr0, epoch=None, train_step=None):
        cur_step = epoch if self.by_epoch else train_step
        max_steps = self.max_epochs if self.by_epoch else self.max_steps
        coeff = (1 - cur_step / max_steps)**self.power
        return (lr0 - self.min_lr) * coeff + self.min_lr

    def call_before_run(self, ctx):
        super().call_before_run(ctx)
        self.max_epochs, self.max_steps = get_max_epochs_and_max_steps(ctx)


@LR_SCHEDULERS.register_class('inv_lr')
class InvLRScheduler(BaseLRScheduler):
    def __init__(self, gamma, power=1., **kwargs):
        self.gamma = gamma
        self.power = power
        super().__init__(**kwargs)

    def compute_lr(self, lr0, epoch=None, train_step=None):
        cur_step = epoch if self.by_epoch else train_step
        return lr0 * (1 + self.gamma * cur_step)**(-self.power)


@LR_SCHEDULERS.register_class('cosine_anneal_lr')
class CosineAnnealLRScheduler(BaseLRScheduler):
    # https://arxiv.org/pdf/1812.01187.pdf
    def __init__(self, min_lr=None, min_lr_ratio=None, **kwargs):
        assert (min_lr is None) ^ (min_lr_ratio is None)
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        super().__init__(**kwargs)

    def compute_lr(self, lr0, epoch=None, train_step=None):
        cur_step = epoch if self.by_epoch else train_step
        max_steps = self.max_epochs if self.by_epoch else self.max_steps
        target_lr = lr0 * self.min_lr_ratio if self.min_lr is None else self.min_lr
        lr = cosine_anneal(lr0, target_lr, cur_step / max_steps)
        return lr

    def call_before_run(self, ctx):
        super().call_before_run(ctx)
        self.max_epochs, self.max_steps = get_max_epochs_and_max_steps(ctx)


@LR_SCHEDULERS.register_class('flat_cosine_anneal_lr')
class FlatCosineAnnealLRScheduler(BaseLRScheduler):
    def __init__(self,
                 start_percent=0.75,  # when to start annealing
                 min_lr=None,
                 min_lr_ratio=None,
                 **kwargs):
        assert 0 <= start_percent < 1
        assert (min_lr is None) ^ (min_lr_ratio is None)
        self.start_percent = start_percent
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        super().__init__(**kwargs)

    def compute_lr(self, lr0, epoch=None, train_step=None):
        cur_step = epoch if self.by_epoch else train_step
        max_steps = self.max_epochs if self.by_epoch else self.max_steps
        start_step = round(max_steps * self.start_percent)
        target_lr = lr0 * self.min_lr_ratio if self.min_lr is None else self.min_lr
        t = (cur_step - start_step) / (max_steps - start_step)
        lr = lr0 if t < 0 else cosine_anneal(lr0, target_lr, t)
        return lr

    def call_before_run(self, ctx):
        super().call_before_run(ctx)
        self.max_epochs, self.max_steps = get_max_epochs_and_max_steps(ctx)


@LR_SCHEDULERS.register_class('cosine_restart_lr')
class CosineRestartLRScheduler(BaseLRScheduler):
    def __init__(self,
                 periods,              # periods of each cycle
                 restart_weights=[1],  # restart weights at each restart step
                 min_lr=None,
                 min_lr_ratio=None,
                 **kwargs):
        assert len(periods) == len(restart_weights)
        assert (min_lr is None) ^ (min_lr_ratio is None)
        self.periods = periods
        self.cumulative_periods = [sum(periods[:i + 1]) for i in range(len(periods))]
        self.restart_weights = restart_weights
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        super().__init__(**kwargs)

    def compute_lr(self, lr0, epoch=None, train_step=None):
        cur_step = epoch if self.by_epoch else train_step
        target_lr = lr0 * self.min_lr_ratio if self.min_lr is None else self.min_lr
        for i, cumulative_period in enumerate(self.cumulative_periods):
            if cur_step < cumulative_period:
                break

        restart_weight = self.restart_weights[i]
        restart_step = 0 if i == 0 else self.cumulative_periods[i - 1]
        cur_period = self.periods[i]
        t = min((cur_step - restart_step) / cur_period, 1)
        lr = cosine_anneal(lr0, target_lr, t, restart_weight)
        return lr


@LR_SCHEDULERS.register_class('cyclic_lr')
class CyclicLRScheduler(BaseLRScheduler):
    def __init__(self,
                 target_ratio=(10, 1e-4),  # ratios of hightest lr and lowest lr to initial lr
                 cyclic_times=1,           # number of cycles
                 step_ratio_up=0.4,        # ratio of increasing lr in one cycle
                 anneal_strategy='cos',    # 'cos' for cosine annealing, 'linear' for linear annealing
                 **kwargs):
        if isinstance(target_ratio, float):
            target_ratio = (target_ratio, target_ratio / 1e5)

        assert isinstance(target_ratio, (tuple, list)) and len(target_ratio) == 2
        assert 0 <= step_ratio_up < 1
        assert anneal_strategy in ('cos', 'linear')

        self.target_ratio = target_ratio
        self.cyclic_times = cyclic_times
        self.step_ratio_up = step_ratio_up
        self.anneal_strategy = anneal_strategy

        kwargs['by_epoch'] = False
        super().__init__(**kwargs)

        self.lr_phases = []

    def call_before_run(self, ctx):
        param_groups = ctx.optimizer.param_groups
        for param_group in param_groups:
            param_group.setdefault('initial_lr', param_group['lr'])
        self.initial_lr = [param_group['initial_lr'] for param_group in param_groups]

        max_steps_per_cycle = ctx.max_steps // self.cyclic_times
        up_phase_steps = int(self.step_ratio_up * max_steps_per_cycle)
        self.lr_phases = [[0, up_phase_steps, max_steps_per_cycle, 1, self.target_ratio[0]],
                          [up_phase_steps, max_steps_per_cycle, max_steps_per_cycle, self.target_ratio[0], self.target_ratio[1]]]

    def compute_lr(self, lr0, epoch=None, train_step=None):
        cur_step = epoch if self.by_epoch else train_step
        for start_step, end_step, max_steps_per_cycle, start_ratio, end_ratio in self.lr_phases:
            step = cur_step % max_steps_per_cycle
            if start_step <= step < end_step:
                t = (step - start_step) / (end_step - start_step)
                if self.anneal_strategy == 'cos':
                    lr = cosine_anneal(lr0 * start_ratio, lr0 * end_ratio, t)
                else:
                    lr = linear_anneal(lr0 * start_ratio, lr0 * end_ratio, t)
                break
        return lr


@LR_SCHEDULERS.register_class('one_cycle_lr')
class OneCycleLRScheduler(BaseLRScheduler):
    def __init__(self,
                 max_lr,
                 total_steps=None,       # total steps in the cycle
                 pct_start=0.3,          # percentage of the cycle to increase lr
                 anneal_strategy='cos',  # 'cos' for cosine annealing, 'linear' for linear annealing
                 div_factor=25,          # initial_lr = max_lr / div_factor
                 final_div_factor=1e4,   # min_lr = initial_lr / final_div_factor
                 three_phase=False,
                 **kwargs):
        assert anneal_strategy in ('cos', 'linear')
        assert 0 <= pct_start < 1

        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.three_phase = three_phase

        kwargs['by_epoch'] = False
        super().__init__(**kwargs)

        self.lr_phases = []

    def call_before_run(self, ctx):
        max_steps = ctx.max_steps
        if not self.total_steps:
            self.total_steps = max_steps
        assert self.total_steps >= max_steps

        param_groups = ctx.optimizer.param_groups
        for param_group in param_groups:
            param_group.setdefault('initial_lr', self.max_lr / self.div_factor)
        self.initial_lr = [param_group['initial_lr'] for param_group in param_groups]

        if self.three_phase:
            self.lr_phases = [[self.pct_start * self.total_steps - 1, 1, self.div_factor],
                              [(self.pct_start * self.total_steps - 1) * 2, self.div_factor, 1],
                              [self.total_steps - 1, 1, 1 / self.final_div_factor]]
        else:
            self.lr_phases = [[self.pct_start * self.total_steps - 1, 1, self.div_factor],
                              [self.total_steps - 1, self.div_factor, 1 / self.final_div_factor]]

    def compute_lr(self, lr0, epoch=None, train_step=None):
        cur_step = epoch if self.by_epoch else train_step
        start_step = 0
        for end_step, start_ratio, end_ratio in self.lr_phases:
            if cur_step <= end_step:
                t = (cur_step - start_step) / (end_step - start_step)
                if self.anneal_strategy == 'cos':
                    lr = cosine_anneal(lr0 * start_ratio, lr0 * end_ratio, t)
                else:
                    lr = linear_anneal(lr0 * start_ratio, lr0 * end_ratio, t)
                break
            start_step = end_step
        return lr
