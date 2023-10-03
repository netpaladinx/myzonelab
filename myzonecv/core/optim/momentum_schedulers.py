from abc import ABCMeta, abstractmethod

from ..registry import MOMENTUM_SCHEDULERS
from .utils import cosine_anneal, linear_anneal, get_max_epochs_and_max_steps, get_train_step


class BaseMomentumScheduler(metaclass=ABCMeta):
    def __init__(self,
                 by_epoch=True,
                 warmup=None,
                 warmup_ratio=0.5,
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

        self.initial_momentum = []  # for different param groups
        self.regular_momentum = []
        self.warmup_momentum = []

    @abstractmethod
    def compute_momentum(self, momentum0, epoch=None, train_step=None):
        pass

    def _update_regular_momentum(self, epoch=None, train_step=None):
        self.regular_momentum = [self.compute_momentum(mom0, epoch, train_step) for mom0 in self.initial_momentum]

    def _update_warmup_momentum(self, train_step):
        warmup_ratio = self.warmup_ratio
        if isinstance(warmup_ratio, (list, tuple)):
            assert warmup_ratio == len(self.regular_momentum)
        else:
            warmup_ratio = [warmup_ratio] * len(self.regular_momentum)

        if self.warmup == 'constant':
            self.warmup_momentum = [mom1 * wr for wr, mom1 in zip(warmup_ratio, self.regular_momentum)]
        elif self.warmup == 'linear':
            self.warmup_momentum = [mom1 * (1 - (1 - train_step / self.warmup_steps) * (1 - wr))
                                    for wr, mom1 in zip(warmup_ratio, self.regular_momentum)]
        elif self.warmup == 'exp':
            self.warmup_momentum = [mom1 * wr**(1 - train_step / self.warmup_steps)
                                    for wr, mom1 in zip(warmup_ratio, self.regular_momentum)]

    @staticmethod
    def update_momentum(optimizer, momentums):
        param_groups = optimizer.param_groups
        for param_group, mom in zip(param_groups, momentums):
            if 'momentum' in param_group:
                param_group['momentum'] = mom
            elif 'betas' in param_group:
                param_group['betas'] = (mom, param_group['betas'][1])

    def call_before_run(self, ctx):
        param_groups = ctx.optimizer.param_groups
        for param_group in param_groups:
            if 'momentum' in param_group:
                param_group.setdefault('initial_momentum', param_group['momentum'])
            elif 'betas' in param_group:
                param_group.setdefault('initial_momentum', param_group['betas'][0])
        self.initial_momentum = [param_group['initial_momentum'] for param_group in param_groups]
        if self.warmup_steps == 0:
            self.warmup_steps = max(self.warmup_epochs * ctx.steps_per_epoch, self.min_warmup_steps)

    def call_before_epoch(self, ctx):
        if self.by_epoch:
            self._update_regular_momentum(epoch=ctx.epoch)
            self.update_momentum(ctx.optimizer, self.regular_momentum)

    def call_before_step(self, ctx):
        train_step = get_train_step(ctx)
        if self.by_epoch:
            if self.warmup:
                if train_step < self.warmup_steps:
                    self._update_warmup_momentum(train_step)
                    self.update_momentum(ctx.optimizer, self.warmup_momentum)
                elif train_step == self.warmup_steps:
                    self.update_momentum(ctx.optimizer, self.regular_momentum)
        else:
            self._update_regular_momentum(train_step=train_step)
            if self.warmup and train_step < self.warmup_steps:
                self._update_warmup_momentum(train_step)
                self.update_momentum(ctx.optimizer, self.warmup_momentum)
            else:
                self.update_momentum(ctx.optimizer, self.regular_momentum)


@MOMENTUM_SCHEDULERS.register_class('fixed_momentum')
class FixedMomentumScheduler(BaseMomentumScheduler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_momentum(self, momentum0, epoch=None, train_step=None):
        return momentum0


@MOMENTUM_SCHEDULERS.register_class('step_momentum')
class StepMomentumScheduler(BaseMomentumScheduler):
    def __init__(self, step, gamma=0.5, min_momentum=None, **kwargs):
        assert isinstance(step, int) or (
            isinstance(step, (list, tuple)) and all([isinstance(s, int) for s in step]))
        self.step = step
        self.gamma = gamma
        self.min_momentum = min_momentum
        super().__init__(**kwargs)

    def compute_momentum(self, momentum0, epoch=None, train_step=None):
        cur_step = epoch if self.by_epoch else train_step
        if isinstance(self.step, int):
            exp = cur_step // self.step
        else:
            exp = len(self.step)
            for i, s in enumerate(self.step):
                if cur_step < s:
                    exp = i
                    break
        momentum = momentum0 * (self.gamma**exp)
        if self.min_momentum is not None:
            momentum = max(momentum, self.min_momentum)
        return momentum


@MOMENTUM_SCHEDULERS.register_class('cosine_anneal_momentum')
class CosineAnnealMomentumScheduler(BaseMomentumScheduler):
    def __init__(self, min_momentum=None, min_momentum_ratio=None, **kwargs):
        assert (min_momentum is None) ^ (min_momentum_ratio is None)
        self.min_momentum = min_momentum
        self.min_momentum_ratio = min_momentum_ratio
        super().__init__(**kwargs)

    def compute_momentum(self, momentum0, epoch=None, train_step=None):
        cur_step = epoch if self.by_epoch else train_step
        max_steps = self.max_epochs if self.by_epoch else self.max_steps
        target_momentum = momentum0 * self.min_momentum_ratio if self.min_momentum is None else self.min_momentum
        momentum = cosine_anneal(momentum0, target_momentum, cur_step / max_steps)
        return momentum

    def call_before_run(self, ctx):
        super().call_before_run(ctx)
        self.max_epochs, self.max_steps = get_max_epochs_and_max_steps(ctx)


@MOMENTUM_SCHEDULERS.register_class('cyclic_momentum')
class CyclicMomentumScheduler(BaseMomentumScheduler):
    def __init__(self,
                 target_ratio=(0.85 / 0.95, 1),  # ratios of lowest momentum and highest momentum to initial momentum
                 cyclic_times=1,         # number of cycles
                 step_ratio_up=0.4,      # ratio of increasing momentum in one cycle
                 anneal_strategy='cos',  # 'cos' for cosine annealing, 'linear' for linear annealing
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

        self.momentum_phases = []

    def call_before_run(self, ctx):
        param_groups = ctx.optimizer.param_groups
        for param_group in param_groups:
            if 'momentum' in param_group:
                param_group.setdefault('initial_momentum', param_group['momentum'])
            elif 'betas' in param_group:
                param_group.setdefault('initial_momentum', param_group['betas'][0])
        self.initial_momentum = [param_group['initial_momentum'] for param_group in param_groups]

        max_steps_per_cycle = ctx.max_steps // self.cyclic_times
        up_phase_steps = int(self.step_ratio_up * max_steps_per_cycle)
        self.momentum_phases = [[0, up_phase_steps, max_steps_per_cycle, 1, self.target_ratio[0]],
                                [up_phase_steps, max_steps_per_cycle, max_steps_per_cycle, self.target_ratio[0], self.target_ratio[1]]]

    def compute_momentum(self, momentum0, epoch=None, train_step=None):
        cur_step = epoch if self.by_epoch else train_step
        for start_step, end_step, max_steps_per_cycle, start_ratio, end_ratio in self.momentum_phases:
            step = cur_step % max_steps_per_cycle
            if start_step <= step < end_step:
                t = (step - start_step) / (end_step - start_step)
                if self.anneal_strategy == 'cos':
                    momentum = cosine_anneal(momentum0 * start_ratio, momentum0 * end_ratio, t)
                else:
                    momentum = linear_anneal(momentum0 * start_ratio, momentum0 * end_ratio, t)
                break
        return momentum


@MOMENTUM_SCHEDULERS.register_class('one_cycle_momentum')
class OneCycleMomentumScheduler(BaseMomentumScheduler):
    def __init__(self,
                 min_momentum=0.85,  # momentum at the peak of a cycle (cycled inversely to lr)
                 max_momentum=0.95,  # momentum at the start of a cycle
                 total_steps=None,
                 pct_start=0.3,
                 anneal_strategy='cos',
                 three_phase=False,
                 **kwargs):
        assert anneal_strategy in ('cos', 'linear')
        assert 0 <= pct_start < 1

        self.min_momentum = min_momentum
        self.max_momentum = max_momentum
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.three_phase = three_phase

        kwargs['by_epoch'] = False
        super().__init__(**kwargs)

        self.momentum_phases = []

    def call_before_run(self, ctx):
        max_steps = ctx.max_steps
        if not self.total_steps:
            self.total_steps = max_steps
        assert self.total_steps >= max_steps

        param_groups = ctx.optimizer.param_groups
        for param_group in param_groups:
            param_group.setdefault('initial_momentum', self.max_momentum)
        self.initial_momentum = [param_group['initial_momentum'] for param_group in param_groups]

        if self.three_phase:
            self.momentum_phases = [[self.pct_start * self.total_steps - 1, 1, self.min_momentum / self.max_momentum],
                                    [(self.pct_start * self.total_steps - 1) * 2, self.min_momentum / self.max_momentum, 1],
                                    [self.total_steps - 1, 1, 1]]
        else:
            self.momentum_phases = [[self.pct_start * self.total_steps - 1, 1, self.min_momentum / self.max_momentum],
                                    [self.total_steps - 1, self.min_momentum / self.max_momentum, 1]]

    def compute_momentum(self, momentum0, epoch=None, train_step=None):
        cur_step = epoch if self.by_epoch else train_step
        start_step = 0
        for end_step, start_ratio, end_ratio in self.momentum_phases:
            if cur_step <= end_step:
                t = (cur_step - start_step) / (end_step - start_step)
                if self.anneal_strategy == 'cos':
                    momentum = cosine_anneal(momentum0 * start_ratio, momentum0 * end_ratio, t)
                else:
                    momentum = linear_anneal(momentum0 * start_ratio, momentum0 * end_ratio, t)
                break
            start_step = end_step
        return momentum
