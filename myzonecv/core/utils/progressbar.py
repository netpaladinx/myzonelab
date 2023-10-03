import sys
from shutil import get_terminal_size

from .timer import Timer


class ProgressBar:
    def __init__(self, N, bar_width=50, start=False, file=sys.stdout, unit='item'):
        self.N = N
        self.bar_width = bar_width
        self.file = file
        self.unit = unit
        self.n = 0
        self.timer = Timer()

        if start:
            self.start()

    def start(self):
        if self.N > 0:
            self.file.write(f'[{" " * self.bar_width}] 0/{self.N}, elapsed: 0s, ETA:')
        else:
            self.file.write(f'completed: 0, elapsed: 0s')

        self.n = 0
        self.file.flush()
        self.timer.restart()

    def update(self, n=1):
        assert n > 0
        self.n += n

        elapsed = self.timer.since_start()
        if elapsed > 0:
            fps = self.n / elapsed
        else:
            fps = float('inf')

        if self.N > 0:
            pct = self.n / self.N
            eta = int(elapsed * (1 - pct) / pct + 0.5)
            msg = f'\r[{{}}] {self.n}/{self.N}, {fps:.1f} {self.unit}/sec, elapsed: {int(elapsed + 0.5)}s, ETA: {eta: 5}s'

            terminal_width = get_terminal_size()[0]
            bar_width = min(self.bar_width, int(terminal_width - len(msg)) + 2, int(terminal_width * 0.6))
            bar_width = max(2, bar_width)
            mark_width = int(bar_width * pct)
            bar_chars = '>' * mark_width + ' ' * (bar_width - mark_width)
            self.file.write(msg.format(bar_chars))
        else:
            self.file.write(f'completed: {self.n}, elapsed: {int(elapsed + 0.5)}s, {fps:.1f} {self.unit}/sec')

        self.file.flush()

    def end(self):
        self.file.write('\n')
        self.file.flush()


def get_progressbar(unit, ctx):
    if unit == 'item':
        assert ctx.has('data_size') and isinstance(ctx.data_size, int) and ctx.data_size > 0
        N = ctx.data_size
    elif unit == 'step':
        if ctx.is_not('steps_in_epoch', None):
            N = ctx.steps_in_epoch
        elif ctx.gt('max_inner_steps', 0):
            N = ctx.max_inner_steps
        else:
            raise ValueError("Can not find either max_inner_steps or steps_in_epoch")
    else:
        raise ValueError(f"Invalid unit {unit} for inner progressbar")
    return ProgressBar(N, unit=unit)


def update_progressbar(progressbar, ctx):
    if progressbar.unit == 'item':
        assert ctx.has('batch_size')
        progressbar.update(n=ctx.batch_size)
    elif progressbar.unit == 'step':
        progressbar.update(n=1)
