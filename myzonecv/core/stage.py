from collections import OrderedDict, deque

from .hook import Hook


class Stage:
    def __init__(self, name, run_name=None):
        self.name = name
        self.run_name = run_name
        self.hooks = OrderedDict()

    def __call__(self, ctx):
        waiting = deque(self.hooks.keys())
        finished = []
        while waiting:
            name = waiting.popleft()
            if self.hooks[name].ready(finished):
                ret = self.hooks[name](ctx)
                if ret and isinstance(ret, dict):
                    ctx.update(ret)

                finished.append(name)
            else:
                waiting.append(name)

    def add_hook(self, name, func, dependency=None, force=False):
        if not force and name in self.hooks:
            raise KeyError(f'hook {name} is already registered in stage {self.name}')

        self.hooks[name] = Hook(name, func, dependency,
                                stage_name=self.name, run_name=self.run_name)


class IterStage(Stage):
    def __call__(self, ctx):
        waiting = deque(self.hooks.keys())
        finished = []
        while waiting:
            name = waiting.popleft()
            if self.hooks[name].ready(finished):
                for ret in self.hooks[name](ctx):
                    if ret and isinstance(ret, dict):
                        ctx.update(ret)
                    yield ret

                finished.append(name)
            else:
                waiting.append(name)
