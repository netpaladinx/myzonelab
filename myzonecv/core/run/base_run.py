from collections import OrderedDict
from abc import ABCMeta, abstractmethod

from ..context import DummyContext
from ..stage import Stage, IterStage


class BaseRun(metaclass=ABCMeta):
    def __init__(self, name, stage_names=None):
        self.name = name
        self.stage_names = stage_names
        self.stages = OrderedDict()
        self(DummyContext())

    @abstractmethod
    def __call__(self, ctx):
        pass

    def call_stage(self, name, ctx):
        if self.stage_names is not None and name in self.stage_names:
            name = self.stage_names[name]

        if isinstance(ctx, DummyContext):
            self.stages[name] = Stage(name, run_name=self.name)
        else:
            self.stages[name](ctx)

    def call_iter_stage(self, name, ctx):
        if self.stage_names is not None and name in self.stage_names:
            name = self.stage_names[name]

        if isinstance(ctx, DummyContext):
            self.stages[name] = IterStage(name, run_name=self.name)
            yield 0
        else:
            for ret in self.stages[name](ctx):
                yield ret
