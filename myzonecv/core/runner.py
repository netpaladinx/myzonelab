from abc import ABCMeta

from .registry import RUNS


class Runner(metaclass=ABCMeta):
    _run = None

    @classmethod
    def get_run(cls, run_cfg=None):
        if not cls._run:
            if not run_cfg:
                assert hasattr(cls, 'run_cfg'), f"{cls.__name__}'s attribute 'run_cfg' is not defined"
                run_cfg = cls.run_cfg
            cls._run = RUNS.create(run_cfg)
        return cls._run

    @classmethod
    def get_stage(cls, name):
        return cls.get_run().stages.get(name)

    @classmethod
    def register_hook(cls, stage_name, name=None, func=None, dependency=None, force=False):
        stage = cls.get_stage(stage_name)
        assert stage is not None, f"stage {stage_name} is not defined in {cls.__name__}"

        if callable(name):
            func = name
            name = func.__name__

        if func is not None:
            cls._register_hook(stage, name, func, dependency, force)
            return func

        def _register(func):
            cls._register_hook(stage, name, func, dependency, force)
            return func

        return _register

    @classmethod
    def _register_hook(cls, stage, name, func, dependency, force):
        if name is None:
            name = func.__name__

        stage.add_hook(name, func, dependency, force)

    def run(self, ctx):
        self.get_run()(ctx)
        return ctx
