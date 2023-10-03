import os.path as osp
from collections import defaultdict

from ..utils import mkdir, Dict, tolist
from .base_plugin import BasePlugin


class Diagnoser(BasePlugin):
    _diagnosis_methods = defaultdict(dict)

    def __init__(self,
                 at_train_epochs=(),
                 at_train_steps=(),
                 at_val_epochs=(),
                 at_val_steps=(),
                 before_run=(),
                 before_step=(),
                 after_step=(),
                 plot_inputs=True,
                 predict_train=True,
                 predict_val=True,
                 ):
        self.at_train_epochs = tolist(at_train_epochs)
        self.at_train_steps = tolist(at_train_steps)
        self.at_val_epochs = tolist(at_val_epochs)
        self.at_val_steps = tolist(at_val_steps)
        self.before_run = tolist(before_run)
        self.before_step = tolist(before_step)
        self.after_step = tolist(after_step)
        self.out_dir = None
        self.plot_inputs = plot_inputs
        self.predict_train = predict_train
        self.predict_val = predict_val

    def determine_trigger(self, ctx):
        mode, epoch, train_step, val_step = None, None, None, None
        triggered = False
        if ctx.mode == 'train':
            mode = ctx.mode
            epoch = ctx.epoch + 1 if ctx.has('epoch') else None
            train_step = ctx.train_inner_step + 1 if ctx.has('train_inner_step') else ctx.train_step + 1
            if (epoch is None or (epoch in self.at_train_epochs)) and (train_step in self.at_train_steps):
                triggered = True
        elif ctx.mode == 'val':
            mode = ctx.mode
            epoch = ctx.epoch + 1 if ctx.has('epoch') else None
            train_step = ctx.train_inner_step + 1 if ctx.has('train_inner_step') else ctx.train_step + 1
            val_step = ctx.val_step if ctx.has('val_step') else None
            if (epoch is None or (epoch in self.at_val_epochs)) and (val_step in self.at_val_steps):
                triggered = True

        epoch_str = f'E{epoch}' if epoch is not None else ''
        train_step_str = f'TS{train_step}' if train_step is not None else ''
        val_step_str = f'VS{val_step}' if val_step is not None else ''
        tag = f'{mode}[{epoch_str}{train_step_str}{val_step_str}]'

        return triggered, Dict(mode=mode,
                               epoch=epoch,
                               train_step=train_step,
                               val_step=val_step,
                               tag=tag,
                               out_dir=self.out_dir)

    def _run_diagnoses(self, diagnoses, *args):
        for diagnosis in diagnoses:
            group, method = diagnosis.split('.')
            assert group in self._diagnosis_methods
            assert method in self._diagnosis_methods[group]
            self._diagnosis_methods[group][method](*args)

    def call_before_run(self, ctx):
        self.out_dir = osp.join(ctx.work_dir, 'diagnosis')
        mkdir(self.out_dir, exist_rm=True)
        self._run_diagnoses(self.before_run, ctx, self)

    def call_before_step(self, ctx):
        triggered, state = self.determine_trigger(ctx)
        if triggered:
            self._run_diagnoses(self.before_step, ctx, state, self)

    def call_after_step(self, ctx):
        triggered, state = self.determine_trigger(ctx)
        if triggered:
            self._run_diagnoses(self.after_step, ctx, state, self)

    @classmethod
    def register_diagnosis(cls, group, name=None, func=None):
        if callable(name):
            func = name
            name = func.__name__

        if func is not None:
            cls._register_diagnosis(group, name, func)
            return func

        def _register(func):
            cls._register_diagnosis(group, name, func)
            return func

        return _register

    @classmethod
    def _register_diagnosis(cls, group, name, func):
        if name is None:
            name = func.__name__

        cls._diagnosis_methods[group][name] = func
