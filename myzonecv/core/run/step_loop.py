from ..registry import RUNS
from .base_run import BaseRun


@RUNS.register_class('step_loop')
class StepLoop(BaseRun):
    def __init__(self, name='step_loop', stage_names=None):
        super().__init__(name, stage_names)

    def __call__(self, ctx):
        self.call_stage('run_begin', ctx)

        for _ in self.call_iter_stage('iter_steps', ctx):
            self.call_stage('step_begin', ctx)
            self.call_stage('step', ctx)
            self.call_stage('step_end', ctx)

        self.call_stage('run_end', ctx)
